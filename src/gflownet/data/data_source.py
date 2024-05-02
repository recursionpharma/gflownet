import warnings
from typing import Callable, Generator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from gflownet import GFNAlgorithm, GFNTask
from gflownet.config import Config
from gflownet.data.replay_buffer import ReplayBuffer, detach_and_cpu
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
from gflownet.utils.misc import get_worker_rng


def cycle_call(it):
    while True:
        for i in it():
            yield i


class DataSource(IterableDataset):
    def __init__(
        self,
        cfg: Config,
        ctx: GraphBuildingEnvContext,
        algo: GFNAlgorithm,
        task: GFNTask,
        replay_buffer: Optional[ReplayBuffer] = None,
        is_algo_eval: bool = False,
        start_at_step: int = 0,
    ):
        """A DataSource mixes multiple iterators into one. These are created with do_* methods."""
        self.iterators: List[Generator] = []
        self.cfg = cfg
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.replay_buffer = replay_buffer
        self.is_algo_eval = is_algo_eval
        self.sampling_hooks: List[Callable] = []
        self.active = True

        self.global_step_count = torch.zeros(1, dtype=torch.int64) + start_at_step
        self.global_step_count.share_memory_()
        self.global_step_count_lock = torch.multiprocessing.Lock()
        self.current_iter = start_at_step

    def add_sampling_hook(self, hook: Callable):
        """Add a hook that is called when sampling new trajectories.

        The hook should take a list of trajectories as input.
        The hook will not be called on trajectories that are sampled from the replay buffer or dataset.
        """
        self.sampling_hooks.append(hook)
        return self

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        self.rng = get_worker_rng()
        its = [i() for i in self.iterators]
        self.algo.set_is_eval(self.is_algo_eval)
        while True:
            with self.global_step_count_lock:
                self.current_iter = self.global_step_count.item()
                self.global_step_count += 1
            iterator_outputs = [next(i, None) for i in its]
            if any(i is None for i in iterator_outputs):
                if not all(i is None for i in iterator_outputs):
                    warnings.warn("Some iterators are done, but not all. You may be mixing incompatible iterators.")
                    iterator_outputs = [i for i in iterator_outputs if i is not None]
                else:
                    break
            traj_lists, batch_infos = zip(*iterator_outputs)
            trajs = sum(traj_lists, [])
            # Merge all the dicts into one
            batch_info = {}
            for d in batch_infos:
                batch_info.update(d)
            yield self.create_batch(trajs, batch_info)

    def do_sample_model(self, model, num_from_policy, num_new_replay_samples=None):
        if num_new_replay_samples is not None:
            assert self.replay_buffer is not None, "num_new_replay_samples specified without a replay buffer"
        if num_new_replay_samples is None:
            assert self.replay_buffer is None, "num_new_replay_samples not specified with a replay buffer"

        num_new_replay_samples = num_new_replay_samples or 0
        num_samples = max(num_from_policy, num_new_replay_samples)

        def iterator():
            while self.active:
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                # TODO: in the cond info refactor, pass the whole thing instead of just the encoding
                trajs = self.algo.create_training_data_from_own_samples(model, num_samples, cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                self.send_to_replay(trajs[:num_new_replay_samples])  # This is a no-op if there is no replay buffer
                batch_info = self.call_sampling_hooks(trajs)
                yield (trajs[:num_from_policy], batch_info)

        self.iterators.append(iterator)
        return self

    def do_sample_model_n_times(self, model, num_samples_per_batch, num_total):
        total = torch.zeros(1, dtype=torch.int64)
        total.share_memory_()
        total_lock = torch.multiprocessing.Lock()
        total_barrier = torch.multiprocessing.Barrier(max(1, self.cfg.num_workers))

        def iterator():
            while self.active:
                with total_lock:
                    n_so_far = total.item()
                    n_this_time = min(num_total - n_so_far, num_samples_per_batch)
                    total[:] += n_this_time
                    if n_this_time == 0:
                        break
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(n_this_time, t)
                # TODO: in the cond info refactor, pass the whole thing instead of just the encoding
                trajs = self.algo.create_training_data_from_own_samples(model, n_this_time, cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                batch_info = self.call_sampling_hooks(trajs)
                yield trajs, batch_info
            total_barrier.wait()  # Wait for all workers to finish before resetting the counter
            total[:] = 0

        self.iterators.append(iterator)
        return self

    def do_sample_replay(self, num_samples):
        def iterator():
            while self.active:
                trajs, *_ = self.replay_buffer.sample(num_samples)
                self.relabel_in_hindsight(trajs)  # This is a no-op if the hindsight ratio is 0
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def do_dataset_in_order(self, data, num_samples, backwards_model):
        def iterator():
            for idcs in self.iterate_indices(len(data), num_samples):
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                objs, props = map(list, zip(*[data[i] for i in idcs])) if len(idcs) else ([], [])
                trajs = self.algo.create_training_data_from_graphs(objs, backwards_model, cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.set_traj_props(trajs, props)
                self.compute_log_rewards(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def do_conditionals_dataset_in_order(self, data, num_samples, model):
        def iterator():
            for idcs in self.iterate_indices(len(data), num_samples):
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                # TODO: when we refactor cond_info, data[i] will probably be a dict? (or CondInfo objects)
                # I'm also not a fan of encode_conditional_information, it assumes lots of things about what's passed to
                # it and the state of the program (e.g. validation mode)
                cond_info = self.task.encode_conditional_information(torch.stack([data[i] for i in idcs]))
                trajs = self.algo.create_training_data_from_own_samples(model, len(idcs), cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                self.send_to_replay(trajs)  # This is a no-op if there is no replay buffer
                # If we're using a dataset of preferences, the user/hooks may want to know the id of the preference
                for i, j in zip(trajs, idcs):
                    i["data_idx"] = j
                batch_info = self.call_sampling_hooks(trajs)
                yield trajs, batch_info

        self.iterators.append(iterator)
        return self

    def do_sample_dataset(self, data, num_samples, backwards_model):
        def iterator():
            while self.active:
                idcs = self.sample_idcs(len(data), num_samples)
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                objs, props = map(list, zip(*[data[i] for i in idcs])) if len(idcs) else ([], [])
                trajs = self.algo.create_training_data_from_graphs(objs, backwards_model, cond_info["encoding"], p)
                self.set_traj_cond_info(trajs, cond_info)  # Attach the cond info to the trajs
                self.set_traj_props(trajs, props)
                self.compute_log_rewards(trajs)
                yield trajs, {}

        self.iterators.append(iterator)
        return self

    def call_sampling_hooks(self, trajs):
        batch_info = {}
        # TODO: just pass trajs to the hooks and deprecate passing all those arguments
        obj_props = torch.stack([t["obj_props"] for t in trajs])
        # convert cond_info back to a dict
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        rewards = torch.exp(log_rewards / (cond_info.get("beta", 1)))
        for hook in self.sampling_hooks:
            batch_info.update(hook(trajs, rewards, obj_props, cond_info))
        return batch_info

    def create_batch(self, trajs, batch_info):
        trajs = detach_and_cpu(trajs)
        ci = torch.stack([t["cond_info"]["encoding"] for t in trajs])
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        batch = self.algo.construct_batch(trajs, ci, log_rewards)
        batch.num_online = sum(t.get("is_online", 0) for t in trajs)
        batch.num_offline = len(trajs) - batch.num_online
        batch.extra_info = batch_info
        if "preferences" in trajs[0]["cond_info"].keys():
            batch.preferences = torch.stack([t["cond_info"]["preferences"] for t in trajs])
        if "focus_dir" in trajs[0]["cond_info"].keys():
            batch.focus_dir = torch.stack([t["cond_info"]["focus_dir"] for t in trajs])

        if self.ctx.has_n() and self.cfg.algo.tb.do_predict_n:
            log_ns = [self.ctx.traj_log_n(i["traj"]) for i in trajs]
            batch.log_n = torch.tensor([i[-1] for i in log_ns], dtype=torch.float32)
            batch.log_ns = torch.tensor(sum(log_ns, start=[]), dtype=torch.float32)
        batch.obj_props = torch.stack([t["obj_props"] for t in trajs])
        return batch

    def compute_properties(self, trajs, mark_as_online=False):
        """Sets trajs' obj_props and is_valid keys by querying the task."""
        # TODO: refactor obj_props into properties
        valid_idcs = torch.tensor([i for i in range(len(trajs)) if trajs[i].get("is_valid", True)]).long()
        # fetch the valid trajectories endpoints
        objs = [self.ctx.graph_to_obj(trajs[i]["result"]) for i in valid_idcs]
        # ask the task to compute their reward
        # TODO: it's really weird that the task is responsible for this and returns a obj_props
        # tensor whose first dimension is possibly not the same as the output???
        obj_props, m_is_valid = self.task.compute_obj_properties(objs)
        assert obj_props.ndim == 2, "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
        # The task may decide some of the objs are invalid, we have to again filter those
        valid_idcs = valid_idcs[m_is_valid]
        all_fr = torch.zeros((len(trajs), obj_props.shape[1]))
        all_fr[valid_idcs] = obj_props
        for i in range(len(trajs)):
            trajs[i]["obj_props"] = all_fr[i]
            trajs[i]["is_online"] = mark_as_online
        # Override the is_valid key in case the task made some objs invalid
        for i in valid_idcs:
            trajs[i]["is_valid"] = True

    def compute_log_rewards(self, trajs):
        """Sets trajs' log_reward key by querying the task."""
        obj_props = torch.stack([t["obj_props"] for t in trajs])
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        log_rewards = self.task.cond_info_to_logreward(cond_info, obj_props)
        min_r = torch.as_tensor(self.cfg.algo.illegal_action_logreward).float()
        for i in range(len(trajs)):
            trajs[i]["log_reward"] = log_rewards[i] if trajs[i].get("is_valid", True) else min_r

    def send_to_replay(self, trajs):
        if self.replay_buffer is not None:
            for t in trajs:
                self.replay_buffer.push(t, t["log_reward"], t["obj_props"], t["cond_info"], t["is_valid"])

    def set_traj_cond_info(self, trajs, cond_info):
        for i in range(len(trajs)):
            trajs[i]["cond_info"] = {k: cond_info[k][i] for k in cond_info}

    def set_traj_props(self, trajs, props):
        for i in range(len(trajs)):
            trajs[i]["obj_props"] = props[i]  # TODO: refactor

    def relabel_in_hindsight(self, trajs):
        if self.cfg.replay.hindsight_ratio == 0:
            return
        assert hasattr(
            self.task, "relabel_condinfo_and_logrewards"
        ), "Hindsight requires the task to implement relabel_condinfo_and_logrewards"
        # samples indexes of trajectories without repeats
        hindsight_idxs = torch.randperm(len(trajs))[: int(len(trajs) * self.cfg.replay.hindsight_ratio)]
        log_rewards = torch.stack([t["log_reward"] for t in trajs])
        obj_props = torch.stack([t["obj_props"] for t in trajs])
        cond_info = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        cond_info, log_rewards = self.task.relabel_condinfo_and_logrewards(
            cond_info, log_rewards, obj_props, hindsight_idxs
        )
        self.set_traj_cond_info(trajs, cond_info)
        for i in range(len(trajs)):
            trajs[i]["log_reward"] = log_rewards[i]

    def sample_idcs(self, n, num_samples):
        return self.rng.choice(n, num_samples, replace=False)

    def iterate_indices(self, n, num_samples):
        worker_info = torch.utils.data.get_worker_info()
        if n == 0:
            # Should we be raising an error here? warning?
            yield np.arange(0, 0)
            return

        if worker_info is None:  # no multi-processing
            start, end, wid = 0, n, -1
        else:  # split the data into chunks (per-worker)
            nw = worker_info.num_workers
            wid = worker_info.id
            start, end = int(np.round(n / nw * wid)), int(np.round(n / nw * (wid + 1)))

        if end - start <= num_samples:
            yield np.arange(start, end)
            return
        for i in range(start, end - num_samples, num_samples):
            yield np.arange(i, i + num_samples)
        if i + num_samples < end:
            yield np.arange(i + num_samples, end)
