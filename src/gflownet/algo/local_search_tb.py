from gflownet import GFNTask
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.data.data_source import DataSource


class LocalSearchTB(TrajectoryBalance):
    requires_task: bool = True
    task: GFNTask

    def __init__(self, env, ctx, cfg):
        super().__init__(env, ctx, cfg)
        self.task = None

    def set_task(self, task):
        self.task = task
        assert self.cfg.do_parameterize_p_b, "LocalSearchTB requires do_parameterize_p_b to be True"

    def create_training_data_from_own_samples(self, model, n, cond_info=None, random_action_prob=0.0):
        assert self.task is not None, "LocalSearchTB requires a task to be set"
        if self.global_cfg.algo.ls.yield_only_accepted:
            n_per_step = n // 2
            assert n % 2 == 0, "n must be divisible by 2"
        else:
            assert n % (1 + self.global_cfg.algo.ls.num_ls_steps) == 0, "n must be divisible by 1 + num_ls_steps"
            n_per_step = n // (1 + self.global_cfg.algo.ls.num_ls_steps)
        cond_info = {k: v[:n_per_step] for k, v in cond_info.items()} if cond_info is not None else None
        random_action_prob = random_action_prob or 0.0
        data, accept_rate = self.graph_sampler.local_search_sample_from_model(
            model,
            n_per_step,
            cond_info,
            random_action_prob,
            self.global_cfg.algo.ls,
            self._compute_log_rewards,
        )
        for t in data:
            t["accept_rate"] = accept_rate
        return data

    def _compute_log_rewards(self, trajs, cond_info):
        """Sets trajs' log_reward key by querying the task."""
        cfg = self.global_cfg
        self.cfg = cfg  # TODO: fix TB so we don't need to do this
        # Doing a bit of hijacking here to compute the log rewards, DataSource implements this for us.
        # May be worth refactoring this to be more general eventually, this depends on `self` having ctx, task, and cfg
        # attributes.
        DataSource.set_traj_cond_info(self, trajs, cond_info)  # type: ignore
        DataSource.compute_properties(self, trajs)  # type: ignore
        DataSource.compute_log_rewards(self, trajs)  # type: ignore
        self.cfg = cfg.algo.tb
        # trajs is modified in place, so no need to return anything
