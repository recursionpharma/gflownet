import copy
import warnings
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gflownet.algo.config import LSTBConfig
from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionCategorical,
    GraphActionType,
    action_type_to_mask,
)
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.utils.misc import get_worker_device, get_worker_rng


def relabel(g: Graph, ga: GraphAction):
    """Relabel the nodes for g to 0-N, and the graph action ga applied to g.
    This is necessary because torch_geometric and EnvironmentContext classes expect nodes to be
    labeled 0-N, whereas GraphBuildingEnv.parent can return parents with e.g. a removed node that
    creates a gap in 0-N, leading to a faulty encoding of the graph.
    """
    rmap = dict(zip(g.nodes, range(len(g.nodes))))
    if not len(g) and ga.action == GraphActionType.AddNode:
        rmap[0] = 0  # AddNode can add to the empty graph, the source is still 0
    g = g.relabel_nodes(rmap)
    if ga.source is not None:
        ga.source = rmap[ga.source]
    if ga.target is not None:
        ga.target = rmap[ga.target]
    return g, ga


class GraphSampler:
    """A helper class to sample from GraphActionCategorical-producing models"""

    def __init__(
        self, ctx, env, max_len, max_nodes, sample_temp=1, correct_idempotent=False, pad_with_terminal_state=False
    ):
        """
        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        sample_temp: float
            [Experimental] Softmax temperature used when sampling
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        pad_with_terminal_state: bool
            [Experimental] If true pads trajectories with a terminal
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

    def sample_from_model(self, model: nn.Module, n: int, cond_info: Optional[Tensor], random_action_prob: float = 0.0):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        random_action_prob: float
            Probability of taking a random action at each step

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - bck_a: List[GraphAction], the reverse actions
           - is_sink: List[int], sink states have P_B = 1
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
           - interm_rewards: List[float], intermediate rewards
        """
        dev = get_worker_device()
        rng = get_worker_rng()
        # This will be returned
        data = [
            {
                "traj": [],
                "bck_a": [GraphAction(GraphActionType.Pad)],  # The reverse actions
                "is_valid": True,
                "is_sink": [],
                "fwd_logprobs": [],
                "U_bck_logprobs": [],
                "interm_rewards": [],
            }
            for _ in range(n)
        ]
        graphs = [self.env.new() for _ in range(n)]
        done = [False for _ in range(n)]

        for t in range(self.max_len):
            # This modifies `data` and `graphs` in place
            self._forward_step(model, data, graphs, cond_info, t, done, rng, dev, random_action_prob)
            if all(done):
                break
        # Note on is_sink and padding:
        # is_sink indicates to a GFN algorithm that P_B(s) must be 1
        #
        # There are 3 types of possible trajectories
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop)], P_B = [..., bck(gp), 1]
        #  B - ends with an invalid action.  = [..., (g, a)],                 = [..., 1]
        #  C - ends at max_len.              = [..., (g, a)],                 = [..., bck(gp)]
        #
        # Let's say we pad terminal states, then:
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop), (gp, None)], P_B = [..., bck(gp), 1, 1]
        #  B - ends with an invalid action.  = [..., (g, a), (g, None)],                  = [..., 1, 1]
        #  C - ends at max_len.              = [..., (g, a), (gp, None)],                 = [..., bck(gp), 1]
        # and then P_F(terminal) "must" be 1

        for i in range(len(data)):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            # TODO: stop using dicts and used typed objects
            data[i]["fwd_logprobs"] = torch.stack(data[i]["fwd_logprobs"]).reshape(-1)
            data[i]["U_bck_logprobs"] = torch.stack(data[i]["U_bck_logprobs"]).reshape(-1)
            data[i]["fwd_logprob"] = data[i]["fwd_logprobs"].sum()  # type: ignore
            data[i]["U_bck_logprob"] = data[i]["U_bck_logprobs"].sum()  # type: ignore
            data[i]["result"] = graphs[i]
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Pad)))  # type: ignore
                data[i]["U_bck_logprobs"] = torch.cat([data[i]["U_bck_logprobs"], torch.tensor([0.0], device=dev)])
                data[i]["is_sink"].append(1)  # type: ignore
                assert len(data[i]["U_bck_logprobs"]) == len(data[i]["bck_a"])  # type: ignore
        return data

    def sample_backward_from_graphs(
        self,
        graphs: List[Graph],
        model: Optional[nn.Module],
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
    ):
        """Sample a model's P_B starting from a list of graphs, or if the model is None, use a uniform distribution
        over legal actions.

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated
        random_action_prob: float
            Probability of taking a random action (only used if model parameterizes P_B)

        """
        starting_graphs = list(graphs)
        dev = get_worker_device()
        n = len(graphs)
        done = [False] * n
        data = [
            {
                "traj": [(graphs[i], GraphAction(GraphActionType.Stop))],
                "is_valid": True,
                "is_sink": [1],
                "bck_a": [GraphAction(GraphActionType.Pad)],
                "bck_logprobs": [0.0],
                "U_bck_logprobs": [0.0],
                "result": graphs[i],
            }
            for i in range(n)
        ]

        # TODO: This should be doable.
        if random_action_prob > 0:
            warnings.warn("Random action not implemented for backward sampling")

        while sum(done) < n:
            self._backward_step(model, data, graphs, cond_info, done, dev)

        for i in range(n):
            # See comments in sample_from_model
            # TODO: stop using dicts and used typed objects
            data[i]["traj"] = data[i]["traj"][::-1]  # type: ignore
            # I think this pad is only necessary if we're padding terminal states???
            data[i]["bck_a"] = [GraphAction(GraphActionType.Pad)] + data[i]["bck_a"][::-1]  # type: ignore
            data[i]["is_sink"] = data[i]["is_sink"][::-1]  # type: ignore
            data[i]["U_bck_logprobs"] = torch.tensor(
                [0] + data[i]["U_bck_logprobs"][::-1], device=dev  # type: ignore
            ).reshape(-1)
            if self.pad_with_terminal_state:
                data[i]["traj"].append((starting_graphs[i], GraphAction(GraphActionType.Pad)))  # type: ignore
                data[i]["U_bck_logprobs"] = torch.cat([data[i]["U_bck_logprobs"], torch.tensor([0.0], device=dev)])
                data[i]["is_sink"].append(1)  # type: ignore
                assert len(data[i]["U_bck_logprobs"]) == len(data[i]["bck_a"])  # type: ignore
        return data

    def local_search_sample_from_model(
        self,
        model: nn.Module,
        n: int,
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
        cfg: LSTBConfig = LSTBConfig(),
        compute_reward: Optional[Callable] = None,
    ):
        dev = get_worker_device()
        rng = get_worker_rng()
        # First get n trajectories
        current_trajs = self.sample_from_model(model, n, cond_info, random_action_prob)
        compute_reward(current_trajs, cond_info)  # in-place
        # Then we're going to perform num_ls_steps of local search, each with num_bck_steps backward steps.
        # Each local search step is a kind of Metropolis-Hastings step, where we propose a new trajectory, which may
        # be accepted or rejected based on the forward and backward probabilities and reward.
        # Finally, we return all the trajectories that were sampled.

        # We keep the initial trajectories to return them at the end. We need to copy 'traj' to avoid modifying it
        initial_trajs = [{k: v if k != "traj" else list(v) for k, v in t.items()} for t in current_trajs]
        sampled_terminals = []
        if self.pad_with_terminal_state:
            for t in current_trajs:
                t["traj"] = t["traj"][:-1]  # Remove the padding state
        num_accepts = 0

        for mcmc_steps in range(cfg.num_ls_steps):
            # First we must do a bit of accounting so that we can later prevent trajectories longer than max_len
            stop = GraphActionType.Stop
            num_pad = [(1 if t["traj"][-1][1].action == stop else 0) for t in current_trajs]
            trunc_lens = [max(0, len(i["traj"]) - cfg.num_bck_steps - pad) for i, pad in zip(current_trajs, num_pad)]

            # Go backwards num_bck_steps steps
            bck_trajs = [
                {"traj": [], "bck_a": [], "is_sink": [], "bck_logprobs": [], "fwd_logprobs": []} for _ in current_trajs
            ]  # type: ignore
            graphs = [i["traj"][-1][0] for i in current_trajs]
            done = [False] * n
            fwd_a: List[GraphAction] = []
            for i in range(cfg.num_bck_steps):
                # This modifies `bck_trajs` & `graphs` in place, passing fwd_a computes P_F(s|s') for the previous step
                self._backward_step(model, bck_trajs, graphs, cond_info, done, dev, fwd_a)
                fwd_a = [t["traj"][-1][1] for t in bck_trajs]
            # Add forward logprobs for the last step
            self._add_fwd_logprobs(bck_trajs, graphs, model, cond_info, [False] * n, dev, fwd_a)
            log_P_B_tau_back = [sum(t["bck_logprobs"]) for t in bck_trajs]
            log_P_F_tau_back = [sum(t["fwd_logprobs"]) for t in bck_trajs]

            # Go forward to get full trajectories
            fwd_trajs = [
                {"traj": [], "bck_a": [], "is_sink": [], "bck_logprobs": [], "fwd_logprobs": []} for _ in current_trajs
            ]  # type: ignore
            done = [False] * n
            bck_a: List[GraphAction] = []
            while not all(done):
                self._forward_step(model, fwd_trajs, graphs, cond_info, 0, done, rng, dev, random_action_prob, bck_a)
                done = [d or (len(t["traj"]) + T) >= self.max_len for d, t, T in zip(done, fwd_trajs, trunc_lens)]
                bck_a = [t["bck_a"][-1] for t in fwd_trajs]
            # Add backward logprobs for the last step; this is only necessary if the last action is not a stop
            done = [t["traj"][-1][1].action == stop for t in fwd_trajs]
            if not all(done):
                self._add_bck_logprobs(fwd_trajs, graphs, model, cond_info, done, dev, bck_a)
            log_P_F_tau_recon = [sum(t["fwd_logprobs"]) for t in fwd_trajs]
            log_P_B_tau_recon = [sum(t["bck_logprobs"]) for t in fwd_trajs]

            # We add those new terminal states to the list of terminal states
            terminals = [t["traj"][-1][0] for t in fwd_trajs]
            sampled_terminals.extend(terminals)
            for traj, term in zip(fwd_trajs, terminals):
                traj["result"] = term
                traj["is_accept"] = False  # type: ignore
            # Compute rewards for the acceptance
            if compute_reward is not None:
                compute_reward(fwd_trajs, cond_info)

            # To end the iteration, we replace the current trajectories with the new ones if they are accepted by MH
            for i in range(n):
                if cfg.accept_criteria == "deterministic":
                    # Keep the highest reward
                    if fwd_trajs[i]["log_reward"] > current_trajs[i]["log_reward"]:
                        current_trajs[i] = fwd_trajs[i]
                        num_accepts += 1
                elif cfg.accept_criteria == "stochastic":
                    # Accept with probability max(1, R(x')/R(x)q(x'|x)/q(x|x'))
                    log_q_xprime_given_x = log_P_B_tau_back[i] + log_P_F_tau_recon[i]
                    log_q_x_given_xprime = log_P_B_tau_recon[i] + log_P_F_tau_back[i]
                    log_R_ratio = fwd_trajs[i]["log_reward"] - current_trajs[i]["log_reward"]
                    log_acceptance_ratio = log_R_ratio + log_q_xprime_given_x - log_q_x_given_xprime
                    if log_acceptance_ratio > 0 or rng.uniform() < torch.exp(log_acceptance_ratio):
                        current_trajs[i] = fwd_trajs[i]
                        num_accepts += 1
                elif cfg.accept_criteria == "always":
                    current_trajs[i] = fwd_trajs[i]
                    num_accepts += 1

        # Finally, we resample new "P_B-on-policy" trajectories from the terminal states
        # If we're only interested in the accepted trajectories, we use them as starting points instead
        if cfg.yield_only_accepted:
            sampled_terminals = [i["traj"][-1][0] for i in current_trajs]
            stacked_ci = cond_info

        if not cfg.yield_only_accepted:
            # In this scenario, the batch is n // num_ls_steps, so we do some stacking
            stacked_ci = (
                {k: cond_info[k].repeat(cfg.num_ls_steps, *((1,) * (cond_info[k].ndim - 1))) for k in cond_info}
                if cond_info is not None
                else None
            )
        returned_trajs = self.sample_backward_from_graphs(sampled_terminals, model, stacked_ci, random_action_prob)
        # TODO: modify the trajs' cond_info!!!
        return initial_trajs + returned_trajs, num_accepts / (cfg.num_ls_steps * n)

    def _forward_step(self, model, data, graphs, cond_info, t, done, rng, dev, random_action_prob, bck_a=[]) -> None:
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]] if not bck_a else lst

        n = len(data)
        # Construct graphs for the trajectories that aren't yet done
        torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
        if not bck_a:
            not_done_mask = torch.tensor(done, device=cond_info["encoding"].device).logical_not()
        else:
            not_done_mask = torch.tensor([True] * n, device=cond_info["encoding"].device)
        # Forward pass to get GraphActionCategorical
        # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
        # TODO: compute bck_cat.log_prob(bck_a) when relevant
        batch = self.ctx.collate(torch_graphs)
        batch.cond_info = cond_info["encoding"][not_done_mask] if cond_info is not None else None
        fwd_cat, bck_cat, *_ = model(batch.to(dev))
        if random_action_prob > 0:
            # Device which graphs in the minibatch will get their action randomized
            is_random_action = torch.tensor(
                rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev
            ).float()
            # Set the logits to some large value to have a uniform distribution
            fwd_cat.logits = [
                is_random_action[b][:, None] * torch.ones_like(i) * 100 + i * (1 - is_random_action[b][:, None])
                for i, b in zip(fwd_cat.logits, fwd_cat.batch)
            ]
        if self.sample_temp != 1:
            sample_cat = copy.copy(fwd_cat)
            sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
            actions = sample_cat.sample()
        else:
            actions = fwd_cat.sample()
        graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a, fwd=True) for g, a in zip(torch_graphs, actions)]
        log_probs = fwd_cat.log_prob(actions)
        if bck_a:
            aidx_bck_a = [self.ctx.GraphAction_to_ActionIndex(g, a) for g, a in zip(torch_graphs, bck_a)]
            bck_logprobs = bck_cat.log_prob(aidx_bck_a)
        # Step each trajectory, and accumulate statistics
        for i, j in zip(not_done(range(n)), range(n)):
            if bck_a and len(data[i]["bck_logprobs"]) < len(data[i]["traj"]):
                data[i]["bck_logprobs"].append(bck_logprobs[j].unsqueeze(0))
            if done[i]:
                continue
            data[i]["fwd_logprobs"].append(log_probs[j].unsqueeze(0))
            data[i]["traj"].append((graphs[i], graph_actions[j]))
            data[i]["bck_a"].append(self.env.reverse(graphs[i], graph_actions[j]))
            if "U_bck_logprobs" not in data[i]:
                data[i]["U_bck_logprobs"] = []
            # Check if we're done
            if graph_actions[j].action is GraphActionType.Stop:
                done[i] = True
                data[i]["U_bck_logprobs"].append(torch.tensor([1.0], device=dev).log())
                data[i]["is_sink"].append(1)
            else:  # If not done, try to step the self.environment
                gp = graphs[i]
                try:
                    # self.env.step can raise AssertionError if the action is illegal
                    gp = self.env.step(graphs[i], graph_actions[j])
                    assert len(gp.nodes) <= self.max_nodes
                except AssertionError:
                    done[i] = True
                    data[i]["is_valid"] = False
                    data[i]["U_bck_logprobs"].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                    continue
                if t == self.max_len - 1:
                    done[i] = True
                # If no error, add to the trajectory
                # P_B = uniform backward
                n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                data[i]["U_bck_logprobs"].append(torch.tensor([1 / n_back], device=dev).log())
                data[i]["is_sink"].append(0)
                graphs[i] = gp
            if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                # check if the graph is sane (e.g. RDKit can  construct a molecule from it) otherwise
                # treat the done action as illegal
                data[i]["is_valid"] = False
        # Nothing is returned, data is modified in place

    def _backward_step(self, model, data, graphs, cond_info, done, dev, fwd_a=[]):
        # fwd_a is a list of GraphActions that are the reverse of the last backwards actions we took.
        # Passing them allows us to compute the forward logprobs of the actions we took.
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]] if not fwd_a else lst

        torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(len(graphs)))]
        if not fwd_a:
            not_done_mask = torch.tensor(done, device=cond_info["encoding"].device).logical_not()
        else:
            not_done_mask = torch.tensor([True] * len(graphs), device=cond_info["encoding"].device)
        if model is not None:
            gbatch = self.ctx.collate(torch_graphs)
            gbatch.cond_info = cond_info["encoding"][not_done_mask] if cond_info is not None else None
            fwd_cat, bck_cat, *_ = model(gbatch.to(dev))
        else:
            gbatch = self.ctx.collate(torch_graphs)
            action_types = self.ctx.bck_action_type_order
            action_masks = [action_type_to_mask(t, gbatch, assert_mask_exists=True) for t in action_types]
            bck_cat = GraphActionCategorical(
                gbatch,
                raw_logits=[torch.ones_like(m) for m in action_masks],
                keys=[GraphTransformerGFN.action_type_to_key(t) for t in action_types],
                action_masks=action_masks,
                types=action_types,
            )
        bck_actions = bck_cat.sample()
        graph_bck_actions = [
            self.ctx.ActionIndex_to_GraphAction(g, a, fwd=False) for g, a in zip(torch_graphs, bck_actions)
        ]
        bck_logprobs = bck_cat.log_prob(bck_actions)
        if fwd_a and model is not None:
            aidx_fwd_a = [self.ctx.GraphAction_to_ActionIndex(g, a) for g, a in zip(torch_graphs, fwd_a)]
            fwd_logprobs = fwd_cat.log_prob(aidx_fwd_a)

        for i, j in zip(not_done(range(len(graphs))), range(len(graphs))):
            if fwd_a and model is not None and len(data[i]["fwd_logprobs"]) < len(data[i]["traj"]):
                data[i]["fwd_logprobs"].append(fwd_logprobs[j].item())
            if done[i]:
                # This can happen when fwd_a is passed, we should optimize this though. The reason is that even
                # if a graph is done, we may still want to compute its forward logprobs.
                continue
            g = graphs[i]
            b_a = graph_bck_actions[j]
            gp = self.env.step(g, b_a)
            f_a = self.env.reverse(g, b_a)
            graphs[i], f_a = relabel(gp, f_a)
            data[i]["traj"].append((graphs[i], f_a))
            data[i]["bck_a"].append(b_a)
            data[i]["is_sink"].append(0)
            data[i]["bck_logprobs"].append(bck_logprobs[j].item())

            if len(graphs[i]) == 0:
                done[i] = True
            if "U_bck_logprobs" not in data[i]:
                data[i]["U_bck_logprobs"] = []
            if not done[i]:
                n_back = self.env.count_backward_transitions(graphs[i], check_idempotent=self.correct_idempotent)
                data[i]["U_bck_logprobs"].append(torch.tensor([1.0 / n_back], device=dev).log())

    def _add_fwd_logprobs(self, data, graphs, model, cond_info, done, dev, fwd_a):
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(len(graphs)))]
        not_done_mask = torch.tensor(done, device=cond_info["encoding"].device).logical_not()
        gbatch = self.ctx.collate(torch_graphs)
        gbatch.cond_info = cond_info["encoding"][not_done_mask] if cond_info is not None else None
        fwd_cat, *_ = model(gbatch.to(dev))
        fwd_actions = [self.ctx.GraphAction_to_ActionIndex(g, a) for g, a in zip(torch_graphs, fwd_a)]
        log_probs = fwd_cat.log_prob(fwd_actions)
        for i, j in zip(not_done(range(len(graphs))), range(len(graphs))):
            data[i]["fwd_logprobs"].append(log_probs[j].item())

    def _add_bck_logprobs(self, data, graphs, model, cond_info, done, dev, bck_a):
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(len(graphs)))]
        not_done_mask = torch.tensor(done, device=cond_info["encoding"].device).logical_not()
        gbatch = self.ctx.collate(torch_graphs)
        gbatch.cond_info = cond_info["encoding"][not_done_mask] if cond_info is not None else None
        fwd_cat, bck_cat, *_ = model(gbatch.to(dev))
        bck_actions = [self.ctx.GraphAction_to_ActionIndex(g, a) for g, a in zip(torch_graphs, bck_a)]
        log_probs = bck_cat.log_prob(bck_actions)
        for i, j in zip(not_done(range(len(graphs))), range(len(graphs))):
            data[i]["bck_logprobs"].append(log_probs[j].item())
