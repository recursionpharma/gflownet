import copy
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

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
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for _ in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for _ in range(n)]

        graphs = [self.env.new() for _ in range(n)]
        done = [False for _ in range(n)]
        # TODO: instead of padding with Stop, we could have a virtual action whose probability
        # always evaluates to 1. Presently, Stop should convert to a (0,0,0) ActionIndex, which should
        # always be at least a valid index, and will be masked out anyways -- but this isn't ideal.
        # Here we have to pad the backward actions with something, since the backward actions are
        # evaluated at s_{t+1} not s_t.
        bck_a = [[GraphAction(GraphActionType.Stop)] for _ in range(n)]

        rng = get_worker_rng()

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
            # TODO: compute bck_cat.log_prob(bck_a) when relevant
            ci = cond_info[not_done_mask] if cond_info is not None else None
            fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), ci)
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
            graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = fwd_cat.log_prob(actions)
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                bck_a[i].append(self.env.reverse(graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
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
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        data[i]["is_sink"].append(1)
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    data[i]["is_sink"].append(0)
                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can  construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]["is_valid"] = False
            if all(done):
                break

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1

        # There are 3 types of possible trajectories
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop)], P_B = [..., bck(gp), 1]
        #  B - ends with an invalid action.  = [..., (g, a)],                 = [..., 1]
        #  C - ends at max_len.              = [..., (g, a)],                 = [..., bck(gp)]

        # Let's say we pad terminal states, then:
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop), (gp, None)], P_B = [..., bck(gp), 1, 1]
        #  B - ends with an invalid action.  = [..., (g, a), (g, None)],                  = [..., 1, 1]
        #  C - ends at max_len.              = [..., (g, a), (gp, None)],                 = [..., bck(gp), 1]
        # and then P_F(terminal) "must" be 1

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                # TODO: instead of padding with Stop, we could have a virtual action whose
                # probability always evaluates to 1.
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
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
        dev = get_worker_device()
        n = len(graphs)
        done = [False] * n
        data = [
            {
                "traj": [(graphs[i], GraphAction(GraphActionType.Stop))],
                "is_valid": True,
                "is_sink": [1],
                "bck_a": [GraphAction(GraphActionType.Stop)],
                "bck_logprobs": [0.0],
                "result": graphs[i],
            }
            for i in range(n)
        ]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        # TODO: This should be doable.
        if random_action_prob > 0:
            warnings.warn("Random action not implemented for backward sampling")

        while sum(done) < n:
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            if model is not None:
                ci = cond_info[not_done_mask] if cond_info is not None else None
                _, bck_cat, *_ = model(self.ctx.collate(torch_graphs).to(dev), ci)
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

            for i, j in zip(not_done(range(n)), range(n)):
                if not done[i]:
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

        for i in range(n):
            # See comments in sample_from_model
            data[i]["traj"] = data[i]["traj"][::-1]
            data[i]["bck_a"] = [GraphAction(GraphActionType.Stop)] + data[i]["bck_a"][::-1]
            data[i]["is_sink"] = data[i]["is_sink"][::-1]
            data[i]["bck_logprobs"] = torch.tensor(data[i]["bck_logprobs"][::-1], device=dev).reshape(-1)
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data
