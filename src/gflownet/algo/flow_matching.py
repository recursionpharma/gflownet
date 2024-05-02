import networkx as nx
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch_scatter import scatter

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.config import Config
from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)


def relabel(ga: GraphAction, g: Graph):
    """Relabel the nodes for g to 0-N, and the graph action ga applied to g.

    This is necessary because torch_geometric and EnvironmentContext classes expect nodes to be
    labeled 0-N, whereas GraphBuildingEnv.parent can return parents with e.g. a removed node that
    creates a gap in 0-N, leading to a faulty encoding of the graph.
    """
    rmap = dict(zip(g.nodes, range(len(g.nodes))))
    if not len(g) and ga.action == GraphActionType.AddNode:
        rmap[0] = 0  # AddNode can add to the empty graph, the source is still 0
    g = nx.relabel_nodes(g, rmap)
    if ga.source is not None:
        ga.source = rmap[ga.source]
    if ga.target is not None:
        ga.target = rmap[ga.target]
    return ga, g


class FlowMatching(TrajectoryBalance):  # TODO: FM inherits from TB but we could have a generic GFNAlgorithm class
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        super().__init__(env, ctx, cfg)
        self.fm_epsilon = torch.as_tensor(cfg.algo.fm.epsilon).log()
        # We include the "balanced loss" as a possibility to reproduce results from the FM paper, but
        # in a number of settings the regular loss is more stable.
        self.fm_balanced_loss = cfg.algo.fm.balanced_loss
        self.fm_leaf_coef = cfg.algo.fm.leaf_coef
        self.correct_idempotent: bool = self.correct_idempotent or cfg.algo.fm.correct_idempotent

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed reward (e.g. log(R(x) ** beta)) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        if not self.correct_idempotent:
            # For every s' (i.e. every state except the first of each trajectory), enumerate parents
            parents = [[relabel(*i) for i in self.env.parents(i[0])] for tj in trajs for i in tj["traj"][1:]]
            # convert parents to Data
            parent_graphs = [self.ctx.graph_to_Data(pstate) for parent in parents for pact, pstate in parent]
        else:
            # Here we again enumerate parents
            states = [i[0] for tj in trajs for i in tj["traj"][1:]]
            base_parents = [[relabel(*i) for i in self.env.parents(i)] for i in states]
            base_parent_graphs = [
                [self.ctx.graph_to_Data(pstate) for pact, pstate in parent_set] for parent_set in base_parents
            ]
            parents = []
            parent_graphs = []
            for state, parent_set, parent_set_graphs in zip(states, base_parents, base_parent_graphs):
                new_parent_set = []
                new_parent_graphs = []
                # But for each parent we add all the possible (action, parent) pairs to the sets of parents
                for (ga, p), pd in zip(parent_set, parent_set_graphs):
                    ipa = self.get_idempotent_actions(p, pd, state, ga, return_aidx=False)
                    new_parent_set += [(a, p) for a in ipa]
                    new_parent_graphs += [pd] * len(ipa)
                parents.append(new_parent_set)
                parent_graphs += new_parent_graphs
            # Implementation Note: no further correction is required for environments where episodes
            # always end in a Stop action. If this is not the case, then this implementation is
            # incorrect in that it doesn't account for the multiple ways that one could reach the
            # terminal state (because it assumes that a terminal state has only one parent and gives
            # 100% of the reward-flow to the edge between that parent and the terminal state, which
            # for stop actions is correct). Notably, this error will happen in environments where
            # there are invalid states that make episodes end prematurely (when those invalid states
            # have multiple possible parents).

        # convert actions to ActionIndex
        parent_actions = [pact for parent in parents for pact, pstate in parent]
        parent_actionidxs = [
            self.ctx.GraphAction_to_ActionIndex(gdata, a) for gdata, a in zip(parent_graphs, parent_actions)
        ]
        # convert state to Data
        state_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"][1:]]
        terminal_actions = [
            self.ctx.GraphAction_to_ActionIndex(self.ctx.graph_to_Data(tj["traj"][-1][0]), tj["traj"][-1][1])
            for tj in trajs
        ]

        # Create a batch from [*parents, *states]. This order will make it easier when computing the loss
        batch = self.ctx.collate(parent_graphs + state_graphs)
        batch.num_parents = torch.tensor([len(i) for i in parents])
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.parent_acts = torch.tensor(parent_actionidxs)
        batch.terminal_acts = torch.tensor(terminal_actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        if self.correct_idempotent:
            raise ValueError("Not implemented")
        return batch

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        dev = batch.x.device
        eps = self.fm_epsilon.to(dev)
        # Compute relevant quantities
        num_trajs = len(batch.log_rewards)
        num_states = int(batch.num_parents.shape[0])
        total_num_parents = batch.num_parents.sum()
        # Compute, for every parent, the index of the state it corresponds to (all states are
        # considered numbered 0..N regardless of which trajectory they correspond to)
        parents_state_idx = torch.arange(num_states, device=dev).repeat_interleave(batch.num_parents)
        # Compute, for every state, the index of the trajectory it corresponds to
        states_traj_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens - 1)
        # Idem for parents
        parents_traj_idx = states_traj_idx.repeat_interleave(batch.num_parents)
        # Compute the index of the first graph of every trajectory via a cumsum of the trajectory
        # lengths. This works because by design the first parent of every trajectory is s0 (i.e. s1
        # only has one parent that is s0)
        num_parents_per_traj = scatter(batch.num_parents, states_traj_idx, 0, reduce="sum")
        first_graph_idx = torch.cumsum(
            torch.cat([torch.zeros_like(num_parents_per_traj[0])[None], num_parents_per_traj]), 0
        )
        # Similarly we want the index of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens - 1, 0) + total_num_parents - 1

        # Query the model for Fsa. The model will output a GraphActionCategorical, but we will
        # simply interpret the logits as F(s, a). Conveniently the policy of a GFN is the softmax of
        # log F(s,a) so we don't have to change anything in the sampling routines.
        cat, graph_out = model(batch, batch.cond_info[torch.cat([parents_traj_idx, states_traj_idx], 0)])
        # We compute \sum_{s,a : T(s,a)=s'} F(s,a), first we index all the parent's outputs by the
        # parent actions. To do so we reuse the log_prob mechanism, but specify that the logprobs
        # tensor is actually just the logits (which we chose to interpret as edge flows F(s,a). We
        # only need the parent's outputs so we specify those batch indices.
        parent_log_F_sa = cat.log_prob(
            batch.parent_acts, logprobs=cat.logits, batch=torch.arange(total_num_parents, device=dev)
        )
        # The inflows is then simply the sum reduction of exponentiating the log edge flows. The
        # indices are the state index that each parent belongs to.
        log_inflows = scatter(parent_log_F_sa.exp(), parents_state_idx, 0, reduce="sum").log()
        # To compute the outflows we can just logsumexp the log F(s,a) predictions. We do so for the
        # entire batch, which is slightly wasteful (TODO). We only take the last outflows here, and
        # later take the log outflows of s0 to estimate logZ.
        all_log_outflows = cat.logsumexp()
        log_outflows = all_log_outflows[total_num_parents:]

        # The loss of intermediary states is inflow - outflow. We use the log-epsilon variant (see FM paper)
        intermediate_loss = (torch.logaddexp(log_inflows, eps) - torch.logaddexp(log_outflows, eps)).pow(2)
        # To compute the loss of the terminal states we match F(s, a'), where a' is the action that
        # terminated the trajectory, to R(s). We again use the mechanism of log_prob
        log_F_s_stop = cat.log_prob(batch.terminal_acts, cat.logits, final_graph_idx)
        terminal_loss = (torch.logaddexp(log_F_s_stop, eps) - torch.logaddexp(batch.log_rewards, eps)).pow(2)

        if self.fm_balanced_loss:
            loss = intermediate_loss.mean() + terminal_loss.mean() * self.fm_leaf_coef
        else:
            loss = (intermediate_loss.sum() + terminal_loss.sum()) / (
                intermediate_loss.shape[0] + terminal_loss.shape[0]
            )

        # logZ is simply the outflow of s0, the first graph of each parent set.
        logZ = all_log_outflows[first_graph_idx]
        info = {
            "intermediate_loss": intermediate_loss.mean().item(),
            "terminal_loss": terminal_loss.mean().item(),
            "loss": loss.item(),
            "logZ": logZ.mean().item(),
        }
        return loss, info
