from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.data as gd
from torch_scatter import scatter
from torch_scatter import scatter_sum

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.envs.graph_building_env import generate_forward_trajectory
from gflownet.envs.graph_building_env import Graph
from gflownet.envs.graph_building_env import GraphAction
from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.graph_building_env import GraphActionType
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
from gflownet.algo.trajectory_balance import TrajectoryBalance


def relabel(ga, g):
    rmap = dict(zip(g.nodes, range(len(g.nodes))))
    if not len(g) and ga.action == GraphActionType.AddNode:
        rmap[0] = 0  # AddNode can add to the empty graph, the source is still 0
    g = nx.relabel_nodes(g, rmap)
    if ga.source is not None:
        ga.source = rmap[ga.source]
    if ga.target is not None:
        ga.target = rmap[ga.target]
    return ga, g


class FlowMatching(TrajectoryBalance):
    def __init__(self, env: GraphBuildingEnv, ctx: GraphBuildingEnvContext, rng: np.random.RandomState,
                 hps: Dict[str, Any], max_len=None, max_nodes=None):
        super().__init__(env, ctx, rng, hps, max_len=max_len, max_nodes=max_nodes)
        self.fm_epsilon = torch.as_tensor(hps.get('fm_epsilon', 1e-38)).log()
        assert ctx.action_type_order.index(GraphActionType.Stop) == 0

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
        parents = [[relabel(*i) for i in self.env.parents(i[0])] for tj in trajs for i in tj['traj'][1:]]
        parent_graphs = [self.ctx.graph_to_Data(pstate) for parent in parents for pact, pstate in parent]
        parent_actions = [pact for parent in parents for pact, pstate in parent]
        parent_actionidcs = [self.ctx.GraphAction_to_aidx(gdata, a) for gdata, a in zip(parent_graphs, parent_actions)]
        state_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj['traj'][1:]]
        batch = self.ctx.collate(parent_graphs + state_graphs)
        batch.num_parents = torch.tensor([len(i) for i in parents])
        batch.traj_lens = torch.tensor([len(i['traj']) for i in trajs])
        batch.parent_acts = parent_actionidcs
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get('is_valid', True) for i in trajs]).float()
        if self.correct_idempotent:
            raise ValueError('Not implemented')
        return batch

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        dev = batch.x.device
        eps = self.fm_epsilon.to(dev)
        num_trajs = len(batch.log_rewards)
        num_states = int(batch.num_parents.shape[0])
        total_num_parents = batch.num_parents.sum()
        states_batch_idx = torch.arange(num_states, device=dev)
        parents_batch_idx = states_batch_idx.repeat_interleave(batch.num_parents)
        states_traj_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens - 1)
        parents_traj_idx = states_traj_idx.repeat_interleave(batch.num_parents)

        num_parents_per_traj = scatter(batch.num_parents, states_traj_idx, 0, reduce='sum')
        first_graph_idx = torch.cumsum(
            torch.cat([torch.zeros_like(num_parents_per_traj[0])[None], num_parents_per_traj]), 0)
        final_graph_idx = torch.cumsum(batch.traj_lens - 1, 0) + total_num_parents - 1
        cat, graph_out = model(batch, batch.cond_info[torch.cat([parents_traj_idx, states_traj_idx], 0)])
        parent_log_F_sa = cat.log_prob(batch.parent_acts, cat.logits, torch.arange(total_num_parents, device=dev))
        log_inflows = scatter(parent_log_F_sa.exp(), parents_batch_idx, 0).log()
        all_log_outflows = cat.logsumexp()
        log_outflows = all_log_outflows[total_num_parents:]
        intermediate_loss = (torch.logaddexp(log_inflows, eps) - torch.logaddexp(log_outflows, eps)).pow(2).mean()

        fake_stop_actions = torch.zeros((num_trajs, 3), dtype=torch.long, device=dev)
        log_F_s_stop = cat.log_prob(fake_stop_actions, cat.logits, final_graph_idx)
        terminal_loss = (torch.logaddexp(log_F_s_stop, eps) - torch.logaddexp(batch.log_rewards, eps)).pow(2).mean()
        loss = intermediate_loss + terminal_loss
        logZ = all_log_outflows[first_graph_idx]
        info = {
            'intermediate_loss': intermediate_loss.item(),
            'terminal_loss': terminal_loss.item(),
            'loss': loss.item(),
            'logZ': logZ.mean().item(),
        }
        return loss, info
