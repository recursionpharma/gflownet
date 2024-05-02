import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.config import Config
from gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    generate_forward_trajectory,
)
from gflownet.models.graph_transformer import GraphTransformer, mlp
from gflownet.trainer import GFNTask
from gflownet.utils.misc import get_worker_device

from .graph_sampling import GraphSampler


# Custom models are necessary for envelope Q Learning
class GraphTransformerFragEnvelopeQL(nn.Module):
    """GraphTransformer class for an EnvelopeQLearning agent

    Outputs Qs for the following actions
    - Stop
    - AddNode
    - SetEdgeAttr
    """

    def __init__(self, env_ctx, num_emb=64, num_layers=3, num_heads=2, num_objectives=2):
        super().__init__()
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        num_final = num_emb * 2
        num_mlp_layers = 0
        self.emb2add_node = mlp(num_final, num_emb, env_ctx.num_new_node_values * num_objectives, num_mlp_layers)
        # Edge attr logits are "sided", so we will compute both sides independently
        self.emb2set_edge_attr = mlp(
            num_emb + num_final, num_emb, env_ctx.num_edge_attr_logits // 2 * num_objectives, num_mlp_layers
        )
        self.emb2stop = mlp(num_emb * 3, num_emb, num_objectives, num_mlp_layers)
        self.emb2reward = mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.edge2emb = mlp(num_final, num_emb, num_emb, num_mlp_layers)
        self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)
        self.action_type_order = env_ctx.action_type_order
        self.mask_value = -10
        self.num_objectives = num_objectives

    def forward(self, g: gd.Batch, cond: torch.Tensor, output_Qs=False):
        """See `GraphTransformer` for argument values"""
        node_embeddings, graph_embeddings = self.transf(g, cond)
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        edge_emb = self.edge2emb(node_embeddings[e_row] + node_embeddings[e_col])
        src_anchor_logits = self.emb2set_edge_attr(torch.cat([edge_emb, node_embeddings[e_row]], 1))
        dst_anchor_logits = self.emb2set_edge_attr(torch.cat([edge_emb, node_embeddings[e_col]], 1))

        cat = GraphActionCategorical(
            g,
            raw_logits=[
                F.relu(self.emb2stop(graph_embeddings)),
                F.relu(self.emb2add_node(node_embeddings)),
                F.relu(torch.cat([src_anchor_logits, dst_anchor_logits], 1)),
            ],
            action_masks=[
                1,
                g.add_node_mask.repeat(1, self.num_objectives),
                g.set_edge_attr_mask.repeat(1, self.num_objectives),
            ],
            keys=[None, "x", "edge_index"],
            types=self.action_type_order,
        )
        r_pred = self.emb2reward(graph_embeddings)
        if output_Qs:
            return cat, r_pred

        else:
            # Compute the greedy policy
            # See algo.envelope_q_learning.EnvelopeQLearning.compute_batch_losses for further explanations
            # TODO: this makes assumptions about how conditional vectors are created! Not robust to upstream changes
            w = cond[:, -self.num_objectives :]
            w_dot_Q = [
                (qi.reshape((qi.shape[0], qi.shape[1] // w.shape[1], w.shape[1])) * w[b][:, None, :]).sum(2)
                for qi, b in zip(cat.logits, cat.batch)
            ]
            cat.action_masks = [1, g.add_node_mask.cpu(), g.set_edge_attr_mask.cpu()]
            # Set the softmax distribution to a very low temperature to make sure only the max gets
            # sampled (and we get random argmax tie breaking for free!):
            cat.logits = [i * 100 for i in w_dot_Q]
            return cat, r_pred


class GraphTransformerEnvelopeQL(nn.Module):
    def __init__(self, env_ctx, num_emb=64, num_layers=3, num_heads=2, num_objectives=2):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        num_final = num_emb * 2
        num_mlp_layers = 0
        self.emb2add_edge = mlp(num_final, num_emb, num_objectives, num_mlp_layers)
        self.emb2add_node = mlp(num_final, num_emb, env_ctx.num_new_node_values * num_objectives, num_mlp_layers)
        self.emb2set_node_attr = mlp(num_final, num_emb, env_ctx.num_node_attr_logits * num_objectives, num_mlp_layers)
        self.emb2set_edge_attr = mlp(num_final, num_emb, env_ctx.num_edge_attr_logits * num_objectives, num_mlp_layers)
        self.emb2stop = mlp(num_emb * 3, num_emb, num_objectives, num_mlp_layers)
        self.emb2reward = mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)
        self.action_type_order = env_ctx.action_type_order
        self.num_objectives = num_objectives

    def forward(self, g: gd.Batch, cond: torch.Tensor, output_Qs=False):
        node_embeddings, graph_embeddings = self.transf(g, cond)
        ne_row, ne_col = g.non_edge_index
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        cat = GraphActionCategorical(
            g,
            raw_logits=[
                self.emb2stop(graph_embeddings),
                self.emb2add_node(node_embeddings),
                self.emb2set_node_attr(node_embeddings),
                self.emb2add_edge(node_embeddings[ne_row] + node_embeddings[ne_col]),
                self.emb2set_edge_attr(node_embeddings[e_row] + node_embeddings[e_col]),
            ],
            keys=[None, "x", "x", "non_edge_index", "edge_index"],
            types=self.action_type_order,
        )
        r_pred = self.emb2reward(graph_embeddings)
        if output_Qs:
            return cat, r_pred
        # Compute the greedy policy
        # See algo.envelope_q_learning.EnvelopeQLearning.compute_batch_losses for further explanations
        # TODO: this makes assumptions about how conditional vectors are created! Not robust to upstream changes
        w = cond[:, -self.num_objectives :]
        w_dot_Q = [
            (qi.reshape((qi.shape[0], qi.shape[1] // w.shape[1], w.shape[1])) * w[b][:, None, :]).sum(2)
            for qi, b in zip(cat.logits, cat.batch)
        ]
        # Set the softmax distribution to a very low temperature to make sure only the max gets
        # sampled (and we get random argmax tie breaking for free!):
        cat.logits = [i * 100 for i in w_dot_Q]
        return cat, r_pred


class EnvelopeQLearning:
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        task: GFNTask,
        cfg: Config,
    ):
        """Envelope Q-Learning implementation, see
        A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation,
        Runzhe Yang, Xingyuan Sun, Karthik Narasimhan,
        NeurIPS 2019,
        https://arxiv.org/abs/1908.08342

        Hyperparameters used:
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        cfg: Config
            The experiment configuration
        """
        self.ctx = ctx
        self.env = env
        self.task = task
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.gamma = cfg.algo.moql.gamma
        self.num_objectives = cfg.algo.moql.num_objectives
        self.num_omega_samples = cfg.algo.moql.num_omega_samples
        self.lambda_decay = cfg.algo.moql.lambda_decay
        self.invalid_penalty = cfg.algo.moql.penalty
        self._num_updates = 0
        assert self.gamma == 1
        self.bootstrap_own_reward = False
        # Experimental flags
        self.sample_temp = 1
        self.do_q_prime_correction = False
        self.graph_sampler = GraphSampler(ctx, env, self.max_len, self.max_nodes, self.sample_temp)

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: nn.Module
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)
        return data

    def create_training_data_from_graphs(self, graphs):
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{"traj": generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
        actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()

        # Now we create a duplicate/repeated batch for Q(s,a,w')
        omega_prime = self.task.sample_conditional_information(self.num_omega_samples * batch.num_graphs)
        torch_graphs = [i for i in torch_graphs for j in range(self.num_omega_samples)]
        actions = [i for i in actions for j in range(self.num_omega_samples)]
        batch_prime = self.ctx.collate(torch_graphs)
        batch_prime.traj_lens = batch.traj_lens.repeat_interleave(self.num_omega_samples)
        batch_prime.actions = torch.tensor(actions)
        batch_prime.cond_info = omega_prime["encoding"]
        batch_prime.preferences = omega_prime["preferences"]
        batch.batch_prime = batch_prime
        return batch

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        cond_info = batch.cond_info
        num_objectives = self.num_objectives

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        num_states = batch_idx.shape[0]
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and per graph predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        # Q(s,a,omega)
        fwd_cat, per_state_preds = model(batch, cond_info[batch_idx], output_Qs=True)
        Q_omega = fwd_cat.logits
        # reshape to List[shape: (num <T> in all graphs, num actions on T, num_objectives) | for all types T]
        Q_omega = [i.reshape((i.shape[0], i.shape[1] // num_objectives, num_objectives)) for i in Q_omega]

        # We do the same for omega', Q(s, a, w')
        batchp = batch.batch_prime
        batchp_num_trajs = int(batchp.traj_lens.shape[0])
        batchp_batch_idx = torch.arange(batchp_num_trajs, device=dev).repeat_interleave(batchp.traj_lens)
        fwd_cat_prime, per_state_preds = model(batchp, batchp.cond_info[batchp_batch_idx], output_Qs=True)
        Q_omega_prime = fwd_cat_prime.logits
        # We've repeated everything N_omega times, so we can reshape the same way as above but with
        # an extra N_omega first dimension
        Q_omega_prime = [i.reshape((i.shape[0], i.shape[1] // num_objectives, num_objectives)) for i in Q_omega_prime]

        # The math is
        #    y = r + \arg_Q \max_{a,w'} w . Q(s', a, w')
        # so we're going to compute all the dot products
        # then take
        w = batch.preferences[batch_idx]  # shape: (num_graphs, num_objectives)
        w_dot_Q = [
            # Broadcast preferences w over the actions axis, then sum.
            # What's going on with the indexing here?
            # here qi has shape (N_omega * num objects, num actions, num objectives)
            # w has shape (num graphs, num objectives), and we index it by b
            # b has shape (num objects), and refers to which graph each object entry corresponds to in fwd_cat.
            # For use in Q_omega_prime, we repeat every state N_omega times,
            # therefore, we repeat_interleave b so that each repeated state has its own w copy
            (
                qi
                * (
                    w[b.repeat_interleave(self.num_omega_samples)].reshape(
                        (self.num_omega_samples * b.shape[0], 1, num_objectives)
                    )
                )
                # then we multiply the Q(s, a, w') and w, take the sum on the right axis for the dot
            ).sum(2)
            for qi, b in zip(Q_omega_prime, fwd_cat.batch)
        ]  # List[shape: (N_omega * num objects, num actions)]

        # Now we need to do an argmax, over actions _and_ omegas, of the dot which has shape
        # List[(N_omega * num objects, num actions).] To do this fortunately we can reuse the
        # mechanisms of GraphActionCategorical. Once again we repeat interleave the batch indices
        # (of fwd_cat) to map back to the original states -- this makes it so that what are
        # considered different states (because they are repeated) in batch_prime are considered the
        # same (and thus the max is over all of the repeats as well).
        # Since the batch slices we will later index to get Q[:, argmax a, argmax omega'] are those
        # of Q_omega_prime, we need to use fwd_cat_prime.
        argmax = fwd_cat_prime.argmax(
            x=w_dot_Q, batch=[b.repeat_interleave(self.num_omega_samples) for b in fwd_cat.batch], dim_size=num_states
        )
        # Now what we want, for each state, is the vector prediction made by Q(s, a, w') for the
        # argmax a,w'. Let's again reuse GraphActionCategorical methods to do the indexing for us.
        # We must again use fwd_cat_prime to use the right slices.
        Q_pareto = fwd_cat_prime.log_prob(actions=argmax, logprobs=Q_omega_prime)
        # shape: (num_graphs, num_objectives)
        # Now we have \arg_Q \max_{a,w'} w . Q(s, a, w') for each state, we really want Q(s', ...)
        # Shift t+1-> t, pad last state with a 0, multiply by gamma
        shifted_Q_pareto = self.gamma * torch.cat([Q_pareto[1:], torch.zeros_like(Q_pareto[:1])], dim=0)
        # Replace Q(s_T) with R(tau). Since we've shifted the values in the array, Q(s_T) is Q(s_0)
        # of the next trajectory in the array, and rewards are terminal (0 except at s_T).
        shifted_Q_pareto[final_graph_idx] = batch.obj_props + ((1 - batch.is_valid) * self.invalid_penalty)[:, None]
        y = shifted_Q_pareto.detach()

        # We now use the same log_prob trick to get Q(s,a,w)
        Q_saw = fwd_cat.log_prob(actions=batch.actions, logprobs=Q_omega)
        # and compute L_A
        loss_A = (y - Q_saw).pow(2).sum(1)
        # and L_B
        loss_B = abs((w * y).sum(1) - (w * Q_saw).sum(1))

        Lambda = 1 - self.lambda_decay / (self.lambda_decay + self._num_updates)
        losses = (1 - Lambda) * loss_A + Lambda * loss_B
        self._num_updates += 1

        traj_losses = scatter(losses, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        loss = losses.mean()
        invalid_mask = 1 - batch.is_valid
        info = {
            "loss": loss.item(),
            "loss_A": loss_A.mean(),
            "loss_B": loss_B.mean(),
            "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError("loss is not finite")
        return loss, info
