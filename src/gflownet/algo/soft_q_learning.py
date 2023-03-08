from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.data as gd
from torch_scatter import scatter

from gflownet.envs.graph_building_env import generate_forward_trajectory
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.graph_building_env import GraphBuildingEnvContext

from .graph_sampling import GraphSampler


class SoftQLearning:
    def __init__(self, env: GraphBuildingEnv, ctx: GraphBuildingEnvContext, rng: np.random.RandomState,
                 hps: Dict[str, Any], max_len=None, max_nodes=None):
        """Soft Q-Learning implementation, see
        xxxxx

        Hyperparameters used:
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions
        sql_alpha: float, the entropy coefficient

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        rng: np.random.RandomState
            rng used to take random actions
        hps: Dict[str, Any]
            Hyperparameter dictionary, see above for used keys.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        """
        self.ctx = ctx
        self.env = env
        self.rng = rng
        self.max_len = max_len
        self.max_nodes = max_nodes
        self.illegal_action_logreward = hps['illegal_action_logreward']
        self.alpha = hps['sql_alpha']
        self.gamma = hps.get('sql_gamma', 1)
        self.invalid_penalty = hps.get('sql_penalty', -10)
        self.bootstrap_own_reward = False
        # Experimental flags
        self.sample_temp = 1
        self.do_q_prime_correction = False
        self.graph_sampler = GraphSampler(ctx, env, max_len, max_nodes, rng, self.sample_temp)

    def create_training_data_from_own_samples(self, model: nn.Module, n: int, cond_info: Tensor,
                                              random_action_prob: float):
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
        dev = self.ctx.device
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, dev, random_action_prob)
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
        return [{'traj': generate_forward_trajectory(i)} for i in graphs]

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
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj['traj']]
        actions = [
            self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj['traj']])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i['traj']) for i in trajs])
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get('is_valid', True) for i in trajs]).float()
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
        rewards = torch.exp(batch.log_rewards)
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and per molecule predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        Q, per_state_preds = model(batch, cond_info[batch_idx])

        if self.do_q_prime_correction:
            # First we need to estimate V_soft. We will use q_a' = \pi
            log_policy = Q.logsoftmax()
            # in Eq (10) we have an expectation E_{a~q_a'}[exp(1/alpha Q(s,a))/q_a'(a)]
            # we rewrite the inner part `exp(a)/b` as `exp(a-log(b))` since we have the log_policy probabilities
            soft_expectation = [Q_sa / self.alpha - logprob for Q_sa, logprob in zip(Q.logits, log_policy)]
            # This allows us to more neatly just call logsumexp on the logits, and then multiply by alpha
            V_soft = self.alpha * Q.logsumexp(soft_expectation).detach()  # shape: (num_graphs,)
        else:
            V_soft = Q.logsumexp(Q.logits).detach()
            rewards = rewards / self.alpha

        # Here were are again hijacking the GraphActionCategorical machinery to get Q[s,a], but
        # instead of logprobs we're just going to use the logits, i.e. the Q values.
        Q_sa = Q.log_prob(batch.actions, logprobs=Q.logits)

        # We now need to compute the target, \hat Q = R_t + V_soft(s_t+1)
        # Shift t+1-> t, pad last state with a 0, multiply by gamma
        shifted_V_soft = self.gamma * torch.cat([V_soft[1:], torch.zeros_like(V_soft[:1])])
        # Replace V(s_T) with R(tau). Since we've shifted the values in the array, V(s_T) is V(s_0)
        # of the next trajectory in the array, and rewards are terminal (0 except at s_T).
        shifted_V_soft[final_graph_idx] = rewards + (1 - batch.is_valid) * self.invalid_penalty
        # The result is \hat Q = R_t + gamma V(s_t+1)
        hat_Q = shifted_V_soft

        losses = (Q_sa - hat_Q).pow(2)
        traj_losses = scatter(losses, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')
        loss = losses.mean()
        invalid_mask = 1 - batch.is_valid
        info = {
            'mean_loss': loss,
            'offline_loss': traj_losses[:batch.num_offline].mean() if batch.num_offline > 0 else 0,
            'online_loss': traj_losses[batch.num_offline:].mean() if batch.num_online > 0 else 0,
            'invalid_trajectories': invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            'invalid_losses': (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')
        return loss, info
