from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.data as gd
from torch_scatter import scatter, scatter_sum

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.envs.graph_building_env import generate_forward_trajectory
from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.graph_building_env import GraphBuildingEnvContext


class TrajectoryBalanceModel(nn.Module):
    def forward(self, batch: gd.Batch) -> Tuple[GraphActionCategorical, Tensor]:
        raise NotImplementedError()

    def logZ(self, cond_info: Tensor) -> Tensor:
        raise NotImplementedError()


class TrajectoryBalance:
    """
    """
    def __init__(self, env: GraphBuildingEnv, ctx: GraphBuildingEnvContext, rng: np.random.RandomState,
                 hps: Dict[str, Any], max_len=None, max_nodes=None):
        """TB implementation, see
        "Trajectory Balance: Improved Credit Assignment in GFlowNets Nikolay Malkin, Moksh Jain,
        Emmanuel Bengio, Chen Sun, Yoshua Bengio"
        https://arxiv.org/abs/2201.13259

        Hyperparameters used:
        random_action_prob: float, probability of taking a uniform random action when sampling
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions
        bootstrap_own_reward: bool, if True, uses the .reward batch data to predict rewards for sampled data
        tb_epsilon: float, if not None, adds this epsilon in the numerator and denominator of the log-ratio
        reward_loss_multiplier: float, multiplying constant for the bootstrap loss.

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
        self.bootstrap_own_reward = hps['bootstrap_own_reward']
        self.epsilon = hps['tb_epsilon']
        self.reward_loss_multiplier = hps['reward_loss_multiplier']
        # Experimental flags
        self.reward_loss_is_mae = True
        self.tb_loss_is_mae = False
        self.tb_loss_is_huber = False
        self.mask_invalid_rewards = False
        self.length_normalize_losses = False
        self.reward_normalize_losses = False
        self.sample_temp = 1
        self.graph_sampler = GraphSampler(ctx, env, max_len, max_nodes, rng, self.sample_temp)
        self.graph_sampler.random_action_prob = hps['random_action_prob']
        self.is_doing_subTB = hps.get('tb_do_subtb', False)
        if self.is_doing_subTB:
            self._subtb_max_len = hps.get('tb_subtb_max_len', max_len + 1 if max_len is not None else 128)
            self._init_subtb(torch.device('cuda'))  # TODO: where are we getting device info?

    def create_training_data_from_own_samples(self, model: TrajectoryBalanceModel, n: int, cond_info: Tensor):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: TrajectoryBalanceModel
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - reward_pred: float, -100 if an illegal action is taken, predicted R(x) if bootstrapping, None otherwise
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - logZ: predicted log Z
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = self.ctx.device
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, dev)
        logZ_pred = model.logZ(cond_info)
        for i in range(n):
            data[i]['logZ'] = logZ_pred[i].item()
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

    def construct_batch(self, trajs, cond_info, rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        rewards: Tensor
            The transformed reward (e.g. R(x) ** beta) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj['traj']]
        actions = [
            self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj['traj']])
        ]
        num_backward = torch.tensor([
            # Count the number of backward transitions from s_{t+1},
            # unless t+1 = T is the last time step
            self.env.count_backward_transitions(tj['traj'][i + 1][0]) if i + 1 < len(tj['traj']) else 1
            for tj in trajs
            for i in range(len(tj['traj']))
        ])
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i['traj']) for i in trajs])
        batch.num_backward = num_backward
        batch.actions = torch.tensor(actions)
        batch.rewards = rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get('is_valid', True) for i in trajs]).float()
        return batch

    def compute_batch_losses(self, model: TrajectoryBalanceModel, batch: gd.Batch, num_bootstrap: int = 0):
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
        rewards = batch.rewards
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and the optional bootstrap predictions
        fwd_cat, per_mol_out = model(batch, cond_info[batch_idx])

        # Retreive the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        log_reward_preds = per_mol_out[final_graph_idx, 0]
        # Compute trajectory balance objective
        Z = model.logZ(cond_info)[:, 0]
        # This is the log prob of each action in the trajectory
        log_prob = fwd_cat.log_prob(batch.actions)
        # The log prob of each backward action
        log_p_B = (1 / batch.num_backward).log()
        # Take log rewards, and clip
        assert rewards.ndim == 1
        Rp = torch.maximum(rewards.log(), torch.tensor(-100.0, device=dev))
        # This is the log probability of each trajectory
        traj_log_prob = scatter(log_prob, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')
        # Compute log numerator and denominator of the TB objective
        numerator = Z + traj_log_prob
        denominator = Rp + scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')

        if self.epsilon is not None:
            # Numerical stability epsilon
            epsilon = torch.tensor([self.epsilon], device=dev).float()
            numerator = torch.logaddexp(numerator, epsilon)
            denominator = torch.logaddexp(denominator, epsilon)

        invalid_mask = 1 - batch.is_valid
        if self.mask_invalid_rewards:
            # Instead of being rude to the model and giving a
            # logreward of -100 what if we say, whatever you think the
            # logprobablity of this trajetcory is it should be smaller
            # (thus the `numerator - 1`). Why 1? Intuition?
            denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

        if self.is_doing_subTB:
            # SubTB interprets the per_mol_out predictions to predict the state flow F(s)
            traj_losses = self.subtb_loss_fast(log_prob, log_p_B, per_mol_out[:, 0], Rp, batch.traj_lens)
            # The position of the first graph of each trajectory
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            first_graph_idx = torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            Z = per_mol_out[first_graph_idx, 0]
        else:
            if self.tb_loss_is_mae:
                traj_losses = abs(numerator - denominator)
            elif self.tb_loss_is_huber:
                pass  # TODO
            else:
                traj_losses = (numerator - denominator).pow(2)

        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens
        if self.reward_normalize_losses:
            # multiply each loss by how important it is, using R as the importance factor
            # factor = Rp.exp() / Rp.exp().sum()
            factor = -Rp.min() + Rp + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            # * num_trajs because we're doing a convex combination, and a .mean() later, which would
            # undercount (by 2N) the contribution of each loss
            traj_losses = factor * traj_losses * num_trajs

        if self.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(rewards)
            if self.reward_loss_is_mae:
                reward_losses = abs(rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap].exp())
            else:
                reward_losses = (rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap].exp()).pow(2)
            reward_loss = reward_losses.mean()
        else:
            reward_loss = 0

        loss = traj_losses.mean() + reward_loss * self.reward_loss_multiplier
        info = {
            'offline_loss': traj_losses[:batch.num_offline].mean() if batch.num_offline > 0 else 0,
            'online_loss': traj_losses[batch.num_offline:].mean() if batch.num_online > 0 else 0,
            'reward_loss': reward_loss,
            'invalid_trajectories': invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            'invalid_logprob': (invalid_mask * traj_log_prob).sum() / (invalid_mask.sum() + 1e-4),
            'invalid_losses': (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            'logZ': Z.mean(),
            'loss': loss.item(),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')
        return loss, info

    def _init_subtb(self, dev):
        # Precompute all possible subtrajectory indices that we will use for computing the loss
        ar = torch.arange(self._subtb_max_len, device=dev)
        tidx = [torch.tril_indices(i, i, device=dev)[1] for i in range(self._subtb_max_len)]
        self._precomp = [
            (torch.cat([i + tidx[T - i]
                        for i in range(T)]),
             torch.cat([ar[:T - i].repeat_interleave(ar[:T - i] + 1) + ar[T - i + 1:T + 1].sum()
                        for i in range(T)]))
            for T in range(1, self._subtb_max_len)
        ]

    def subtb_loss_fast(self, P_F, P_B, F, R, traj_lengths):
        num_trajs = int(traj_lengths.shape[0])
        max_len = int(traj_lengths.max() + 1)
        dev = traj_lengths.device
        cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=dev), traj_lengths]), 0).long()
        total_loss = torch.zeros(num_trajs, device=dev)
        ar = torch.arange(max_len, device=dev)
        car = torch.cumsum(ar, 0)
        F_and_R = torch.cat([F, R])
        R_start = F.shape[0]
        for ep in range(traj_lengths.shape[0]):
            offset = cumul_lens[ep]
            T = int(traj_lengths[ep])
            idces, dests = self._precomp[T - 1]
            fidces = torch.cat(
                [torch.cat([ar[i + 1:T] + offset, torch.tensor([R_start + ep], device=dev)]) for i in range(T)])
            P_F_sums = scatter_sum(P_F[idces + offset], dests)
            P_B_sums = scatter_sum(P_B[idces + offset], dests)
            F_start = F[offset:offset + T].repeat_interleave(T - ar[:T])
            F_end = F_and_R[fidces]
            total_loss[ep] = (F_start - F_end + P_F_sums - P_B_sums).pow(2).sum() / car[T]
        return total_loss
