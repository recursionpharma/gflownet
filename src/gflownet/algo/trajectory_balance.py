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
        self.reward_loss_multiplier = hps.get('reward_loss_multiplier', 1)
        # Experimental flags
        self.reward_loss_is_mae = True
        self.tb_loss_is_mae = False
        self.tb_loss_is_huber = False
        self.mask_invalid_rewards = False
        self.length_normalize_losses = False
        self.reward_normalize_losses = False
        self.sample_temp = 1
        self.is_doing_subTB = hps.get('tb_do_subtb', False)
        self.correct_idempotent = hps.get('tb_correct_idempotent', False)

        self.graph_sampler = GraphSampler(ctx, env, max_len, max_nodes, rng, self.sample_temp,
                                          correct_idempotent=self.correct_idempotent)
        self.graph_sampler.random_action_prob = hps['random_action_prob']
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

    def get_idempotent_actions(self, g: Graph, gd: gd.Data, gp: Graph, action: GraphAction):
        """Returns the list of idempotent actions for a given transition.

        Note, this is slow! Correcting for idempotency is needed to estimate p(x) correctly, but
        isn't generally necessary if we mostly care about sampling approximately from the modes
        of p(x).

        Parameters
        ----------
        g: Graph
            The state graph
        gd: gd.Data
            The Data instance corresponding to g
        gp: Graph
            The next state's graph
        action: GraphAction
            Action leading from g to gp

        Returns
        -------
        actions: List[Tuple[int,int,int]]
            The list of idempotent actions that all lead from g to gp.

        """
        iaction = self.ctx.GraphAction_to_aidx(gd, action)
        if action.action == GraphActionType.Stop:
            return [iaction]
        mask_name = self.ctx.action_mask_names[iaction[0]]
        lmask = getattr(gd, mask_name)
        nz = lmask.nonzero()
        actions = [iaction]
        for i in nz:
            aidx = (iaction[0], i[0].item(), i[1].item())
            if aidx == iaction:
                continue
            ga = self.ctx.aidx_to_GraphAction(gd, aidx)
            child = self.env.step(g, ga)
            if nx.algorithms.is_isomorphic(child, gp, lambda a, b: a == b, lambda a, b: a == b):
                actions.append(aidx)
        return actions

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
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj['traj']]
        actions = [
            self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj['traj']])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i['traj']) for i in trajs])
        batch.log_p_B = torch.cat([i['bck_logprobs'] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get('is_valid', True) for i in trajs]).float()
        if self.correct_idempotent:
            agraphs = [i[0] for tj in trajs for i in tj['traj']]
            bgraphs = sum([[i[0] for i in tj['traj'][1:]] + [tj['result']] for tj in trajs], [])
            gactions = [i[1] for tj in trajs for i in tj['traj']]
            ipa = [
                self.get_idempotent_actions(g, gd, gp, a)
                for g, gd, gp, a in zip(agraphs, torch_graphs, bgraphs, gactions)
            ]
            batch.ip_actions = torch.tensor(sum(ipa, []))
            batch.ip_lens = torch.tensor([len(i) for i in ipa])
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
        log_rewards = batch.log_rewards
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
        log_Z = model.logZ(cond_info)[:, 0]
        # Compute the log prob of each action in the trajectory
        if self.correct_idempotent:
            # If we want to correct for idempotent actions, we need to sum probabilities
            # i.e. to compute P(s' | s) = sum_{a that lead to s'} P(a|s)
            # here we compute the indices of the graph that each action corresponds to, ip_lens
            # contains the number of  idempotent actions for each transition, so we
            # repeat_interleave as with batch_idx
            ip_idces = torch.arange(batch.ip_lens.shape[0], device=dev).repeat_interleave(batch.ip_lens)
            # get the full logprobs
            logprobs = fwd_cat.logsoftmax()
            # batch.ip_actions lists those actions, so we index the logprobs appropriately
            log_prob = torch.stack(
                [logprobs[t][row + fwd_cat.slice[t][i], col] for i, (t, row, col) in zip(ip_idces, batch.ip_actions)])
            # take the logsumexp (because we want to sum probabilities, not log probabilities)
            # TODO: numerically stable version:
            log_prob = scatter(log_prob.exp(), ip_idces, dim=0, dim_size=batch_idx.shape[0], reduce='sum').log()
        else:
            # Else just naively take the logprob of the actions we took
            log_prob = fwd_cat.log_prob(batch.actions)
        # The log prob of each backward action
        log_p_B = batch.log_p_B
        assert log_prob.shape == log_p_B.shape
        # Clip rewards
        assert log_rewards.ndim == 1
        clip_log_R = torch.maximum(log_rewards, torch.tensor(self.illegal_action_logreward, device=dev))
        # This is the log probability of each trajectory
        traj_log_prob = scatter(log_prob, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')
        # Compute log numerator and denominator of the TB objective
        numerator = log_Z + traj_log_prob
        denominator = clip_log_R + scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')

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
            traj_losses = self.subtb_loss_fast(log_prob, log_p_B, per_mol_out[:, 0], clip_log_R, batch.traj_lens)
            # The position of the first graph of each trajectory
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            first_graph_idx = torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            log_Z = per_mol_out[first_graph_idx, 0]
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
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            # * num_trajs because we're doing a convex combination, and a .mean() later, which would
            # undercount (by 2N) the contribution of each loss
            traj_losses = factor * traj_losses * num_trajs

        if self.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(log_rewards)
            if self.reward_loss_is_mae:
                reward_losses = abs(log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap])
            else:
                reward_losses = (log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap]).pow(2)
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
            'logZ': log_Z.mean(),
            'loss': loss.item(),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')
        return loss, info

    def _init_subtb(self, dev):
        r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
            \sum_{m=1}^{T-1} \sum_{n=m+1}^T
                \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                           {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
        """
        ar = torch.arange(self._subtb_max_len, device=dev)
        # This will contain a sequence of repeated ranges, e.g.
        # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
        tidx = [torch.tril_indices(i, i, device=dev)[1] for i in range(self._subtb_max_len)]
        # We need two sets of indices, the first are the source indices, the second the destination
        # indices. We precompute such indices for every possible trajectory length.

        # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
        # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
        # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

        # The destination indices indicate the index of the subsequence the source indices correspond to.
        # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
        # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

        # And so with these indices, for example for m=0, n=3, the forward probability
        # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

        self._precomp = [
            (torch.cat([i + tidx[T - i]
                        for i in range(T)]),
             torch.cat([ar[:T - i].repeat_interleave(ar[:T - i] + 1) + ar[T - i + 1:T + 1].sum()
                        for i in range(T)]))
            for T in range(1, self._subtb_max_len)
        ]

    def subtb_loss_fast(self, P_F, P_B, F, R, traj_lengths):
        r"""Computes the full SubTB(1) loss (all arguments on log-scale).

        Computes:
            \sum_{m=1}^{T-1} \sum_{n=m+1}^T
                \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                           {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
            where T is the length of the trajectory, for every trajectory.

        The shape of P_F, P_B, and F should be (total num steps,), i.e. sum(traj_lengths). The shape
        of R and traj_lengths should be (num trajs,).

        Parameters
        ----------
        P_F: Tensor
            Forward policy log-probabilities
        P_B: Tensor
            Backward policy log-probabilities
        F: Tensor
            Log-scale flow predictions
        R: Tensor
            The reward of each trajectory
        traj_lengths: Tensor
            The lenght of each trajectory

        Returns
        -------
        losses: Tensor
            The SubTB(1) loss of each trajectory.
        """
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
