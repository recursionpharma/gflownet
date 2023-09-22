from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.config import Config
from gflownet.envs.graph_building_env import (
    Graph,
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    generate_forward_trajectory,
)
from gflownet.trainer import GFNAlgorithm
from gflownet.utils.transforms import thermometer


class QLearning(GFNAlgorithm):
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        rng: np.random.RandomState,
        cfg: Config,
    ):
        """Classic Q-Learning implementation

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        rng: np.random.RandomState
            rng used to take random actions
        cfg: Config
            The experiment configuration
        """
        self.ctx = ctx
        self.env = env
        self.rng = rng
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.mellowmax_omega = cfg.mellowmax_omega
        self.graph_sampler = GraphSampler(
            ctx, env, self.max_len, self.max_nodes, rng, input_timestep=cfg.algo.input_timestep
        )
        self.graph_sampler.sample_temp = 0  # Greedy policy == infinitely low temperature
        self.gamma = 1
        self.type = "ddqn"  # TODO: add to config

    def create_training_data_from_own_samples(
        self,
        model: nn.Module,
        batch_size: int,
        cond_info: Tensor,
        random_action_prob: float = 0.0,
        starts: Optional[List[Graph]] = None,
    ) -> List[Dict[str, Tensor]]:
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
        data = self.graph_sampler.sample_from_model(
            model, batch_size, cond_info, dev, random_action_prob, starts=starts
        )
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
        torch_graphs = [self.ctx.graph_to_Data(i[0], timestep) for tj in trajs for timestep, i in enumerate(tj["traj"])]
        actions = [
            self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        if self.graph_sampler.input_timestep:
            batch.timesteps = torch.tensor(
                [min(1, (len(tj["traj"]) - t) / self.max_len) for tj in trajs for t in range(len(tj["traj"]))]
            )
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        batch.trajs = trajs
        return batch

    def compute_batch_losses(  # type: ignore
        self, model: nn.Module, batch: gd.Batch, lagged_model: nn.Module, num_bootstrap: int = 0
    ) -> Tuple[Any, Dict[str, Any]]:
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
        # rewards = torch.exp(batch.log_rewards)
        rewards = batch.log_rewards

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and per molecule predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        if self.graph_sampler.input_timestep:
            ci = torch.cat([batch.cond_info[batch_idx], thermometer(batch.timesteps, 32)], dim=1)
        else:
            ci = batch.cond_info[batch_idx]
        Q: GraphActionCategorical
        Q, per_state_preds = model(batch, ci)
        with torch.no_grad():
            Qp, _ = lagged_model(batch, ci)

        if self.type == "dqn":
            V_s = Qp.max(Qp.logits).values.detach()
        elif self.type == "ddqn":
            # Q(s, a) = r + γ * Q'(s', argmax Q(s', a'))
            # Q: (num-states, num-actions)
            # V = Q[arange(sum(batch.traj_lens)), actions]
            # V_s : (sum(batch.traj_lens),)
            V_s = Qp.log_prob(Q.argmax(Q.logits), logprobs=Qp.logits)
        elif self.type == "mellowmax":
            V_s = Q.logsumexp([i * self.mellowmax_omega for i in Q.logits]).detach() / self.mellowmax_omega

        # Here were are again hijacking the GraphActionCategorical machinery to get Q[s,a], but
        # instead of logprobs we're just going to use the logits, i.e. the Q values.
        Q_sa = Q.log_prob(batch.actions, logprobs=Q.logits)

        # We now need to compute the target, \hat Q = R_t + V_soft(s_t+1)
        # Shift t+1->t, pad last state with a 0, multiply by gamma
        shifted_V = self.gamma * torch.cat([V_s[1:], torch.zeros_like(V_s[:1])])
        # batch_lens = [3,4]
        # V = [0,1,2, 3,4,5,6]
        # shifted_V = [1,2,3, 4,5,6,0]
        # Replace V(s_T) with R(tau). Since we've shifted the values in the array, V(s_T) is V(s_0)
        # of the next trajectory in the array, and rewards are terminal (0 except at s_T).
        shifted_V[final_graph_idx] = rewards * batch.is_valid + (1 - batch.is_valid) * self.illegal_action_logreward
        # shifted_V = [1,2,R1, 4,5,6,R2]
        # The result is \hat Q = R_t + gamma V(s_t+1) * non_terminal
        hat_Q = shifted_V

        losses = nn.functional.huber_loss(Q_sa, hat_Q, reduction="none")

        loss = losses.mean()
        invalid_mask = 1 - batch.is_valid
        info = {
            "loss": loss,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            # "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            "Q_sa": Q_sa.mean().item(),
        }

        return loss, info
