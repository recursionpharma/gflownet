import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext, generate_forward_trajectory
from gflownet.utils.misc import get_worker_device

from .graph_sampling import GraphSampler


class A2C:
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        """Advantage Actor-Critic implementation, see
          Asynchronous Methods for Deep Reinforcement Learning,
          Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim
          Harley, David Silver, Koray Kavukcuoglu
          Proceedings of The 33rd International Conference on Machine Learning, 2016

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
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.entropy_coef = cfg.algo.a2c.entropy
        self.gamma = cfg.algo.a2c.gamma
        self.invalid_penalty = cfg.algo.a2c.penalty
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

        # Forward pass of the model, returns a GraphActionCategorical and per graph predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        policy, per_state_preds = model(batch, cond_info[batch_idx])
        V = per_state_preds[:, 0]
        G = rewards[batch_idx]  # The return is the terminal reward everywhere, we're using gamma==1
        G = G + (1 - batch.is_valid[batch_idx]) * self.invalid_penalty  # Add in penalty for invalid object
        A = G - V
        log_probs = policy.log_prob(batch.actions)

        V_loss = A.pow(2).mean()
        pol_objective = (log_probs * A.detach()).mean() + self.entropy_coef * policy.entropy().mean()
        pol_loss = -pol_objective

        loss = V_loss + pol_loss
        invalid_mask = 1 - batch.is_valid
        info = {
            "V_loss": V_loss,
            "A": A.mean(),
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "loss": loss.item(),
        }

        if not torch.isfinite(loss).all():
            raise ValueError("loss is not finite")
        return loss, info
