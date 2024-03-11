import torch
import torch_geometric.data as gd
from torch_scatter import scatter

from gflownet.algo.trajectory_balance import TrajectoryBalance, TrajectoryBalanceModel
from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext


class MultiObjectiveReinforce(TrajectoryBalance):
    """
    Class that inherits from TrajectoryBalance and implements the multi-objective reinforce algorithm
    """

    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        super().__init__(env, ctx, cfg)

    def compute_batch_losses(self, model: TrajectoryBalanceModel, batch: gd.Batch, num_bootstrap: int = 0):
        """Compute  multi objective REINFORCE loss over trajectories contained in the batch"""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        rewards = torch.exp(batch.log_rewards)
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)

        # Forward pass of the model, returns a GraphActionCategorical and the optional bootstrap predictions
        fwd_cat, log_reward_preds = model(batch, cond_info[batch_idx])

        # This is the log prob of each action in the trajectory
        log_prob = fwd_cat.log_prob(batch.actions)

        # Take log rewards, and clip
        assert rewards.ndim == 1
        traj_log_prob = scatter(log_prob, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        traj_losses = traj_log_prob * (-rewards - (-1) * rewards.mean())

        loss = traj_losses.mean()
        info = {
            "loss": loss.item(),
        }
        if not torch.isfinite(traj_losses).all():
            raise ValueError("loss is not finite")
        return loss, info
