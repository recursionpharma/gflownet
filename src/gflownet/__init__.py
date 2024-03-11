from typing import Dict, List, NewType, Optional, Tuple

import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor, nn

from .config import Config

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore


class GFNAlgorithm:
    updates: int = 0
    global_cfg: Config
    is_eval: bool = False

    def step(self):
        self.updates += 1  # This isn't used anywhere?

    def compute_batch_losses(
        self, model: nn.Module, batch: gd.Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes the loss for a batch of data, and proves logging informations

        Parameters
        ----------
        model: nn.Module
            The model being trained or evaluated
        batch: gd.Batch
            A batch of graphs
        num_bootstrap: Optional[int]
            The number of trajectories with reward targets in the batch (if applicable).

        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, Tensor]
            Logged information about model predictions.
        """
        raise NotImplementedError()

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Typically calls ctx.graph_to_Data and ctx.collate to convert the trajectories into
        a batch of graphs and adds the necessary attributes for training.

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
        raise NotImplementedError()

    def get_random_action_prob(self, it: int):
        if self.is_eval:
            return self.global_cfg.algo.valid_random_action_prob
        if self.global_cfg.algo.train_det_after is None or it < self.global_cfg.algo.train_det_after:
            return self.global_cfg.algo.train_random_action_prob
        return 0


class GFNTask:
    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Parameters
        ----------
        cond_info: Dict[str, Tensor]
            A dictionary with various conditional informations (e.g. temperature)
        flat_reward: FlatRewards
            A 2d tensor where each row represents a series of flat rewards.

        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar log-reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies

        Parameters
        ----------
        mols: List[RDMol]
            A list of RDKit molecules.
        Returns
        -------
        reward: FlatRewards
            A 2d tensor, a vector of scalar reward for valid each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()
