import os
import shutil
import socket
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor

from gflownet.config import Config, init_empty
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar


class MakeRingsTask(GFNTask):
    """A toy task where the reward is the number of rings in the molecule."""

    def __init__(
        self,
        rng: np.random.Generator,
    ):
        self.rng = rng

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(y)

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return {"beta": torch.ones(n), "encoding": torch.ones(n, 1)}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        scalar_logreward = torch.as_tensor(flat_reward).squeeze().clamp(min=1e-30).log()
        return RewardScalar(scalar_logreward.flatten())

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        rs = torch.tensor([m.GetRingInfo().NumRings() for m in mols]).float()
        return FlatRewards(rs.reshape((-1, 1))), torch.ones(len(mols)).bool()


class MakeRingsTrainer(StandardOnlineTrainer):
    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.num_workers = 8
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 6
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.do_parameterize_p_b = True

        cfg.replay.use = False

    def setup_task(self):
        self.task = MakeRingsTask(rng=self.rng)

    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ["C"],
            charges=[0],  # disable charge
            chiral_types=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED],  # disable chirality
            num_rw_feat=0,
            max_nodes=self.cfg.algo.max_nodes,
            num_cond_dim=1,
        )


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug_run_mr4"
    config.device = "cuda"
    config.num_training_steps = 10_000
    config.num_workers = 8
    config.algo.tb.do_parameterize_p_b = True

    if os.path.exists(config.log_dir):
        if config.overwrite_existing_exp:
            shutil.rmtree(config.log_dir)
        else:
            raise ValueError(f"Log dir {config.log_dir} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(config.log_dir)

    trial = MakeRingsTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
