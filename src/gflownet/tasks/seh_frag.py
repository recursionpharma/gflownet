import os
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.config import Config
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext, Graph
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward


class SEHTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid


SOME_MOLS = [
    "O=C(NCc1cc(CCc2cccc(N3CCC(c4cc(-c5cc(-c6cncnc6)[nH]n5)ccn4)CC3)c2)ccn1)c1cccc2ccccc12",
    "O=c1nc2[nH]c3cc(-c4cc(C5CC(c6ccc(CNC7CCOC7c7csc(C8=CC(c9ccc%10ccccc%10c9)CCC8)n7)cc6)CO5)c[nH]4)ccc3nc-2c(=O)[nH]1",
    "c1ccc(-c2cnn(-c3cc(-c4cc(CCc5cc(C6CCC(c7cc(-c8ccccc8)[nH]n7)CO6)ncn5)n[nH]4)ccn3)c2)cc1",
    "O=C(NCc1cc(C2CCNC2C2CCNC2)ncn1)c1cccc(-c2cccc(-c3cccc(C4CCC(c5ccccc5)CO4)c3)c2)c1",
    "O=C(NCc1cccc(C2COC(c3ccc4nc5c(=O)[nH]c(=O)nc-5[nH]c4c3)C2)c1)c1cccc(CCc2cccc(-c3ncnc4c3ncn4C3CCCN3)c2)c1",
    "O=C(NCc1ccc(OCc2ccc(-c3ccncc3C3CCNCC3)cn2)cc1)c1cccc(N2CCC(C3CCCN3)CC2)n1",
    "O=C(NCc1ccc(C2CCC(c3cccc(-c4cccc(C5CCOC5)c4)c3)CO2)cn1)c1ccc(-n2ccc(-c3ccc4nc5c(=O)[nH]c(=O)nc-5[nH]c4c3)n2)cn1",
    "O=C(NCc1nc2c(c(=O)[nH]1)NC(c1cn(N3CCN(c4ccc5nc6c(=O)[nH]c(=O)nc-6[nH]c5c4)CC3)c(=O)[nH]c1=O)CN2)c1ccc[n+](-c2cccc(-c3nccc(-c4ccccc4)n3)c2)c1",
    "C1=C(C2CCC(c3ccnc(-c4ccc(CNC5CCC(c6ccncc6)OC5)cc4)n3)CO2)CCC(c2cc(-c3cncnc3)c3ccccc3c2)C1",
    "O=C(NCc1cccc(-c2nccc(-c3cc(-c4ccc5ccccc5c4)n[nH]3)n2)c1)C1CCC(C2CCC(c3cn(-c4ccc5nc6c(=O)[nH]c(=O)nc-6[nH]c5c4)c(=O)[nH]c3=O)OC2)O1",
    "O=C(Nc1ccc2ccccc2c1)c1cccc(-c2cccc(CNN3CCN(C4CCCC(c5cccc(C6CCCN6)c5)C4)CC3)c2)c1",
    "O=C(NCC1CC=C(c2cc(CCc3c[nH]c(-c4cccc(-c5ccccc5)c4)c3)n[nH]2)CC1)c1cccc(C2CCNC2)n1",
    "O=C(Nc1nccc(CNc2cc(C3CCNC3)n[nH]2)n1)c1nccc(C2CCC(C3CCNCC3c3ccc4ccccc4c3)CO2)n1",
    "C1=C(C2CCC(c3ccc(-c4cc(C5CCCNC5)n[nH]4)cc3)OC2)CCCC1CCc1cccc(-c2cccc(-c3ncnc4[nH]cnc34)c2)n1",
    "O=C(NCc1cc(C2CCC(C3CCN(c4cc(-c5nccc(-c6cccc(-c7ccccc7)c6)n5)c[nH]4)CC3)CO2)ccn1)c1ccccc1",
    "O=C(NCc1cccc(-c2ccn(NCc3ccc(-c4cc(C5CNC(c6ccncc6)C5)c[nH]4)cc3)n2)c1)c1ccc2ccccc2c1",
    "O=c1nc2n(-c3cccc(OCc4cccc(CNC5CCC(c6cccc(-c7ccc(C8CCNC8)cc7)c6)OC5)c4)n3)c3ccccc3nc-2c(=O)[nH]1",
    "O=C(NCc1ccc(C2OCCC2C2CC(c3ccnc(-c4ccc5ccccc5c4)c3)CO2)cc1)c1nccc(N2C=CCC(c3ccccc3)=C2)n1",
    "O=C(NCNC(=O)c1cccc(C(=O)NCc2cccc(-c3ccc4[nH]c5nc(=O)[nH]c(=O)c-5nc4c3)c2)n1)c1ccnc(-c2nccc(C3CCCN3)n2)c1",
    "O=c1nc2[nH]c3cc(C4CCC(c5ccc(-c6cc(C7CCC(C8CCCC(C9CCC(c%10ccc(-c%11cncnc%11)cc%10)O9)O8)OC7)ccn6)cn5)CO4)ccc3nc-2c(=O)[nH]1",
    "O=c1[nH]c(CNc2cc(-c3cccc(-n4ccc(-c5ccc6ccccc6c5)n4)c3)c[nH]2)nc2c1NC(n1ccc(C3CCC(c4cccnc4)CO3)n1)CN2",
    "O=c1nc2[nH]c3cc(C=CC4COC(C5CCCC(C6CCOC(C7CCC(c8cccc(-c9ccnc(-c%10ccc%11ccccc%11c%10)n9)c8)CO7)C6)O5)C4)ccc3nc-2c(=O)[nH]1",
    "c1ccc2c(C3CC(CNc4ccnc(C5CCNC5)c4)CO3)cc(NCc3ccc(-c4cc(C5CCNC5)c[nH]4)cc3)cc2c1",
    "O=C(NCc1nccc(C2CC(C(=O)NC3CCC(c4ccc5nc6c(=O)[nH]c(=O)nc-6[nH]c5c4)CO3)CCO2)n1)c1ccnc(-n2cc(-n3cnc4cncnc43)cn2)n1",
    "O=C(NCc1ccc(-c2ccccc2)cc1)c1cccc(C(=O)NCc2nccc(N3C=CCC(c4ncnc5c4ncn5-c4cccc5ccccc45)=C3)n2)c1",
]


class LittleSEHDataset(Dataset):
    """Note: this dataset isn't used by default, but turning it on showcases some features of this codebase.

    To turn on, self `cfg.algo.offline_ratio > 0`"""

    def __init__(self) -> None:
        super().__init__()
        self.props: List[Tensor] = []
        self.mols: List[Graph] = []

    def setup(self, task, ctx):
        rdmols = [Chem.MolFromSmiles(i) for i in SOME_MOLS]
        self.mols = [ctx.mol_to_graph(i) for i in rdmols]
        self.props = task.compute_flat_rewards(rdmols)[0]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, index):
        return self.mols[index], self.props[index]


class SEHFragTrainer(StandardOnlineTrainer):
    task: SEHTask
    training_data: LittleSEHDataset

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_task(self):
        self.task = SEHTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_data(self):
        super().setup_data()
        self.training_data = LittleSEHDataset()

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.task.num_cond_dim)

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)


def main():
    """Example of how this trainer can be run"""
    hps = {
        "log_dir": "./logs/debug_run_seh_frag_pb",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 10_000,
        "num_workers": 8,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {"sampling_tau": 0.99, "offline_ratio": 0.0},
        "cond": {
            "temperature": {
                "sample_dist": "uniform",
                "dist_params": [0, 64.0],
            }
        },
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHFragTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
