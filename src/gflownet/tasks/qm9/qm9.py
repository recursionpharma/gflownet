import ast
import copy
import os
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from ruamel.yaml import YAML
import scipy.stats as stats
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.data as gd

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.data.qm9 import QM9Dataset
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.models.graph_transformer import GraphTransformerGFN
import gflownet.models.mxmnet as mxmnet
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import GFNTrainer
from gflownet.train import RewardScalar
from gflownet.utils.transforms import thermometer


class QM9GapTask(GFNTask):
    """This class captures conditional information generation and reward transforms"""
    def __init__(self, dataset: Dataset, temperature_distribution: str, 
                 temperature_parameters: Tuple[float, float],
                 num_thermometer_dim: int, rng: np.random.Generator = None, 
                 wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self.load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim
        # TODO: fix interface
        self._min, self._max, self._percentile_95 = self.dataset.get_stats(percentile=0.05)  # type: ignore
        self._width = self._max - self._min
        self._rtrans = 'unit+95p'  # TODO: hyperparameter

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        """Transforms a target quantity y (e.g. the LUMO energy in QM9) to a positive reward scalar"""
        if self._rtrans == 'exp':
            flat_r = np.exp(-(y - self._min) / self._width)
        elif self._rtrans == 'unit':
            flat_r = 1 - (y - self._min) / self._width
        elif self._rtrans == 'unit+95p':
            # Add constant such that 5% of rewards are > 1
            flat_r = 1 - (y - self._percentile_95) / self._width
        else:
            raise ValueError(self._rtrans)
        return FlatRewards(flat_r)

    def inverse_flat_reward_transform(self, rp):
        if self._rtrans == 'exp':
            return -np.log(rp) * self._width + self._min
        elif self._rtrans == 'unit':
            return (1 - rp) * self._width + self._min
        elif self._rtrans == 'unit+95p':
            return (1 - rp + (1 - self._percentile_95)) * self._width + self._min

    def load_task_models(self):
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        # TODO: this path should be part of the config?
        state_dict = torch.load('/data/chem/qm9/mxmnet_gap_model.pt')
        gap_model.load_state_dict(state_dict)
        gap_model.cuda()
        gap_model, self.device = self._wrap_model(gap_model)
        return {'mxmnet_gap': gap_model}

    def sample_conditional_information(self, n: int) -> Dict[str, Tensor]:
        beta = None
        if self.temperature_sample_dist == 'constant':
            assert type(self.temperature_dist_params) in [float, int]
            beta = np.array(self.temperature_dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            if self.temperature_sample_dist == 'gamma':
                loc, scale = self.temperature_dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.temperature_sample_dist == 'uniform':
                beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'loguniform':
                beta = np.exp(self.rng.uniform(*np.log(self.temperature_dist_params), n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'beta':
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)
        
        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {'beta': beta, 'encoding': beta_enc}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        flat_reward = flat_reward.squeeze()
        assert len(flat_reward.shape) == len(cond_info['beta'].shape), \
            f"dangerous shape mismatch: {flat_reward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(torch.log(flat_reward + 1e-8) * cond_info['beta'])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [mxmnet.mol2graph(i) for i in mols]  # type: ignore[attr-defined]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models['mxmnet_gap'](batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV  # type: ignore[attr-defined]
        preds[preds.isnan()] = 1
        preds = self.flat_reward_transform(preds).clip(1e-4, 2).reshape((-1, 1))
        return FlatRewards(preds), is_valid


class QM9GapTrainer(GFNTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            'bootstrap_own_reward': False,
            'learning_rate': 1e-4,
            'global_batch_size': 64,
            'num_emb': 128,
            'num_layers': 4,
            'tb_epsilon': None,
            'illegal_action_logreward': -75,
            'reward_loss_multiplier': 1,
            'temperature_sample_dist': 'uniform',
            'temperature_dist_params': (.5, 32.),
            'weight_decay': 1e-8,
            'num_data_loader_workers': 8,
            'momentum': 0.9,
            'adam_eps': 1e-8,
            'lr_decay': 20000,
            'Z_lr_decay': 20000,
            'clip_grad_type': 'norm',
            'clip_grad_param': 10,
            'random_action_prob': .001,
            'sampling_tau': 0.,
            'num_thermometer_dim': 32,
        }

    def setup(self):
        hps = self.hps
        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.ctx = MolBuildingEnvContext(['H', 'C', 'N', 'F', 'O'], num_cond_dim=32)
        self.training_data = QM9Dataset(hps['qm9_h5_path'], train=True, target='gap')
        self.test_data = QM9Dataset(hps['qm9_h5_path'], train=False, target='gap')

        model = GraphTransformerGFN(self.ctx, num_emb=hps['num_emb'], num_layers=hps['num_layers'])
        self.model = model
        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(non_Z_params, hps['learning_rate'], (hps['momentum'], 0.999),
                                    weight_decay=hps['weight_decay'], eps=hps['adam_eps'])
        self.opt_Z = torch.optim.Adam(Z_params, hps['learning_rate'], (0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2**(-steps / hps['lr_decay']))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2**(-steps / hps['Z_lr_decay']))

        self.sampling_tau = hps['sampling_tau']
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(model)
        else:
            self.sampling_model = self.model
        eps = hps['tb_epsilon']
        hps['tb_epsilon'] = ast.literal_eval(eps) if isinstance(eps, str) else eps
        self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, hps, max_nodes=9)

        self.task = QM9GapTask(dataset=self.training_data, temperature_distribution=hps['temperature_sample_dist'],
                               temperature_parameters=hps['temperature_dist_params'],
                               num_thermometer_dim=hps['num_thermometer_dim'], wrap_model=self._wrap_model_mp)
        self.mb_size = hps['global_batch_size']
        self.clip_grad_param = hps['clip_grad_param']
        self.clip_grad_callback = {
            'value': (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            'norm': (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            'none': (lambda x: None)
        }[hps['clip_grad_type']]

    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))


def main():
    """Example of how this model can be run outside of Determined"""
    yaml = YAML(typ="safe", pure=True)
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qm9.yaml')
    with open(config_file, 'r') as f:
        hps = yaml.load(f)
    trial = QM9GapTrainer(hps, torch.device('cpu'))
    trial.run()


if __name__ == '__main__':
    main()
