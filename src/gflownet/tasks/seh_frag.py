import ast
import copy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
import scipy.stats as stats
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.data as gd

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerFragGFN
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import GFNTrainer
from gflownet.train import RewardScalar
from gflownet.utils.transforms import thermometer


class SEHTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching (TODO: port to this repo).
    """
    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float],
                 wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model)
        return {'seh': model}

    def sample_conditional_information(self, n):
        beta = None
        if self.temperature_sample_dist == 'gamma':
            loc, scale = self.temperature_dist_params
            beta = self.rng.gamma(loc, scale, n).astype(np.float32)
            upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif self.temperature_sample_dist == 'uniform':
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = 1
        beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)  # TODO: hyperparameters
        return {'beta': torch.tensor(beta), 'encoding': beta_enc}

    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        return flat_reward**cond_info['beta']

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0,))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models['seh'](batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid


class SEHFragTrainer(GFNTrainer):
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
            'temperature_dist_params': '(.5, 32)',
            'weight_decay': 1e-8,
            'num_data_loader_workers': 8,
            'momentum': 0.9,
            'adam_eps': 1e-8,
            'lr_decay': 20000,
            'Z_lr_decay': 20000,
            'clip_grad_type': 'norm',
            'clip_grad_param': 10,
            'random_action_prob': 0.,
            'sampling_tau': 0.,
            'num_cond_dim': 32,
        }

    def setup(self):
        hps = self.hps
        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.ctx = FragMolBuildingEnvContext(max_frags=9, num_cond_dim=hps['num_cond_dim'])
        self.training_data = []
        self.test_data = []
        self.offline_ratio = 0
        self.valid_offline_ratio = 0

        model = GraphTransformerFragGFN(self.ctx, num_emb=hps['num_emb'], num_layers=hps['num_layers'])
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

        self.task = SEHTask(self.training_data, hps['temperature_sample_dist'],
                            ast.literal_eval(hps['temperature_dist_params']), wrap_model=self._wrap_model_mp)
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
    hps = {
        'lr_decay': 10000,
        'qm9_h5_path': '/data/chem/qm9/qm9.h5',
        'log_dir': '/scratch/logs/seh_frag/run_0/',
        'num_training_steps': 10_000,
        'validate_every': 100,
        'sampling_tau': 0.99,
        'temperature_dist_params': '(0, 64)',
    }
    trial = SEHFragTrainer(hps, torch.device('cuda'))
    trial.verbose = True
    trial.run()


if __name__ == '__main__':
    main()
