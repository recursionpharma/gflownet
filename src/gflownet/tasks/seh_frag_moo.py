import ast
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit.Chem.rdchem import Mol as RDMol
import scipy.stats as stats
import torch
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.data as gd

from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import RewardScalar
from gflownet.utils import metrics
from gflownet.utils import sascore
from gflownet.utils.transforms import thermometer


class SEHMOOTask(GFNTask):
    """Sets up a multiobjective task where the rewards are (functions of):
    - the the binding energy of a molecule to Soluble Epoxide Hydrolases.
    - its QED
    - its synthetic accessibility
    - its molecular weight

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """
    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float],
                 wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

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
        if self.preference_distribution == 'Dir[0.95]':
            m = Dirichlet(torch.FloatTensor([0.95] * 4))  # TODO: this hyperparameter might matter as well
            preferences = m.sample([n])
        elif self.preference_distribution == 'const':
            preferences = torch.ones((n, 4)).float() / 4
        else:
            a = np.random.dirichlet([1] * 4, n)
            b = np.random.exponential(1, n)[:, None]
            preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': torch.tensor(beta), 'encoding': encoding, 'preferences': preferences}

    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_reward = (flat_reward * cond_info['preferences']).sum(1)
        return scalar_reward**cond_info['beta']

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 4))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        seh_preds = self.models['seh'](batch).reshape((-1,)).clip(1e-4, 100).data.cpu() / 8
        seh_preds[seh_preds.isnan()] = 0

        def safe(f, x, default):
            try:
                return f(x)
            except Exception:
                return default

        qeds = torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v.item()])
        sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i, v in zip(mols, is_valid) if v.item()])
        sas = (10 - sas) / 9  # Turn into a [0-1] reward
        molwts = torch.tensor([safe(Descriptors.MolWt, i, 1000) for i, v in zip(mols, is_valid) if v.item()])
        molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
        flat_rewards = torch.stack([seh_preds, qeds, sas, molwts], 1)
        return FlatRewards(flat_rewards), is_valid


class SEHMOOFragTrainer(SEHFragTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            **super().default_hps(),
            'use_fixed_weight': False,
            'num_cond_dim': 32 + 4,  # thermometer encoding of beta + 4 preferences
            'preference_distribution': 'Dir[0.95]',
        }

    def setup(self):
        super().setup()
        self.task = SEHMOOTask(self.training_data, self.hps['temperature_sample_dist'],
                               ast.literal_eval(self.hps['temperature_dist_params']), wrap_model=self._wrap_model_mp)
        self.task.preference_distribution = self.hps['preference_distribution']
        self.sampling_hooks.append(MultiObjectiveStatsHook(256))


class MultiObjectiveStatsHook:
    def __init__(self, num_to_keep: int):
        self.num_to_keep = num_to_keep
        self.all_flat_rewards: List[Tensor] = []
        self.hsri_epsilon = 0.3

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        self.all_flat_rewards = self.all_flat_rewards + list(flat_rewards)
        if len(self.all_flat_rewards) > self.num_to_keep:
            self.all_flat_rewards = self.all_flat_rewards[-self.num_to_keep:]

        flat_rewards = torch.stack(self.all_flat_rewards).numpy()
        target_min = flat_rewards.min(0).copy()
        target_range = flat_rewards.max(0).copy() - target_min
        hypercube_transform = metrics.Normalizer(
            loc=target_min,
            scale=target_range,
        )
        gfn_pareto = metrics.pareto_frontier(flat_rewards)
        normed_gfn_pareto = hypercube_transform(gfn_pareto)
        hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=True)
        hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=False)
        unnorm_hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=True)
        unnorm_hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=False)

        upper = np.zeros(normed_gfn_pareto.shape[-1]) + self.hsri_epsilon
        lower = np.ones(normed_gfn_pareto.shape[-1]) * -1 - self.hsri_epsilon
        hsr_indicator = metrics.HSR_Calculator(lower, upper)
        try:
            hsri_w_pareto, x = hsr_indicator.calculate_hsr(-1 * gfn_pareto)
        except Exception:
            hsri_w_pareto = 0
        try:
            hsri_on_flat, _ = hsr_indicator.calculate_hsr(-1 * flat_rewards)
        except Exception:
            hsri_on_flat = 0

        return {
            'HV with zero ref': hypervolume_with_zero_ref,
            'HV w/o zero ref': hypervolume_wo_zero_ref,
            'Unnormalized HV with zero ref': unnorm_hypervolume_with_zero_ref,
            'Unnormalized HV w/o zero ref': unnorm_hypervolume_wo_zero_ref,
            'hsri_with_pareto': hsri_w_pareto,
            'hsri_on_flat_rew': hsri_on_flat,
        }


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        'lr_decay': 10000,
        'log_dir': '/scratch/logs/seh_frag_moo/run_18/',
        'num_training_steps': 50_000,
        'validate_every': 500,
        'sampling_tau': 0.95,
        'num_layers': 5,
        'num_emb': 96,
        'weight_decay': 1e-4,
        'num_data_loader_workers': 12,
        'temperature_dist_params': '(1, 192)',
        'global_batch_size': 256,
    }
    trial = SEHMOOFragTrainer(hps, torch.device('cuda'))
    trial.verbose = True
    trial.run()


if __name__ == '__main__':
    main()
