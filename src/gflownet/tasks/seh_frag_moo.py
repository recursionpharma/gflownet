import ast
import json
import math
import os
import pathlib
import shutil
from typing import Any, Callable, Dict, List, Tuple, Union
import math

import git
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

from gflownet.algo.advantage_actor_critic import A2C
from gflownet.algo.envelope_q_learning import EnvelopeQLearning
from gflownet.algo.envelope_q_learning import GraphTransformerFragEnvelopeQL
from gflownet.algo.multiobjective_reinforce import MultiObjectiveReinforce
from gflownet.algo.soft_q_learning import SoftQLearning
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import RewardScalar
from gflownet.utils import metrics
from gflownet.utils import sascore
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook
from gflownet.utils.multiobjective_hooks import TopKHook
from gflownet.utils.transforms import thermometer


class SEHMOOTask(GFNTask):
    """Sets up a multiobjective task where the rewards are (functions of):
    - the the binding energy of a molecule to Soluble Epoxide Hydrolases.
    - its QED
    - its synthetic accessibility
    - its molecular weight

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """
    def __init__(self, objectives: List[str], dataset: Dataset, temperature_sample_dist: str,
                 temperature_parameters: Tuple[float], num_thermometer_dim: int,
                 preference_type: str = None, focus_dir: Tuple[float] = None, focus_cosim: float = None,
                 illegal_action_logreward: float = None, wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.models = self._load_task_models()
        self.objectives = objectives
        self.dataset = dataset
        self.temperature_sample_dist = temperature_sample_dist
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim
        self.preference_type = preference_type
        self.seeded_preference = None
        self.experimental_dirichlet = False
        self.focus_dir = focus_dir
        self.focus_cosim = focus_cosim
        self.illegal_action_logreward = illegal_action_logreward
        assert not ((self.focus_dir is None) ^ (self.focus_cosim is None))
        assert set(objectives) <= {'seh', 'qed', 'sa', 'mw'} and len(objectives) == len(set(objectives))

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model)
        return {'seh': model}

    def sample_conditional_information(self, n: int) -> Dict[str, Tensor]:
        beta = None
        if self.temperature_sample_dist == 'constant':
            assert type(self.temperature_dist_params) in [float, int]
            beta = np.array(self.temperature_dist_params, dtype=np.float32).repeat(n)
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

        if self.preference_type is None:
            preferences = torch.ones((n, len(self.objectives)))
        else:
            if self.seeded_preference is not None:
                preferences = torch.tensor([self.seeded_preference] * n).float()
            elif self.experimental_dirichlet:
                a = np.random.dirichlet([1] * len(self.objectives), n)
                b = np.random.exponential(1, n)[:, None]
                preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
            else:
                m = Dirichlet(torch.FloatTensor([1.] * len(self.objectives)))
                preferences = m.sample([n])

        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': torch.tensor(beta), 'encoding': encoding, 'preferences': preferences}

    def encode_conditional_information(self, preferences: torch.TensorType) -> Dict[str, Tensor]:
        if self.temperature_sample_dist == 'constant':
            beta = torch.ones(len(preferences)) * self.temperature_dist_params
            beta_enc = torch.zeros((len(preferences), self.num_thermometer_dim))
        else:
            beta = torch.ones(len(preferences)) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((len(preferences), self.num_thermometer_dim))

        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': beta, 'encoding': encoding.float(), 'preferences': preferences.float()}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_logreward = torch.log((flat_reward * cond_info['preferences']).sum(1) + 1e-8)
        if self.focus_dir is not None:
            cosim = nn.functional.cosine_similarity(flat_reward, torch.tensor(self.focus_dir), dim=1)
            scalar_logreward[cosim < self.focus_cosim] = self.illegal_action_logreward
        return RewardScalar(scalar_logreward * cond_info['beta'])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, len(self.objectives)))), is_valid

        else:
            flat_rewards = []
            if 'seh' in self.objectives:
                batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
                batch.to(self.device)
                seh_preds = self.models['seh'](batch).reshape((-1,)).clip(1e-4, 100).data.cpu() / 8
                seh_preds[seh_preds.isnan()] = 0
                flat_rewards.append(seh_preds)

            def safe(f, x, default):
                try:
                    return f(x)
                except Exception:
                    return default

            if "qed" in self.objectives:
                qeds = torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v.item()])
                flat_rewards.append(qeds)

            if "sa" in self.objectives:
                sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i, v in zip(mols, is_valid) if v.item()])
                sas = (10 - sas) / 9  # Turn into a [0-1] reward
                flat_rewards.append(sas)

            if "mw" in self.objectives:
                molwts = torch.tensor([safe(Descriptors.MolWt, i, 1000) for i, v in zip(mols, is_valid) if v.item()])
                molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
                flat_rewards.append(molwts)

            flat_rewards = torch.stack(flat_rewards, dim=1)
            return FlatRewards(flat_rewards), is_valid


class SEHMOOFragTrainer(SEHFragTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            **super().default_hps(),
            'use_fixed_weight': False,
            'objectives': ['seh', 'qed', 'sa', 'mw'],
            'sampling_tau': 0.95,
            'valid_sample_cond_info': False,
            'n_valid_prefs': 15,
            'n_valid_repeats_per_pref': 128,
            'preference_type': 'dirichlet',
            'focus_dir': None,
            'focus_cosim': None,
        }

    def setup_algo(self):
        hps = self.hps
        if hps['algo'] == 'TB':
            self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps['algo'] == 'SQL':
            self.algo = SoftQLearning(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps['algo'] == 'A2C':
            self.algo = A2C(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps['algo'] == 'MOREINFORCE':
            self.algo = MultiObjectiveReinforce(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps['algo'] == 'MOQL':
            self.algo = EnvelopeQLearning(self.env, self.ctx, self.rng, hps, max_nodes=9)

    def setup_task(self):
        self.task = SEHMOOTask(objectives=self.hps['objectives'], dataset=self.training_data,
                               temperature_sample_dist=self.hps['temperature_sample_dist'],
                               temperature_parameters=ast.literal_eval(self.hps['temperature_dist_params']),
                               num_thermometer_dim=self.hps['num_thermometer_dim'],
                               preference_type=self.hps['preference_type'],
                               focus_dir=ast.literal_eval(self.hps['focus_dir']),
                               focus_cosim=self.hps['focus_cosim'],
                               illegal_action_logreward=self.hps['illegal_action_logreward'],
                               wrap_model=self._wrap_model_mp)

    def setup_model(self):
        if self.hps['algo'] == 'MOQL':
            model = GraphTransformerFragEnvelopeQL(self.ctx, num_emb=self.hps['num_emb'],
                                                   num_layers=self.hps['num_layers'],
                                                   num_objectives=len(self.hps['objectives']))
        else:
            model = GraphTransformerGFN(self.ctx, num_emb=self.hps['num_emb'], num_layers=self.hps['num_layers'])

        if self.hps['algo'] in ['A2C', 'MOQL']:
            model.do_mask = False
        self.model = model

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(max_frags=9,
                                             num_cond_dim=self.hps['num_thermometer_dim'] + len(self.hps['objectives']))

    def setup(self):
        super().setup()
        self.task = SEHMOOTask(objectives=self.hps['objectives'], dataset=self.training_data,
                               temperature_sample_dist=self.hps['temperature_sample_dist'],
                               temperature_parameters=ast.literal_eval(self.hps['temperature_dist_params']),
                               num_thermometer_dim=self.hps['num_thermometer_dim'],
                               preference_type=self.hps['preference_type'],
                               focus_dir=ast.literal_eval(self.hps['focus_dir']),
                               focus_cosim=self.hps['focus_cosim'],
                               illegal_action_logreward=self.hps['illegal_action_logreward'],
                               wrap_model=self._wrap_model_mp)

        self.sampling_hooks.append(MultiObjectiveStatsHook(256, self.hps['log_dir']))

        n_obj = len(self.hps['objectives'])
        if self.hps['preference_type'] is None:
            valid_preferences = np.ones((self.hps['n_valid_prefs'], n_obj))
        elif self.hps['preference_type'] == 'dirichlet':
            valid_preferences = metrics.generate_simplex(n_obj, n_per_dim=math.ceil(self.hps['n_valid_prefs'] / n_obj))
        elif self.hps['preference_type'] == 'seeded_single':
            seeded_prefs = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet([1] * n_obj,
                                                                                           self.hps['n_valid_prefs'])
            valid_preferences = seeded_prefs[0].reshape((1, n_obj))
            self.task.seeded_preference = valid_preferences[0]
        elif self.hps['preference_type'] == 'seeded_many':
            valid_preferences = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet(
                [1] * n_obj, self.hps['n_valid_prefs'])
        else:
            raise NotImplementedError(f"Unknown preference type {self.hps['preference_type']}")

        self._top_k_hook = TopKHook(10, self.hps['n_valid_repeats_per_pref'], len(valid_preferences))
        self.test_data = RepeatedPreferenceDataset(valid_preferences, self.hps['n_valid_repeats_per_pref'])
        self.valid_sampling_hooks.append(self._top_k_hook)

        self.algo.task = self.task

        git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        self.hps['gflownet_git_hash'] = git_hash

        os.makedirs(self.hps['log_dir'], exist_ok=True)
        fmt_hps = '\n'.join([f"{f'{k}':40}:\t{f'({type(v).__name__})':10}\t{v}" for k, v in self.hps.items()])
        print(f"\n\nHyperparameters:\n{'-'*50}\n{fmt_hps}\n{'-'*50}\n\n")
        json.dump(self.hps, open(pathlib.Path(self.hps['log_dir']) / 'hps.json', 'w'))

    def build_callbacks(self):
        # We use this class-based setup to be compatible with the DeterminedAI API, but no direct
        # dependency is required.
        parent = self

        class TopKMetricCB:
            def on_validation_end(self, metrics: Dict[str, Any]):
                top_k = parent._top_k_hook.finalize()
                for i in range(len(top_k)):
                    metrics[f'topk_rewards_{i}'] = top_k[i]
                print('validation end', metrics)

        return {'topk': TopKMetricCB()}


class RepeatedPreferenceDataset:
    def __init__(self, preferences, repeat):
        self.prefs = preferences
        self.repeat = repeat

    def __len__(self):
        return len(self.prefs) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return torch.tensor(self.prefs[int(idx // self.repeat)])


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        'objectives': ['seh', 'qed', 'sa'],
        'focus_dir': '(1., 1., 1.)',
        'focus_cosim': 0.99,
        'log_dir': '/mnt/ps/home/CORP/julien.roy/logs/seh_frag_moo/debug_run/',
        'num_training_steps': 20_000,
        'validate_every': 1,
        'sampling_tau': 0.95,
        'num_layers': 4,
        'num_data_loader_workers': 8,
        'temperature_sample_dist': 'constant',
        'temperature_dist_params': '60.',
        'num_thermometer_dim': 32,
        'global_batch_size': 64,
        'algo': 'TB',
        'sql_alpha': 0.01,
        'seed': 0,
        'preference_type': 'dirichlet',
        'n_valid_prefs': 15,
        'n_valid_repeats_per_pref': 8,
    }
    if os.path.exists(hps['log_dir']):
        shutil.rmtree(hps['log_dir'])
    trial = SEHMOOFragTrainer(hps, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    trial.verbose = True
    trial.run()


if __name__ == '__main__':
    try:
        main()
    except Warning as e:
        print(e)
        exit(1)
