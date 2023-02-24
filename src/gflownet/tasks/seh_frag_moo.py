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

from gflownet.algo.advantage_actor_critic import A2C
from gflownet.algo.envelope_q_learning import EnvelopeQLearning
from gflownet.algo.envelope_q_learning import GraphTransformerFragEnvelopeQL
from gflownet.algo.multiobjective_reinforce import MultiObjectiveReinforce
from gflownet.algo.soft_q_learning import SoftQLearning
from gflownet.algo.trajectory_balance import TrajectoryBalance
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
    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float],
                 wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.seeded_preference = None
        self.experimental_dirichlet = False

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
        elif self.temperature_sample_dist == 'loguniform':
            beta = np.exp(self.rng.uniform(*np.log(self.temperature_dist_params), n).astype(np.float32))
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = 1
        beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)  # TODO: hyperparameters
        if self.seeded_preference is not None:
            preferences = torch.tensor([self.seeded_preference] * n).float()
        elif self.experimental_dirichlet:
            a = np.random.dirichlet([1] * 4, n)
            b = np.random.exponential(1, n)[:, None]
            preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
        else:
            m = Dirichlet(torch.FloatTensor([1.] * 4))
            preferences = m.sample([n])
        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': torch.tensor(beta), 'encoding': encoding, 'preferences': preferences}

    def encode_conditional_information(self, info):
        # This assumes we're using a constant (max) beta and that info is the preferences
        encoding = torch.cat([torch.ones((len(info), 32)), info], 1)
        return {
            'beta': torch.ones(len(info)) * self.temperature_dist_params[-1],
            'encoding': encoding.float(),
            'preferences': info.float()
        }

    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_reward = (flat_reward * cond_info['preferences']).sum(1).log()
        return RewardScalar(scalar_reward * cond_info['beta'])

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
            'sampling_tau': 0.95,
            'valid_sample_cond_info': False,
            'preference_type': 'dirichlet',
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
        self.task = SEHMOOTask(self.training_data, self.hps['temperature_sample_dist'],
                               ast.literal_eval(self.hps['temperature_dist_params']), wrap_model=self._wrap_model_mp)

    def setup_model(self):
        if self.hps['algo'] == 'MOQL':
            model = GraphTransformerFragEnvelopeQL(self.ctx, num_emb=self.hps['num_emb'],
                                                   num_layers=self.hps['num_layers'], num_objectives=4)
        else:
            model = GraphTransformerGFN(self.ctx, num_emb=self.hps['num_emb'], num_layers=self.hps['num_layers'])

        if self.hps['algo'] in ['A2C', 'MOQL']:
            model.do_mask = False
        self.model = model

    def setup(self):
        super().setup()
        self.task = SEHMOOTask(self.training_data, self.hps['temperature_sample_dist'],
                               ast.literal_eval(self.hps['temperature_dist_params']), wrap_model=self._wrap_model_mp)
        self.sampling_hooks.append(MultiObjectiveStatsHook(256, self.hps['log_dir']))
        if self.hps['preference_type'] == 'dirichlet':
            valid_preferences = metrics.generate_simplex(4, 5)  # This yields 35 points of dimension 4
        elif self.hps['preference_type'] == 'seeded_single':
            seeded_prefs = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet([1] * 4, 10)
            valid_preferences = seeded_prefs[int(self.hps['single_pref_target_idx'])].reshape((1, 4))
            self.task.seeded_preference = valid_preferences[0]
        elif self.hps['preference_type'] == 'seeded_many':
            valid_preferences = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet([1] * 4, 10)
        self._top_k_hook = TopKHook(10, 128, len(valid_preferences))
        self.test_data = RepeatedPreferenceDataset(valid_preferences, 128)
        self.valid_sampling_hooks.append(self._top_k_hook)

        self.algo.task = self.task

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
        'lr_decay': 10000,
        'log_dir': '/scratch/emmanuel.bengio/logs/seh_frag_moo/run_tmp/',
        'num_training_steps': 20_000,
        'validate_every': 500,
        'sampling_tau': 0.95,
        'num_layers': 6,
        'num_data_loader_workers': 12,
        'temperature_dist_params': '(1, 192)',
        'global_batch_size': 256,
        'algo': 'TB',
        'sql_alpha': 0.01,
        'seed': 0,
        'preference_type': 'seeded_many',
    }
    trial = SEHMOOFragTrainer(hps, torch.device('cuda'))
    trial.verbose = True
    trial.run()


if __name__ == '__main__':
    main()
