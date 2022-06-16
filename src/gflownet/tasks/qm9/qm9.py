import ast
import copy
from typing import Any, Callable, Dict, List, Tuple, Union, NewType

from determined.pytorch import LRScheduler
from determined.pytorch import PyTorchTrial
from determined.pytorch import PyTorchTrialContext
import numpy as np
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem import Descriptors
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.data as gd
from torch.distributions.dirichlet import Dirichlet
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.data.qm9 import QM9Dataset
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.models import mxmnet
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import GFNTrainer
from gflownet.train import RewardScalar

'''
gap: minimize
logP: around 4
QED: maximize
molecular_weight: median

'''
def thermometer(v: Tensor, n_bins=50, vmin=0, vmax=1) -> Tensor:
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap

class RewardInfo:
    def __init__(self, _min: float, _max: float, _median:float,  _percentile_95: float):
        self._min = _min
        self._max = _max
        self._percentile_95 = _percentile_95
        self._median = _median
        self._width = self._max - self._min

def gap_reward(_input, _info):
    return 1 - (_input - _info._percentile_95) / _info._width

def logP_reward(_input, _info):
    return np.exp(-(_input  - 1) ** 2 / (2 * np.pi * _info._width))

def molecular_weight_reward(_input, _info):
    return np.exp(-(_input-_info._median)**2 /(2 * np.pi * _info._width))

def qed_reward(_input, _info):
    return _input / _info._width

class QM9GapTask(GFNTask):
    """This class captures conditional information generation and reward transforms"""
    def __init__(
            self,
            dataset: Dataset,
            temperature_distribution: str,
            uniform_temperature_parameters: Tuple[float],
            gamma_temperature_parameters: Tuple[float],
            beta_temperature_parameters: Tuple[float],
            temperature_max_min: Tuple[int],
            const_temp: int,
            number_of_objectives: int,
            reward_transform: str,
            targets: list,
            wrap_model: Callable[[nn.Module], nn.Module] = None
    ):
        self._wrap_model = wrap_model
        self.models = self.load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        if temperature_distribution == "uniform":
            self.temperature_dist_params = uniform_temperature_parameters
        elif temperature_distribution == "gamma":
            self.temperature_dist_params = gamma_temperature_parameters
        elif temperature_distribution == "beta":
            self.temperature_dist_params = beta_temperature_parameters
        else:
            self.temperature_dist_params = None
        self.temperature_max_min = list(map(tuple, temperature_max_min))
        self.const_temp = const_temp
        self.number_of_objectives = number_of_objectives
        self._rtrans = reward_transform
        self.reward_stat_info = []
        self.targets = targets
        for i in range(number_of_objectives):
            _min, _max, _median,  _percentile_95 = self.dataset.get_stats(percentile=0.05, target=targets[i])
            self.reward_stat_info.append(RewardInfo(_min=_min, _max=_max, _median=_median, _percentile_95=_percentile_95))

    def flat_reward_transform(self, y: Union[list, np.ndarray, float, Tensor]) -> FlatRewards:
        """Transforms a target quantity y (e.g. the LUMO energy in QM9) to a positive reward scalar"""
        # This assumes that y is a list. Check if that is indeed a case. If yes, turn that into an array
        if isinstance(y, list):
            y = np.vstack(y)
        rewards = []
        for i in range(self.number_of_objectives):
            rew = self._transform(y[:, i], self.reward_stat_info[i], self.targets[i])
            rewards.append(rew)
        rewards = np.vstack(rewards).reshape(-1, self.number_of_objectives)
        return rewards

    def _transform(self, y: Union[float, Tensor], reward_stat_info, target: str):
        if target == 'gap':
            return gap_reward(y, reward_stat_info)
        elif target == 'logP':
            return logP_reward(y, reward_stat_info)
        elif target == 'molecular_weight':
            return molecular_weight_reward(y, reward_stat_info)
        elif target == 'QED':
            return qed_reward(y, reward_stat_info)
        raise ValueError(self._rtrans)

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

    def sample_conditional_information(self, n):
        # Beta Info
        beta = None
        # TODO Sharath: Inlcude Annealing
        if self.temperature_sample_dist == 'gamma':
            beta = self.rng.gamma(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == 'uniform':
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == 'beta':
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == 'const':
            beta = np.ones(n).astype(np.float32)
            beta = beta * self.const_temp
        # Thermometer encode the beta
        beta_enc = thermometer(torch.tensor(beta), 32, 0, 32)
        # Get the preferences
        m = Dirichlet(torch.FloatTensor([1.5] * self.number_of_objectives))
        preferences = m.sample([n])
        encoding = torch.cat([preferences, beta_enc], dim=-1)
        return {'beta': torch.tensor(beta).unsqueeze(1), 'encoding': encoding, 'preferences': preferences}

    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        # print(f"Flat Before: {flat_reward}")
        if isinstance(flat_reward, list):
            flat_reward = np.vstack(flat_reward)
            flat_reward = torch.bmm(
                torch.FloatTensor(flat_reward).unsqueeze(1), cond_info["preferences"].unsqueeze(1).permute(0, 2, 1)
            ).squeeze(1)
        else:
            flat_reward = flat_reward.dot(cond_info["preferences"])
        return flat_reward**cond_info['beta']

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[RewardScalar, Tensor]:
        all_preds = []
        graphs = [mxmnet.mol2graph(i) for i in mols]  # type: ignore[attr-defined]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return RewardScalar(torch.zeros((0,))), is_valid
        for target in self.targets:
            if target == 'gap':
                batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
                batch.to(self.device)
                preds = self.models['mxmnet_gap'](batch).reshape((-1, 1)).data.cpu().numpy() / mxmnet.HAR2EV  # type: ignore[attr-defined]                
            elif target == 'logP':
                preds = np.asarray([Descriptors.MolLogP(i) for idx, i in enumerate(mols) if graphs[idx] is not None]).reshape((-1,1 ))
            elif target == 'molecular_weight':
                preds = np.asarray([Descriptors.MolWt(i) for idx, i in enumerate(mols) if graphs[idx] is not None]).reshape((-1, 1))
            elif target == 'QED':
                preds = np.asarray([Descriptors.qed(i) for idx, i in enumerate(mols) if graphs[idx] is not None]).reshape((-1, 1))
            else:
                preds = []
            preds[np.isnan(preds)] = 1
            all_preds.append(preds)
        all_preds = np.hstack(all_preds)
        preds = self.flat_reward_transform(all_preds).clip(1e-4, 2) # TODO: Is this clipping valid for all the rewards?
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
            'illegal_action_logreward': -50,
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
            'random_action_prob': .001,
            'sampling_tau': 0.,
        }

    def setup(self):
        hps = self.hps
        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        cond_dim = 32 + self.hps["number_of_objectives"]
        self.ctx = MolBuildingEnvContext(['H', 'C', 'N', 'F', 'O'], num_cond_dim=cond_dim)
        self.training_data = QM9Dataset(hps['qm9_h5_path'], train=True, targets=hps['targets'][:hps['number_of_objectives']])
        self.test_data = QM9Dataset(hps['qm9_h5_path'], train=False, targets=hps['targets'][:hps['number_of_objectives']])

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
        self.task = QM9GapTask(
            dataset=self.training_data,
            temperature_distribution=hps['temperature_sample_dist'],
            uniform_temperature_parameters=ast.literal_eval(hps['uniform_temperature_dist_params']),
            gamma_temperature_parameters=ast.literal_eval(hps['gamma_temperature_dist_params']),
            beta_temperature_parameters=ast.literal_eval(hps['beta_temperature_dist_params']),
            temperature_max_min=hps['temperature_max_min'],
            const_temp=hps['const_temp'],
            number_of_objectives=hps['number_of_objectives'],
            reward_transform=hps['reward_transform'],
            targets=hps['targets'][:hps['number_of_objectives']],
            wrap_model=self._wrap_model_mp
        )
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


# Determined-specific code:
class QM9Trial(QM9GapTrainer, PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        QM9GapTrainer.__init__(self, context.get_hparams(), context.device)  # type: ignore
        self.mb_size = context.get_per_slot_batch_size()
        self.context = context
        self.model = context.wrap_model(self.model)
        if context.get_hparam('sampling_tau') > 0:
            self.sampling_model = context.wrap_model(self.sampling_model)

        self._opt = context.wrap_optimizer(self.opt)
        self._opt_Z = context.wrap_optimizer(self.opt_Z)
        context.wrap_lr_scheduler(self.lr_sched, LRScheduler.StepMode.STEP_EVERY_BATCH)
        context.wrap_lr_scheduler(self.lr_sched_Z, LRScheduler.StepMode.STEP_EVERY_BATCH)

        # See docs.determined.ai/latest/training-apis/api-pytorch-advanced.html#customizing-a-reproducible-dataset
        if isinstance(context, PyTorchTrialContext):
            context.experimental.disable_dataset_reproducibility_checks()

    def get_batch_length(self, batch):
        return batch.traj_lens.shape[0]

    def log(self, info, index, key):
        pass  # Override this method since Determined is doing the logging for us

    def step(self, loss):
        self.context.backward(loss)
        self.context.step_optimizer(self._opt, clip_grads=self.clip_grad_callback)
        self.context.step_optimizer(self._opt_Z, clip_grads=self.clip_grad_callback)
        # This isn't wrapped in self.context, would probably break the Trial API
        # TODO: fix, if we go to multi-gpu
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        'lr_decay': 10000,
        'qm9_h5_path': '/data/chem/qm9/qm9.h5',
        'log_dir': '/mnt/ps/home/CORP/sharath.raparthy/sandbox/qm9_results/logs/',
        'num_training_steps': 100_000,
        'validate_every': 100,
        'number_of_objectives': 2,
        'targets': ["gap", "logP", "QED", "molecular_weight"],
        'temperature_sample_dist': 'gamma',
        'temperature_max_min': '(0, 32)',
        'reward_transform':  'unit+95p',
        'tb_epsilon': None,
        'weight_decay': 1e-8,
        'lr_decay': 100000,
        'Z_lr_decay': 100000,
        'temperature_dist_params': '(2, 2)',
        'bootstrap_own_reward': False,
        'clip_grad_type': 'norm',
        'clip_grad_param': 10,
        'global_batch_size': 64,
        'illegal_action_logreward': -50,
        'learning_rate': 1e-4,
        'momentum': 0.9,
        'num_data_loader_workers': 1,
        'num_emb': 128,
        'num_layers': 4,
        'const_temp': 8,
    }
    trial = QM9GapTrainer(hps, torch.device('cuda'))
    trial.run()


if __name__ == '__main__':
    main()
