from copy import deepcopy
import json
import os
import pathlib
import shutil
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit.Chem.rdchem import Mol as RDMol
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
from gflownet.tasks.seh_frag import SEHTask
from gflownet.train import FlatRewards
from gflownet.train import RewardScalar
from gflownet.utils import metrics
from gflownet.utils import sascore
from gflownet.utils.focus_model import FocusModel
from gflownet.utils.focus_model import TabularFocusModel
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook
from gflownet.utils.multiobjective_hooks import TopKHook


class SEHMOOTask(SEHTask):
    """Sets up a multiobjective task where the rewards are (functions of):
    - the the binding energy of a molecule to Soluble Epoxide Hydrolases.
    - its QED
    - its synthetic accessibility
    - its molecular weight

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """

    def __init__(
        self,
        objectives: List[str],
        dataset: Dataset,
        temperature_sample_dist: str,
        temperature_parameters: Tuple[float, float],
        num_thermometer_dim: int,
        preference_type: str = None,
        focus_type: Union[list, str] = None,
        focus_cosim: float = 0.0,
        focus_limit_coef: float = 1.0,
        fixed_focus_dirs: torch.Tensor = None,
        illegal_action_logreward: float = None,
        focus_model: FocusModel = None,
        focus_model_training_limits: Tuple[int, int] = None,
        max_train_it: int = None,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.objectives = objectives
        self.dataset = dataset
        self.temperature_sample_dist = temperature_sample_dist
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim
        self.preference_type = preference_type
        self.seeded_preference = None
        self.experimental_dirichlet = False
        self.focus_type = focus_type
        self.focus_cosim = focus_cosim
        self.focus_limit_coef = focus_limit_coef
        self.fixed_focus_dirs = fixed_focus_dirs
        self.illegal_action_logreward = illegal_action_logreward
        self.focus_model = focus_model
        self.focus_model_training_limits = focus_model_training_limits
        self.max_train_it = max_train_it
        assert set(objectives) <= {"seh", "qed", "sa", "mw"} and len(objectives) == len(set(objectives))

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n)

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
                m = Dirichlet(torch.FloatTensor([1.0] * len(self.objectives)))
                preferences = m.sample([n])

        if self.fixed_focus_dirs is not None:
            focus_dir = torch.tensor(
                np.array(self.fixed_focus_dirs)[self.rng.choice(len(self.fixed_focus_dirs), n)].astype(np.float32)
            )
        elif self.focus_type == "dirichlet":
            m = Dirichlet(torch.FloatTensor([1.0] * len(self.objectives)))
            focus_dir = m.sample([n])
        elif self.focus_type == "hyperspherical":
            focus_dir = torch.tensor(
                metrics.sample_positiveQuadrant_ndim_sphere(n, len(self.objectives), normalisation="l2")
            ).float()
        elif self.focus_type is not None and "learned" in self.focus_type:
            if (
                self.focus_model is not None
                and train_it >= self.focus_model_training_limits[0] * self.max_train_it
            ):
                focus_dir = self.focus_model.sample_focus_directions(n)
            else:
                focus_dir = torch.tensor(
                    metrics.sample_positiveQuadrant_ndim_sphere(n, len(self.objectives), normalisation="l2")
                ).float()
        else:
            raise NotImplementedError(f"Unsupported focus_type={type(self.focus_type)}")

        cond_info["encoding"] = torch.cat([cond_info["encoding"], preferences, focus_dir], 1)
        cond_info["preferences"] = preferences
        cond_info["focus_dir"] = focus_dir
        return cond_info

    def encode_conditional_information(self, cond_info: Tensor) -> Dict[str, Tensor]:
        if self.temperature_sample_dist == "constant":
            beta = torch.ones(len(cond_info)) * self.temperature_dist_params
            beta_enc = torch.zeros((len(cond_info), self.num_thermometer_dim))
        else:
            beta = torch.ones(len(cond_info)) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((len(cond_info), self.num_thermometer_dim))
        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"

        preferences = cond_info[:, : len(self.objectives)].float()
        focus_dir = cond_info[:, len(self.objectives) :].float()
        encoding = torch.cat([beta_enc, cond_info], 1).float()

        return {
            "beta": beta,
            "encoding": encoding,
            "preferences": preferences,
            "focus_dir": focus_dir,
        }

    def relabel_condinfo_and_logrewards(
        self, cond_info: Dict[str, Tensor], log_rewards: Tensor, flat_rewards: FlatRewards, hindsight_idxs: Tensor
    ):
        if self.focus_type is None:
            return cond_info, log_rewards
        # only keep hindsight_idxs that actually correspond to a violated constraint
        _, in_focus_mask = metrics.compute_focus_coef(flat_rewards, cond_info["focus_dir"], self.focus_cosim)
        out_focus_mask = torch.logical_not(in_focus_mask)
        hindsight_idxs = hindsight_idxs[out_focus_mask[hindsight_idxs]]
        # relabels the focus_dirs and log_rewards
        cond_info["focus_dir"][hindsight_idxs] = nn.functional.normalize(flat_rewards[hindsight_idxs], dim=1)
        cond_info["encoding"] = torch.cat(
            [cond_info["encoding"][:, : self.num_thermometer_dim + len(self.objectives)], cond_info["focus_dir"]], 1
        )
        log_rewards = self.cond_info_to_logreward(cond_info, flat_rewards)
        return cond_info, log_rewards

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_logreward = (flat_reward * cond_info["preferences"]).sum(1).clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"

        if self.focus_type is not None:
            focus_coef, in_focus_mask = metrics.compute_focus_coef(
                flat_reward, cond_info["focus_dir"], self.focus_cosim, self.focus_limit_coef
            )
            scalar_logreward[in_focus_mask] += torch.log(focus_coef[in_focus_mask])
            scalar_logreward[~in_focus_mask] = self.illegal_action_logreward

        return RewardScalar(scalar_logreward * cond_info["beta"])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, len(self.objectives)))), is_valid

        else:
            flat_r = []
            if "seh" in self.objectives:
                batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
                batch.to(self.device)
                seh_preds = self.models["seh"](batch).reshape((-1,)).clip(1e-4, 100).data.cpu() / 8
                seh_preds[seh_preds.isnan()] = 0
                flat_r.append(seh_preds)

            def safe(f, x, default):
                try:
                    return f(x)
                except Exception:
                    return default

            if "qed" in self.objectives:
                qeds = torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v.item()])
                flat_r.append(qeds)

            if "sa" in self.objectives:
                sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i, v in zip(mols, is_valid) if v.item()])
                sas = (10 - sas) / 9  # Turn into a [0-1] reward
                flat_r.append(sas)

            if "mw" in self.objectives:
                molwts = torch.tensor([safe(Descriptors.MolWt, i, 1000) for i, v in zip(mols, is_valid) if v.item()])
                molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
                flat_r.append(molwts)

            flat_rewards = torch.stack(flat_r, dim=1)
            return FlatRewards(flat_rewards), is_valid


class SEHMOOFragTrainer(SEHFragTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            **super().default_hps(),
            "use_fixed_weight": False,
            "objectives": ["seh", "qed", "sa", "mw"],
            "sampling_tau": 0.95,
            "valid_sample_cond_info": False,
            "n_valid": 15,
            "n_valid_repeats": 128,
            "preference_type": "dirichlet",
            "focus_type": None,
            "focus_cosim": None,
            "focus_limit_coef": 1.0,
            "hindsight_ratio": 0.0,
            "focus_model_training_limits": None,
            "focus_model_state_space_res": None,
        }

    def setup_algo(self):
        hps = self.hps
        if hps["algo"] == "TB":
            self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps["algo"] == "SQL":
            self.algo = SoftQLearning(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps["algo"] == "A2C":
            self.algo = A2C(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps["algo"] == "MOREINFORCE":
            self.algo = MultiObjectiveReinforce(self.env, self.ctx, self.rng, hps, max_nodes=9)
        elif hps["algo"] == "MOQL":
            self.algo = EnvelopeQLearning(self.env, self.ctx, self.rng, hps, max_nodes=9)

        if hps["focus_type"] is not None and "learned" in hps["focus_type"]:
            if hps["focus_type"] == "learned-tabular":
                self.focus_model = TabularFocusModel(
                    device=self.device,
                    n_objectives=len(hps["objectives"]),
                    state_space_res=hps["focus_model_state_space_res"],
                    focus_cosim=hps["focus_cosim"],
                )
            else:
                raise NotImplementedError("Unknown focus model type {self.focus_type}")
        else:
            self.focus_model = None

    def setup_task(self):
        self.task = SEHMOOTask(
            objectives=self.hps["objectives"],
            dataset=self.training_data,
            temperature_sample_dist=self.hps["temperature_sample_dist"],
            temperature_parameters=self.hps["temperature_dist_params"],
            num_thermometer_dim=self.hps["num_thermometer_dim"],
            preference_type=self.hps["preference_type"],
            focus_type=self.hps["focus_type"],
            focus_cosim=self.hps["focus_cosim"],
            focus_limit_coef=self.hps["focus_limit_coef"],
            illegal_action_logreward=self.hps["illegal_action_logreward"],
            focus_model=self.focus_model,
            focus_model_training_limits=self.hps["focus_model_training_limits"],
            max_train_it=self.hps["num_training_steps"],
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_model(self):
        if self.hps["algo"] == "MOQL":
            model = GraphTransformerFragEnvelopeQL(
                self.ctx,
                num_emb=self.hps["num_emb"],
                num_layers=self.hps["num_layers"],
                num_objectives=len(self.hps["objectives"]),
            )
        else:
            model = GraphTransformerGFN(self.ctx, num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"])

        if self.hps["algo"] in ["A2C", "MOQL"]:
            model.do_mask = False
        self.model = model

    def setup_env_context(self):
        n_cond = self.hps["num_thermometer_dim"] + 2 * len(self.hps["objectives"])  # 1 for prefs and 1 for focus region
        self.ctx = FragMolBuildingEnvContext(max_frags=9, num_cond_dim=n_cond)

    def setup(self):
        super().setup()
        self.sampling_hooks.append(
            MultiObjectiveStatsHook(
                256,
                self.hps["log_dir"],
                compute_igd=True,
                compute_pc_entropy=True,
                compute_focus_accuracy=True if self.hps["focus_type"] is not None else False,
                focus_cosim=self.hps["focus_cosim"],
            )
        )

        # instantiate preference and focus conditioning vectors for validation

        n_obj = len(self.hps["objectives"])
        n_valid = self.hps["n_valid"]

        # making sure hyperparameters for preferences and focus regions are consistent
        if not (
            self.hps["focus_type"] is None
            or self.hps["focus_type"] == "centered"
            or (type(self.hps["focus_type"]) is list and len(self.hps["focus_type"]) == 1)
        ):
            assert self.hps["preference_type"] is None, (
                f"Cannot use preferences with multiple focus regions, here focus_type={self.hps['focus_type']} "
                f"and preference_type={self.hps['preference_type']}"
            )

        if type(self.hps["focus_type"]) is list and len(self.hps["focus_type"]) > 1:
            n_valid = len(self.hps["focus_type"])

        # preference vectors
        if self.hps["preference_type"] is None:
            valid_preferences = np.ones((n_valid, n_obj))
        elif self.hps["preference_type"] == "dirichlet":
            valid_preferences = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l1")
        elif self.hps["preference_type"] == "seeded_single":
            seeded_prefs = np.random.default_rng(142857 + int(self.hps["seed"])).dirichlet([1] * n_obj, n_valid)
            valid_preferences = seeded_prefs[0].reshape((1, n_obj))
            self.task.seeded_preference = valid_preferences[0]
        elif self.hps["preference_type"] == "seeded_many":
            valid_preferences = np.random.default_rng(142857 + int(self.hps["seed"])).dirichlet([1] * n_obj, n_valid)
        else:
            raise NotImplementedError(f"Unknown preference type {self.hps['preference_type']}")

        # focus regions
        if self.hps["focus_type"] is None:
            valid_focus_dirs = np.zeros((n_valid, n_obj))
            self.task.fixed_focus_dirs = valid_focus_dirs
        elif self.hps["focus_type"] == "centered":
            valid_focus_dirs = np.ones((n_valid, n_obj))
            self.task.fixed_focus_dirs = valid_focus_dirs
        elif self.hps["focus_type"] == "partitioned":
            valid_focus_dirs = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l2")
            self.task.fixed_focus_dirs = valid_focus_dirs
        elif self.hps["focus_type"] in ["dirichlet", "learned-gfn"]:
            valid_focus_dirs = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l1")
            self.task.fixed_focus_dirs = None
        elif self.hps["focus_type"] in ["hyperspherical", "learned-tabular"]:
            valid_focus_dirs = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l2")
            self.task.fixed_focus_dirs = None
        elif type(self.hps["focus_type"]) is list:
            if len(self.hps["focus_type"]) == 1:
                valid_focus_dirs = np.array([self.hps["focus_type"][0]] * n_valid)
                self.task.fixed_focus_dirs = valid_focus_dirs
            else:
                valid_focus_dirs = np.array(self.hps["focus_type"])
                self.task.fixed_focus_dirs = valid_focus_dirs
        else:
            raise NotImplementedError(
                f"focus_type should be None, a list of fixed_focus_dirs, or a string describing one of the supported "
                f"focus_type, but here: {self.hps['focus_type']}"
            )

        self.hps["fixed_focus_dirs"] = (
            np.unique(self.task.fixed_focus_dirs, axis=0).tolist() if self.task.fixed_focus_dirs is not None else None
        )
        with open(pathlib.Path(self.hps["log_dir"]) / "hps.json", "w") as f:
            json.dump(self.hps, f)
        assert valid_focus_dirs.shape == (
            n_valid,
            n_obj,
        ), f"Invalid shape for valid_preferences, {valid_focus_dirs.shape} != ({n_valid}, {n_obj})"

        # combine preferences and focus directions (fixed focus cosim)
        valid_cond_vector = np.concatenate([valid_preferences, valid_focus_dirs], axis=1)

        self._top_k_hook = TopKHook(10, self.hps["n_valid_repeats"], len(valid_cond_vector))
        self.test_data = RepeatedCondInfoDataset(valid_cond_vector, repeat=self.hps["n_valid_repeats"])
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
                    metrics[f"topk_rewards_{i}"] = top_k[i]
                print("validation end", metrics)

        return {"topk": TopKMetricCB()}

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        focus_model_training_limits = self.hps["focus_model_training_limits"]
        max_train_it = self.hps["num_training_steps"]
        if (
            self.focus_model is not None
            and train_it >= focus_model_training_limits[0] * max_train_it
            and train_it <= focus_model_training_limits[1] * max_train_it
        ):
            self.focus_model.update_belief(deepcopy(batch.focus_dir), deepcopy(batch.flat_rewards))
        return super().train_batch(batch, epoch_idx, batch_idx, train_it)

    def _save_state(self, it):
        if self.focus_model is not None:
            self.focus_model.save(pathlib.Path(self.hps["log_dir"]))
        return super()._save_state(it)


class RepeatedCondInfoDataset:
    def __init__(self, cond_info_vectors, repeat):
        self.cond_info_vectors = cond_info_vectors
        self.repeat = repeat

    def __len__(self):
        return len(self.cond_info_vectors) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return torch.tensor(self.cond_info_vectors[int(idx // self.repeat)])


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "./logs/debug_run",
        "overwrite_existing_exp": True,
        "seed": 0,
        "global_batch_size": 8,
        "num_training_steps": 4,
        "num_final_gen_steps": 3,
        "validate_every": 10,
        "num_layers": 2,
        "num_emb": 256,
        "algo": "TB",
        "objectives": ["seh", "qed"],
        "learning_rate": 1e-4,
        "Z_learning_rate": 1e-3,
        "lr_decay": 20000,
        "Z_lr_decay": 50000,
        "sampling_tau": 0.95,
        "random_action_prob": 0.1,
        "num_data_loader_workers": 0,
        "temperature_sample_dist": "constant",
        "temperature_dist_params": 60.0,
        "num_thermometer_dim": 32,
        "preference_type": None,
        "focus_type": "learned-tabular",
        "focus_cosim": 0.98,
        "focus_limit_coef": 1e-1,
        "n_valid": 15,
        "n_valid_repeats": 8,
        "use_replay_buffer": True,
        "replay_buffer_warmup": 0,
        "hindsight_ratio": 0.3,
        "mp_pickle_messages": True,
        "focus_model_training_limits": [0.25, 0.75],
        "focus_model_state_space_res": 10,
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHMOOFragTrainer(hps, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    try:
        main()
    except Warning as e:
        print(e)
        exit(1)
