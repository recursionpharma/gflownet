import os
import pathlib
import shutil
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import Dataset

from gflownet.algo.envelope_q_learning import EnvelopeQLearning, GraphTransformerFragEnvelopeQL
from gflownet.algo.multiobjective_reinforce import MultiObjectiveReinforce
from gflownet.config import Config
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag import SEHFragTrainer, SEHTask
from gflownet.train import FlatRewards, RewardScalar
from gflownet.utils import metrics, sascore
from gflownet.utils.focus_model import FocusModel, TabularFocusModel
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook, TopKHook
from gflownet.utils.transforms import thermometer


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
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        focus_model: Optional[FocusModel] = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.cfg = cfg
        mcfg = self.cfg.task.seh_moo
        self.rng = rng
        self.models = self._load_task_models()
        self.objectives = cfg.task.seh_moo.objectives
        self.dataset = dataset
        self.temperature_sample_dist = mcfg.temperature_sample_dist
        self.temperature_dist_params = mcfg.temperature_dist_params
        self.num_thermometer_dim = mcfg.num_thermometer_dim
        self.use_steer_thermometer = mcfg.use_steer_thermometer
        self.preference_type = mcfg.preference_type
        self.seeded_preference = None
        self.experimental_dirichlet = False
        self.focus_type = mcfg.focus_type
        self.focus_cosim = mcfg.focus_cosim
        self.focus_limit_coef = mcfg.focus_limit_coef
        self.focus_model = focus_model
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.focus_model_training_limits = mcfg.focus_model_training_limits
        self.max_train_it = mcfg.max_train_it
        self.setup_focus_regions()
        assert set(self.objectives) <= {"seh", "qed", "sa", "mw"} and len(self.objectives) == len(set(self.objectives))

    def setup_focus_regions(self):
        mcfg = self.cfg.task.seh_moo
        n_valid = mcfg.n_valid
        n_obj = len(self.objectives)
        # focus regions
        if mcfg.focus_type is None:
            valid_focus_dirs = np.zeros((n_valid, n_obj))
            self.fixed_focus_dirs = valid_focus_dirs
        elif mcfg.focus_type == "centered":
            valid_focus_dirs = np.ones((n_valid, n_obj))
            self.fixed_focus_dirs = valid_focus_dirs
        elif mcfg.focus_type == "partitioned":
            valid_focus_dirs = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l2")
            self.fixed_focus_dirs = valid_focus_dirs
        elif mcfg.focus_type in ["dirichlet", "learned-gfn"]:
            valid_focus_dirs = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l1")
            self.fixed_focus_dirs = None
        elif mcfg.focus_type in ["hyperspherical", "learned-tabular"]:
            valid_focus_dirs = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l2")
            self.fixed_focus_dirs = None
        elif mcfg.focus_type == "listed":
            if len(mcfg.focus_type) == 1:
                valid_focus_dirs = np.array([mcfg.focus_dirs_listed[0]] * n_valid)
                self.fixed_focus_dirs = valid_focus_dirs
            else:
                valid_focus_dirs = np.array(mcfg.focus_dirs_listed)
                self.fixed_focus_dirs = valid_focus_dirs
        else:
            raise NotImplementedError(
                f"focus_type should be None, a list of fixed_focus_dirs, or a string describing one of the supported "
                f"focus_type, but here: {mcfg.focus_type}"
            )
        self.valid_focus_dirs = valid_focus_dirs

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"seh": model}

    def get_steer_encodings(self, preferences, focus_dirs):
        n = len(preferences)
        if self.use_steer_thermometer:
            pref_enc = thermometer(preferences, self.num_thermometer_dim, 0, 1).reshape(n, -1)
            focus_enc = thermometer(focus_dirs, self.num_thermometer_dim, 0, 1).reshape(n, -1)
        else:
            pref_enc = preferences
            focus_enc = focus_dirs
        return pref_enc, focus_enc

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it)

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
            if self.focus_model is not None and train_it >= self.focus_model_training_limits[0] * self.max_train_it:
                focus_dir = self.focus_model.sample_focus_directions(n)
            else:
                focus_dir = torch.tensor(
                    metrics.sample_positiveQuadrant_ndim_sphere(n, len(self.objectives), normalisation="l2")
                ).float()
        else:
            raise NotImplementedError(f"Unsupported focus_type={type(self.focus_type)}")

        preferences_enc, focus_enc = self.get_steer_encodings(preferences, focus_dir)
        cond_info["encoding"] = torch.cat([cond_info["encoding"], preferences_enc, focus_enc], 1)
        cond_info["preferences"] = preferences
        cond_info["focus_dir"] = focus_dir
        return cond_info

    def encode_conditional_information(self, steer_info: Tensor) -> Dict[str, Tensor]:
        """
        Encode conditional information at validation-time
        We use the maximum temperature beta for inference
        Args:
            steer_info: Tensor of shape (Batch, 2 * n_objectives) containing the preferences and focus_dirs
            in that order
        Returns:
            Dict[str, Tensor]: Dictionary containing the encoded conditional information
        """
        n = len(steer_info)
        if self.temperature_sample_dist == "constant":
            beta = torch.ones(n) * self.temperature_dist_params
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            beta = torch.ones(n) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((n, self.num_thermometer_dim))

        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"

        # TODO: positional assumption here, should have something cleaner
        preferences = steer_info[:, : len(self.objectives)].float()
        focus_dir = steer_info[:, len(self.objectives) :].float()

        preferences_enc, focus_enc = self.get_steer_encodings(preferences, focus_dir)
        encoding = torch.cat([beta_enc, preferences_enc, focus_enc], 1).float()

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

        preferences_enc, focus_enc = self.get_steer_encodings(cond_info["preferences"], cond_info["focus_dir"])
        cond_info["encoding"] = torch.cat(
            [cond_info["encoding"][:, : self.num_thermometer_dim], preferences_enc, focus_enc], 1
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
            flat_r: List[Tensor] = []
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
    task: SEHMOOTask

    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.algo.sampling_tau = 0.95
        cfg.algo.valid_sample_cond_info = False

    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "MOREINFORCE":
            self.algo = MultiObjectiveReinforce(self.env, self.ctx, self.rng, self.cfg)
        elif algo == "MOQL":
            self.algo = EnvelopeQLearning(self.env, self.ctx, self.task, self.rng, self.cfg)
        else:
            super().setup_algo()

        focus_type = self.cfg.task.seh_moo.focus_type
        if focus_type is not None and "learned" in focus_type:
            if focus_type == "learned-tabular":
                self.focus_model = TabularFocusModel(
                    device=self.device,
                    n_objectives=len(self.cfg.task.seh_moo.objectives),
                    state_space_res=self.cfg.task.seh_moo.focus_model_state_space_res,
                )
            else:
                raise NotImplementedError("Unknown focus model type {self.focus_type}")
        else:
            self.focus_model = None

    def setup_task(self):
        self.task = SEHMOOTask(
            dataset=self.training_data,
            cfg=self.cfg,
            focus_model=self.focus_model,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        if self.cfg.task.seh_moo.use_steer_thermometer:
            ncd = self.cfg.task.seh_moo.num_thermometer_dim * (1 + 2 * len(self.cfg.task.seh_moo.objectives))
        else:
            # 1 for prefs and 1 for focus region
            ncd = self.cfg.task.seh_moo.num_thermometer_dim + 2 * len(self.cfg.task.seh_moo.objectives)
        self.ctx = FragMolBuildingEnvContext(max_frags=self.cfg.algo.max_nodes, num_cond_dim=ncd)

    def setup_model(self):
        if self.cfg.algo.method == "MOQL":
            self.model = GraphTransformerFragEnvelopeQL(
                self.ctx,
                num_emb=self.cfg.model.num_emb,
                num_layers=self.cfg.model.num_layers,
                num_heads=self.cfg.model.graph_transformer.num_heads,
                num_objectives=len(self.cfg.task.seh_moo.objectives),
            )
        else:
            super().setup_model()

    def setup(self):
        super().setup()
        self.sampling_hooks.append(
            MultiObjectiveStatsHook(
                256,
                self.cfg.log_dir,
                compute_igd=True,
                compute_pc_entropy=True,
                compute_focus_accuracy=True if self.cfg.task.seh_moo.focus_type is not None else False,
                focus_cosim=self.cfg.task.seh_moo.focus_cosim,
            )
        )
        # instantiate preference and focus conditioning vectors for validation

        tcfg = self.cfg.task.seh_moo
        n_obj = len(tcfg.objectives)

        # making sure hyperparameters for preferences and focus regions are consistent
        if not (
            tcfg.focus_type is None
            or tcfg.focus_type == "centered"
            or (type(tcfg.focus_type) is list and len(tcfg.focus_type) == 1)
        ):
            assert tcfg.preference_type is None, (
                f"Cannot use preferences with multiple focus regions, here focus_type={tcfg.focus_type} "
                f"and preference_type={tcfg.preference_type}"
            )

        if type(tcfg.focus_type) is list and len(tcfg.focus_type) > 1:
            n_valid = len(tcfg.focus_type)
        else:
            n_valid = tcfg.n_valid

        # preference vectors
        if tcfg.preference_type is None:
            valid_preferences = np.ones((n_valid, n_obj))
        elif tcfg.preference_type == "dirichlet":
            valid_preferences = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l1")
        elif tcfg.preference_type == "seeded_single":
            seeded_prefs = np.random.default_rng(142857 + int(self.cfg.seed)).dirichlet([1] * n_obj, n_valid)
            valid_preferences = seeded_prefs[0].reshape((1, n_obj))
            self.task.seeded_preference = valid_preferences[0]
        elif tcfg.preference_type == "seeded_many":
            valid_preferences = np.random.default_rng(142857 + int(self.cfg.seed)).dirichlet([1] * n_obj, n_valid)
        else:
            raise NotImplementedError(f"Unknown preference type {self.cfg.task.seh_moo.preference_type}")

        # TODO: this was previously reported, would be nice to serialize it
        # hps["fixed_focus_dirs"] = (
        #    np.unique(self.task.fixed_focus_dirs, axis=0).tolist() if self.task.fixed_focus_dirs is not None else None
        # )
        assert self.task.valid_focus_dirs.shape == (
            n_valid,
            n_obj,
        ), f"Invalid shape for valid_preferences, {self.task.valid_focus_dirs.shape} != ({n_valid}, {n_obj})"

        # combine preferences and focus directions (fixed focus cosim) since they could be used together (not either/or)
        # TODO: this relies on positional assumptions, should have something cleaner
        valid_cond_vector = np.concatenate([valid_preferences, self.task.valid_focus_dirs], axis=1)

        self._top_k_hook = TopKHook(10, tcfg.n_valid_repeats, len(valid_cond_vector))
        self.test_data = RepeatedCondInfoDataset(valid_cond_vector, repeat=tcfg.n_valid_repeats)
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
        focus_model_training_limits = self.cfg.task.seh_moo.focus_model_training_limits
        max_train_it = self.cfg.num_training_steps
        if (
            self.focus_model is not None
            and train_it >= focus_model_training_limits[0] * max_train_it
            and train_it <= focus_model_training_limits[1] * max_train_it
        ):
            self.focus_model.update_belief(deepcopy(batch.focus_dir), deepcopy(batch.flat_rewards))
        return super().train_batch(batch, epoch_idx, batch_idx, train_it)

    def _save_state(self, it):
        if self.focus_model is not None:
            self.focus_model.save(pathlib.Path(self.cfg.log_dir))
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
        "pickle_mp_messages": True,
        "overwrite_existing_exp": True,
        "seed": 0,
        "num_training_steps": 20_000,
        "num_final_gen_steps": 500,
        "validate_every": 500,
        "num_workers": 0,
        "algo.global_batch_size": 64,
        "algo.method": "TB",
        "model.num_layers": 2,
        "model.num_emb": 256,
        "task.seh_moo.objectives": ["seh", "qed"],
        "opt.learning_rate": 1e-4,
        "algo.tb.Z_learning_rate": 1e-3,
        "opt.lr_decay": 20000,
        "algo.tb.Z_lr_decay": 50000,
        "algo.sampling_tau": 0.95,
        "algo.train_random_action_prob": 0.01,
        "task.seh_moo.temperature_sample_dist": "constant",
        "task.seh_moo.temperature_dist_params": 60.0,
        "task.seh_moo.num_thermometer_dim": 32,
        "task.seh_moo.use_steer_thermometer": False,
        "task.seh_moo.preference_type": None,
        "task.seh_moo.focus_type": "learned-tabular",
        "task.seh_moo.focus_cosim": 0.98,
        "task.seh_moo.focus_limit_coef": 1e-1,
        "task.seh_moo.n_valid": 15,
        "task.seh_moo.n_valid_repeats": 128,
        "replay.use": True,
        "replay.warmup": 1000,
        "replay.hindsight_ratio": 0.3,
        "task.seh_moo.focus_model_training_limits": [0.25, 0.75],
        "task.seh_moo.focus_model_state_space_res": 30,
        "task.seh_moo.max_train_it": 20_000,
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
    main()
