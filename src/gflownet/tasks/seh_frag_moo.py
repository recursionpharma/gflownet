import os
import pathlib
import shutil
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.algo.envelope_q_learning import EnvelopeQLearning, GraphTransformerFragEnvelopeQL
from gflownet.algo.multiobjective_reinforce import MultiObjectiveReinforce
from gflownet.config import Config
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag import SEHFragTrainer, SEHTask
from gflownet.trainer import FlatRewards, RewardScalar
from gflownet.utils import metrics, sascore
from gflownet.utils.conditioning import FocusRegionConditional, MultiObjectiveWeightedPreferences
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook, TopKHook


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
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        super().__init__(dataset, cfg, rng, wrap_model)
        self.cfg = cfg
        mcfg = self.cfg.task.seh_moo
        self.objectives = cfg.task.seh_moo.objectives
        self.dataset = dataset
        if self.cfg.cond.focus_region.focus_type is not None:
            self.focus_cond = FocusRegionConditional(self.cfg, mcfg.n_valid, rng)
        else:
            self.focus_cond = None
        self.pref_cond = MultiObjectiveWeightedPreferences(self.cfg)
        self.temperature_sample_dist = cfg.cond.temperature.sample_dist
        self.temperature_dist_params = cfg.cond.temperature.dist_params
        self.num_thermometer_dim = cfg.cond.temperature.num_thermometer_dim
        self.num_cond_dim = (
            self.temperature_conditional.encoding_size()
            + self.pref_cond.encoding_size()
            + (self.focus_cond.encoding_size() if self.focus_cond is not None else 0)
        )
        assert set(self.objectives) <= {"seh", "qed", "sa", "mw"} and len(self.objectives) == len(set(self.objectives))

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it)
        pref_ci = self.pref_cond.sample(n)
        focus_ci = (
            self.focus_cond.sample(n, train_it) if self.focus_cond is not None else {"encoding": torch.zeros(n, 0)}
        )
        cond_info = {
            **cond_info,
            **pref_ci,
            **focus_ci,
            "encoding": torch.cat([cond_info["encoding"], pref_ci["encoding"], focus_ci["encoding"]], dim=1),
        }
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
            beta = torch.ones(n) * self.temperature_dist_params[0]
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            beta = torch.ones(n) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((n, self.num_thermometer_dim))

        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"

        # TODO: positional assumption here, should have something cleaner
        preferences = steer_info[:, : len(self.objectives)].float()
        focus_dir = steer_info[:, len(self.objectives) :].float()

        preferences_enc = self.pref_cond.encode(preferences)
        if self.focus_cond is not None:
            focus_enc = self.focus_cond.encode(focus_dir)
            encoding = torch.cat([beta_enc, preferences_enc, focus_enc], 1).float()
        else:
            encoding = torch.cat([beta_enc, preferences_enc], 1).float()
        return {
            "beta": beta,
            "encoding": encoding,
            "preferences": preferences,
            "focus_dir": focus_dir,
        }

    def relabel_condinfo_and_logrewards(
        self, cond_info: Dict[str, Tensor], log_rewards: Tensor, flat_rewards: FlatRewards, hindsight_idxs: Tensor
    ):
        # TODO: we seem to be relabeling tensors in place, could that cause a problem?
        if self.focus_cond is None:
            raise NotImplementedError("Hindsight relabeling only implemented for focus conditioning")
        if self.focus_cond.cfg.focus_type is None:
            return cond_info, log_rewards
        # only keep hindsight_idxs that actually correspond to a violated constraint
        _, in_focus_mask = metrics.compute_focus_coef(
            flat_rewards, cond_info["focus_dir"], self.focus_cond.cfg.focus_cosim
        )
        out_focus_mask = torch.logical_not(in_focus_mask)
        hindsight_idxs = hindsight_idxs[out_focus_mask[hindsight_idxs]]

        # relabels the focus_dirs and log_rewards
        cond_info["focus_dir"][hindsight_idxs] = nn.functional.normalize(flat_rewards[hindsight_idxs], dim=1)

        preferences_enc = self.pref_cond.encode(cond_info["preferences"])
        focus_enc = self.focus_cond.encode(cond_info["focus_dir"])
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

        scalarized_reward = self.pref_cond.transform(cond_info, flat_reward)
        focused_reward = (
            self.focus_cond.transform(cond_info, flat_reward, scalarized_reward)
            if self.focus_cond is not None
            else scalarized_reward
        )
        tempered_reward = self.temperature_conditional.transform(cond_info, focused_reward)
        return RewardScalar(tempered_reward)

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
    ctx: FragMolBuildingEnvContext

    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.algo.sampling_tau = 0.95
        # We use a fixed set of preferences as our "validation set", so we must disable the preference (cond_info)
        # sampling and set the offline ratio to 1
        cfg.algo.valid_sample_cond_info = False
        cfg.algo.valid_offline_ratio = 1

    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "MOREINFORCE":
            self.algo = MultiObjectiveReinforce(self.env, self.ctx, self.rng, self.cfg)
        elif algo == "MOQL":
            self.algo = EnvelopeQLearning(self.env, self.ctx, self.task, self.rng, self.cfg)
        else:
            super().setup_algo()

    def setup_task(self):
        self.task = SEHMOOTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.task.num_cond_dim)

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
        if self.task.focus_cond is not None:
            assert self.task.focus_cond.valid_focus_dirs.shape == (
                n_valid,
                n_obj,
            ), (
                "Invalid shape for valid_preferences, "
                f"{self.task.focus_cond.valid_focus_dirs.shape} != ({n_valid}, {n_obj})"
            )

            # combine preferences and focus directions (fixed focus cosim) since they could be used together
            # (not either/or). TODO: this relies on positional assumptions, should have something cleaner
            valid_cond_vector = np.concatenate([valid_preferences, self.task.focus_cond.valid_focus_dirs], axis=1)
        else:
            valid_cond_vector = valid_preferences

        self._top_k_hook = TopKHook(10, tcfg.n_valid_repeats, n_valid)
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
        if self.task.focus_cond is not None:
            self.task.focus_cond.step_focus_model(batch, train_it)
        return super().train_batch(batch, epoch_idx, batch_idx, train_it)

    def _save_state(self, it):
        if self.task.focus_cond is not None and self.task.focus_cond.focus_model is not None:
            self.task.focus_cond.focus_model.save(pathlib.Path(self.cfg.log_dir))
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
    """Example of how this model can be run."""
    hps = {
        "log_dir": "./logs/debug_run_sfm",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pickle_mp_messages": True,
        "overwrite_existing_exp": True,
        "seed": 0,
        "num_training_steps": 500,
        "num_final_gen_steps": 50,
        "validate_every": 100,
        "num_workers": 0,
        "algo": {
            "global_batch_size": 64,
            "method": "TB",
            "sampling_tau": 0.95,
            "train_random_action_prob": 0.01,
            "tb": {
                "Z_learning_rate": 1e-3,
                "Z_lr_decay": 50000,
            },
        },
        "model": {
            "num_layers": 2,
            "num_emb": 256,
        },
        "task": {
            "seh_moo": {
                "objectives": ["seh", "qed"],
                "n_valid": 15,
                "n_valid_repeats": 128,
            },
        },
        "opt": {
            "learning_rate": 1e-4,
            "lr_decay": 20000,
        },
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [60.0],
                "num_thermometer_dim": 32,
            },
            "weighted_prefs": {
                "preference_type": "dirichlet",
            },
            "focus_region": {
                "focus_type": None,  # "learned-tabular",
                "focus_cosim": 0.98,
                "focus_limit_coef": 1e-1,
                "focus_model_training_limits": (0.25, 0.75),
                "focus_model_state_space_res": 30,
                "max_train_it": 5_000,
            },
        },
        "replay": {
            "use": False,
            "warmup": 1000,
            "hindsight_ratio": 0.0,
        },
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHMOOFragTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
