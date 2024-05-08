import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader

from gflownet import LogScalar, ObjectProperties
from gflownet.algo.envelope_q_learning import EnvelopeQLearning, GraphTransformerFragEnvelopeQL
from gflownet.algo.multiobjective_reinforce import MultiObjectiveReinforce
from gflownet.config import Config, init_empty
from gflownet.data.data_source import DataSource
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag import SEHFragTrainer, SEHTask
from gflownet.utils import metrics, sascore
from gflownet.utils.conditioning import FocusRegionConditional, MultiObjectiveWeightedPreferences
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook, TopKHook
from gflownet.utils.sqlite_log import SQLiteLogHook
from gflownet.utils.transforms import to_logreward


def safe(f, x, default):
    try:
        return f(x)
    except Exception:
        return default


def mol2mw(mols: list[RDMol], is_valid: list[bool], default=1000):
    molwts = torch.tensor([safe(Descriptors.MolWt, i, default) for i, v in zip(mols, is_valid) if v])
    molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
    return molwts


def mol2sas(mols: list[RDMol], is_valid: list[bool], default=10):
    sas = torch.tensor([safe(sascore.calculateScore, i, default) for i, v in zip(mols, is_valid) if v])
    sas = (10 - sas) / 9  # Turn into a [0-1] reward
    return sas


def mol2qed(mols: list[RDMol], is_valid: list[bool], default=0):
    return torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v])


aux_tasks = {"qed": mol2qed, "sa": mol2sas, "mw": mol2mw}


class SEHMOOTask(SEHTask):
    """Sets up a multiobjective task where the rewards are (functions of):
    - the binding energy of a molecule to Soluble Epoxide Hydrolases,
    - its QED,
    - its synthetic accessibility,
    - and its molecular weight.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """

    def __init__(
        self,
        cfg: Config,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(cfg, wrap_model)
        self.cfg = cfg
        mcfg = self.cfg.task.seh_moo
        self.objectives = cfg.task.seh_moo.objectives
        cfg.cond.moo.num_objectives = len(self.objectives)  # This value is used by the focus_cond and pref_cond
        if self.cfg.cond.focus_region.focus_type is not None:
            self.focus_cond = FocusRegionConditional(self.cfg, mcfg.n_valid)
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
        self, cond_info: Dict[str, Tensor], log_rewards: Tensor, obj_props: ObjectProperties, hindsight_idxs: Tensor
    ):
        # TODO: we seem to be relabeling tensors in place, could that cause a problem?
        if self.focus_cond is None:
            raise NotImplementedError("Hindsight relabeling only implemented for focus conditioning")
        if self.focus_cond.cfg.focus_type is None:
            return cond_info, log_rewards
        # only keep hindsight_idxs that actually correspond to a violated constraint
        _, in_focus_mask = metrics.compute_focus_coef(
            obj_props, cond_info["focus_dir"], self.focus_cond.cfg.focus_cosim
        )
        out_focus_mask = torch.logical_not(in_focus_mask)
        hindsight_idxs = hindsight_idxs[out_focus_mask[hindsight_idxs]]

        # relabels the focus_dirs and log_rewards
        cond_info["focus_dir"][hindsight_idxs] = nn.functional.normalize(obj_props[hindsight_idxs], dim=1)

        preferences_enc = self.pref_cond.encode(cond_info["preferences"])
        focus_enc = self.focus_cond.encode(cond_info["focus_dir"])
        cond_info["encoding"] = torch.cat(
            [cond_info["encoding"][:, : self.num_thermometer_dim], preferences_enc, focus_enc], 1
        )

        log_rewards = self.cond_info_to_logreward(cond_info, obj_props)
        return cond_info, log_rewards

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        """
        Compute the logreward from the object properties, which we interpret as each objective, and the conditional
        information
        """
        flat_reward = obj_props
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)

        scalarized_rewards = self.pref_cond.transform(cond_info, flat_reward)
        scalarized_logrewards = to_logreward(scalarized_rewards)
        focused_logreward = (
            self.focus_cond.transform(cond_info, (flat_reward, scalarized_logrewards))
            if self.focus_cond is not None
            else scalarized_logrewards
        )
        tempered_logreward = self.temperature_conditional.transform(cond_info, focused_logreward)
        clamped_logreward = tempered_logreward.clamp(min=self.cfg.algo.illegal_action_logreward)

        return LogScalar(clamped_logreward)

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        assert len(graphs) == len(mols)
        is_valid = [i is not None for i in graphs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        if not any(is_valid):
            return ObjectProperties(torch.zeros((0, len(self.objectives)))), is_valid_t
        else:
            flat_r: List[Tensor] = []
            for obj in self.objectives:
                if obj == "seh":
                    flat_r.append(super().compute_reward_from_graph(graphs))
                else:
                    flat_r.append(aux_tasks[obj](mols, is_valid))

            flat_rewards = torch.stack(flat_r, dim=1)
            assert flat_rewards.shape[0] == len(mols)
            return ObjectProperties(flat_rewards), is_valid_t


class SEHMOOFragTrainer(SEHFragTrainer):
    task: SEHMOOTask
    ctx: FragMolBuildingEnvContext

    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.algo.sampling_tau = 0.95
        # We sample from a dataset of valid conditional information, so we set this, and override
        # build_validation_data_loader to use the dataset
        cfg.cond.valid_sample_cond_info = False  # TODO deprecate this?
        cfg.algo.valid_num_from_dataset = 64

    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "MOREINFORCE":
            self.algo = MultiObjectiveReinforce(self.env, self.ctx, self.cfg)
        elif algo == "MOQL":
            self.algo = EnvelopeQLearning(self.env, self.ctx, self.task, self.cfg)
        else:
            super().setup_algo()

    def setup_task(self):
        self.cfg.cond.moo.num_objectives = len(self.cfg.task.seh_moo.objectives)
        self.task = SEHMOOTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

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
        if self.cfg.task.seh_moo.online_pareto_front:
            self.sampling_hooks.append(
                MultiObjectiveStatsHook(
                    256,
                    self.cfg.log_dir,
                    compute_igd=True,
                    compute_pc_entropy=True,
                    compute_focus_accuracy=True if self.cfg.cond.focus_region.focus_type is not None else False,
                    focus_cosim=self.cfg.cond.focus_region.focus_cosim,
                )
            )
            self.to_terminate.append(self.sampling_hooks[-1].terminate)
        # instantiate preference and focus conditioning vectors for validation

        n_obj = len(self.cfg.task.seh_moo.objectives)
        cond_cfg = self.cfg.cond

        # making sure hyperparameters for preferences and focus regions are consistent
        if not (
            cond_cfg.focus_region.focus_type is None
            or cond_cfg.focus_region.focus_type == "centered"
            or (isinstance(cond_cfg.focus_region.focus_type, list) and len(cond_cfg.focus_region.focus_type) == 1)
        ):
            assert cond_cfg.weighted_prefs.preference_type is None, (
                f"Cannot use preferences with multiple focus regions, "
                f"here focus_type={cond_cfg.focus_region.focus_type} "
                f"and preference_type={cond_cfg.weighted_prefs.preference_type }"
            )

        if isinstance(cond_cfg.focus_region.focus_type, list) and len(cond_cfg.focus_region.focus_type) > 1:
            n_valid = len(cond_cfg.focus_region.focus_type)
        else:
            n_valid = self.cfg.task.seh_moo.n_valid

        # preference vectors
        if cond_cfg.weighted_prefs.preference_type is None:
            valid_preferences = np.ones((n_valid, n_obj))
        elif cond_cfg.weighted_prefs.preference_type == "dirichlet":
            valid_preferences = metrics.partition_hypersphere(d=n_obj, k=n_valid, normalisation="l1")
        elif cond_cfg.weighted_prefs.preference_type == "seeded_single":
            seeded_prefs = np.random.default_rng(142857 + int(self.cfg.seed)).dirichlet([1] * n_obj, n_valid)
            valid_preferences = seeded_prefs[0].reshape((1, n_obj))
            self.task.seeded_preference = valid_preferences[0]
        elif cond_cfg.weighted_prefs.preference_type == "seeded_many":
            valid_preferences = np.random.default_rng(142857 + int(self.cfg.seed)).dirichlet([1] * n_obj, n_valid)
        else:
            raise NotImplementedError(f"Unknown preference type {cond_cfg.weighted_prefs.preference_type}")

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

        self.test_data = RepeatedCondInfoDataset(valid_cond_vector, repeat=self.cfg.task.seh_moo.n_valid_repeats)

        self._top_k_hook = TopKHook(10, self.cfg.task.seh_moo.n_valid_repeats, n_valid)
        if self.cfg.task.seh_moo.log_topk:
            self.valid_sampling_hooks.append(self._top_k_hook)

        self.algo.task = self.task

    def build_callbacks(self):
        # We use this class-based setup to be compatible with the DeterminedAI API, but no direct
        # dependency is required.
        parent = self
        callback_dict = {}

        if self.cfg.task.seh_moo.log_topk:

            class TopKMetricCB:
                def on_validation_end(self, metrics: Dict[str, Any]):
                    top_k = parent._top_k_hook.finalize()
                    for i in range(len(top_k)):
                        metrics[f"topk_rewards_{i}"] = top_k[i]
                    print("validation end", metrics)

            callback_dict["topk"] = TopKMetricCB()

        return callback_dict

    def build_validation_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        src.do_conditionals_dataset_in_order(self.test_data, self.cfg.algo.valid_num_from_dataset, model)

        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "valid"), self.ctx))
        for hook in self.valid_sampling_hooks:
            src.add_sampling_hook(hook)

        return self._make_data_loader(src)

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
        self.cond_info_vectors = torch.as_tensor(cond_info_vectors).float()
        self.repeat = repeat

    def __len__(self):
        return len(self.cond_info_vectors) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return self.cond_info_vectors[int(idx // self.repeat)]


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.desc = "debug_seh_frag_moo"
    config.log_dir = "./logs/debug_run_sfm"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.num_workers = 0
    config.print_every = 1
    config.algo.num_from_policy = 2
    config.validate_every = 1
    config.num_final_gen_steps = 5
    config.num_training_steps = 3
    config.pickle_mp_messages = True
    config.overwrite_existing_exp = True
    config.algo.sampling_tau = 0.95
    config.algo.train_random_action_prob = 0.01
    config.algo.tb.Z_learning_rate = 1e-3
    config.task.seh_moo.objectives = ["seh", "qed"]
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [60.0]
    config.cond.weighted_prefs.preference_type = "dirichlet"
    config.cond.focus_region.focus_type = None
    config.replay.use = False
    config.task.seh_moo.n_valid = 15
    config.task.seh_moo.n_valid_repeats = 2

    trial = SEHMOOFragTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
