from .trainer import GFNTrainer
import git
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.algo.advantage_actor_critic import A2C
from gflownet.algo.flow_matching import FlowMatching
from gflownet.algo.soft_q_learning import SoftQLearning
from gflownet.algo.trajectory_balance import TrajectoryBalance


class StandardOnlineTrainer(GFNTrainer):
    def setup_model(self):
        self.model = GraphTransformerGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
        )

    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "TB":
            algo = TrajectoryBalance
        elif algo == "FM":
            algo = FlowMatching
        elif algo == "A2C":
            algo = A2C
        elif algo == "SQL":
            algo = SoftQLearning
        else:
            raise ValueError(algo)
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)

    def setup(self):
        super().setup()
        self.training_data = []
        self.test_data = []
        self.offline_ratio = 0
        self.valid_offline_ratio = 0
        self.replay_buffer = ReplayBuffer(self.cfg, self.rng) if self.cfg.replay.use else None

        # Separate Z parameters from non-Z to allow for LR decay on the former
        if hasattr(self.model, "logZ"):
            Z_params = list(self.model.logZ.parameters())
            non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        else:
            Z_params = []
            non_Z_params = list(self.model.parameters())
        self.opt = torch.optim.Adam(
            non_Z_params,
            self.cfg.opt.learning_rate,
            (self.cfg.opt.momentum, 0.999),
            weight_decay=self.cfg.opt.weight_decay,
            eps=self.cfg.opt.adam_eps,
        )
        self.opt_Z = torch.optim.Adam(Z_params, self.cfg.algo.tb.Z_learning_rate, (0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(
            self.opt_Z, lambda steps: 2 ** (-steps / self.cfg.algo.tb.Z_lr_decay)
        )

        self.sampling_tau = self.cfg.algo.sampling_tau
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

        self.mb_size = self.cfg.algo.global_batch_size
        self.clip_grad_callback = {
            "value": (lambda params: torch.nn.utils.clip_grad_value_(params, self.cfg.opt.clip_grad_param)),
            "norm": (lambda params: torch.nn.utils.clip_grad_norm_(params, self.cfg.opt.clip_grad_param)),
            "none": (lambda x: None),
        }[self.cfg.opt.clip_grad_type]

        # saving hyperparameters
        git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        self.cfg.git_hash = git_hash

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        cd = config_to_dict(self.cfg)
        fmt_hps = "\n".join([f"{f'{k}':40}:\t{f'({type(v).__name__})':10}\t{v}" for k, v in sorted(cd.items())])
        print(f"\n\nHyperparameters:\n{'-'*50}\n{fmt_hps}\n{'-'*50}\n\n")
        with open(pathlib.Path(self.cfg.log_dir) / "hps.json", "w") as f:
            json.dump(cd, f)

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
