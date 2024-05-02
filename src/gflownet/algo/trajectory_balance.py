from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter, scatter_sum

from gflownet.algo.config import Backward, LossFN, NLoss, TBVariant
from gflownet.algo.graph_sampling import GraphSampler
from gflownet.config import Config
from gflownet.envs.graph_building_env import (
    ActionIndex,
    Graph,
    GraphAction,
    GraphActionCategorical,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    generate_forward_trajectory,
)
from gflownet.trainer import GFNAlgorithm
from gflownet.utils.misc import get_worker_device


def shift_right(x: torch.Tensor, z=0):
    "Shift x right by 1, and put z in the first position"
    x = torch.roll(x, 1, dims=0)
    x[0] = z
    return x


def cross(x: torch.Tensor):
    """
    Calculate $y_{ij} = \sum_{t=i}^j x_t$.
    The lower triangular portion is the inverse of the upper triangular one.
    """
    assert x.ndim == 1
    y = torch.cumsum(x, 0)
    return y[None] - shift_right(y)[:, None]


def subTB(v: torch.Tensor, x: torch.Tensor):
    r"""
    Compute the SubTB(1):
    $\forall i \leq j: D[i,j] =
        \log \frac{F(s_i) \prod_{k=i}^{j} P_F(s_{k+1}|s_k)}
        {F(s_{j + 1}) \prod_{k=i}^{j} P_B(s_k|s_{k+1})}$
      for a single trajectory.
    Note that x_k should be P_F(s_{k+1}|s_k) - P_B(s_k|s_{k+1}).
    """
    assert v.ndim == x.ndim == 1
    # D[i,j] = V[i] - V[j + 1]
    D = v[:-1, None] - v[None, 1:]
    # cross(x)[i, j] = sum(x[i:j+1])
    D = D + cross(x)
    return torch.triu(D)


class TrajectoryBalanceModel(nn.Module):
    def forward(self, batch: gd.Batch) -> Tuple[GraphActionCategorical, Tensor]:
        raise NotImplementedError()

    def logZ(self, cond_info: Tensor) -> Tensor:
        raise NotImplementedError()


class TrajectoryBalance(GFNAlgorithm):
    """Trajectory-based GFN loss implementations. Implements
    - TB: Trajectory Balance: Improved Credit Assignment in GFlowNets Nikolay Malkin, Moksh Jain,
    Emmanuel Bengio, Chen Sun, Yoshua Bengio
    https://arxiv.org/abs/2201.13259

    - SubTB(1): Learning GFlowNets from partial episodes for improved convergence and stability, Kanika Madan, Jarrid
    Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Cristian Nica, Tom Bosc, Yoshua Bengio,
    Nikolay Malkin
    https://arxiv.org/abs/2209.12782
    Note: We implement the lambda=1 version of SubTB here (this choice is based on empirical results from the paper)

    - DB: GFlowNet Foundations, Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, Emmanuel Bengio
    https://arxiv.org/abs/2111.09266
    Note: This is the trajectory version of Detailed Balance (i.e. transitions are not iid, but trajectories are).
    Empirical results in subsequent papers suggest that DB may be improved by training on iid transitions (sampled from
    a replay buffer) instead of trajectories.

    - Maxent[A], GSQL[A], TermTB1, StartTB1: Sobhan Mohammadpour, Emmanuel Bengio, Emma Frejinger, Pierre-Luc Bacon
    https://arxiv.org/abs/2312.14331
    """

    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ) -> None:
        """Instanciate a TB algorithm.

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        cfg: Config
            Hyperparameters
        """
        self.ctx = ctx
        self.env = env
        self.global_cfg = cfg
        self.cfg = cfg.algo.tb
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.length_normalize_losses = cfg.algo.tb.do_length_normalize
        # Experimental flags
        self.reward_loss = self.cfg.loss_fn
        self.tb_loss = self.cfg.loss_fn
        self.mask_invalid_rewards = False
        self.reward_normalize_losses = False
        self.sample_temp = 1
        self.bootstrap_own_reward = self.cfg.bootstrap_own_reward
        # When the model is autoregressive, we can avoid giving it ["A", "AB", "ABC", ...] as a sequence of inputs, and
        # instead give "ABC...Z" as a single input, but grab the logits at every timestep. Only works if using something
        # like a transformer with causal self-attention.
        self.model_is_autoregressive = False
        assert (
            self.cfg.backward_policy not in [Backward.Maxent, Backward.GSQL] or self.cfg.n_loss != NLoss.none
        ), "can't do maxent w/o learning or knowing $n$"
        assert self.ctx.has_n() or (
            self.cfg.backward_policy not in [Backward.MaxentA, Backward.GSQLA]
        ), "can't do analytical maxent/GSQL w/o knowing $n$"
        assert self.cfg.do_predict_n or self.cfg.n_loss == NLoss.none, "`n_loss != NLoss.none` requires `do_predict_n`"
        self.random_action_prob = [cfg.algo.train_random_action_prob, cfg.algo.valid_random_action_prob]

        self.graph_sampler = GraphSampler(
            ctx,
            env,
            cfg.algo.max_len,
            cfg.algo.max_nodes,
            self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
        )
        if self.cfg.variant == TBVariant.SubTB1:
            self._subtb_max_len = self.global_cfg.algo.max_len + 2
            self._init_subtb(get_worker_device())

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def create_training_data_from_own_samples(
        self,
        model: TrajectoryBalanceModel,
        n: int,
        cond_info: Optional[Tensor] = None,
        random_action_prob: Optional[float] = 0.0,
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: TrajectoryBalanceModel
           The model being sampled
        n: int
            Number of trajectories to sample
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - reward_pred: float, -100 if an illegal action is taken, predicted R(x) if bootstrapping, None otherwise
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - logZ: predicted log Z
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        cond_info = cond_info.to(dev) if cond_info is not None else None
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)
        if cond_info is not None:
            logZ_pred = model.logZ(cond_info)
            for i in range(n):
                data[i]["logZ"] = logZ_pred[i].item()
        return data

    def create_training_data_from_graphs(
        self,
        graphs,
        model: Optional[TrajectoryBalanceModel] = None,
        cond_info: Optional[Tensor] = None,
        random_action_prob: Optional[float] = None,
    ):
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: TrajectoryBalanceModel
           The model being sampled
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        if self.cfg.do_sample_p_b:
            assert model is not None and cond_info is not None and random_action_prob is not None
            dev = get_worker_device()
            cond_info = cond_info.to(dev)
            return self.graph_sampler.sample_backward_from_graphs(
                graphs, model if self.cfg.do_parameterize_p_b else None, cond_info, random_action_prob
            )
        trajs: List[Dict[str, Any]] = [{"traj": generate_forward_trajectory(i)} for i in graphs]
        for traj in trajs:
            n_back = [
                self.env.count_backward_transitions(gp, check_idempotent=self.cfg.do_correct_idempotent)
                for gp, _ in traj["traj"][1:]
            ] + [1]
            traj["bck_logprobs"] = (1 / torch.tensor(n_back).float()).log().to(get_worker_device())
            traj["result"] = traj["traj"][-1][0]
            if self.cfg.do_parameterize_p_b:
                traj["bck_a"] = [GraphAction(GraphActionType.Stop)] + [self.env.reverse(g, a) for g, a in traj["traj"]]
                # There needs to be an additonal node when we're parameterizing P_B,
                # See sampling with parametrized P_B
                traj["traj"].append(deepcopy(traj["traj"][-1]))
                traj["is_sink"] = [0 for _ in traj["traj"]]
                traj["is_sink"][-1] = 1
                traj["is_sink"][-2] = 1
                assert len(traj["bck_a"]) == len(traj["traj"]) == len(traj["is_sink"])
        return trajs

    def get_idempotent_actions(self, g: Graph, gd: gd.Data, gp: Graph, action: GraphAction, return_aidx: bool = True):
        """Returns the list of idempotent actions for a given transition.

        Note, this is slow! Correcting for idempotency is needed to estimate p(x) correctly, but
        isn't generally necessary if we mostly care about sampling approximately from the modes
        of p(x).

        Parameters
        ----------
        g: Graph
            The state graph
        gd: gd.Data
            The Data instance corresponding to g
        gp: Graph
            The next state's graph
        action: GraphAction
            Action leading from g to gp
        return_aidx: bool
            If true returns of list of ActionIndex, else a list of GraphAction

        Returns
        -------
        actions: Union[List[ActionIndex], List[GraphAction]]
            The list of idempotent actions that all lead from g to gp.

        """
        iaction = self.ctx.GraphAction_to_ActionIndex(gd, action)
        if action.action == GraphActionType.Stop:
            return [iaction if return_aidx else action]
        # Here we're looking for potential idempotent actions by looking at legal actions of the
        # same type. This assumes that this is the only way to get to a similar parent. Perhaps
        # there are edges cases where this is not true...?
        lmask = getattr(gd, action.action.mask_name)
        nz = lmask.nonzero()  # Legal actions are those with a nonzero mask value
        actions = [iaction if return_aidx else action]
        for i in nz:
            aidx = ActionIndex(action_type=iaction[0], row_idx=i[0].item(), col_idx=i[1].item())
            if aidx == iaction:
                continue
            ga = self.ctx.ActionIndex_to_GraphAction(gd, aidx, fwd=not action.action.is_backward)
            child = self.env.step(g, ga)
            if nx.algorithms.is_isomorphic(child, gp, lambda a, b: a == b, lambda a, b: a == b):
                actions.append(aidx if return_aidx else ga)
        return actions

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        if self.model_is_autoregressive:
            torch_graphs = [self.ctx.graph_to_Data(tj["traj"][-1][0]) for tj in trajs]
            actions = [
                self.ctx.GraphAction_to_ActionIndex(g, i[1]) for g, tj in zip(torch_graphs, trajs) for i in tj["traj"]
            ]
        else:
            torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
            actions = [
                self.ctx.GraphAction_to_ActionIndex(g, a)
                for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
            ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        if self.cfg.do_parameterize_p_b:
            batch.bck_actions = torch.tensor(
                [
                    self.ctx.GraphAction_to_ActionIndex(g, a)
                    for g, a in zip(torch_graphs, [i for tj in trajs for i in tj["bck_a"]])
                ]
            )
            batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        if self.cfg.do_correct_idempotent:
            # Every timestep is a (graph_a, action, graph_b) triple
            agraphs = [i[0] for tj in trajs for i in tj["traj"]]
            # Here we start at the 1th timestep and append the result
            bgraphs = sum([[i[0] for i in tj["traj"][1:]] + [tj["result"]] for tj in trajs], [])
            gactions = [i[1] for tj in trajs for i in tj["traj"]]
            ipa = [
                self.get_idempotent_actions(g, gd, gp, a)
                for g, gd, gp, a in zip(agraphs, torch_graphs, bgraphs, gactions)
            ]
            batch.ip_actions = torch.tensor(sum(ipa, []))
            batch.ip_lens = torch.tensor([len(i) for i in ipa])
            if self.cfg.do_parameterize_p_b:
                # Here we start at the 0th timestep and prepend None (it will be unused)
                bgraphs = sum([[None] + [i[0] for i in tj["traj"][:-1]] for tj in trajs], [])
                gactions = [i for tj in trajs for i in tj["bck_a"]]
                bck_ipa = [
                    self.get_idempotent_actions(g, gd, gp, a)
                    for g, gd, gp, a in zip(agraphs, torch_graphs, bgraphs, gactions)
                ]
                batch.bck_ip_actions = torch.tensor(sum(bck_ipa, []))
                batch.bck_ip_lens = torch.tensor([len(i) for i in bck_ipa])

        # compute_batch_losses expects these two optional values, if someone else doesn't fill them in, default to 0
        batch.num_offline = 0
        batch.num_online = 0
        return batch

    def compute_batch_losses(
        self,
        model: TrajectoryBalanceModel,
        batch: gd.Batch,
        num_bootstrap: int = 0,  # type: ignore[override]
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        log_rewards = batch.log_rewards
        # Clip rewards
        assert log_rewards.ndim == 1
        clip_log_R = torch.maximum(
            log_rewards, torch.tensor(self.global_cfg.algo.illegal_action_logreward, device=dev)
        ).float()
        cond_info = getattr(batch, "cond_info", None)
        invalid_mask = 1 - batch.is_valid

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        traj_cumlen = torch.cumsum(batch.traj_lens, 0)
        final_graph_idx = traj_cumlen - 1
        # The position of the first graph of each trajectory
        first_graph_idx = shift_right(traj_cumlen)
        final_graph_idx_1 = torch.maximum(final_graph_idx - 1, first_graph_idx)

        fwd_cat: GraphActionCategorical  # The per-state cond_info
        batched_cond_info = cond_info[batch_idx] if cond_info is not None else None

        # Forward pass of the model, returns a GraphActionCategorical representing the forward
        # policy P_F, optionally a backward policy P_B, and per-graph outputs (e.g. F(s) in SubTB).
        if self.cfg.do_parameterize_p_b:
            fwd_cat, bck_cat, per_graph_out = model(batch, batched_cond_info)
        else:
            if self.model_is_autoregressive:
                fwd_cat, per_graph_out = model(batch, cond_info, batched=True)
            else:
                fwd_cat, per_graph_out = model(batch, batched_cond_info)
        # Retreive the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        log_reward_preds = per_graph_out[final_graph_idx, 0]
        if self.cfg.do_predict_n:
            log_n_preds = per_graph_out[:, 1]
            log_n_preds[first_graph_idx] = 0
        else:
            log_n_preds = None

        # Compute trajectory balance objective
        log_Z = model.logZ(cond_info)[:, 0]
        # Compute the log prob of each action in the trajectory
        if self.cfg.do_correct_idempotent:
            # If we want to correct for idempotent actions, we need to sum probabilities
            # i.e. to compute P(s' | s) = sum_{a that lead to s'} P(a|s)
            # here we compute the indices of the graph that each action corresponds to, ip_lens
            # contains the number of idempotent actions for each transition, so we
            # repeat_interleave as with batch_idx
            ip_batch_idces = torch.arange(batch.ip_lens.shape[0], device=dev).repeat_interleave(batch.ip_lens)
            # Indicate that the `batch` corresponding to each action is the above
            ip_log_prob = fwd_cat.log_prob(batch.ip_actions, batch=ip_batch_idces)
            # take the logsumexp (because we want to sum probabilities, not log probabilities)
            # TODO: numerically stable version:
            p = scatter(ip_log_prob.exp(), ip_batch_idces, dim=0, dim_size=batch_idx.shape[0], reduce="sum")
            # As a (reasonable) band-aid, ignore p < 1e-30, this will prevent underflows due to
            # scatter(small number) = 0 on CUDA
            log_p_F = p.clamp(1e-30).log()

            if self.cfg.do_parameterize_p_b:
                # Now we repeat this but for the backward policy
                bck_ip_batch_idces = torch.arange(batch.bck_ip_lens.shape[0], device=dev).repeat_interleave(
                    batch.bck_ip_lens
                )
                bck_ip_log_prob = bck_cat.log_prob(batch.bck_ip_actions, batch=bck_ip_batch_idces)
                bck_p = scatter(
                    bck_ip_log_prob.exp(), bck_ip_batch_idces, dim=0, dim_size=batch_idx.shape[0], reduce="sum"
                )
                log_p_B = bck_p.clamp(1e-30).log()
        else:
            # Else just naively take the logprob of the actions we took
            log_p_F = fwd_cat.log_prob(batch.actions)
            if self.cfg.do_parameterize_p_b:
                log_p_B = bck_cat.log_prob(batch.bck_actions)

        if self.cfg.do_parameterize_p_b:
            # If we're modeling P_B then trajectories are padded with a virtual terminal state sF,
            # zero-out the logP_F of those states
            log_p_F[final_graph_idx] = 0
            if self.cfg.variant == TBVariant.SubTB1 or self.cfg.variant == TBVariant.DB:
                # Force the pad states' F(s) prediction to be R
                per_graph_out[final_graph_idx, 0] = clip_log_R

            # To get the correct P_B we need to shift all predictions by 1 state, and ignore the
            # first P_B prediction of every trajectory.
            # Our batch looks like this:
            # [(s1, a1), (s2, a2), ..., (st, at), (sF, None),   (s1, a1), ...]
            #                                                   ^ new trajectory begins
            # For the P_B of s1, we need the output of the model at s2.

            # We also have access to the is_sink attribute, which tells us when P_B must = 1, which
            # we'll use to ignore the last padding state(s) of each trajectory. This by the same
            # occasion masks out the first P_B of the "next" trajectory that we've shifted.
            log_p_B = torch.roll(log_p_B, -1, 0) * (1 - batch.is_sink)
        else:
            log_p_B = batch.log_p_B
        assert log_p_F.shape == log_p_B.shape

        if self.cfg.n_loss == NLoss.TB:
            log_traj_n = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
            n_loss = self._loss(log_traj_n + log_n_preds[final_graph_idx_1])
        else:
            n_loss = self.n_loss(log_p_B, log_n_preds, batch.traj_lens)

        if self.ctx.has_n() and self.cfg.do_predict_n:
            analytical_maxent_backward = self.analytical_maxent_backward(batch, first_graph_idx)
            if self.cfg.do_parameterize_p_b:
                analytical_maxent_backward = torch.roll(analytical_maxent_backward, -1, 0) * (1 - batch.is_sink)
        else:
            analytical_maxent_backward = None

        if self.cfg.backward_policy in [Backward.GSQL, Backward.GSQLA]:
            log_p_B = torch.zeros_like(log_p_B)
            nzf = torch.maximum(first_graph_idx, final_graph_idx - 1)
            if self.cfg.backward_policy == Backward.GSQLA:
                log_p_B[nzf] = -batch.log_n
            else:
                log_p_B[nzf] = -log_n_preds[
                    nzf
                ]  # this is due to the fact that n(s_0)/n(s1) * n(s1)/ n(s2) = n(s_0)/n(s2) = 1 / n(s)
            # this is not final_graph_idx because we throw away the last thing
        elif self.cfg.backward_policy == Backward.MaxentA:
            log_p_B = analytical_maxent_backward

        if self.cfg.do_parameterize_p_b:
            # Life is pain, log_p_B is one unit too short for all trajs

            log_p_B_unif = torch.zeros_like(log_p_B)
            for i, (s, e) in enumerate(zip(first_graph_idx, traj_cumlen)):
                log_p_B_unif[s : e - 1] = batch.log_p_B[s - i : e - 1 - i]

            if self.cfg.backward_policy == Backward.Uniform:
                log_p_B = log_p_B_unif
        else:
            log_p_B_unif = log_p_B

        if self.cfg.backward_policy in [Backward.Maxent, Backward.GSQL]:
            log_p_B = log_p_B.detach()
        # This is the log probability of each trajectory
        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_unif_log_p_B = scatter(log_p_B_unif, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        if self.cfg.variant == TBVariant.SubTB1:
            # SubTB interprets the per_graph_out predictions to predict the state flow F(s)
            if self.cfg.cum_subtb:
                traj_losses = self.subtb_cum(log_p_F, log_p_B, per_graph_out[:, 0], clip_log_R, batch.traj_lens)
            else:
                traj_losses = self.subtb_loss_fast(log_p_F, log_p_B, per_graph_out[:, 0], clip_log_R, batch.traj_lens)

            # The position of the first graph of each trajectory
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            log_Z = per_graph_out[first_graph_idx, 0]
        elif self.cfg.variant == TBVariant.DB:
            F_sn = per_graph_out[:, 0]
            F_sm = per_graph_out[:, 0].roll(-1)
            F_sm[final_graph_idx] = clip_log_R
            transition_losses = self._loss(F_sn + log_p_F - F_sm - log_p_B)
            traj_losses = scatter(transition_losses, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            log_Z = per_graph_out[first_graph_idx, 0]
        else:
            # Compute log numerator and denominator of the TB objective
            numerator = log_Z + traj_log_p_F
            denominator = clip_log_R + traj_log_p_B

            if self.mask_invalid_rewards:
                # Instead of being rude to the model and giving a
                # logreward of -100 what if we say, whatever you think the
                # logprobablity of this trajetcory is it should be smaller
                # (thus the `numerator - 1`). Why 1? Intuition?
                denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

            if self.cfg.epsilon is not None:
                # Numerical stability epsilon
                epsilon = torch.tensor([self.cfg.epsilon], device=dev).float()
                numerator = torch.logaddexp(numerator, epsilon)
                denominator = torch.logaddexp(denominator, epsilon)
            traj_losses = self._loss(numerator - denominator, self.tb_loss)

        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens
        if self.reward_normalize_losses:
            # multiply each loss by how important it is, using R as the importance factor
            # factor = Rp.exp() / Rp.exp().sum()
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            # * num_trajs because we're doing a convex combination, and a .mean() later, which would
            # undercount (by 2N) the contribution of each loss
            traj_losses = factor * traj_losses * num_trajs

        if self.cfg.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(log_rewards)
            reward_losses = self._loss(log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap], self.reward_loss)

            reward_loss = reward_losses.mean() * self.cfg.reward_loss_multiplier
        else:
            reward_loss = 0

        n_loss = n_loss.mean()
        tb_loss = traj_losses.mean()
        loss = tb_loss + reward_loss + self.cfg.n_loss_multiplier * n_loss
        info = {
            "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "reward_loss": reward_loss,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_logprob": (invalid_mask * traj_log_p_F).sum() / (invalid_mask.sum() + 1e-4),
            "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            "backward_vs_unif": (traj_unif_log_p_B - traj_log_p_B).pow(2).mean(),
            "logZ": log_Z.mean(),
            "loss": loss.item(),
            "n_loss": n_loss,
            "tb_loss": tb_loss.item(),
            "batch_entropy": -traj_log_p_F.mean(),
            "traj_lens": batch.traj_lens.float().mean(),
        }
        if self.ctx.has_n() and self.cfg.do_predict_n:
            info["n_loss_pred"] = scatter(
                (log_n_preds - batch.log_ns) ** 2, batch_idx, dim=0, dim_size=num_trajs, reduce="sum"
            ).mean()
            info["n_final_loss"] = torch.mean((log_n_preds[final_graph_idx] - batch.log_n) ** 2)
            if self.cfg.do_parameterize_p_b:
                info["n_loss_tgsql"] = torch.mean((-batch.log_n - traj_log_p_B) ** 2)
                d = analytical_maxent_backward - log_p_B
                d = d * d
                d[final_graph_idx] = 0
                info["n_loss_maxent"] = scatter(d, batch_idx, dim=0, dim_size=num_trajs, reduce="sum").mean()

        return loss, info

    def analytical_maxent_backward(self, batch, first_graph_idx):
        s = shift_right(batch.log_ns)
        s[first_graph_idx] = 0
        return s - batch.log_ns

    def _init_subtb(self, dev):
        r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
        \sum_{m=1}^{T-1} \sum_{n=m+1}^T
            \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                       {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
        """
        ar = torch.arange(self._subtb_max_len, device=dev)
        # This will contain a sequence of repeated ranges, e.g.
        # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
        tidx = [torch.tril_indices(i, i, device=dev)[1] for i in range(self._subtb_max_len)]
        # We need two sets of indices, the first are the source indices, the second the destination
        # indices. We precompute such indices for every possible trajectory length.

        # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
        # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
        # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

        # The destination indices indicate the index of the subsequence the source indices correspond to.
        # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
        # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

        # And so with these indices, for example for m=0, n=3, the forward probability
        # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

        self._precomp = [
            (
                torch.cat([i + tidx[T - i] for i in range(T)]),
                torch.cat(
                    [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
                ),
            )
            for T in range(1, self._subtb_max_len)
        ]

    def subtb_loss_fast(self, P_F, P_B, F, R, traj_lengths):
        r"""Computes the full SubTB(1) loss (all arguments on log-scale).

        Computes:
            \sum_{m=1}^{T-1} \sum_{n=m+1}^T
                \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                           {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
            where T is the length of the trajectory, for every trajectory.

        The shape of P_F, P_B, and F should be (total num steps,), i.e. sum(traj_lengths). The shape
        of R and traj_lengths should be (num trajs,).

        Parameters
        ----------
        P_F: Tensor
            Forward policy log-probabilities
        P_B: Tensor
            Backward policy log-probabilities
        F: Tensor
            Log-scale flow predictions
        R: Tensor
            The log-reward of each trajectory
        traj_lengths: Tensor
            The length of each trajectory

        Returns
        -------
        losses: Tensor
            The SubTB(1) loss of each trajectory.
        """
        num_trajs = int(traj_lengths.shape[0])
        max_len = int(traj_lengths.max() + 1)
        dev = traj_lengths.device
        cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=dev), traj_lengths]), 0).long()
        total_loss = torch.zeros(num_trajs, device=dev)
        ar = torch.arange(max_len, device=dev)
        car = torch.cumsum(ar, 0)
        F_and_R = torch.cat([F, R])
        R_start = F.shape[0]
        for ep in range(traj_lengths.shape[0]):
            offset = cumul_lens[ep]
            T = int(traj_lengths[ep])
            if self.cfg.do_parameterize_p_b:
                # The length of the trajectory is the padded length, reduce by 1
                T -= 1
            idces, dests = self._precomp[T - 1]
            fidces = torch.cat(
                [torch.cat([ar[i + 1 : T] + offset, torch.tensor([R_start + ep], device=dev)]) for i in range(T)]
            )
            P_F_sums = scatter_sum(P_F[idces + offset], dests)
            P_B_sums = scatter_sum(P_B[idces + offset], dests)
            F_start = F[offset : offset + T].repeat_interleave(T - ar[:T])
            F_end = F_and_R[fidces]
            total_loss[ep] = self._loss(F_start - F_end + P_F_sums - P_B_sums).sum() / car[T]
        return total_loss

    def n_loss(self, P_N, N, traj_lengths):
        dev = traj_lengths.device
        num_trajs = len(traj_lengths)
        total_loss = torch.zeros(num_trajs, device=dev)
        if self.cfg.n_loss == NLoss.none:
            return total_loss
        assert self.cfg.do_parameterize_p_b

        x = torch.cumsum(traj_lengths, 0)
        for ep, (s_idx, e_idx) in enumerate(zip(shift_right(x), x)):
            # the last state is the same as the first state
            e_idx -= 1
            total_loss[ep] = self._n_loss(self.cfg.n_loss, P_N[s_idx : e_idx - 1], N[s_idx:e_idx])
        return total_loss

    def _loss(self, x, loss_fn=None):
        if loss_fn is None:
            loss_fn = self.cfg.loss_fn
        if loss_fn == LossFN.MSE:
            return x * x
        elif loss_fn == LossFN.MAE:
            return torch.abs(x)
        elif loss_fn == LossFN.HUB:
            ax = torch.abs(x)
            d = self.cfg.loss_fn_par
            return torch.where(ax < 1, 0.5 * x * x / d, ax / d - 0.5 / d)
        elif loss_fn == LossFN.GHL:
            ax = self.cfg.loss_fn_par * x
            return torch.logaddexp(ax, -ax) - np.log(2)
        else:
            raise NotImplementedError()

    def _n_loss(self, method, P_N, N):
        n = len(N)
        if method == NLoss.SubTB1:
            return self._loss(subTB(N, -P_N)).sum() / (n * n - n) * 2
        elif method == NLoss.TermTB1:
            return self._loss(subTB(N, -P_N)[:, 0]).mean()
        elif method == NLoss.StartTB1:
            # return self._loss(subTB(N, -P_N)[0, :]).mean()
            return self._loss(N[1:] + torch.cumsum(P_N, -1)).mean()
        elif method == NLoss.TB:
            return self._loss(P_N.sum() + N[-1])
        elif method == NLoss.Transition:
            return self._loss(N[1:] + P_N - N[:-1]).mean()
        else:
            raise NotImplementedError()

    def subtb_cum(self, P_F, P_B, F, R, traj_lengths):
        """
        Calcualte the subTB(1) loss (all arguments on log-scale) using dynamic programming.

        See also `subTB`
        """
        dev = traj_lengths.device
        num_trajs = len(traj_lengths)
        total_loss = torch.zeros(num_trajs, device=dev)
        x = torch.cumsum(traj_lengths, 0)
        # P_B is already shifted
        pdiff = P_F - P_B
        for ep, (s_idx, e_idx) in enumerate(zip(shift_right(x), x)):
            if self.cfg.do_parameterize_p_b:
                e_idx -= 1
            n = e_idx - s_idx
            fr = torch.cat([F[s_idx:e_idx], torch.tensor([R[ep]], device=F.device)])
            p = pdiff[s_idx:e_idx]
            total_loss[ep] = self._loss(subTB(fr, p)).sum() / (n * n + n) * 2
        return total_loss
