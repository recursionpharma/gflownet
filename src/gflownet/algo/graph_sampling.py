import copy
from typing import List

import torch
from torch import Tensor
import torch.nn as nn

from gflownet.envs.graph_building_env import GraphActionType


class GraphSampler:
    """A helper class to sample from GraphActionCategorical-producing models"""
    def __init__(self, ctx, env, max_len, max_nodes, rng, sample_temp=1, correct_idempotent=False):
        """
        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        rng: np.random.RandomState
            rng used to take random actions
        sample_temp: float
            [Experimental] Softmax temperature used when sampling
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        self.rng = rng
        # Experimental flags
        self.sample_temp = sample_temp
        self.random_action_prob = 0
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent

    def sample_from_model(self, model: nn.Module, n: int, cond_info: Tensor, dev: torch.device):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        # This will be returned
        data = [{'traj': [], 'reward_pred': None, 'is_valid': True} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for i in range(n)]

        graphs = [self.env.new() for i in range(n)]
        done = [False] * n

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            fwd_cat, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            if self.random_action_prob > 0:
                masks = [1] * len(fwd_cat.logits) if fwd_cat.masks is None else fwd_cat.masks
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < self.random_action_prob, device=dev).float()
                # Set the logits to some large value if they're not masked, this way the masked
                # actions have no probability of getting sampled, and there is a uniform
                # distribution over the rest
                fwd_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                    for i, m, b in zip(fwd_cat.logits, masks, fwd_cat.batch)
                ]
            if self.sample_temp != 1:
                sample_cat = copy.copy(fwd_cat)
                sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
                actions = sample_cat.sample()
            else:
                actions = fwd_cat.sample()
            graph_actions = [self.ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = fwd_cat.log_prob(actions)
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]['traj'].append((graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]['is_valid'] = False
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can
                    # construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]['is_valid'] = False
            if all(done):
                break

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]['fwd_logprob'] = sum(fwd_logprob[i])
            data[i]['bck_logprob'] = sum(bck_logprob[i])
            data[i]['bck_logprobs'] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]['result'] = graphs[i]
        return data
