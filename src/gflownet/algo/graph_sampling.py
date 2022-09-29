import copy
from itertools import count
from typing import List

import torch
from torch import Tensor
import torch.nn as nn

from gflownet.envs.graph_building_env import GraphActionType


class GraphSampler:
    def __init__(self, ctx, env, max_len, max_nodes, rng, sample_temp=1):
        self.ctx = ctx
        self.env = env
        self.max_len = max_len
        self.max_nodes = max_nodes
        self.rng = rng
        # Experimental flags
        self.sample_temp = sample_temp
        self.random_action_prob = 0
        self.sanitize_samples = True

    def sample_from_model(self, model: nn.Module, n: int, cond_info: Tensor, dev: torch.device):
        # This will be returned as training data
        data = [{'traj': [], 'reward_pred': None, 'is_valid': True} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        zero = torch.tensor([0], device=dev).float()
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[zero] for i in range(n)]  # zero in case there is a single invalid action

        graphs = [self.env.new() for i in range(n)]
        done = [False] * n

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for t in (range(self.max_len) if self.max_len is not None else count(0)):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            fwd_cat, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            if self.random_action_prob > 0:
                masks = [1] * len(fwd_cat.logits) if fwd_cat.masks is None else fwd_cat.masks
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < self.random_action_prob, device=dev).float()
                fwd_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m + i * (1 - is_random_action[b][:, None])
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
            for i, j in zip(not_done(range(n)), range(n)):
                # Step each trajectory, and accumulate statistics
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]['traj'].append((graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop or (self.max_len and t == self.max_len - 1):
                    done[i] = True
                    if self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                        # check if the graph is sane (e.g. RDKit can
                        # construct a molecule from it) otherwise
                        # treat the done action as illegal
                        data[i]['is_valid'] = False
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        if self.max_nodes is not None:
                            assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]['is_valid'] = False
                        continue
                    # Add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    graphs[i] = gp
            if all(done):
                break

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward flows
            data[i]['fwd_logprob'] = sum(fwd_logprob[i])
            data[i]['bck_logprob'] = sum(bck_logprob[i])
        return data
