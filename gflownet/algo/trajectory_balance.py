import numpy as np
from itertools import count

import torch
from torch_scatter import scatter

from gflownet.envs.graph_building_env import GraphActionType, generate_forward_trajectory


class TrajectoryBalance:
    """
    See, Trajectory Balance: Improved Credit Assignment in GFlowNets
    Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio
    https://arxiv.org/abs/2201.13259
    """
    def __init__(self, rng, max_len=None, random_action_prob=None, max_nodes=None):
        self.max_len = max_len
        self.random_action_prob = random_action_prob
        self.illegal_action_logreward = -100
        self.bootstrap_own_reward = True
        self.sanitize_samples = True
        self.max_nodes = max_nodes
        self.rng = rng

    def _corrupt_actions(self, actions, cat):
        """Sample from the uniform policy with probability `self.random_action_prob`"""
        # Should this be a method of GraphActionCategorical?
        corrupted, = (np.random.uniform(size=len(actions)) < self.random_action_prob).nonzero()
        for i in corrupted:
            n_in_batch = [(b == i).sum() for b in cat.batch]
            n_each = np.float32([l.shape[1] * nb for l, nb in zip(cat.logits, n_in_batch)])
            which = self.rng.choice(len(n_each), p=n_each / n_each.sum())
            row = self.rng.choice(n_in_batch[which])
            col = self.rng.choice(cat.logits[which].shape[1])
            actions[i] = (which, row, col)

    def sample_model_losses(self, env, ctx, model, n):
        loss_items = [[model.logZ] for i in range(n)]
        graphs = [env.new() for i in range(n)]
        done = [False] * n

        def not_done(l):
            return [l[i] for i in range(n) if not done[i]]

        final_rewards = [None] * n
        dev = model.device
        illegal_action_logreward = torch.tensor([self.illegal_action_logreward], device=dev)
        for t in (range(self.max_len) if self.max_len is not None else count(0)):
            torch_graphs = [ctx.graph_to_Data(i) for i in not_done(graphs)]
            fwd_cat, log_reward_preds = model(ctx.collate(torch_graphs))
            actions = fwd_cat.sample()
            self._corrupt_actions(actions, fwd_cat)
            graph_actions = [
                ctx.aidx_to_GraphAction(g, a, model.action_type_order[a[0]]) for g, a in zip(torch_graphs, actions)
            ]
            log_probs = fwd_cat.log_prob(actions)
            for i, j, li, lp, ga in zip(not_done(list(range(n))), range(n), not_done(loss_items), log_probs,
                                        graph_actions):
                li.append(lp.unsqueeze(0))
                if ga.action is GraphActionType.Stop:
                    done[i] = True
                else:
                    try:
                        gp = env.step(graphs[i], ga)
                        if self.max_nodes is None or len(gp.nodes) < self.max_nodes:
                            # P_B
                            li.append(-torch.tensor([1 / env.count_backward_transitions(gp)], device=dev).log())
                        else:
                            done[i] = True
                        graphs[i] = gp
                    except AssertionError:
                        done[i] = True
                        final_rewards[i] = illegal_action_logreward
                if done[i] and final_rewards[i] is None:
                    if self.sanitize_samples and not ctx.is_sane(graphs[i]):
                        final_rewards[i] = illegal_action_logreward
                    elif self.bootstrap_own_reward:
                        final_rewards[i] = log_reward_preds[j].detach()
            if all(done):
                break
        losses = []
        for i in range(n):
            loss_items[i].append(-final_rewards[i])
            losses.append(torch.stack(loss_items[i]).sum().pow(2))
        return torch.stack(losses)

    def compute_data_losses(self, env, ctx, model, graphs, rewards):
        trajs = [generate_forward_trajectory(i) for i in graphs]
        torch_graphs = [ctx.graph_to_Data(i[0]) for tj in trajs for i in tj]
        actions = [i[1] for tj in trajs for i in tj]
        actions = [ctx.GraphAction_to_aidx(g, a, model.action_type_order) for g, a in zip(torch_graphs, actions)]
        batch = ctx.collate(torch_graphs).to(model.device)
        batch_idx = torch.tensor(sum(([i] * len(trajs[i]) for i in range(len(trajs))), []), device=model.device)
        fwd_cat, log_reward_preds = model(batch)
        log_prob = fwd_cat.log_prob(actions)
        num_backward = torch.tensor([
            env.count_backward_transitions(tj[i + 1][0]) if tj[i][1].action is not GraphActionType.Stop else 1
            for tj in trajs
            for i in range(len(tj))
        ], device=model.device)
        log_p_B = (1 / num_backward).log()

        traj_losses = (torch.stack([model.logZ - r for r in rewards.log()]).flatten() +
                       scatter(log_prob - log_p_B, batch_idx, dim=0, dim_size=len(trajs), reduce='sum')).pow(2)
        print(traj_losses.shape)
        if self.bootstrap_own_reward:
            traj_losses = traj_losses + (rewards.log() - log_reward_preds).pow(2)
        return traj_losses
