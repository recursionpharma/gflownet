import numpy as np
from itertools import count

import torch
from torch_scatter import scatter

from gflownet.envs.graph_building_env import Graph, GraphActionType, generate_forward_trajectory


class TrajectoryBalance:
    """
    See, Trajectory Balance: Improved Credit Assignment in GFlowNets
    Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio
    https://arxiv.org/abs/2201.13259
    """
    def __init__(self, rng, max_len=None, random_action_prob=None, max_nodes=None,
                 epsilon=-60):
        self.max_len = max_len
        self.random_action_prob = random_action_prob
        self.illegal_action_logreward = -100
        self.bootstrap_own_reward = True
        self.sanitize_samples = True
        self.max_nodes = max_nodes
        self.rng = rng
        self.epsilon = epsilon
        self.reward_loss_multiplier = 1

    def _corrupt_actions(self, actions, cat):
        """Sample from the uniform policy with probability `self.random_action_prob`"""
        # Should this be a method of GraphActionCategorical?
        corrupted, = (np.random.uniform(size=len(actions)) < self.random_action_prob).nonzero()
        for i in corrupted:
            n_in_batch = [(b == i).sum().item() for b in cat.batch]
            n_each = np.float32([l.shape[1] * nb for l, nb in zip(cat.logits, n_in_batch)])
            which = self.rng.choice(len(n_each), p=n_each / n_each.sum())
            row = self.rng.choice(n_in_batch[which])
            col = self.rng.choice(cat.logits[which].shape[1])
            actions[i] = (which, row, col)

    def sample_model_losses(self, env, ctx, model, n, cond_info=None, generated_molecules=None):
        if cond_info is None:
            loss_items = [([model.logZ], []) for i in range(n)]
        else:
            logZ_pred = model.logZ(cond_info)
            loss_items = [([logZ_pred[i]], []) for i in range(n)]
        graphs = [env.new() for i in range(n)]
        done = [False] * n

        def not_done(l):
            return [l[i] for i in range(n) if not done[i]]

        final_rewards = [None] * n
        dev = model.device
        illegal_action_logreward = torch.tensor([self.illegal_action_logreward], device=dev)
        epsilon = torch.tensor([self.epsilon], device=dev).float()
        for t in (range(self.max_len) if self.max_len is not None else count(0)):
            torch_graphs = [ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            fwd_cat, log_reward_preds = model(ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            actions = fwd_cat.sample()
            self._corrupt_actions(actions, fwd_cat)
            graph_actions = [
                ctx.aidx_to_GraphAction(g, a, model.action_type_order[a[0]]) for g, a in zip(torch_graphs, actions)
            ]
            log_probs = fwd_cat.log_prob(actions)
            for i, j, li, lp, ga in zip(not_done(list(range(n))), range(n), not_done(loss_items), log_probs,
                                        graph_actions):
                li[0].append(lp.unsqueeze(0))
                if ga.action is GraphActionType.Stop:
                    done[i] = True
                    #print('done', i, t)
                else:
                    try:
                        gp = env.step(graphs[i], ga)
                        if self.max_nodes is None or len(gp.nodes) < self.max_nodes:
                            # P_B
                            li[1].append(torch.tensor([1 / env.count_backward_transitions(gp)], device=dev).log())
                        else:
                            done[i] = True
                            final_rewards[i] = gp
                        graphs[i] = gp
                    except AssertionError:
                        #print('fail', i, t)
                        done[i] = True
                        final_rewards[i] = illegal_action_logreward
                if done[i] and final_rewards[i] is None:
                    if self.sanitize_samples and not ctx.is_sane(graphs[i]):
                        final_rewards[i] = illegal_action_logreward
                    elif self.bootstrap_own_reward:
                        final_rewards[i] = log_reward_preds[j].detach()
            if all(done):
                break
        graphs_to_fill = []
        idx_to_fill = []
        for i, r in enumerate(final_rewards):
            if isinstance(r, Graph):
                graphs_to_fill.append(r)
                idx_to_fill.append(i)
        if len(graphs_to_fill):
            with torch.no_grad():
                _, log_reward_preds = model(ctx.collate([ctx.graph_to_Data(i) for i in graphs_to_fill]).to(dev), cond_info[torch.tensor(idx_to_fill, device=dev)])
            for i, r in zip(idx_to_fill, log_reward_preds):
                final_rewards[i] = r
        
        losses = []
        for i in range(n):
            if generated_molecules is not None:
                generated_molecules[i] = (graphs[i], final_rewards[i])
            loss_items[i][1].append(final_rewards[i])
            numerator = torch.logaddexp(sum(loss_items[i][0]), epsilon)
            denominator = torch.logaddexp(sum(loss_items[i][1]), epsilon)
            #print(sum(loss_items[i][0]), sum(loss_items[i][1]), numerator, denominator)
            losses.append((numerator - denominator).pow(2) / len(loss_items[i][0]))
            #losses.append(torch.stack(loss_items[i]).sum().pow(2))
        return torch.stack(losses)

    def compute_data_losses(self, env, ctx, model, graphs, rewards, cond_info=None):
        epsilon = torch.tensor([self.epsilon], device=model.device).float()
        trajs = [generate_forward_trajectory(i) for i in graphs]
        torch_graphs = [ctx.graph_to_Data(i[0]) for tj in trajs for i in tj]
        actions = [i[1] for tj in trajs for i in tj]
        actions = [ctx.GraphAction_to_aidx(g, a, model.action_type_order) for g, a in zip(torch_graphs, actions)]
        batch = ctx.collate(torch_graphs).to(model.device)
        batch_idx = torch.tensor(sum(([i] * len(trajs[i]) for i in range(len(trajs))), []), device=model.device)
        final_graph_idx = torch.tensor(np.cumsum([len(i) for i in trajs]) - 1, device=model.device)
        fwd_cat, log_reward_preds = model(batch, cond_info[batch_idx])
        log_reward_preds = log_reward_preds[final_graph_idx, 0]
        log_prob = fwd_cat.log_prob(actions)
        num_backward = torch.tensor([
            env.count_backward_transitions(tj[i + 1][0]) if tj[i][1].action is not GraphActionType.Stop else 1
            for tj in trajs
            for i in range(len(tj))
        ], device=model.device)
        log_p_B = (1 / num_backward).log()
        if cond_info is None:
            # TODO: redo this
            Z_minus_r = torch.stack([model.logZ - r for r in rewards.log()]).flatten()
        else:
            Z = model.logZ(cond_info)[:, 0]
            
        numerator = Z + scatter(log_prob, batch_idx, dim=0, dim_size=len(trajs), reduce='sum')
        denominator = rewards.log() + scatter(log_p_B, batch_idx, dim=0, dim_size=len(trajs), reduce='sum') 
        numerator = torch.logaddexp(numerator, epsilon)
        denominator = torch.logaddexp(denominator, epsilon)
        lens = torch.tensor([len(t) for t in trajs], device=model.device)
        unnorm = traj_losses = (numerator - denominator).pow(2)
        traj_losses = traj_losses / lens
        info = {'unnorm_traj_losses': unnorm}
        if self.bootstrap_own_reward:
            info['reward_losses'] = reward_losses = (rewards - log_reward_preds.exp()).pow(2)
            #info['reward_losses'] = reward_losses = (rewards.log() - log_reward_preds).pow(2)
            traj_losses = traj_losses + reward_losses * self.reward_loss_multiplier
        return traj_losses, info
