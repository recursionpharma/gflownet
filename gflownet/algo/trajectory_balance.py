import time
import queue
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch_geometric.data as gd
from torch_scatter import scatter

from gflownet.envs.graph_building_env import Graph, GraphActionType, generate_forward_trajectory


class TrajectoryBalance:
    """
    See, Trajectory Balance: Improved Credit Assignment in GFlowNets
    Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio
    https://arxiv.org/abs/2201.13259
    """
    def __init__(self, env, ctx, rng, max_len=None, random_action_prob=None, max_nodes=None,
                 epsilon=-60):
        self.ctx = ctx
        self.env = env
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
            
    def create_training_data_from_own_samples(self, model, n, cond_info):
        """Generate trajectories by sampling a model
        
        Parameters
        ----------
        model: nn.Module
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
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
        """
        ctx = self.ctx
        env = self.env
        dev = model.device
        cond_info = cond_info.to(dev)
        logZ_pred = model.logZ(cond_info)
        # This will be returned as training data
        data = [{'traj': [], 'reward_pred': None} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob = [[logZ_pred[i]] for i in range(n)]
        zero = torch.tensor([0], device=dev).float()
        bck_logprob = [[zero] for i in range(n)] # zero in case there is a single invalid action
        
        graphs = [env.new() for i in range(n)]
        done = [False] * n
        def not_done(l):
            return [l[i] for i in range(n) if not done[i]]

        final_rewards = [None] * n
        illegal_action_logreward = torch.tensor([self.illegal_action_logreward], device=dev)
        epsilon = torch.tensor([self.epsilon], device=dev).float()
        for t in (range(self.max_len) if self.max_len is not None else count(0)):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            fwd_cat, log_reward_preds = model(ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            actions = fwd_cat.sample()
            self._corrupt_actions(actions, fwd_cat)
            graph_actions = [
                ctx.aidx_to_GraphAction(g, a, model.action_type_order[a[0]])
                for g, a in zip(torch_graphs, actions)
            ]
            log_probs = fwd_cat.log_prob(actions)
            for i, j, ga in zip(not_done(list(range(n))), range(n), graph_actions):
                # Step each trajectory, and accumulate statistics
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]['traj'].append((graphs[i], ga))
                if ga.action is GraphActionType.Stop:
                    done[i] = True
                else:
                    try:
                        # env.step can raise AssertionError if the action is illegal
                        gp = env.step(graphs[i], ga)
                        if self.max_nodes is not None:
                            assert len(gp.nodes) <= self.max_nodes
                        # P_B = uniform backward
                        n_back = env.count_backward_transitions(gp)
                        bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                        graphs[i] = gp
                    except AssertionError:
                        done[i] = True
                        data[i]['reward_pred'] = illegal_action_logreward.exp()
                if done[i] and data[i]['reward_pred'] is None:
                    # If we're not done and we haven't performed an illegal action
                    if self.sanitize_samples and not ctx.is_sane(graphs[i]):
                        # check if the graph is sane (e.g. RDKit can
                        # construct a molecule from it) otherwise
                        # treat the done action as illegal
                        data[i]['reward_pred'] = illegal_action_logreward.exp()
                    elif self.bootstrap_own_reward:
                        # if we're bootstrapping, extract reward prediction
                        data[i]['reward_pred'] = log_reward_preds[j].detach().exp()
            if all(done):
                break
        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward flows
            data[i]['logZ'] = fwd_logprob[i][0].item()
            data[i]['fwd_logprob'] = sum(fwd_logprob[i])
            data[i]['bck_logprob'] = sum(bck_logprob[i])
            if self.bootstrap_own_reward:
                # If we are bootstrapping, we can report the theoretical loss as well
                numerator = torch.logaddexp(data[i]['fwd_logprob'], epsilon)
                denominator = torch.logaddexp(data[i]['bck_logprob'] + data[i]['reward_pred'].log(), epsilon)
                data[i]['loss'] = (numerator - denominator).pow(2) / len(fwd_logprob[i])
        return data

    def create_training_data_from_graphs(self, graphs):
        """Generate trajectories from known endpoints
        
        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{'traj': generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, rewards, action_type_order):
        """Construct a batch from a list of trajectories and their information
        
        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: torch.Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        rewards: torch.Tensor
            The transformed reward (e.g. R(x) ** beta) for each trajectory. Shape (N,)
        action_type_order: List[GraphActionType]
            The order in which models output graph action logits.

        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj['traj']]
        actions = [self.ctx.GraphAction_to_aidx(g, a, action_type_order)
                   for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj['traj']])]
        num_backward = torch.tensor([
            # Count the number of backward transitions from s_{t+1},
            # unless t+1 = T is the last time step
            self.env.count_backward_transitions(tj['traj'][i + 1][0]) if i + 1 < len(tj['traj']) else 1
            for tj in trajs for i in range(len(tj['traj']))
        ])
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i['traj']) for i in trajs])
        batch.num_backward = num_backward
        batch.actions = torch.tensor(actions)
        batch.rewards = rewards
        batch.cond_info = cond_info
        return batch

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap:int=0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: nn.Module
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0.        
        """
        dev = model.device
        # A single trajectory is comprised of many graphs
        num_trajs = batch.traj_lens.shape[0]
        epsilon = torch.tensor([self.epsilon], device=dev).float()
        rewards = batch.rewards
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(batch.traj_lens.shape[0], device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and the optional bootstrap predictions
        fwd_cat, log_reward_preds = model(batch, cond_info[batch_idx])

        # Retreive the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        log_reward_preds = log_reward_preds[final_graph_idx, 0]
        # Compute trajectory balance objective
        Z = model.logZ(cond_info)[:, 0]
        log_prob = fwd_cat.log_prob(batch.actions)
        log_p_B = (1 / batch.num_backward).log()
        numerator = Z + scatter(log_prob, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')
        denominator = rewards.log() + scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')
        # Numerical stability epsilon
        numerator = torch.logaddexp(numerator, epsilon)
        denominator = torch.logaddexp(denominator, epsilon)
        if 1:
            # Instead of being rude to the model and giving a
            # logreward of -100 what if we say, whatever you think the
            # logprobablity of this trajectory is it should be smaller
            # (thus the `numerator - 0.1`). Why 0.1? Intuition, and
            # also 1 didn't appear to work as well.
            mask = (rewards < 1e-30).float()
            #denominator = denominator * (1 - mask) + mask * (numerator.detach() - 10)
            
        unnorm = traj_losses = (numerator - denominator).pow(2)
        # Normalize losses by trajectory length
        traj_losses = traj_losses / batch.traj_lens
        info = {'unnorm_traj_losses': unnorm,
                'invalid_trajectories': mask.mean() * 2}
        
        if self.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(rewards)
            reward_losses = (rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap].exp()).pow(2)
            info['reward_losses'] = reward_losses
            
        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')
        return traj_losses, info
        
