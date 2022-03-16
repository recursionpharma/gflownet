import time
import queue
import numpy as np
from itertools import count

import torch
import torch.multiprocessing as mp
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
        self.max_len = max_len
        self.random_action_prob = random_action_prob
        self.illegal_action_logreward = -100
        self.bootstrap_own_reward = True
        self.sanitize_samples = True
        self.max_nodes = max_nodes
        self.rng = rng
        self.epsilon = epsilon
        self.reward_loss_multiplier = 1
        self.pool = TrajectoryBuildingPool(64, env, ctx)

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
        t = [time.time()]
        epsilon = torch.tensor([self.epsilon], device=model.device).float()
        if 1:
            trajs = [generate_forward_trajectory(i) for i in graphs]
            torch_graphs = [ctx.graph_to_Data(i[0]) for tj in trajs for i in tj]
            actions = [i[1] for tj in trajs for i in tj]
            actions = [ctx.GraphAction_to_aidx(g, a, model.action_type_order) for g, a in zip(torch_graphs, actions)]
            num_backward = torch.tensor([
                env.count_backward_transitions(tj[i + 1][0]) if tj[i][1].action is not GraphActionType.Stop else 1
                for tj in trajs
                for i in range(len(tj))
            ], device=model.device)
            t += [time.time()]
            batch = ctx.collate(torch_graphs).to(model.device)
        else:
            trajs, torch_graphs, actions, num_backward = self.pool.build_batch(graphs, rewards, model.action_type_order)
            t += [time.time()]
            batch = ctx.collate(torch_graphs)
        t += [time.time()]
        batch_idx = torch.tensor(sum(([i] * len(trajs[i]) for i in range(len(trajs))), []), device=model.device)
        final_graph_idx = torch.tensor(np.cumsum([len(i) for i in trajs]) - 1, device=model.device)
        fwd_cat, log_reward_preds = model(batch, cond_info[batch_idx])
        t += [time.time()]
        log_reward_preds = log_reward_preds[final_graph_idx, 0]
        log_prob = fwd_cat.log_prob(actions)
        t += [time.time()]
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
        t += [time.time()]
        #print('compute_data_losses', ' '.join(f"{t[i+1]-t[i]:.3f}" for i in range(len(t)-1)))
        return traj_losses, info

def _trajectory_building_process(qin, qout, pid, env, ctx, pq):
    np.random.seed(pid)
    refs = []
    time_here = 0
    torch_graph = None
    asd = None
    while True:
        try:
            msg = pq.get(block=False)
            if msg == 'stop':
                break
            elif msg == 'free':
                qout.put(time_here)
                time_here = 0
                refs = []
        except queue.Empty:
            pass
        try:
            msg = qin.get(timeout=0.005)
        except queue.Empty:
            continue
        if msg is None:
            break
        t0 = time.time()
        action, idx, arg = msg
        #print(f'[{pid}]', action, idx, arg)
        if action == 'make_graph':
            (g, a), action_type_order = arg
            torch_graph = ctx.graph_to_Data(g)
            action = ctx.GraphAction_to_aidx(torch_graph, a, action_type_order)
            asd = torch_graph = torch_graph.cuda()
            # We must hold on to this and delete it in this process
            refs.append(torch_graph)
            qout.put((idx, (torch_graph, action, env.count_backward_transitions(g))))
        elif action == 'test':
            if asd is not None:
                qout.put((idx, (asd.x, asd.edge_index, asd.edge_attr, asd.non_edge_index)))
            else:
                qout.put((idx, None))
        t1 = time.time()
        time_here += t1-t0
    
class TrajectoryBuildingPool:

    def __init__(self, num_processes, env, ctx):
        self.env = env
        self.ctx = ctx
        self.num = num_processes
        # Global work queues
        self.qin = mp.Queue()
        self.qout = mp.Queue()
        # This queue is used per-process (e.g. to stop them or free data)
        self.pq = [mp.Queue() for i in range(self.num)]
        self.procs = [mp.Process(target=_trajectory_building_process,
                                 args=(self.qin, self.qout, i, env, ctx, self.pq[i]),
                                 daemon=True)
                      for i in range(self.num)]
        for p in self.procs:
            p.start()
        self.time_in_proc = 0
        
    def map(self, action, args):
        for idx, a in enumerate(args):
            self.qin.put((action, idx, a))
        results = [None] * len(args)
        for i in range(len(args)):
            idx, res = self.qout.get()
            results[idx] = res
        return results

    def free(self):
        self.time_in_proc = 0
        for i in range(self.num):
            self.pq[i].put('free') # Free the last batch
        for i in range(self.num):
            self.time_in_proc += self.qout.get()
        print('time_in_proc', self.time_in_proc / self.num)

    def build_batch(self, graphs, rewards, action_type_order):
        t = [time.time()]
        self.free()
        trajs = [generate_forward_trajectory(i) for i in graphs]
        t += [time.time()]
        data = self.map('test', [(i, action_type_order) for tj in trajs for i in tj])
        t += [time.time()]
        self.free()
        print('test', t[-1] - t[-2])
        data = self.map('make_graph', [(i, action_type_order) for tj in trajs for i in tj])
        torch_graphs, actions, num_back = zip(*data)
        t += [time.time()]
        stop_idx = action_type_order.index(GraphActionType.Stop)
        num_backward = torch.tensor(
            [num_back[i + 1] if actions[i][0] != stop_idx else 1
             for i in range(len(num_back))], device=torch.device('cuda'))
        t += [time.time()]
        print('build_batch', ' '.join(f"{t[i+1]-t[i]:.3f}" for i in range(len(t)-1)))
        return trajs, torch_graphs, actions, num_backward
        
        
    def __del__(self):
        print('deleting pools')
        for i in range(self.num):
            self.pq[i].put('stop')
        for i in range(self.num):
            self.qin.put(None)
        for i in range(self.num):
            self.procs[i].join()
        
