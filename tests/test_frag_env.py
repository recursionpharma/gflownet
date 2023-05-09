import base64
import pickle
from collections import defaultdict

import networkx as nx
import pytest

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv


def build_two_node_states():
    # TODO: This is actually fairly generic code that will probably be reused by other tests in the future.
    # Having a proper class to handle graph-indexed hash maps would probably be good.
    graph_cache = {}
    graph_by_idx = {}
    _graph_cache_buckets = {}

    # We're enumerating all states of length two, but we could've just as well randomly sampled
    # some states.
    env = GraphBuildingEnv()
    ctx = FragMolBuildingEnvContext(max_frags=2)

    def g2h(g):
        gc = g.to_directed()
        for e in gc.edges:
            gc.edges[e]["v"] = (
                str(gc.edges[e].get(f"{e[0]}_attach", -1)) + "_" + str(gc.edges[e].get(f"{e[1]}_attach", -1))
            )
        h = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(gc, "v", "v")
        if h not in _graph_cache_buckets:
            _graph_cache_buckets[h] = [g]
            return h + "_0"
        else:
            bucket = _graph_cache_buckets[h]
            for i, gp in enumerate(bucket):
                if nx.algorithms.isomorphism.is_isomorphic(g, gp, lambda a, b: a == b, lambda a, b: a == b):
                    return h + "_" + str(i)
            # Nothing was isomorphic
            bucket.append(g)
            return h + "_" + str(len(bucket) - 1)

    mdp_graph = nx.DiGraph()
    mdp_graph.add_node(0)
    graph_by_idx[0] = nx.Graph()

    def expand(s, idx):
        # Recursively expand all children of s
        gd = ctx.graph_to_Data(s)
        masks = [getattr(gd, gat.mask_name) for gat in ctx.action_type_order]
        for at, mask in enumerate(masks):
            if at == 0:  # Ignore Stop action
                continue
            nz = mask.nonzero()
            for i in nz:  # Only expand non-masked legal actions
                aidx = (at, i[0].item(), i[1].item())
                ga = ctx.aidx_to_GraphAction(gd, aidx)
                sp = env.step(s, ga)
                h = g2h(sp)
                if h in graph_cache:
                    idxp = graph_cache[h][1]
                else:
                    idxp = len(mdp_graph)
                    mdp_graph.add_node(idxp)
                    graph_cache[h] = (sp, idxp)
                    graph_by_idx[idxp] = sp
                    expand(sp, idxp)
                mdp_graph.add_edge(idx, idxp)

    expand(graph_by_idx[0], 0)
    return [graph_by_idx[i] for i in list(nx.topological_sort(mdp_graph))]


@pytest.fixture
def two_node_states(request):
    data = request.config.cache.get("frag_env/two_node_states", None)
    if data is None:
        data = build_two_node_states()
        # pytest caches through JSON so we have to make a clean enough string
        request.config.cache.set("frag_env/two_node_states", base64.b64encode(pickle.dumps(data)).decode())
    else:
        data = pickle.loads(base64.b64decode(data))
    return data


def test_backwards_mask_equivalence(two_node_states):
    """This tests that FragMolBuildingEnvContext implements backwards masks correctly. It treats
    GraphBuildingEnv.count_backward_transitions as the ground truth and raises an error if there is
    a different number of actions leading to the parents of any state.
    """
    env = GraphBuildingEnv()
    ctx = FragMolBuildingEnvContext(max_frags=2)
    for i in range(1, len(two_node_states)):
        g = two_node_states[i]
        n = env.count_backward_transitions(g, check_idempotent=False)
        nm = 0
        gd = ctx.graph_to_Data(g)
        for u, k in enumerate(ctx.bck_action_type_order):
            m = getattr(gd, k.mask_name)
            nm += m.sum()
        if n != nm:
            raise ValueError()


def test_backwards_mask_equivalence_ipa(two_node_states):
    """This tests that FragMolBuildingEnvContext implements backwards masks correctly. It treats
    GraphBuildingEnv.count_backward_transitions as the ground truth and raises an error if there is
    a different number of actions leading to the parents of any state.

    This test also accounts for idempotent actions.
    """
    env = GraphBuildingEnv()
    ctx = FragMolBuildingEnvContext(max_frags=2)
    algo = TrajectoryBalance(env, ctx, None, defaultdict(int), max_nodes=2)
    for i in range(1, len(two_node_states)):
        g = two_node_states[i]
        n = env.count_backward_transitions(g, check_idempotent=True)
        gd = ctx.graph_to_Data(g)
        # To check that we're computing masks correctly, we need to check that there is the same
        # number of idempotent action classes, i.e. groups of actions that lead to the same parent.
        equivalence_classes = []
        for u, k in enumerate(ctx.bck_action_type_order):
            m = getattr(gd, k.mask_name)
            for a in m.nonzero():
                a = (u, a[0].item(), a[1].item())
                for c in equivalence_classes:
                    # Here `a` could have been added in another equivalence class by
                    # get_idempotent_actions. If so, no need to check it.
                    if a in c:
                        break
                else:
                    ga = ctx.aidx_to_GraphAction(gd, a, fwd=False)
                    gp = env.step(g, ga)
                    # TODO: It is a bit weird that get_idempotent_actions is in an algo class,
                    # probably also belongs in a graph utils file.
                    ipa = algo.get_idempotent_actions(g, gd, gp, ga)
                    equivalence_classes.append(ipa)
        if n != len(equivalence_classes):
            raise ValueError()
