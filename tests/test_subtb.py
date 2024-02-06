from functools import reduce

import networkx as nx
import numpy as np
import torch

from gflownet.algo.trajectory_balance import subTB
from gflownet.envs.frag_mol_env import NCounter


def subTB_ref(P_F, P_B, F):
    h = F.shape[0] - 1
    assert P_F.shape == P_B.shape == (h,)
    assert F.ndim == 1

    dtype = reduce(torch.promote_types, [P_F.dtype, P_B.dtype, F.dtype])
    D = torch.zeros(h, h, dtype=dtype)
    for i in range(h):
        for j in range(i, h):
            D[i, j] = F[i] - F[j + 1]
            D[i, j] = D[i, j] + P_F[i : j + 1].sum()
            D[i, j] = D[i, j] - P_B[i : j + 1].sum()
    return D


def test_subTB():
    for T in range(5, 20):
        T = 10
        P_F = torch.randint(1, 10, (T,))
        P_B = torch.randint(1, 10, (T,))
        F = torch.randint(1, 10, (T + 1,))
        assert (subTB(F, P_F - P_B) == subTB_ref(P_F, P_B, F)).all()


def test_n():
    n = NCounter()
    x = 0
    for i in range(1, 10):
        x += np.log(i)
        assert np.isclose(n.lfac(i), x)

    assert np.isclose(n.lcomb(5, 2), np.log(10))


def test_g1():
    n = NCounter()
    g = nx.Graph()
    for i in range(3):
        g.add_node(i)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    rg = n.root_tree(g, 0)
    assert n.f(rg, 0) == 0
    rg = n.root_tree(g, 2)
    assert n.f(rg, 2) == 0
    rg = n.root_tree(g, 1)
    assert np.isclose(n.f(rg, 1), np.log(2))

    assert np.isclose(n(g), np.log(4))


def test_g():
    n = NCounter()
    g = nx.Graph()
    for i in range(3):
        g.add_node(i)
    g.add_edge(0, 1)
    g.add_edge(1, 2, weight=2)
    rg = n.root_tree(g, 0)
    assert n.f(rg, 0) == 0
    rg = n.root_tree(g, 2)
    assert np.isclose(n.f(rg, 2), np.log(2))
    rg = n.root_tree(g, 1)
    assert np.isclose(n.f(rg, 1), np.log(3))

    assert np.isclose(n(g), np.log(6))
