from functools import reduce

import torch

from gflownet.algo.trajectory_balance import subTB


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
