import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_add


def random_walk_probs(g: Data, k: int, skip_odd=False):
    source, _ = g.edge_index[0], g.edge_index[1]
    deg = scatter_add(torch.ones_like(source), source, dim=0, dim_size=g.num_nodes)
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if g.edge_index.shape[1] == 0:
        P = g.edge_index.new_zeros((1, g.num_nodes, g.num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)  # (1, num nodes, num nodes)
    diags = []
    if skip_odd:
        Pmult = P @ P
    else:
        Pmult = P
    Pk = Pmult
    for _ in range(k):
        diags.append(torch.diagonal(Pk, dim1=-2, dim2=-1))
        Pk = Pk @ Pmult
    p = torch.cat(diags, dim=0).transpose(0, 1)  # (num nodes, k)
    return p
