from typing import Any, List, Tuple
from copy import deepcopy

import torch
import torch_geometric.data as gd
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence

from gflownet.envs.graph_building_env import Graph, GraphAction
from .graph_building_env import Graph, GraphAction, GraphActionType, GraphBuildingEnv, GraphBuildingEnvContext


# For typing's sake, we'll pretend that a sequence is a graph.
class Seq(Graph):
    def __init__(self):
        self.seq: list[Any] = []

    def __repr__(self):
        return "".join(self.seq)

    @property
    def nodes(self):
        return self.seq


class SeqBuildingEnv(GraphBuildingEnv):
    """This class masquerades as a GraphBuildingEnv, but actually generates sequences of tokens."""

    def __init__(self, variant):
        super().__init__()

    def new(self):
        return Seq()

    def step(self, g: Graph, a: GraphAction):
        s: Seq = deepcopy(g)  # type: ignore
        if a.action == GraphActionType.AddNode:
            s.seq.append(a.value)
        return s

    def count_backward_transitions(self, g: Graph, check_idempotent: bool = False):
        return 1

    def parents(self, g: Graph):
        s: Seq = deepcopy(g)  # type: ignore
        if not len(s.seq):
            return []
        v = s.seq.pop()
        return [(GraphAction(GraphActionType.AddNode, value=v), s)]

    def reverse(self, g: Graph, ga: GraphAction):
        # TODO: if we implement non-LR variants we'll need to do something here
        return GraphAction(GraphActionType.Stop)


class SeqBatch:
    def __init__(self, seqs: List[torch.Tensor], pad: int):
        self.seqs = seqs
        self.x = pad_sequence(seqs, batch_first=False, padding_value=pad)
        self.mask = self.x.eq(pad).T
        self.lens = torch.tensor([len(i) for i in seqs], dtype=torch.long)
        self.logit_idx = self.x.ne(pad).flatten().nonzero().flatten()
        self.num_graphs = self.lens.sum().item()
        # self.batch = torch.tensor([[i] * len(s) for i, s in enumerate(seqs)])

    def to(self, device):
        for name in dir(self):
            x = getattr(self, name)
            if isinstance(x, torch.Tensor):
                setattr(self, name, x.to(device))
        return self


class GenericSeqBuildingContext(GraphBuildingEnvContext):
    def __init__(self, alphabet, num_cond_dim=0):
        self.alphabet = alphabet
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode]

        self.num_tokens = len(alphabet) + 2  # Alphabet + BOS + PAD
        self.bos_token = len(alphabet)
        self.pad_token = len(alphabet) + 1
        self.num_outputs = len(alphabet) + 1  # Alphabet + Stop
        self.num_cond_dim = num_cond_dim

    def aidx_to_GraphAction(self, g: Data, action_idx: Tuple[int, int, int], fwd: bool = True) -> GraphAction:
        act_type, _, act_col = [int(i) for i in action_idx]
        t = self.action_type_order[act_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, value=act_col)
        raise ValueError(action_idx)

    def GraphAction_to_aidx(self, g: Data, action: GraphAction) -> Tuple[int, int, int]:
        if action.action is GraphActionType.Stop:
            col = 0
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.AddNode:
            col = action.value
            type_idx = self.action_type_order.index(action.action)
        else:
            raise ValueError(action)
        return (type_idx, 0, int(col))

    def graph_to_Data(self, g: Graph):
        s: Seq = g  # type: ignore
        return torch.tensor([self.bos_token] + s.seq, dtype=torch.long)

    def collate(self, graphs: List[Data]):
        return SeqBatch(graphs, pad=self.pad_token)

    def is_sane(self, g: Graph) -> bool:
        return True

    def graph_to_mol(self, g: Graph):
        s: Seq = g  # type: ignore
        return "".join(self.alphabet[i] for i in s.seq)

    def object_to_log_repr(self, g: Graph):
        return self.graph_to_mol(g)
