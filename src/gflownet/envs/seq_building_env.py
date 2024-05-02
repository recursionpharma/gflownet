from copy import deepcopy
from typing import Any, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data

from gflownet.envs.graph_building_env import (
    ActionIndex,
    Graph,
    GraphAction,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)


# For typing's sake, we'll pretend that a sequence is a graph.
class Seq(Graph):
    def __init__(self):
        self.seq: list[Any] = []

    def __repr__(self):
        return "".join(map(str, self.seq))

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
        # This tells where (in the flattened array of outputs) the non-masked outputs are.
        # E.g. if the batch is [["ABC", "VWXYZ"]], logit_idx would be [0, 1, 2, 5, 6, 7, 8, 9]
        self.logit_idx = self.x.ne(pad).T.flatten().nonzero().flatten()
        # Since we're feeding this batch object to graph-based algorithms, we have to use this naming, but this
        # is the total number of timesteps.
        self.num_graphs = self.lens.sum().item()

    def to(self, device):
        for name in dir(self):
            x = getattr(self, name)
            if isinstance(x, torch.Tensor):
                setattr(self, name, x.to(device))
        return self


class AutoregressiveSeqBuildingContext(GraphBuildingEnvContext):
    """This class masquerades as a GraphBuildingEnvContext, but actually generates sequences of tokens.

    This context gets an agent to generate sequences of tokens from left to right, i.e. in an autoregressive fashion.
    """

    def __init__(self, alphabet: Sequence[str], num_cond_dim=0):
        self.alphabet = alphabet
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode]

        self.num_tokens = len(alphabet) + 2  # Alphabet + BOS + PAD
        self.bos_token = len(alphabet)
        self.pad_token = len(alphabet) + 1
        self.num_actions = len(alphabet) + 1  # Alphabet + Stop
        self.num_cond_dim = num_cond_dim

    def ActionIndex_to_GraphAction(self, g: Data, aidx: ActionIndex, fwd: bool = True) -> GraphAction:
        # Since there's only one "object" per timestep to act upon (in graph parlance), the row is always == 0
        t = self.action_type_order[aidx.action_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, value=aidx.col_idx)
        raise ValueError(aidx)

    def GraphAction_to_ActionIndex(self, g: Data, action: GraphAction) -> ActionIndex:
        if action.action is GraphActionType.Stop:
            col = 0
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.AddNode:
            col = action.value
            type_idx = self.action_type_order.index(action.action)
        else:
            raise ValueError(action)
        return ActionIndex(action_type=type_idx, row_idx=0, col_idx=int(col))

    def graph_to_Data(self, g: Graph):
        s: Seq = g  # type: ignore
        return torch.tensor([self.bos_token] + s.seq, dtype=torch.long)

    def collate(self, graphs: List[Data]):
        return SeqBatch(graphs, pad=self.pad_token)

    def is_sane(self, g: Graph) -> bool:
        return True

    def graph_to_obj(self, g: Graph):
        s: Seq = g  # type: ignore
        return "".join(self.alphabet[int(i)] for i in s.seq)

    def object_to_log_repr(self, g: Graph):
        return self.graph_to_obj(g)
