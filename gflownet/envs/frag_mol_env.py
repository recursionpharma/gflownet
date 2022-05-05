import os
from typing import List, Tuple

import numpy as np
import rdkit.Chem as Chem
import torch
import torch_geometric.data as gd
from gflownet.envs.graph_building_env import (Graph, GraphAction, GraphActionType, GraphBuildingEnvContext)


class FragMolBuildingEnvContext(GraphBuildingEnvContext):
    def __init__(self, num_cond_dim=0):
        self.frags_smi = open(os.path.split(__file__)[0] + '/frags_72.txt', 'r').read().splitlines()
        self.frags_mol = [Chem.MolFromSmiles(i) for i in self.frags_smi]
        self.frags_stems = [[
            atomidx for atomidx in range(m.GetNumAtoms()) if m.GetAtomWithIdx(atomidx).GetTotalNumHs() > 0
        ] for m in self.frags_mol]
        self.frags_numatm = [m.GetNumAtoms() for m in self.frags_mol]
        self.num_stem_acts = most_stems = max(map(len, self.frags_stems))
        self.action_map = [(fragidx, stemidx)
                           for fragidx in range(len(self.frags_stems))
                           for stemidx in range(len(self.frags_stems[fragidx]))]
        self.num_actions = len(self.action_map)
        # These values are used by Models to know how many inputs/logits to produce
        self.num_new_node_values = len(self.frags_smi)
        self.num_node_attr_logits = 0
        self.num_node_dim = len(self.frags_smi) + 1
        self.num_edge_attr_logits = most_stems * 2
        self.num_edge_dim = most_stems * 2
        self.num_cond_dim = num_cond_dim

        # Order in which models have to output logits
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.SetEdgeAttr]
        self.device = torch.device('cpu')

    def aidx_to_GraphAction(self, g: gd.Data, action_idx: Tuple[int, int, int]):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction"""
        act_type, act_row, act_col = [int(i) for i in action_idx]
        t = self.action_type_order[act_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=act_row, value=act_col)
        elif t is GraphActionType.SetEdgeAttr:
            a, b = g.edge_index[:, act_row * 2]  # Edges are duplicated to get undirected GNN, deduplicated for logits
            if act_col < self.num_stem_acts:
                attr = f'{int(a)}_attach'
                val = act_col
            else:
                attr = f'{int(b)}_attach'
                val = act_col - self.num_stem_acts
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr, value=val)

    def GraphAction_to_aidx(self, g: gd.Data, action: GraphAction) -> Tuple[int, int, int]:
        """Translate a GraphAction to an index tuple"""
        if action.action is GraphActionType.Stop:
            row = col = 0
        elif action.action is GraphActionType.AddNode:
            row = action.source
            col = action.value
        elif action.action is GraphActionType.SetEdgeAttr:
            # Here the edges are duplicated, both (i,j) and (j,i) are in edge_index
            # so no need for a double check.
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            # Because edges are duplicated but logits aren't, divide by two
            row = row.div(2, rounding_mode='floor')  # type: ignore
            if action.attr == f'{int(action.source)}_attach':
                col = action.value
            else:
                col = action.value + self.num_stem_acts
        type_idx = self.action_type_order.index(action.action)
        return (type_idx, int(row), int(col))

    def graph_to_Data(self, g: Graph):
        """Convert a networkx Graph to a torch geometric Data instance"""
        x = torch.zeros((max(1, len(g.nodes)), self.num_node_dim))
        x[0, -1] = len(g.nodes) == 0
        for i, n in enumerate(g.nodes):
            x[i, g.nodes[n]['v']] = 1
        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            a, b = e
            for n, offset in zip(e, [0, self.num_stem_acts]):
                idx = ad.get('{int(n)}_attach', 0) + offset
                edge_attr[i * 2, idx] = 1
                edge_attr[i * 2 + 1, idx] = 1
        edge_index = torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).reshape(
            (-1, 2)).T
        return gd.Data(x, edge_index, edge_attr)

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        return gd.Batch.from_data_list(graphs, follow_batch=['edge_index'])

    def mol_to_graph(self, mol):
        """Convert an RDMol to a Graph"""
        raise NotImplementedError()
        g = Graph()
        # Only set an attribute tag if it is not the default attribute
        for a in mol.GetAtoms():
            attrs = {
                'chi': a.GetChiralTag(),
                'charge': a.GetFormalCharge(),
                'expl_H': a.GetNumExplicitHs(),
                'no_impl': a.GetNoImplicit()
            }
            g.add_node(a.GetIdx(), v=a.GetSymbol(),
                       **{attr: val for attr, val in attrs.items() if val != self.atom_attr_defaults[attr]})
        for b in mol.GetBonds():
            attrs = {'type': b.GetBondType()}
            g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                       **{attr: val for attr, val in attrs.items() if val != self.bond_attr_defaults[attr]})
        return g

    def graph_to_mol(self, g):
        offsets = np.cumsum([0] + [self.frags_numatm[g.nodes[i]['v']] for i in g])
        mol = None
        for i in g.nodes:
            if mol is None:
                mol = self.frags_mol[g.nodes[i]['v']]
            else:
                mol = Chem.CombineMols(mol, self.frags_mol[g.nodes[i]['v']])

        mol = Chem.EditableMol(mol)
        for a, b in g.edges:
            afrag = g.nodes[a]['v']
            bfrag = g.nodes[b]['v']
            mol.AddBond(int(self.frags_stems[afrag][g.edges[(a, b)].get(f'{a}_attach', 0)] + offsets[a]),
                        int(self.frags_stems[bfrag][g.edges[(a, b)].get(f'{b}_attach', 0)] + offsets[b]))
        mol = mol.GetMol()
        return mol

    def is_sane(self, g):
        try:
            mol = self.graph_to_mol(g)
            assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None
        except Exception:
            return False
        if mol is None:
            return False
        return True
