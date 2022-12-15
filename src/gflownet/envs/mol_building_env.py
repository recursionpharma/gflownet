from typing import List, Tuple

import networkx as nx
import numpy as np
from rdkit.Chem import Mol
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import ChiralType
import torch
import torch_geometric.data as gd

from gflownet.envs.graph_building_env import Graph
from gflownet.envs.graph_building_env import GraphAction
from gflownet.envs.graph_building_env import GraphActionType
from gflownet.envs.graph_building_env import GraphBuildingEnvContext

DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]


class MolBuildingEnvContext(GraphBuildingEnvContext):
    """A specification of what is being generated for a GraphBuildingEnv

    This context specifies how to create molecules atom-by-atom (and attribute-by-attribute).

    """
    def __init__(self, atoms=['H', 'C', 'N', 'O', 'F'], num_cond_dim=0, chiral_types=DEFAULT_CHIRAL_TYPES,
                 charges=[0, 1, -1], expl_H_range=[0, 1], allow_explicitly_aromatic=False):
        # idx 0 has to coincide with the default value
        self.atom_attr_values = {
            'v': atoms + ['*'],
            'chi': chiral_types,
            'charge': charges,
            'expl_H': expl_H_range,
            'no_impl': [False, True],
            'fill_wildcard': [None] + atoms,  # default is, there is nothing
        }
        self.default_wildcard_replacement = 'C'
        self.negative_attrs = ['fill_wildcard']
        self.atom_attr_defaults = {k: self.atom_attr_values[k][0] for k in self.atom_attr_values}
        # The size of the input vector for each atom
        self.atom_attr_size = sum(len(i) for i in self.atom_attr_values.values())
        self.atom_attrs = sorted(self.atom_attr_values.keys())
        # The beginning position within the input vector of each attribute
        self.atom_attr_slice = [0] + list(np.cumsum([len(self.atom_attr_values[i]) for i in self.atom_attrs]))
        # The beginning position within the logit vector of each attribute
        num_atom_logits = [len(self.atom_attr_values[i]) - 1 for i in self.atom_attrs]
        self.atom_attr_logit_slice = {
            k: (s, e)
            for k, s, e in zip(self.atom_attrs, [0] + list(np.cumsum(num_atom_logits)), np.cumsum(num_atom_logits))
        }
        # The attribute and value each logit dimension maps back to
        self.atom_attr_logit_map = [
            (k, v) for k in self.atom_attrs if k != 'v'
            # index 0 is skipped because it is the default value
            for v in self.atom_attr_values[k][1:]
        ]

        # By default, instead of allowing/generating aromatic bonds, we instead "ask of" the
        # generative process to generate a Kekulized form of the molecule. RDKit is capable of
        # recovering aromatic ring, and so we leave it at that.
        self.allow_explicitly_aromatic = allow_explicitly_aromatic
        aromatic_optional = [BondType.AROMATIC] if allow_explicitly_aromatic else []
        self.bond_attr_values = {
            'type': [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE] + aromatic_optional,
        }
        self.bond_attr_defaults = {k: self.bond_attr_values[k][0] for k in self.bond_attr_values}
        self.bond_attr_size = sum(len(i) for i in self.bond_attr_values.values())
        self.bond_attrs = sorted(self.bond_attr_values.keys())
        self.bond_attr_slice = [0] + list(np.cumsum([len(self.bond_attr_values[i]) for i in self.bond_attrs]))
        num_bond_logits = [len(self.bond_attr_values[i]) - 1 for i in self.bond_attrs]
        self.bond_attr_logit_slice = {
            k: (s, e)
            for k, s, e in zip(self.bond_attrs, [0] + list(np.cumsum(num_bond_logits)), np.cumsum(num_bond_logits))
        }
        self.bond_attr_logit_map = [(k, v) for k in self.bond_attrs for v in self.bond_attr_values[k][1:]]
        self._bond_valence = {
            BondType.SINGLE: 1,
            BondType.DOUBLE: 2,
            BondType.TRIPLE: 3,
            BondType.AROMATIC: 1.5,
        }
        pt = Chem.GetPeriodicTable()
        self._max_atom_valence = {
            **{a: max(pt.GetValenceList(a)) for a in atoms},
            'N': 5,  # allow nitro groups by allowing the 5-valent N (perhaps there's a better way?)
            '*': 0,  # wildcard atoms have 0 valence until filled in
        }

        # These values are used by Models to know how many inputs/logits to produce
        self.num_new_node_values = len(atoms)
        self.num_node_attr_logits = len(self.atom_attr_logit_map)
        self.num_node_dim = self.atom_attr_size + 1
        self.num_edge_attr_logits = len(self.bond_attr_logit_map)
        self.num_edge_dim = self.bond_attr_size
        self.num_cond_dim = num_cond_dim

        # Order in which models have to output logits
        self.action_type_order = [
            GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.SetNodeAttr, GraphActionType.AddEdge,
            GraphActionType.SetEdgeAttr
        ]
        self.device = torch.device('cpu')

    def aidx_to_GraphAction(self, g: gd.Data, action_idx: Tuple[int, int, int]):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction"""
        act_type, act_row, act_col = [int(i) for i in action_idx]
        t = self.action_type_order[act_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=act_row, value=self.atom_attr_values['v'][act_col])
        elif t is GraphActionType.SetNodeAttr:
            attr, val = self.atom_attr_logit_map[act_col]
            return GraphAction(t, source=act_row, attr=attr, value=val)
        elif t is GraphActionType.AddEdge:
            a, b = g.non_edge_index[:, act_row]
            return GraphAction(t, source=a.item(), target=b.item())
        elif t is GraphActionType.SetEdgeAttr:
            a, b = g.edge_index[:, act_row * 2]  # Edges are duplicated to get undirected GNN, deduplicated for logits
            attr, val = self.bond_attr_logit_map[act_col]
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr, value=val)

    def GraphAction_to_aidx(self, g: gd.Data, action: GraphAction) -> Tuple[int, int, int]:
        """Translate a GraphAction to an index tuple"""
        if action.action is GraphActionType.Stop:
            row = col = 0
        elif action.action is GraphActionType.AddNode:
            row = action.source
            col = self.atom_attr_values['v'].index(action.value)
        elif action.action is GraphActionType.SetNodeAttr:
            row = action.source
            # - 1 because the default is index 0
            col = self.atom_attr_values[action.attr].index(
                action.value) - 1 + self.atom_attr_logit_slice[action.attr][0]
        elif action.action is GraphActionType.AddEdge:
            # Here we have to retrieve the index in non_edge_index of an edge (s,t)
            # that's also possibly in the reverse order (t,s).
            # That's definitely not too efficient, can we do better?
            row = ((g.non_edge_index.T == torch.tensor([(action.source, action.target)])).prod(1) +
                   (g.non_edge_index.T == torch.tensor([(action.target, action.source)])).prod(1)).argmax()
            col = 0
        elif action.action is GraphActionType.SetEdgeAttr:
            # Here the edges are duplicated, both (i,j) and (j,i) are in edge_index
            # so no need for a double check.
            # row = ((g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1) +
            #       (g.edge_index.T == torch.tensor([(action.target, action.source)])).prod(1)).argmax()
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            # Because edges are duplicated but logits aren't, divide by two
            row = row.div(2, rounding_mode='floor')  # type: ignore
            col = self.bond_attr_values[action.attr].index(
                action.value) - 1 + self.bond_attr_logit_slice[action.attr][0]
        type_idx = self.action_type_order.index(action.action)
        return (type_idx, int(row), int(col))

    def graph_to_Data(self, g: Graph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        x = torch.zeros((max(1, len(g.nodes)), self.num_node_dim))
        x[0, -1] = len(g.nodes) == 0
        add_node_mask = torch.ones((x.shape[0], self.num_new_node_values))
        explicit_valence = {}
        max_valence = {}
        set_node_attr_mask = torch.ones((x.shape[0], self.num_node_attr_logits))
        if not len(g.nodes):
            set_node_attr_mask *= 0
        for i, n in enumerate(g.nodes):
            ad = g.nodes[n]
            for k, sl in zip(self.atom_attrs, self.atom_attr_slice):
                idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                x[i, sl + idx] = 1
                # If the attribute is already there, mask out logits
                # (or if the attribute is a negative attribute and has been filled)
                if k in self.negative_attrs:
                    if k in ad and idx > 0 or k not in ad:
                        s, e = self.atom_attr_logit_slice[k]
                        set_node_attr_mask[i, s:e] = 0
                elif k in ad:
                    s, e = self.atom_attr_logit_slice[k]
                    set_node_attr_mask[i, s:e] = 0
            # Account for charge and explicit Hs in atom as limiting the total valence
            max_atom_valence = self._max_atom_valence[ad.get('fill_wildcard', None) or ad['v']]
            max_valence[n] = max_atom_valence - abs(ad.get('charge', 0)) - ad.get('expl_H', 0)
            # Compute explicitly defined valence:
            explicit_valence[n] = 0
            for ne in g[n]:
                explicit_valence[n] += self._bond_valence[g.edges[(n, ne)].get('type', self.bond_attr_defaults['type'])]
            # If the valence is maxed out, mask out logits that would add a new atom + single bond to this node
            if explicit_valence[n] >= max_valence[n]:
                add_node_mask[i, :] = 0
            # If charge is not yet defined make sure there is room in the valence
            if 'charge' not in ad and explicit_valence[n] + 1 > max_valence[n]:
                s, e = self.atom_attr_logit_slice['charge']
                set_node_attr_mask[i, s:e] = 0
            # idem for explicit hydrogens
            if 'expl_H' not in ad and explicit_valence[n] + 1 > max_valence[n]:
                s, e = self.atom_attr_logit_slice['expl_H']
                set_node_attr_mask[i, s:e] = 0

        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        set_edge_attr_mask = torch.zeros((len(g.edges), self.num_edge_attr_logits))
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for k, sl in zip(self.bond_attrs, self.bond_attr_slice):
                idx = self.bond_attr_values[k].index(ad[k]) if k in ad else 0
                edge_attr[i * 2, sl + idx] = 1
                edge_attr[i * 2 + 1, sl + idx] = 1
                if k in ad:  # If the attribute is already there, mask out logits
                    s, e = self.bond_attr_logit_slice[k]
                    set_edge_attr_mask[i, s:e] = 0
            # Check which bonds don't bust the valence of their atoms
            if 'type' not in ad:  # Only if type isn't already set
                sl, _ = self.bond_attr_logit_slice['type']
                for ti, bond_type in enumerate(self.bond_attr_values['type'][1:]):  # [1:] because 0th is default
                    # -1 because we'd be removing the single bond and replacing it with a double/triple/aromatic bond
                    is_ok = all([explicit_valence[n] + self._bond_valence[bond_type] - 1 <= max_valence[n] for n in e])
                    set_edge_attr_mask[i, sl + ti] = float(is_ok)
        edge_index = torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).reshape(
            (-1, 2)).T
        gc = nx.complement(g)

        def is_ok_non_edge(e):
            return all([explicit_valence[i] + 1 <= max_valence[i] for i in e])

        non_edge_index = torch.tensor([i for i in gc.edges if is_ok_non_edge(i)], dtype=torch.long).T.reshape((2, -1))

        return gd.Data(
            x,
            edge_index,
            edge_attr,
            non_edge_index=non_edge_index,
            add_node_mask=add_node_mask,
            set_node_attr_mask=set_node_attr_mask,
            add_edge_mask=torch.ones((non_edge_index.shape[1], 1)),  # Already filtered by is_ok_non_edge
            set_edge_attr_mask=set_edge_attr_mask,
        )

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        return gd.Batch.from_data_list(graphs, follow_batch=['edge_index', 'non_edge_index'])

    def mol_to_graph(self, mol: Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = Graph()
        mol = Mol(mol)  # Make a copy
        if not self.allow_explicitly_aromatic:
            # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
            Chem.Kekulize(mol, clearAromaticFlags=True)
        # Only set an attribute tag if it is not the default attribute
        for a in mol.GetAtoms():
            attrs = {
                'chi': a.GetChiralTag(),
                'charge': a.GetFormalCharge(),
                'expl_H': a.GetNumExplicitHs(),
                # RDKit makes * atoms have no implicit Hs, but we don't want this to trickle down.
                'no_impl': a.GetNoImplicit() and a.GetSymbol() != '*',
            }
            g.add_node(a.GetIdx(), v=a.GetSymbol(),
                       **{attr: val for attr, val in attrs.items() if val != self.atom_attr_defaults[attr]},
                       **({
                           'fill_wildcard': None
                       } if a.GetSymbol() == '*' else {}))
        for b in mol.GetBonds():
            attrs = {'type': b.GetBondType()}
            g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                       **{attr: val for attr, val in attrs.items() if val != self.bond_attr_defaults[attr]})
        return g

    def graph_to_mol(self, g: Graph) -> Mol:
        mp = Chem.RWMol()
        mp.BeginBatchEdit()
        for i in range(len(g.nodes)):
            d = g.nodes[i]
            s = d.get('fill_wildcard', d['v'])
            a = Chem.Atom(s if s is not None else self.default_wildcard_replacement)
            if 'chi' in d:
                a.SetChiralTag(d['chi'])
            if 'charge' in d:
                a.SetFormalCharge(d['charge'])
            if 'expl_H' in d:
                a.SetNumExplicitHs(d['expl_H'])
            if 'no_impl' in d:
                a.SetNoImplicit(d['no_impl'])
            mp.AddAtom(a)
        for e in g.edges:
            d = g.edges[e]
            mp.AddBond(e[0], e[1], d.get('type', BondType.SINGLE))
        mp.CommitBatchEdit()
        Chem.SanitizeMol(mp)
        return mp

    def is_sane(self, g: Graph) -> bool:
        try:
            mol = self.graph_to_mol(g)
            assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None
        except Exception:
            return False
        if mol is None:
            return False
        return True
