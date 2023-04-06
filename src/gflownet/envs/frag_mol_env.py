from collections import defaultdict
from typing import List, Tuple

import numpy as np
import rdkit.Chem as Chem
import torch
import torch_geometric.data as gd

from gflownet.envs.graph_building_env import Graph
from gflownet.envs.graph_building_env import GraphAction
from gflownet.envs.graph_building_env import GraphActionType
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
from gflownet.models import bengio2021flow


class FragMolBuildingEnvContext(GraphBuildingEnvContext):
    """A specification of what is being generated for a GraphBuildingEnv

    This context specifies how to create molecules fragment by fragment as encoded by a junction tree.
    Fragments are obtained from the original GFlowNet paper, Bengio et al., 2021.

    This context works by having the agent generate a (tree) graph of fragments, and by then having
    the agent specify which atom each edge uses as an attachment point (single bond) between
    fragments. Masks ensure that the agent can only perform chemically valid attachments.
    """
    def __init__(self, max_frags: int = 9, num_cond_dim: int = 0, fragments: List[Tuple[str, List[int]]] = None):
        """Construct a fragment environment
        Parameters
        ----------
        max_frags: int
            The maximum number of fragments the agent is allowed to insert.
        num_cond_dim: int
            The dimensionality of the observations' conditional information vector (if >0)
        fragments: List[Tuple[str, List[int]]]
            A list of (SMILES, List[attachment atom idx]) fragments. If None the default is to use
            the fragments of Bengio et al., 2021.
        """
        self.max_frags = max_frags
        if fragments is None:
            smi, stems = zip(*bengio2021flow.FRAGMENTS)
        else:
            smi, stems = zip(*fragments)
        self.frags_smi = smi
        self.frags_mol = [Chem.MolFromSmiles(i) for i in self.frags_smi]
        self.frags_stems = stems
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
        self.num_edge_dim = (most_stems + 1) * 2
        self.num_cond_dim = num_cond_dim
        self.edges_are_duplicated = True
        self.edges_are_unordered = False

        # Order in which models have to output logits
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.SetEdgeAttr]
        self.device = torch.device('cpu')

    def aidx_to_GraphAction(self, g: gd.Data, action_idx: Tuple[int, int, int]):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction

        Parameters
        ----------
        g: gd.Data
            The graph object on which this action would be applied.
        action_idx: Tuple[int, int, int]
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.

        Returns
        action: GraphAction
            A graph action whose type is one of Stop, AddNode, or SetEdgeAttr.
        """
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
        """Translate a GraphAction to an index tuple

        Parameters
        ----------
        g: gd.Data
            The graph object on which this action would be applied.
        action: GraphAction
            A graph action whose type is one of Stop, AddNode, or SetEdgeAttr.

        Returns
        -------
        action_idx: Tuple[int, int, int]
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.
        """
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

    def graph_to_Data(self, g: Graph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance
        Parameters
        ----------
        g: Graph
            A Graph object representing a fragment junction tree

        Returns
        -------
        data:  gd.Data
            The corresponding torch_geometric object.
        """
        x = torch.zeros((max(1, len(g.nodes)), self.num_node_dim))
        x[0, -1] = len(g.nodes) == 0
        for i, n in enumerate(g.nodes):
            x[i, g.nodes[n]['v']] = 1
        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        set_edge_attr_mask = torch.zeros((len(g.edges), self.num_edge_attr_logits))
        if len(g):
            degrees = torch.tensor(list(g.degree))[:, 1]
            max_degrees = torch.tensor([len(self.frags_stems[g.nodes[n]['v']]) for n in g.nodes])
        else:
            degrees = max_degrees = torch.zeros((0,))

        # We compute the attachment points of fragments that have been already used so that we can
        # mask them out for the agent (so that only one neighbor can be attached to one attachment
        # point at a time).
        attached = defaultdict(list)
        # If there are unspecified attachment points, we will disallow the agent from using the stop
        # action.
        has_unfilled_attach = False
        for e in g.edges:
            ed = g.edges[e]
            a = ed.get(f'{int(e[0])}_attach', -1)
            b = ed.get(f'{int(e[1])}_attach', -1)
            if a >= 0:
                attached[e[0]].append(a)
            else:
                has_unfilled_attach = True
            if b >= 0:
                attached[e[1]].append(b)
            else:
                has_unfilled_attach = True
        # Here we encode the attached atoms in the edge features, as well as mask out attached
        # atoms.
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for j, n in enumerate(e):
                idx = ad.get(f'{int(n)}_attach', -1) + 1
                edge_attr[i * 2, idx + (self.num_stem_acts + 1) * j] = 1
                edge_attr[i * 2 + 1, idx + (self.num_stem_acts + 1) * (1 - j)] = 1
                if f'{int(n)}_attach' not in ad:
                    for attach_point in range(max_degrees[n]):
                        if attach_point not in attached[n]:
                            set_edge_attr_mask[i, attach_point + self.num_stem_acts * j] = 1
        edge_index = torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).reshape(
            (-1, 2)).T
        if x.shape[0] == self.max_frags:
            add_node_mask = torch.zeros((x.shape[0], 1))
        else:
            add_node_mask = (degrees < max_degrees).float()[:, None] if len(g.nodes) else torch.ones((1, 1))
        stop_mask = torch.zeros((1, 1)) if has_unfilled_attach or not len(g) else torch.ones((1, 1))

        return gd.Data(x, edge_index, edge_attr, stop_mask=stop_mask, add_node_mask=add_node_mask,
                       set_edge_attr_mask=set_edge_attr_mask)

    def collate(self, graphs: List[gd.Data]) -> gd.Batch:
        """Batch Data instances

        Parameters
        ----------
        graphs: List[gd.Data]
            A list of gd.Data objects (e.g. given by graph_to_Data).

        Returns
        batch: gd.Batch
            A torch_geometric Batch object
        """
        return gd.Batch.from_data_list(graphs, follow_batch=['edge_index'])

    def mol_to_graph(self, mol):
        """Convert an RDMol to a Graph"""
        raise NotImplementedError()

    def graph_to_mol(self, g: Graph) -> Chem.Mol:
        """Convert a Graph to an RDKit molecule

        Parameters
        ----------
        g: Graph
            A Graph instance representing a fragment junction tree.

        Returns
        -------
        m: Chem.Mol
            The corresponding RDKit molecule
        """
        offsets = np.cumsum([0] + [self.frags_numatm[g.nodes[i]['v']] for i in g])
        mol = None
        for i in g.nodes:
            if mol is None:
                mol = self.frags_mol[g.nodes[i]['v']]
            else:
                mol = Chem.CombineMols(mol, self.frags_mol[g.nodes[i]['v']])

        mol = Chem.EditableMol(mol)
        bond_atoms = []
        for a, b in g.edges:
            afrag = g.nodes[a]['v']
            bfrag = g.nodes[b]['v']
            u, v = (int(self.frags_stems[afrag][g.edges[(a, b)].get(f'{a}_attach', 0)] + offsets[a]),
                    int(self.frags_stems[bfrag][g.edges[(a, b)].get(f'{b}_attach', 0)] + offsets[b]))
            bond_atoms += [u, v]
            mol.AddBond(u, v, Chem.BondType.SINGLE)
        mol = mol.GetMol()

        def _pop_H(atom):
            atom = mol.GetAtomWithIdx(atom)
            nh = atom.GetNumExplicitHs()
            if nh > 0:
                atom.SetNumExplicitHs(nh - 1)

        list(map(_pop_H, bond_atoms))
        Chem.SanitizeMol(mol)
        return mol

    def is_sane(self, g: Graph) -> bool:
        """Verifies whether the given Graph is valid according to RDKit"""
        try:
            mol = self.graph_to_mol(g)
            assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None
        except Exception:
            return False
        if mol is None:
            return False
        return True
