from collections import defaultdict
from typing import List, Tuple

import numpy as np
import rdkit.Chem as Chem
import torch
import torch_geometric.data as gd

from gflownet.envs.graph_building_env import Graph, GraphAction, GraphActionType, GraphBuildingEnvContext
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
        self.action_map = [
            (fragidx, stemidx)
            for fragidx in range(len(self.frags_stems))
            for stemidx in range(len(self.frags_stems[fragidx]))
        ]
        self.num_actions = len(self.action_map)

        # These values are used by Models to know how many inputs/logits to produce
        self.edges_are_duplicated = True
        # The ordering in which the model sees & produces logits for edges matters,
        # i.e. action(u, v) != action(v, u).
        # This is because of the way we encode attachment points (see below on semantics of SetEdgeAttr)
        self.edges_are_unordered = False
        self.num_new_node_values = len(self.frags_smi)
        self.num_node_attrs = 1
        self.num_node_attr_logits = 0
        self.num_node_dim = len(self.frags_smi) + 1

        # The semantics of the SetEdgeAttr indices is that, for edge (u, v), we use the first half
        # for u and the second half for v. Each logit i in the first half for a given edge
        # corresponds to setting the stem atom of fragment u used to attach between u and v to be i
        # (named f'{u}_attach') and vice versa for the second half and v, u.
        # Note to self: this choice results in a special case in generate_forward_trajectory for these
        # edge attributes. See PR#83 for details.
        self.num_edge_attr_logits = most_stems * 2
        # There are thus up to 2 edge attributes, the stem of u and the stem of v.
        self.num_edge_attrs = 2
        # The + 1 is for an extra dimension to indicate when the attribute isn't yet set
        self.num_edge_dim = (most_stems + 1) * 2
        self.num_cond_dim = num_cond_dim
        self.edges_are_duplicated = True
        self.edges_are_unordered = False

        # Order in which models have to output logits
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.SetEdgeAttr]
        self.bck_action_type_order = [
            GraphActionType.RemoveNode,
            GraphActionType.RemoveEdgeAttr,
        ]
        self.device = torch.device("cpu")
        self.sorted_frags = sorted(list(enumerate(self.frags_mol)), key=lambda x: -x[1].GetNumAtoms())

    def aidx_to_GraphAction(self, g: gd.Data, action_idx: Tuple[int, int, int], fwd: bool = True):
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
        if fwd:
            t = self.action_type_order[act_type]
        else:
            t = self.bck_action_type_order[act_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=act_row, value=act_col)
        elif t is GraphActionType.SetEdgeAttr:
            a, b = g.edge_index[:, act_row * 2]  # Edges are duplicated to get undirected GNN, deduplicated for logits
            if act_col < self.num_stem_acts:
                attr = f"{int(a)}_attach"
                val = act_col
            else:
                attr = f"{int(b)}_attach"
                val = act_col - self.num_stem_acts
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr, value=val)
        elif t is GraphActionType.RemoveNode:
            return GraphAction(t, source=act_row)
        elif t is GraphActionType.RemoveEdgeAttr:
            a, b = g.edge_index[:, act_row * 2]
            attr = f"{int(a)}_attach" if act_col == 0 else f"{int(b)}_attach"
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr)

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
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.AddNode:
            row = action.source
            col = action.value
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.SetEdgeAttr:
            # Here the edges are duplicated, both (i,j) and (j,i) are in edge_index
            # so no need for a double check.
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            # Because edges are duplicated but logits aren't, divide by two
            row = row.div(2, rounding_mode="floor")  # type: ignore
            if action.attr == f"{int(action.source)}_attach":
                col = action.value
            else:
                col = action.value + self.num_stem_acts
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.RemoveNode:
            row = action.source
            col = 0
            type_idx = self.bck_action_type_order.index(action.action)
        elif action.action is GraphActionType.RemoveEdgeAttr:
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            row = row.div(2, rounding_mode="floor")  # type: ignore
            if action.attr == f"{int(action.source)}_attach":
                col = 0
            else:
                col = 1
            type_idx = self.bck_action_type_order.index(action.action)
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
        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        set_edge_attr_mask = torch.zeros((len(g.edges), self.num_edge_attr_logits))
        # TODO: This is a bit silly but we have to do +1 when the graph is empty because the default
        # padding action is a [0, 0, 0], which needs to be legal for the empty state. Should be
        # fixable with a bit of smarts & refactoring.
        remove_node_mask = torch.zeros((x.shape[0], 1)) + (1 if len(g) == 0 else 0)
        remove_edge_attr_mask = torch.zeros((len(g.edges), self.num_edge_attrs))
        if len(g):
            degrees = torch.tensor(list(g.degree))[:, 1]
            max_degrees = torch.tensor([len(self.frags_stems[g.nodes[n]["v"]]) for n in g.nodes])
        else:
            degrees = max_degrees = torch.zeros((0,))
        for i, n in enumerate(g.nodes):
            x[i, g.nodes[n]["v"]] = 1
            # The node must be connected to at most 1 other node and in the case where it is
            # connected to exactly one other node, the edge connecting them must not have any
            # attributes.
            edge_has_no_attr = bool(len(g.edges[list(g.edges(i))[0]]) == 0 if degrees[i] == 1 else degrees[i] == 0)
            remove_node_mask[i, 0] = degrees[i] <= 1 and edge_has_no_attr

        # We compute the attachment points of fragments that have been already used so that we can
        # mask them out for the agent (so that only one neighbor can be attached to one attachment
        # point at a time).
        attached = defaultdict(list)
        # If there are unspecified attachment points, we will disallow the agent from using the stop
        # action.
        has_unfilled_attach = False
        for i, e in enumerate(g.edges):
            ed = g.edges[e]
            a = ed.get(f"{int(e[0])}_attach", -1)
            b = ed.get(f"{int(e[1])}_attach", -1)
            if a >= 0:
                attached[e[0]].append(a)
                remove_edge_attr_mask[i, 0] = 1
            else:
                has_unfilled_attach = True
            if b >= 0:
                attached[e[1]].append(b)
                remove_edge_attr_mask[i, 1] = 1
            else:
                has_unfilled_attach = True
        # Here we encode the attached atoms in the edge features, as well as mask out attached
        # atoms.
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for j, n in enumerate(e):
                idx = ad.get(f"{int(n)}_attach", -1) + 1
                edge_attr[i * 2, idx + (self.num_stem_acts + 1) * j] = 1
                edge_attr[i * 2 + 1, idx + (self.num_stem_acts + 1) * (1 - j)] = 1
                if f"{int(n)}_attach" not in ad:
                    for attach_point in range(max_degrees[n]):
                        if attach_point not in attached[n]:
                            set_edge_attr_mask[i, attach_point + self.num_stem_acts * j] = 1
        edge_index = (
            torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).reshape((-1, 2)).T
        )
        if x.shape[0] == self.max_frags:
            add_node_mask = torch.zeros((x.shape[0], self.num_new_node_values))
        else:
            add_node_mask = (degrees < max_degrees).float()[:, None] if len(g.nodes) else torch.ones((1, 1))
            add_node_mask = add_node_mask * torch.ones((x.shape[0], self.num_new_node_values))
        stop_mask = torch.zeros((1, 1)) if has_unfilled_attach or not len(g) else torch.ones((1, 1))

        return gd.Data(
            x,
            edge_index,
            edge_attr,
            stop_mask=stop_mask,
            add_node_mask=add_node_mask,
            set_edge_attr_mask=set_edge_attr_mask,
            remove_node_mask=remove_node_mask,
            remove_edge_attr_mask=remove_edge_attr_mask,
        )

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
        return gd.Batch.from_data_list(graphs, follow_batch=["edge_index"])

    def mol_to_graph(self, mol):
        """Convert an RDMol to a Graph"""
        assert type(mol) is Chem.Mol
        all_matches = {}
        for fragidx, frag in self.sorted_frags:
            all_matches[fragidx] = mol.GetSubstructMatches(frag, uniquify=False)
        return _recursive_decompose(self, mol, all_matches, {}, [], [], 9)

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
        offsets = np.cumsum([0] + [self.frags_numatm[g.nodes[i]["v"]] for i in g])
        mol = None
        for i in g.nodes:
            if mol is None:
                mol = self.frags_mol[g.nodes[i]["v"]]
            else:
                mol = Chem.CombineMols(mol, self.frags_mol[g.nodes[i]["v"]])

        mol = Chem.EditableMol(mol)
        bond_atoms = []
        for a, b in g.edges:
            afrag = g.nodes[a]["v"]
            bfrag = g.nodes[b]["v"]
            u, v = (
                int(self.frags_stems[afrag][g.edges[(a, b)].get(f"{a}_attach", 0)] + offsets[a]),
                int(self.frags_stems[bfrag][g.edges[(a, b)].get(f"{b}_attach", 0)] + offsets[b]),
            )
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

    def object_to_log_repr(self, g: Graph):
        """Convert a Graph to a string representation"""
        return Chem.MolToSmiles(self.graph_to_mol(g))


def _recursive_decompose(ctx, m, all_matches, a2f, frags, bonds, max_depth=9, numiters=None):
    if numiters is None:
        numiters = [0]
    numiters[0] += 1
    if numiters[0] > 1_000:
        raise ValueError("too many iterations")
    if max_depth == 0 or len(a2f) == m.GetNumAtoms():
        # try to make a mol, does it work?
        # Did we match all the atoms?
        if len(a2f) < m.GetNumAtoms():
            return None
        # graph is a tree, e = n - 1
        if len(bonds) != len(frags) - 1:
            return None
        g = nx.Graph()
        g.add_nodes_from(range(len(frags)))
        g.add_edges_from([(i[0], i[1]) for i in bonds])
        assert nx.is_connected(g), "Somehow we got here but fragments dont connect?"
        for fi, f in enumerate(frags):
            g.nodes[fi]["v"] = f
        for a, b, stemidx_a, stemidx_b, _, _ in bonds:
            g.edges[(a, b)][f"{a}_attach"] = stemidx_a
            g.edges[(a, b)][f"{b}_attach"] = stemidx_b
        m2 = ctx.graph_to_mol(g)
        if m2.HasSubstructMatch(m) and m.HasSubstructMatch(m2):
            return g
        return None
    for fragidx, frag in ctx.sorted_frags:
        # Some fragments have symmetric versions, so we need all matches up to isomorphism!
        matches = all_matches[fragidx]
        for match in matches:
            if any(i in a2f for i in match):
                continue
            # Verify that atoms actually have the same charge
            if any(
                frag.GetAtomWithIdx(ai).GetFormalCharge() != m.GetAtomWithIdx(bi).GetFormalCharge()
                for ai, bi in enumerate(match)
            ):
                continue
            new_frag_idx = len(frags)
            new_frags = frags + [fragidx]
            new_a2f = {**a2f, **{i: (fi, new_frag_idx) for fi, i in enumerate(match)}}
            possible_bonds = []
            is_valid_match = True
            # Is every atom that has a bond outside of this fragment also a stem atom?
            for fi, i in enumerate(match):
                for j in m.GetAtomWithIdx(i).GetNeighbors():
                    j = j.GetIdx()
                    if j in match:
                        continue
                    # There should only be single bonds between fragments
                    if m.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE:
                        is_valid_match = False
                        break
                    # At this point, we know (i, j) is a single bond that goes outside the fragment
                    # so we check if the fragment we chose has that atom as a stem atom
                    if fi not in ctx.frags_stems[fragidx]:
                        is_valid_match = False
                        break
                if not is_valid_match:
                    break
            if not is_valid_match:
                continue
            for this_frag_stemidx, i in enumerate([match[s] for s in ctx.frags_stems[fragidx]]):
                for j in m.GetAtomWithIdx(i).GetNeighbors():
                    j = j.GetIdx()
                    if j in match:
                        continue
                    if m.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE:
                        continue
                    # Make sure the neighbor is part of an already identified fragment
                    if j in a2f and a2f[j] != new_frag_idx:
                        other_frag_atomidx, other_frag_idx = a2f[j]
                        try:
                            # Make sure that fragment has that atom as a stem atom
                            other_frag_stemidx = ctx.frags_stems[frags[other_frag_idx]].index(other_frag_atomidx)
                        except ValueError as e:
                            continue
                        # Make sure that that fragment's stem atom isn't already used
                        for b in bonds + possible_bonds:
                            if b[0] == other_frag_idx and b[2] == other_frag_stemidx:
                                break
                            if b[1] == other_frag_idx and b[3] == other_frag_stemidx:
                                break
                            if b[0] == new_frag_idx and b[2] == this_frag_stemidx:
                                break
                            if b[1] == new_frag_idx and b[3] == this_frag_stemidx:
                                break
                        else:
                            possible_bonds.append(
                                (other_frag_idx, new_frag_idx, other_frag_stemidx, this_frag_stemidx, i, j)
                            )
            new_bonds = bonds + possible_bonds
            dec = _recursive_decompose(ctx, m, all_matches, new_a2f, new_frags, new_bonds, max_depth - 1, numiters)
            if dec:
                return dec
