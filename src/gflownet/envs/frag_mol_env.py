from collections import defaultdict
from math import log
from typing import List, Tuple

import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import torch
import torch_geometric.data as gd
from scipy import special

from gflownet.envs.graph_building_env import ActionIndex, Graph, GraphAction, GraphActionType, GraphBuildingEnvContext
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
        # (named 'src_attach') and vice versa for the second half for v (named 'dst_attach').
        # Note to self: this choice results in a special case in generate_forward_trajectory for these
        # edge attributes. See PR#83 for details.
        # Note to self: PR#XXX solves this issue by using src_attach/dst_attach as edge attributes
        self.num_edge_attr_logits = most_stems * 2
        # There are thus up to 2 edge attributes, the stem of u and the stem of v.
        self.num_edge_attrs = 2
        # The + 1 is for an extra dimension to indicate when the attribute isn't yet set
        self.num_edge_dim = (most_stems + 1) * 2
        self.num_cond_dim = num_cond_dim
        self.edges_are_duplicated = True
        self.edges_are_unordered = False
        self.fail_on_missing_attr = True

        # Order in which models have to output logits
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.SetEdgeAttr]
        self.bck_action_type_order = [
            GraphActionType.RemoveNode,
            GraphActionType.RemoveEdgeAttr,
        ]
        self.n_counter = NCounter()
        self.sorted_frags = sorted(list(enumerate(self.frags_mol)), key=lambda x: -x[1].GetNumAtoms())

    def ActionIndex_to_GraphAction(self, g: gd.Data, aidx: ActionIndex, fwd: bool = True):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction

        Parameters
        ----------
        g: gd.Data
            The graph object on which this action would be applied.
        aidx: ActionIndex
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.

        Returns
        action: GraphAction
            A graph action whose type is one of Stop, AddNode, or SetEdgeAttr.
        """
        if fwd:
            t = self.action_type_order[aidx.action_type]
        else:
            t = self.bck_action_type_order[aidx.action_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=aidx.row_idx, value=aidx.col_idx)
        elif t is GraphActionType.SetEdgeAttr:
            a, b = g.edge_index[
                :, aidx.row_idx * 2
            ]  # Edges are duplicated to get undirected GNN, deduplicated for logits
            if aidx.col_idx < self.num_stem_acts:
                attr = "src_attach"
                val = aidx.col_idx
            else:
                attr = "dst_attach"
                val = aidx.col_idx - self.num_stem_acts
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr, value=val)
        elif t is GraphActionType.RemoveNode:
            return GraphAction(t, source=aidx.row_idx)
        elif t is GraphActionType.RemoveEdgeAttr:
            a, b = g.edge_index[:, aidx.row_idx * 2]
            attr = "src_attach" if aidx.col_idx == 0 else "dst_attach"
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr)

    def GraphAction_to_ActionIndex(self, g: gd.Data, action: GraphAction) -> ActionIndex:
        """Translate a GraphAction to an ActionIndex

        Parameters
        ----------
        g: gd.Data
            The graph object on which this action would be applied.
        action: GraphAction
            A graph action whose type is one of Stop, AddNode, or SetEdgeAttr.

        Returns
        -------
        action_idx: ActionIndex
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.
        """
        # Find the index of the action type, privileging the forward actions
        for u in [self.action_type_order, self.bck_action_type_order]:
            if action.action in u:
                type_idx = u.index(action.action)
                break
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
            row = row.div(2, rounding_mode="floor")  # type: ignore
            if action.attr == "src_attach":
                col = action.value
            else:
                col = action.value + self.num_stem_acts
        elif action.action is GraphActionType.RemoveNode:
            row = action.source
            col = 0
        elif action.action is GraphActionType.RemoveEdgeAttr:
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            row = row.div(2, rounding_mode="floor")  # type: ignore
            if action.attr == "src_attach":
                col = 0
            else:
                col = 1
        return ActionIndex(action_type=type_idx, row_idx=int(row), col_idx=int(col))

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
        if hasattr(g, "_Data_cache") and g._Data_cache is not None:
            return g._Data_cache
        zeros = lambda x: np.zeros(x, dtype=np.float32)  # noqa: E731
        ones = lambda x: np.ones(x, dtype=np.float32)  # noqa: E731
        x = zeros((max(1, len(g.nodes)), self.num_node_dim))
        x[0, -1] = len(g.nodes) == 0
        edge_attr = zeros((len(g.edges) * 2, self.num_edge_dim))
        set_edge_attr_mask = zeros((len(g.edges), self.num_edge_attr_logits))
        # TODO: This is a bit silly but we have to do +1 when the graph is empty because the default
        # padding action is a [0, 0, 0], which needs to be legal for the empty state. Should be
        # fixable with a bit of smarts & refactoring.
        remove_node_mask = ones((x.shape[0], 1)) if len(g) == 0 else zeros((x.shape[0], 1))
        remove_edge_attr_mask = zeros((len(g.edges), self.num_edge_attrs))
        if len(g):
            degrees = np.array(list(g.degree), dtype=np.int32)[:, 1]  # type: ignore
            max_degrees = np.array([len(self.frags_stems[g.nodes[n]["v"]]) for n in g.nodes])  # type: ignore
        else:
            degrees = max_degrees = np.zeros((0,), dtype=np.int32)

        node_is_connected = degrees <= 1
        node_is_connected_to_edge_with_attr = zeros(degrees.shape[0])
        for i, n in enumerate(g.nodes):
            x[i, g.nodes[n]["v"]] = 1

        # We compute the attachment points of fragments that have been already used so that we can
        # mask them out for the agent (so that only one neighbor can be attached to one attachment
        # point at a time).
        attached = defaultdict(list)
        # If there are unspecified attachment points, we will disallow the agent from using the stop
        # action.
        has_unfilled_attach = False
        for i, e in enumerate(g.edges):
            ed = g.edges[e]
            if len(ed):
                node_is_connected_to_edge_with_attr[e[0]] = 1
                node_is_connected_to_edge_with_attr[e[1]] = 1
            a = ed.get("src_attach", -1)
            b = ed.get("dst_attach", -1)
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

        # The node must be connected to at most 1 other node and in the case where it is
        # connected to exactly one other node, the edge connecting them must not have any
        # attributes.
        if len(g):
            remove_node_mask = (node_is_connected * (1 - node_is_connected_to_edge_with_attr)).reshape((-1, 1))

        # Here we encode the attached atoms in the edge features, as well as mask out attached
        # atoms.
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for j, n in enumerate(e):
                attach_name = ["src_attach", "dst_attach"][j]
                idx = ad.get(attach_name, -1) + 1
                edge_attr[i * 2, idx + (self.num_stem_acts + 1) * j] = 1
                edge_attr[i * 2 + 1, idx + (self.num_stem_acts + 1) * (1 - j)] = 1
                if attach_name not in ad:
                    for attach_point in range(max_degrees[n]):
                        if attach_point not in attached[n]:
                            set_edge_attr_mask[i, attach_point + self.num_stem_acts * j] = 1
        # Since this is a DiGraph, make sure to put (i, j) first and (j, i) second
        edge_index = np.array([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=np.int64).reshape((-1, 2)).T

        if x.shape[0] == self.max_frags:
            add_node_mask = zeros((x.shape[0], self.num_new_node_values))
        else:
            add_node_mask = (
                np.array(degrees < max_degrees, dtype=np.float32)[:, None]
                if len(g.nodes)
                else np.ones((1, 1), np.float32)
            )
            add_node_mask = add_node_mask * np.ones((x.shape[0], self.num_new_node_values), np.float32)
        stop_mask = zeros((1, 1)) if has_unfilled_attach or not len(g) else ones((1, 1))

        data = gd.Data(
            **{
                k: torch.from_numpy(v)
                for k, v in dict(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    stop_mask=stop_mask,
                    add_node_mask=add_node_mask,
                    set_edge_attr_mask=set_edge_attr_mask,
                    remove_node_mask=remove_node_mask,
                    remove_edge_attr_mask=remove_edge_attr_mask,
                ).items()
            }
        )
        g._Data_cache = data
        return data

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

    def obj_to_graph(self, mol):
        """Convert an RDMol to a Graph"""
        assert type(mol) is Chem.Mol
        all_matches = {}
        for fragidx, frag in self.sorted_frags:
            all_matches[fragidx] = mol.GetSubstructMatches(frag, uniquify=False)
        return _recursive_decompose(self, mol, all_matches, {}, [], [], self.max_frags)

    def graph_to_obj(self, g: Graph) -> Chem.Mol:
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
            if self.fail_on_missing_attr:
                assert "src_attach" in g.edges[(a, b)] and "dst_attach" in g.edges[(a, b)]
            u, v = (
                int(self.frags_stems[afrag][g.edges[(a, b)].get("src_attach", 0)] + offsets[a]),
                int(self.frags_stems[bfrag][g.edges[(a, b)].get("dst_attach", 0)] + offsets[b]),
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
            mol = self.graph_to_obj(g)
            assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None
        except Exception:
            return False
        if mol is None:
            return False
        return True

    def object_to_log_repr(self, g: Graph):
        """Convert a Graph to a string representation"""
        return Chem.MolToSmiles(self.graph_to_obj(g))

    def has_n(self) -> bool:
        return True

    def log_n(self, g: Graph) -> int:
        return self.n_counter(g)


class NCounter:
    """
    Dynamic program to calculate the number of trajectories to a state.
    See Appendix D of "Maximum entropy GFlowNets with soft Q-learning"
    by Mohammadpour et al 2024 (https://arxiv.org/abs/2312.14331) for a proof.
    """

    def __init__(self):
        # Hold the log factorial
        self.cache = [0.0, 0.0]

    def lfac(self, arg: int):
        while arg >= len(self.cache):
            self.cache.append(log(len(self.cache)) + self.cache[-1])
        return self.cache[arg]

    def lcomb(self, x, y):
        # log c(x, y) = log (x! / (y! (x - y)!))
        assert x >= y
        return self.lfac(x) - self.lfac(y) - self.lfac(x - y)

    @staticmethod
    def root_tree(og: nx.Graph, x):
        g = nx.DiGraph(nx.create_empty_copy(og))
        visited = np.zeros(len(g), bool)
        visited[x] = True
        q = [x]
        while len(q) > 0:  # print(i, x)
            x = q.pop()
            for i in nx.neighbors(og, x):
                if not visited[i]:
                    visited[i] = True
                    g.add_edge(x, i, **(og.get_edge_data(x, i) | og.get_edge_data(i, x)))
                    q.append(i)

        return g

    def f(self, g, x):
        elem = np.full((len(g),), -1, int)
        ways = np.full((len(g),), -1, float)

        def _f(x):
            if elem[x] < 0:
                e, w = 0, 0
                for i in nx.neighbors(g, x):
                    e1, w1 = _f(i)
                    # edge feature
                    f = len(g.get_edge_data(x, i))
                    for i in range(f):
                        w1 += np.log(e1 + i)
                    e1 += f

                    w = w + w1 + self.lcomb(e + e1, e)
                    e = e + e1

                elem[x] = e + 1
                ways[x] = w
            return elem[x], ways[x]

        return _f(x)[1]

    def __call__(self, g):
        if len(g) == 0:
            return 0

        acc = []
        for i in nx.nodes(g):
            rg = self.root_tree(g, i)
            x = self.f(rg, i)
            acc.append(x)

        return special.logsumexp(acc)


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
        g = Graph()
        g.add_nodes_from(range(len(frags)))
        g.add_edges_from([(i[0], i[1]) for i in bonds])
        # assert nx.is_connected(g), "Somehow we got here but fragments dont connect?"
        for fi, f in enumerate(frags):
            g.nodes[fi]["v"] = f
        for a, b, stemidx_a, stemidx_b, _, _ in bonds:
            g.edges[(a, b)]["src_attach"] = stemidx_a  # TODO: verify src/dst is correct?
            g.edges[(a, b)]["dst_attach"] = stemidx_b
        m2 = ctx.graph_to_obj(g)
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
                        except ValueError:
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
