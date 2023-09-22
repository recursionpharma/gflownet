import copy
import torch
from gflownet._C import Graph, GraphDef, mol_graph_to_Data
import networkx as nx
from gflownet.envs.mol_building_env import MolBuildingEnvContext, Graph as PyGraph
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphAction, GraphActionType
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, ChiralType
import pickle


def test_Graph():
    g = nx.Graph()
    g.add_node(0, v=20, q="a")
    g.add_node(4, v=21)
    g.add_node(1, v=21)
    g.add_edge(0, 4, z="foo")
    g.add_edge(1, 4)
    b = list(nx.bridges(g))

    gdef = GraphDef({"v": [20, 21, 22], "q": ["a", "b"]}, {"z": ["foo", "bar", "qwe"]})
    g = Graph(gdef)
    g.add_node(0, v=20, q="a")
    g.add_node(4, v=21)
    assert g.nodes[4]["v"] == 21

    g.add_edge(0, 4, z="foo")
    assert 0 in g.nodes
    assert 1 not in g.nodes
    assert 4 in g.nodes
    assert "v" in g.nodes[0]
    assert "v" in g.nodes[4]
    assert "q" in g.nodes[0]
    assert "q" not in g.nodes[4]
    g.nodes[0]["v"] = 22
    g.nodes[4]["q"] = "b"
    assert g.nodes[0]["q"] == "a"
    del g.nodes[0]["q"]
    assert "q" not in g.nodes[0]
    assert g.nodes[0].get("q", 12345) == 12345
    assert g.nodes[4]["q"] == "b"
    assert g.edges[0, 4]["z"] == "foo"
    g.edges[0, 4]["z"] = "bar"
    assert g.edges[0, 4]["z"] == "bar"
    del g.edges[0, 4]["z"]
    assert "z" not in g.edges[0, 4]
    assert g.edges[0, 4].get("z", 12345) == 12345

    g.add_node(1, v=21)
    g.add_edge(1, 4)
    assert "z" not in g.edges[1, 4]
    g.edges[4, 1]["z"] = "bar"
    assert "z" in g.edges[1, 4]
    assert g.edges[1, 4]["z"] == "bar"

    g = Graph(gdef)
    g.add_node(10, v=20, q="a")
    g.add_node(11, v=21)
    g.add_node(12, v=22)
    g.add_edge(10, 11, z="foo")
    g.add_edge(11, 12, z="bar")
    g._inspect()
    g.remove_node(10)
    assert 10 not in g.nodes
    assert 11 in g.nodes
    assert 12 in g.nodes
    # print(list(g.edges), list(g.nodes))
    g._inspect()

    g = Graph(gdef)
    g.add_node(10, v=20, q="a")
    g.add_node(11, v=21)
    g.add_node(12, v=22)
    g.add_node(13, v=20)
    g.add_edge(10, 11)
    g.add_edge(11, 12, z="bar")
    g.add_edge(11, 13, z="qwe")
    g._inspect()
    g.remove_edge(10, 11)
    assert 10 in g.nodes
    assert 11 in g.nodes
    assert 12 in g.nodes
    assert 13 in g.nodes
    g._inspect()

    gnx = nx.generators.random_graphs.connected_watts_strogatz_graph(60, 3, 0.2, seed=42)
    assert nx.is_connected(gnx)
    g = Graph(GraphDef({"v": [0]}, {}))
    for i in gnx.nodes:
        g.add_node(i)
    for i, j in gnx.edges:
        g.add_edge(i, j)
    assert set([tuple(sorted(i)) for i in nx.bridges(gnx)]) == set([tuple(sorted(i)) for i in g.bridges()])
    assert sorted(gnx.edges) == sorted(g.edges)


def timing_tests():
    # print("bridges", list(nx.bridges(gnx)))
    # print("bridges", g.bridges())
    # print(list(gnx.edges))
    import time

    """
    t0 = time.time()
    for i in range(1000):
        list(nx.bridges(gnx))
    t1 = time.time()
    print("nx", t1 - t0)
    t0 = time.time()
    for i in range(1000):
        g.bridges()
    t1 = time.time()
    print("g", t1 - t0)"""
    for smi in ["CCOC1NC=CC=1", "C1C=CNC1NC1NCC=C1CC1CC1CN" * 4]:
        print(smi)
        ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F"], max_nodes=10)
        mol_o = ctx.mol_to_graph(Chem.MolFromSmiles(smi))
        data_o = ctx.graph_to_Data(mol_o)
        t0 = time.time()
        for i in range(1000):
            ctx.graph_to_Data(mol_o)
        t1 = time.time()
        print("nx + numpy", t1 - t0)

        ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F"], max_nodes=10)
        moldef = GraphDef(ctx.atom_attr_values, ctx.bond_attr_values)
        ctx.graph_cls = lambda: Graph(moldef)
        mol = ctx.mol_to_graph(Chem.MolFromSmiles(smi))
        print(mol.bridges())
        data = mol_graph_to_Data(mol, ctx, torch)
        for i in data:
            pass
            # print(torch.cat([data[i], data_o[i]], 0 if "index" in i else 1))
        t0 = time.time()
        for i in range(10000):
            mol_graph_to_Data(mol, ctx, torch)
        t1 = time.time()
        print("C", t1 - t0)


import time


def test_Graph2():
    for smi in ["CCOC1NC=CC=1", "C1C=CNC1NC1NCC=C1CC1CC1CN" * 4]:
        ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F"], max_nodes=10)
        moldef = GraphDef(ctx.atom_attr_values, ctx.bond_attr_values)
        ctx.graph_cls = lambda: Graph(moldef)
        mol = ctx.mol_to_graph(Chem.MolFromSmiles(smi))
        t0 = time.time()
        for i in range(100000):
            mol_graph_to_Data(mol, ctx, torch)
        t1 = time.time()
        print("C", t1 - t0)


def test_Graph3():
    ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F"], max_nodes=10)
    moldef = GraphDef(ctx.atom_attr_values, ctx.bond_attr_values)
    # ctx._graph_cls = lambda: Graph(moldef)
    mol = Graph(moldef)
    data = ctx.graph_to_Data(mol)
    print(data)
    print(data.x, data.edge_index, data.edge_attr, data.add_node_mask, data.set_node_attr_mask, data.remove_node_mask)

    ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F"], max_nodes=10)
    ctx.graph_cls = PyGraph
    import gflownet.envs.mol_building_env as mbe

    mbe.C_Graph_available = False
    mol = PyGraph()
    data = ctx.graph_to_Data(mol)
    print(data)
    print(data.x, data.edge_index, data.edge_attr, data.add_node_mask, data.set_node_attr_mask, data.remove_node_mask)


def test_Graph_step():
    env = GraphBuildingEnv()
    ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F", "Cl"], max_nodes=10)
    C_graph_cls = ctx.graph_cls
    ctx.graph_cls = PyGraph
    g0 = env.new()
    g0.add_node(0, v="C")
    g0.add_node(1, v="C")
    g0.add_node(2, v="N")
    g0.add_node(3, v="O")
    g0.add_edge(0, 1)
    g0.add_edge(1, 2, type=BondType.DOUBLE)
    g0.add_edge(0, 3)
    data0 = ctx.graph_to_Data(g0)

    ctx.graph_cls = C_graph_cls
    env.graph_cls = ctx.graph_cls
    g = Graph(ctx.graph_def)  # env.new()
    g.add_node(0, v="C")
    print(g.nodes[0])
    print(list(g.nodes))
    g = env.new()
    g = env.step(g, GraphAction(GraphActionType.AddNode, source=0, value="C"))
    for i in g.nodes:
        print(i, g.nodes[i])
    print(list(g.nodes), list(g.edges))

    datas = []
    for i in range(2):
        if i == 0:
            ctx.graph_cls = PyGraph
            env.graph_cls = PyGraph
        if i == 1:
            ctx.graph_cls = C_graph_cls
            env.graph_cls = C_graph_cls
        g = env.new().copy()
        g = env.step(g, GraphAction(GraphActionType.AddNode, source=0, value="C"))
        print(list(g.nodes), list(g.edges))
        gp = copy.deepcopy(g)

        gp.add_node(1, v="C")
        gp.add_node(2, v="N")
        gp.add_node(3, v="O")
        gp.add_node(4, v="F")
        gp.add_edge(0, 1)
        gp.add_edge(1, 2, type=BondType.DOUBLE)
        gp.add_edge(0, 3)
        gp.add_edge(3, 4)
        gp.remove_edge(3, 4)
        gp.remove_node(4)
        n, e = list(gp.nodes), list(gp.edges)
        g = gp
        mol = ctx.graph_to_mol(g)
        _data = ctx.graph_to_Data(g)
        datas.append(_data)
    data = _data.as_torch()
    assert data.set_edge_attr_mask[0].sum() == 1

    odata = datas[0]
    edge_reindexing = []
    for e in data.edge_index.T:
        for i, ep in enumerate(odata.edge_index.T):
            if e[0] == ep[0] and e[1] == ep[1]:
                edge_reindexing.append(i)
    edge_reindexing = torch.tensor(edge_reindexing)
    for k in odata.keys:
        fixed = getattr(odata, k)
        if k == "non_edge_index":
            # let's just check they're all there
            assert set(map(tuple, fixed.T.tolist())) == set(map(tuple, data.non_edge_index.T.tolist()))
            continue
        if "edge" in k:
            if "index" in k:
                fixed = fixed[:, edge_reindexing]
            elif k == "edge_attr":
                fixed = fixed[edge_reindexing]
            else:
                fixed = fixed[edge_reindexing[::2] // 2]
        assert (getattr(data, k) == fixed).all()

    masks = [getattr(data, a.mask_name) for a in ctx.action_type_order]
    asd = [
        GraphActionType.Stop,
        GraphActionType.AddNode,
        GraphActionType.AddEdge,
        GraphActionType.RemoveEdge,
        GraphActionType.RemoveNode,
        GraphActionType.SetEdgeAttr,
        GraphActionType.SetNodeAttr,
        GraphActionType.RemoveNodeAttr,
        GraphActionType.RemoveEdgeAttr,
    ]

    for t, m in enumerate(masks):
        nz = m.nonzero()
        for r, c in nz:
            ga = ctx.aidx_to_GraphAction(_data, (t, r.item(), c.item()))
            new_aidx = ctx.GraphAction_to_aidx(_data, ga)
            assert new_aidx == (t, r.item(), c.item())


def test_Graph_nitrogen():
    ctx = MolBuildingEnvContext(
        ["C", "N", "O", "S", "F", "Cl", "Br"], charges=[0], chiral_types=[ChiralType.CHI_UNSPECIFIED], max_nodes=10
    )
    pdef = pickle.dumps(ctx.graph_def)
    ctx.graph_def = pickle.loads(pdef)
    g = Graph(ctx.graph_def)
    g.add_node(0, v="N")
    g.add_node(1, v="C")
    g.add_node(2, v="F")
    g.add_node(3, v="Cl")
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    mol = ctx.graph_to_mol(g)
    data = ctx.graph_to_Data(g).as_torch()
    assert data.set_edge_attr_mask[0].sum() == 0


def test_Graph_collate():
    env = GraphBuildingEnv()
    ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F", "Cl"], max_nodes=10)
    batches = []
    o_cls = ctx.graph_cls
    for i in range(2):
        if i == 0:
            ctx.graph_cls = PyGraph
            env.graph_cls = PyGraph
        if i == 1:
            ctx.graph_cls = o_cls
            env.graph_cls = o_cls
        g0 = env.new()
        g0.add_node(0, v="C")
        g0.add_node(1, v="C")
        g0.add_node(2, v="N")
        g0.add_node(3, v="O")
        g0.add_edge(0, 1)
        g0.add_edge(1, 2, type=BondType.DOUBLE)
        g0.add_edge(0, 3)
        data0 = ctx.graph_to_Data(g0)
        g1 = g0.copy()
        g1.add_node(4, v="F")
        g1.add_edge(0, 4)
        data1 = ctx.graph_to_Data(g1)
        batch = ctx.collate([data0, data1])
        batches.append(batch)

    for k in batches[0].keys:
        a = getattr(batches[0], k)
        if k not in batches[1].keys:
            continue
        b = getattr(batches[1], k)

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    from gflownet.envs.mol_building_env import MolBuildingEnvContext
    from gflownet.envs.graph_building_env import Graph as PyGraph

    test_Graph()
    test_Graph_step()
    test_Graph_nitrogen()
    test_Graph_collate()
