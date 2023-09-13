import copy
import torch
from gflownet._C import Graph, GraphDef, mol_graph_to_Data
import networkx as nx
from gflownet.envs.mol_building_env import MolBuildingEnvContext, Graph as PyGraph
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphAction, GraphActionType
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, ChiralType


def test_Graph():
    g = nx.Graph()
    g.add_node(0, v=20, q="a")
    g.add_node(4, v=21)
    g.add_node(1, v=21)
    g.add_edge(0, 4, z="foo")
    g.add_edge(1, 4)
    b = list(nx.bridges(g))

    gdef = GraphDef({"v": [20, 21, 22], "q": ["a", "b"]}, {"z": ["foo", "bar"]})
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
    assert g.nodes[4]["q"] == "b"
    assert g.edges[0, 4]["z"] == "foo"
    g.edges[0, 4]["z"] = "bar"
    assert g.edges[0, 4]["z"] == "bar"
    g.add_node(1, v=21)
    g.add_edge(1, 4)
    assert "z" not in g.edges[1, 4]
    g.edges[4, 1]["z"] = "bar"
    assert "z" in g.edges[1, 4]
    assert g.edges[1, 4]["z"] == "bar"

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
        ctx._graph_cls = lambda: Graph(moldef)
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
        ctx._graph_cls = lambda: Graph(moldef)
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
    ctx._graph_cls = PyGraph
    import gflownet.envs.mol_building_env as mbe

    mbe.C_Graph_available = False
    mol = PyGraph()
    data = ctx.graph_to_Data(mol)
    print(data)
    print(data.x, data.edge_index, data.edge_attr, data.add_node_mask, data.set_node_attr_mask, data.remove_node_mask)


def test_Graph_step():
    env = GraphBuildingEnv()
    ctx = MolBuildingEnvContext(atoms=["C", "N", "O", "F"], max_nodes=10)
    env.graph_def = ctx.graph_def
    g = Graph(env.graph_def)  # env.new()
    g.add_node(0, v="C")
    print(g.nodes[0])
    print(list(g.nodes))
    g = env.new()
    g = env.step(g, GraphAction(GraphActionType.AddNode, source=0, value="C"))
    for i in g.nodes:
        print(i, g.nodes[i])
    print(list(g.nodes), list(g.edges))
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
    print(n, e)
    g = gp
    mol = ctx.graph_to_mol(g)
    data = ctx.graph_to_Data(g)
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    test_Graph_step()
