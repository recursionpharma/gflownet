import torch
from gflownet._C import Graph, GraphDef, mol_graph_to_Data
import networkx as nx
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from rdkit import Chem


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
    print(g.nodes[4]["v"], "should be 21")

    g.add_edge(0, 4, z="foo")
    print(g)
    print(g.nodes)
    print(list(g))
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
    print(g.edges)
    print(g.edges[0, 4])
    print(g.edges[4, 0]["z"])
    assert g.edges[0, 4]["z"] == "foo"
    g.edges[0, 4]["z"] = "bar"
    assert g.edges[0, 4]["z"] == "bar"
    g.add_node(1, v=21)
    g.add_edge(1, 4)
    assert "z" not in g.edges[1, 4]
    g.edges[4, 1]["z"] = "bar"
    assert "z" in g.edges[1, 4]
    assert g.edges[1, 4]["z"] == "bar"
    print("Doing bridges now")
    print(b)
    print(g.bridges())

    gnx = nx.generators.random_graphs.connected_watts_strogatz_graph(60, 3, 0.2, seed=42)
    # gnx = nx.generators.random_graphs.gnp_random_graph(60, 0.08, seed=43)
    assert nx.is_connected(gnx)
    g = Graph(GraphDef({"v": [0]}, {}))
    for i in gnx.nodes:
        g.add_node(i)
    for i, j in gnx.edges:
        g.add_edge(i, j)
    assert set([tuple(sorted(i)) for i in nx.bridges(gnx)]) == set([tuple(sorted(i)) for i in g.bridges()])
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


if __name__ == "__main__":
    test_Graph2()
