import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from preprocessing.brf import CurvaturePlainGraph

g = nx.Graph()
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(3, 1)
g.add_edge(1, 4)
g.add_edge(3, 4)
g.add_edge(4, 5)
g.add_edge(5, 1)
g = g.to_directed()

cpg = CurvaturePlainGraph(g)
C, PI = cpg.edge_curvatures(return_transport_cost=True)

orc = OllivierRicci(g, alpha=0)
orc.compute_ricci_curvature()

for src, target in C.keys():
    print(C[(src, target)], orc.G[src][target]['ricciCurvature']['rc_curvature'])

