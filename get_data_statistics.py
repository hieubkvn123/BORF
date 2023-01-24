"""
Calculate: 
number of graphs,
number of nodes (range - average),
number of edges (range - average),
number of features,
number of classes,
whether the graph is directed
"""
import numpy as np
from torch_geometric.utils import is_undirected
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset

mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
datasets = {"mutag" : mutag, "enzymes" : enzymes, "imdb": imdb, "proteins": proteins}

for key in datasets:
    dataset = datasets[key]
    num_graphs = len(dataset)

    nodes = []
    edges = []
    classes = []
    for i in range(num_graphs):
        G = to_networkx(dataset[i])
        nodes.append(len(G.nodes))
        edges.append(len(G.edges))
        classes.append(dataset[i].y.item())

    nodes = np.array(nodes)
    edges = np.array(edges)
    directed = not is_undirected(dataset[0].edge_index)

    print(key.upper())
    print(f'Number of graphs : {num_graphs}')
    print(f'Number of nodes : {nodes.min()} - {nodes.max()}')
    print(f'Number of edges : {edges.min()} - {edges.max()}')
    print(f'Avg number of nodes : {nodes.mean()}')
    print(f'Avg number of edges : {edges.mean()}')
    print(f'Number of classes : {len(np.unique(classes))}')
    print(f'Is directed : {directed}')
