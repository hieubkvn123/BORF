from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment

import copy
import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, borf, borf2
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

largest_cc = LargestConnectedComponents()
cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
chameleon = WikipediaNetwork(root="data", name="chameleon")
squirrel = WikipediaNetwork(root="data", name="squirrel")
actor = Actor(root="data")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
pubmed = Planetoid(root="data", name="pubmed")
ogbn_arxiv = PygNodePropPredDataset(name='ogbn-arxiv', root='data')
datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "squirrel": squirrel, "actor": actor, "cora": cora, "citeseer": citeseer, "pubmed": pubmed, 'ogbn-arxiv' : ogbn_arxiv}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.2,
    "num_layers": 5,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 10,
    "eval_every": 1,
    "rewiring": "fosr",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 100,
    "dataset": None,
    "borf_batch_add" : 4,
    "borf_batch_remove" : 2,
    "sdrf_remove_edges" : True
})


results = []
args = default_args
args += get_args_from_input()

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

key = 'texas'
borf_iters = 3
borf_ba = 30
borf_br = 10

accuracies = []
print(f"TESTING: {key} ({args.rewiring})")
dataset = datasets[key]

# BORF
print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
dataset.data.edge_index, dataset.data.edge_type = borf.borf_optimized(dataset.data, 
        loops=borf_iters,
        remove_edges=False, 
        is_undirected=True,
        batch_add=borf_ba,
        batch_remove=borf_br,
        dataset_name=key,
        graph_index=0)