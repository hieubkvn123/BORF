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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, borf, borf2
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

# Configure matplotlib
import matplotlib
font = {'size'   : 35}
matplotlib.rc('font', **font)


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
    name = args.dataset
    datasets = {name: datasets[name]}

key = 'chameleon'
sdrf_iters = 50
sdrf_rm_edges = True
fosr_iters = 50
borf_iters = 3
borf_ba = 20
borf_br = 20

print(f"TESTING: {key} ({args.rewiring})")
dataset = datasets[key]
original = copy.deepcopy(dataset)

# SDRF
curvature_type = "bfc"
ds_sdrf = copy.deepcopy(original)
ds_sdrf.data.edge_index, ds_sdrf.data.edge_type = sdrf.sdrf(ds_sdrf.data, loops=sdrf_iters, remove_edges=sdrf_rm_edges, 
        is_undirected=True, curvature=curvature_type)

# FoSR
ds_fosr = copy.deepcopy(original)
edge_index, edge_type, _ = fosr.edge_rewire(ds_fosr.data.edge_index.numpy(), num_iterations=fosr_iters)
ds_fosr.data.edge_index = torch.tensor(edge_index)
ds_fosr.data.edge_type = torch.tensor(edge_type)

# BORF
ds_borf = copy.deepcopy(original)
ds_borf.data.edge_index, ds_borf.data.edge_type = borf.borf_optimized(ds_borf.data, 
        loops=borf_iters,
        remove_edges=False, 
        is_undirected=True,
        batch_add=borf_ba,
        batch_remove=borf_br,
        dataset_name=key,
        graph_index=0)

# Convert to networkx
G_original = to_networkx(original.data)
G_sdrf = to_networkx(ds_sdrf.data)
G_fosr = to_networkx(ds_fosr.data)
G_borf = to_networkx(ds_borf.data)

# Compute degree distribution
dg_original = np.array([np.log2(G_original.degree[x]+0.001) for x in G_original.nodes])
dg_sdrf = np.array([np.log2(G_sdrf.degree[x]+0.001) for x in G_sdrf.nodes])
dg_fosr = np.array([np.log2(G_fosr.degree[x]+0.001) for x in G_fosr.nodes])
dg_borf = np.array([np.log2(G_borf.degree[x]+0.001) for x in G_borf.nodes])

# Get the kde
print(dg_original[dg_original == np.inf])
kde_original = stats.gaussian_kde(dg_original[~np.isnan(dg_original)])
kde_sdrf = stats.gaussian_kde(dg_sdrf[~np.isnan(dg_sdrf)])
kde_fosr = stats.gaussian_kde(dg_fosr[~np.isnan(dg_fosr)])
kde_borf = stats.gaussian_kde(dg_borf[~np.isnan(dg_borf)])

# Get the W1 distance
x = np.linspace(0, 10)
W1_sdrf = stats.wasserstein_distance(kde_original(x), kde_sdrf(x))
W1_fosr = stats.wasserstein_distance(kde_original(x), kde_fosr(x))
W1_borf = stats.wasserstein_distance(kde_original(x), kde_borf(x))

# Report W1
print(f'W1(original, SDRF) = {W1_sdrf}')
print(f'W1(original, FoSR) = {W1_fosr}')
print(f'W1(original, BORF) = {W1_borf}')

# Plot
fig, ax = plt.subplots(figsize=(15, 8))
sns.kdeplot(data=dg_original, label='Original', linewidth=2)
sns.kdeplot(data=dg_fosr, label='FoSR', linewidth=2)
sns.kdeplot(data=dg_sdrf, label='SDRF', linewidth=2)
sns.kdeplot(data=dg_borf, label='BORF', linewidth=2)

ax.set_ylabel("")
plt.legend()
plt.savefig(f'BORF_vs_SDRF_vs_FoSR_degrees_{key}.png')

