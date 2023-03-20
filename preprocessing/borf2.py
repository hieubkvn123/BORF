import os
import ot
import copy
import time
import glob
import torch
import pickle
import pathlib
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
import networkx as nx
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from torch_geometric.datasets import TUDataset
from preprocessing.orc import BORFOllivierRicciCurvature
warnings.filterwarnings("ignore")

def _softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()

def _preprocess_data(data, is_undirected=False):
    # Get necessary data information
    N = data.x.shape[0]
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()

    return G, N, edge_type

def _get_neighbors(x, G, is_undirected=False, is_source=False):
    if is_undirected:
        x_neighbors = list(G.neighbors(x)) #+ [x]
    else:
        if(is_source):
          x_neighbors = list(G.successors(x)) #+ [x]
        else:
          x_neighbors = list(G.predecessors(x)) #+ [x]
    return x_neighbors

def _get_rewire_candidates(G, x_neighbors, y_neighbors):
    candidates = []
    for i in x_neighbors:
        for j in y_neighbors:
            if (i != j) and (not G.has_edge(i, j)):
                candidates.append((i, j))
    return candidates

def _calculate_improvement(graph, C, x, y, x_neighbors, y_neighbors, k, l):
    """
    Calculate the curvature performance of x -> y when k -> l is added.
    """
    graph.add_edge(k, l)
    old_curvature = C[(x, y)]

    new_curvature, _ = graph.curvature_uv(x, y, u_neighbors=x_neighbors, v_neighbors=y_neighbors)
    improvement = new_curvature - old_curvature
    graph.remove_edge(k, l)

    return new_curvature, old_curvature

### Optimized code for rebuttal ###
def save_graph(G1, G2, graph_idx, fname, iters, added, removed):
    with open(fname, 'a') as f:
        # Put number of iterations
        if(os.path.getsize(fname) <= 0):
            f.write(f'num_iters={iters}\n')

        # Put graph ID
        f.write(f'{graph_idx}\n')

        # Put added edges
        f.write('\tadded\n')
        for (p, q) in added:
            if(not G1.has_edge(p, q)):
                f.write(f'\t\t{p} {q}\n')

        # Put removed edges
        f.write('\tremoved\n')
        for (u, v) in removed:
            if(G1.has_edge(u, v)):
                f.write(f'\t\t{u} {v}\n')

        # Put final graph edges
        f.write('\tedges\n')
        for (k, l) in G2.edges:
            f.write(f'\t\t{k} {l}\n')

def load_saved_graph(fname):
    curr_index = None
    curr_mode = None
    latest_iters = 0
    result = {}
    with open(fname, 'r') as f:
        lines = f.readlines()
        latest_iters = int(lines[0].split('=')[1])
        for line in lines[1:]:
            if(not line.startswith('\t') and not line.startswith('\t\t')):
                curr_index = int(line.strip())
                result[curr_index] = {'added':[], 'removed':[], 'edges':[]}
            elif(line.startswith('\t') and not line.startswith('\t\t')):
                curr_mode = line.strip()
            elif(line.startswith('\t\t')):
                src, dst = line.strip().split(' ')
                result[curr_index][curr_mode].append((int(src), int(dst)))

    return result, latest_iters

def load_latest_checkpoint(G, graph_idx, fname):
    checkpoint, latest_iters = load_saved_graph(fname)
    if(graph_idx not in checkpoint):
        return G, 0, [], []

    checkpoint = checkpoint[graph_idx]
    original_edges = copy.deepcopy(G.edges)
    
    for (p, q) in checkpoint['edges']:
        if(not G.has_edge(p, q)):
            G.add_edge(p, q)

    for (u, v) in original_edges:
        if((int(u), int(v)) not in checkpoint['edges']):
            G.remove_edge(u, v)

    return G, latest_iters, checkpoint['added'], checkpoint['removed']

def borf_optimized(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs_new',
    curvature_dir='curvatures_new',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    curvature_dirname = f'{curvature_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    pathlib.Path(curvature_dirname).mkdir(parents=True, exist_ok=True)

    # Create graph checkpoint file
    fname = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}.txt')
    curvature_fname = os.path.join(curvature_dirname, f'{graph_index}.pkl')
    latest_iters = 0
    added = []
    removed = []

    # Preprocess data
    G, N, edge_type = _preprocess_data(data)
    original_G = copy.deepcopy(G)
    
    # Load the latest checkpoint
    checkpoints = [x for x in glob.glob(os.path.join(dirname, f'iters_*_add_{batch_add}_remove_{batch_remove}.txt')) if x <= fname]
    checkpoints = sorted(checkpoints)
    
    if(len(checkpoints) >= 1):
        latest_checkpoint = checkpoints[-1]
        if(str(graph_index) in [x.strip() for x in open(latest_checkpoint, 'r').readlines()]):
            latest_checkpoint = checkpoints[-1]
        else:
            if(len(checkpoints) >= 2):
                latest_checkpoint = checkpoints[-2]
        G, latest_iters, added, removed = load_latest_checkpoint(G, graph_index, latest_checkpoint)

    # Rewiring begins
    for i in range(latest_iters, loops):
        # Compute ORC
        if(i == 0 and os.path.exists(curvature_fname)): # Always save the first curvature
            orc = pickle.load(open(curvature_fname, 'rb'))
        else:
            orc = BORFOllivierRicciCurvature(G, device=device, chunk_size=1) 
            if(i == 0):
                pickle.dump(orc, open(curvature_fname, 'wb'))
    
        # Compute curvatures + transport plans
        _C, _PI = orc.edge_curvatures(method='OTD')
        _C = sorted(_C, key=_C.get)

        # Collect garbage
        gc.collect()
        torch.cuda.empty_cach()

        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]

        # Add edges
        for (u, v) in most_neg_edges:
            pi = _PI[(u, v)] 
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)
                added.append((p, q))

        # Remove edges
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)
                removed.append((u, v))

    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    
    # Save rewired graph
    save_graph(original_G, G, graph_index, fname, loops, set(added), set(removed))

    return edge_index, edge_type
