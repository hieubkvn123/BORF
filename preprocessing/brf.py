# SDRF preprocessing, from https://github.com/jctops/understanding-oversquashing

import torch
import numpy as np
import networkx as nx
from numba import jit, prange
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from GraphRicciCurvature.OllivierRicci import OllivierRicci

class CurvaturePlainGraph():
    def __init__(self, V, E):
        self.V = V 
        self.E = E
        self.adjacency_matrix = np.full((V,V),np.inf)
        for index in range(V):
            self.adjacency_matrix[index, index] = 0
        for index, edge in enumerate(E):
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1
        # Floyd Warshall
        self.dist = self.adjacency_matrix.copy()
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][k] + self.dist[k][j])
    
    def __str__(self):
        return f'The graph contains {self.V} nodes and {len(self.E)} edges {self.E}. '
    
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.E)
        nx.draw_networkx(G)
        plt.show()

    def _transport_plan_uv(self, u, v, method = 'OTD'):
        u_neighbors = [p for p in range(self.V) if self.adjacency_matrix[u][p] == 1]
        v_neighbors = [q for q in range(self.V) if self.adjacency_matrix[v][q] == 1]
        u_deg = len(u_neighbors)
        v_deg = len(v_neighbors)
        # Instead of using fractions [1/n,...,1/n], [1/m,...,1/m], we use [m,...,m], [n,...,n] and then divides by mn
        mu = np.full(u_deg, v_deg)
        mv = np.full(v_deg, u_deg)
        sub_indices = np.ix_(u_neighbors, v_neighbors)
        dist_matrix = self.dist[sub_indices]
        if method == 'OTD':
            optimal_plan = ot.emd(mu, mv, dist_matrix)
        elif method == 'Sinkhorn':
            optimal_plan = ot.sinkhorn(x, y, d, 1e-1, method='sinkhorn')
        else:
            raise NotImplemented
        optimal_plan = optimal_plan/(u_deg*v_deg)
        optimal_cost = np.sum(optimal_plan*dist_matrix)
        return optimal_cost, optimal_plan

    def curvature_uv(self, u, v, method = 'OTD'):
        optimal_cost, _ = self._transport_plan_uv(u, v, method)
        return 1 - optimal_cost/self.dist[u,v]

    def edge_curvatures(self, method = 'OTD'):
        edge_curvature_dict = {}
        for edge in self.E:
            edge_curvature_dict[edge] = self.curvature_uv(edge[0], edge[1], method)
        return edge_curvature_dict
        
    def all_curvatures(self, method = 'OTD'):
        C = np.zeros((self.V, self.V))
        for u in range(self.V):
            for v in range(u+1, self.V):
                C[u,v] = self.curvature_uv(u,v,method)
        C = C + np.transpose(C) + np.eye(self.V)
        C = np.hstack((np.reshape([str(u) for u in np.arange(self.V)],(self.V,1)), C))
        head = ['C'] + [str(u) for u in range(self.V)]
        print(tabulate(C, floatfmt=".2f", headers=head, tablefmt="presto"))

def balanced_forman_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = np.zeros((len(i_neighbors), len(j_neighbors)))

    _balanced_forman_post_delta(
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D


def bfr(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    curvature='bfc'
):
    # Get necessary data information
    N = data.x.shape[0]
    A = np.zeros(shape=(N, N))
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type
    if is_undirected:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = A[j, i] = 1.0
    else:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = 1.0
    N = A.shape[0]

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    C = np.zeros((N, N))

    # Rewiring begins
    for _ in range(loops):
        # Compute ORC
        can_add = True
        ollivier_ricci_curvature(A, C=C)
        
        # Get the neighbors of an edge x-y with
        # minimum curvature (most negative).
        ix_min = C.argmin()
        x = ix_min // N
        y = ix_min % N
        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        
        # Get all node pairs from two nodes' neighborhood
        # that are not connected
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        # If there are candidates for edge removal
        if len(candidates):
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)]
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            edge_type = np.append(edge_type, 1)
            edge_type = np.append(edge_type, 1)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound and G.has_edge(x, y):
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break
    return from_networkx(G).edge_index, torch.tensor(edge_type)
