import ot
import time
import torch
import numpy as np
import multiprocessing as mp
import networkx as nx
from numba import jit, prange
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from torch_geometric.datasets import TUDataset

class CurvaturePlainGraph():
    def __init__(self, V, E, device=None):
        self.V = V
        self.E = E
        self.adjacency_matrix = np.full((V,V),np.inf)

        if(device is None):
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
          self.device = device
        
        for index in range(V):
            self.adjacency_matrix[index, index] = 0
        for index, edge in enumerate(E):
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1
        
        # Floyd Warshall
        self.dist = self._floyd_warshall()

    def __str__(self):
        return f'The graph contains {self.V} nodes and {len(self.E)} edges {self.E}. '

    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.E)
        nx.draw_networkx(G)
        plt.show()

    def _floyd_warshall(self):
        self.dist = self.adjacency_matrix.copy()
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][k] + self.dist[k][j])
        return self.dist

    def _to_tensor(self, x):
        x = torch.Tensor(x).to(self.device)
        return x

    def _to_numpy(self, x):
        if(torch.cuda.is_available()):
            return x.cpu().detach().numpy()
        return x.detach().numpy()

    def _transport_plan_uv(self, u, v, method = 'OTD', u_neighbors=None, v_neighbors=None):
        u_neighbors = [p for p in range(self.V) if self.adjacency_matrix[u][p] == 1] if u_neighbors is None else u_neighbors
        v_neighbors = [q for q in range(self.V) if self.adjacency_matrix[v][q] == 1] if v_neighbors is None else v_neighbors
        u_deg = len(u_neighbors)
        v_deg = len(v_neighbors)

        # Instead of using fractions [1/n,...,1/n], [1/m,...,1/m], we use [m,...,m], [n,...,n] and then divides by mn
        mu = self._to_tensor(np.full(u_deg, v_deg))
        mv = self._to_tensor(np.full(v_deg, u_deg))
        sub_indices = np.ix_(u_neighbors, v_neighbors)
        dist_matrix = self._to_tensor(self.dist[sub_indices])
        dist_matrix[dist_matrix == np.inf] = 0 # Correct the dist matrix
        if method == 'OTD':
            optimal_plan = self._to_numpy(ot.emd(mu, mv, dist_matrix))
        elif method == 'Sinkhorn':
            optimal_plan = ot.sinkhorn(x, y, d, 1e-1, method='sinkhorn')
        else:
            raise NotImplemented
        optimal_plan = optimal_plan/(u_deg*v_deg)
        optimal_cost = np.sum(optimal_plan*self._to_numpy(dist_matrix))
        return optimal_cost, optimal_plan

    def add_edge(self, i, j):
        # TODO : Need to replace with a more efficient algorithm
        self.adjacency_matrix[i, j] = 1
        self.adjacency_matrix[j, i] = 1
        
        # self.dist = self._floyd_warshall()
        self.dist[i, j] = 1
        self.dist[j, i] = 1

    def remove_edge(self, i, j):
        self.adjacency_matrix[i, j] = 0
        self.adjacency_matrix[j, i] = 0
        # self.dist = self._floyd_warshall()
        self.dist[i, j] = np.inf
        self.dist[j, i] = np.inf

    def curvature_uv(self, u, v, method = 'OTD', u_neighbors=None, v_neighbors=None):
        optimal_cost, _ = self._transport_plan_uv(u, v, method, u_neighbors=u_neighbors, v_neighbors=v_neighbors)
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

def _softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()

def _preprocess_data(data, is_undirected=False):
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

    return G, N, A, m, C, edge_type

def _get_neighbors(x, G, is_undirected=False, is_source=False):
    if is_undirected:
        x_neighbors = list(G.neighbors(x)) + [x]
    else:
        if(is_source):
          x_neighbors = list(G.successors(x)) + [x]
        else:
          x_neighbors = list(G.predecessors(x)) + [x]
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

  # This step should be removed by something more efficient
  # start = time.time()
  new_curvature = graph.curvature_uv(x, y, u_neighbors=x_neighbors, v_neighbors=y_neighbors)
  # end = time.time()
  # print('Time taken to re-calculate curvature : ', end-start)
  improvement = new_curvature - old_curvature
  graph.remove_edge(k, l)

  return new_curvature, old_curvature

def bfr(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    num_add_per_iter=4,
    num_rmv_per_iter=2,
    device=None
):
    # Preprocess data
    G, N, A, m, C, edge_type = _preprocess_data(data)
    graph = CurvaturePlainGraph(N, list(G.edges), device=device)
    C = graph.edge_curvatures(method='OTD')

    # Rewiring begins
    for _ in range(loops):
        # Compute ORC
        can_add = True
        C = graph.edge_curvatures(method='OTD')

        # Get the neighbors of an edge x-y with
        # minimum curvature (most negative).
        x, y = min(C, key=C.get)
        x_neighbors = _get_neighbors(x, G, is_undirected=is_undirected, is_source=True)
        y_neighbors = _get_neighbors(y, G, is_undirected=is_undirected, is_source=False)

        # Get all node pairs from two nodes' neighborhood
        # that are not connected
        candidates = _get_rewire_candidates(G, x_neighbors, y_neighbors)
        print('Number of candidates : ', len(candidates))
        candidates = candidates[:100]

        # If there are candidates for edge removal
        if len(candidates):
            improvements = []
            for (i, j) in candidates:
                new_curvature, old_curvature = _calculate_improvement(graph, C, x, y, x_neighbors, y_neighbors, i, j)
                improvement = new_curvature - old_curvature
                improvements.append(improvement)

            candidate_idx = np.random.choice(range(len(candidates)), size=(num_add_per_iter,),
                                             p = _softmax(np.array(improvements), tau=tau))

            for c in candidate_idx:
                k, l = candidates[c]
                print(f'Adding edge ({k} -> {l}), improvement = {improvements[c]}')

                # Add edge to both networkx and meta graphs
                G.add_edge(k, l)
                graph.add_edge(k, l)
                edge_type = np.append(edge_type, 1)
                edge_type = np.append(edge_type, 1)

                # Update adjacency
                if is_undirected:
                    A[k, l] = A[l, k] = 1
                else:
                    A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break
    return from_networkx(G).edge_index, torch.tensor(edge_type)

if __name__ == '__main__':
    ### Benchmark our performance vs. their performance ###
    # Calculate curvature
    G = nx.karate_club_graph()
    graph = CurvaturePlainGraph(len(G.nodes), list(G.edges), device=torch.device('cpu'))
    start = time.time()
    C = graph.edge_curvatures(method = 'OTD')
    end = time.time()
    print(f'Time taken (Ours) : {end - start}')

    ### Check the add_edge function ###
    # Get min curvature
    u, v = min(C, key=C.get)
    min_curvature = C[(u, v)]
    print(f'Minimum curvature ({u} -> {v}) : {min_curvature}')

    # Add edge
    graph.add_edge(30, 31)
    c_uv = graph.curvature_uv(u, v)
    print('Curvature after adding edge : ', c_uv)

    # Remove edge
    graph.remove_edge(30, 31)
    c_uv = graph.curvature_uv(u, v)
    print('Curvature after removing edge : ', c_uv)

    ### Check the calculate_improvement method ###
    # Calculate curvature improvements
    G = nx.karate_club_graph()
    graph = CurvaturePlainGraph(len(G.nodes), list(G.edges), device=torch.device('cuda'))
    C = graph.edge_curvatures(method = 'OTD')
    x, y = 0 , 31
    k, l = 30, 31
    neighbors_x = _get_neighbors(x, G, is_undirected=True)
    neighbors_y = _get_neighbors(y, G, is_undirected=True)

    new_curvature, old_curvature = _calculate_improvement(graph, C, x, y, neighbors_x, neighbors_y, k, l)
    improvement = new_curvature - old_curvature
    print(f'Curvature improvement of ({x} -> {y}) after adding ({k} -> {l}) is {improvement}')
    print(f'Curvature of ({x} -> {y}) after calling _curvature_improvement : {graph.curvature_uv(x, y)}')

    ### Test rewiring ###
    dataset = list(TUDataset(root="data", name="REDDIT-BINARY"))
    for graph in dataset:
        n = graph.num_nodes
        graph.x = torch.ones((n,1))

    start = time.time()
    bfr(dataset[0], device=torch.device('cpu'))
    end = time.time()

    print(f'Time taken for rewiring = {end - start}')
