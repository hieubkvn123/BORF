import ot
import torch
import threading
import numpy as np
import pandas as pd
import networkx as nx

class BORFOllivierRicciCurvature():
    def __init__(self, G, device=None, is_undirected=False, chunk_size=2):
        self.G = G
        self.V = len(G.nodes)
        self.E = list(G.edges)
        self.chunk_size = chunk_size
        self.adjacency_matrix = np.full((self.V,self.V),np.inf)

        if(device is None):
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
          self.device = torch.device(device)
        
        for index in range(self.V):
            self.adjacency_matrix[index, index] = 0
        for index, edge in enumerate(self.E):
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1

    def __str__(self):
        return f'The graph contains {self.V} nodes and {len(self.E)} edges {self.E}. '

    def _to_tensor(self, x):
        x = torch.Tensor(x).to(self.device)
        return x

    def _to_numpy(self, x):
        if(torch.cuda.is_available()):
            return x.cpu().detach().numpy()
        return x.detach().numpy()

    def _dist_matrix_uv(self, u, v, u_neighbors, v_neighbors):
        A = self.adjacency_matrix.copy()
        A[A == np.inf] = 0.0
        dist = np.zeros(shape=(len(u_neighbors), len(v_neighbors)))
        for i, _u in enumerate(u_neighbors):
            for j, _v in enumerate(v_neighbors):
                # If same node -> distance is zero
                if(_u == _v):
                    dist[i, j] = 0
                    continue
                
                # If different nodes
                # 1. If edge exists -> distance is one
                if(self.G.has_edge(_u, _v)):
                    dist[i, j] = 1
                    continue
                else:
                    # 2. If edge does not exists
                    # 2.1. If have adjacent to same node -> distance is two
                    # 2.2. If no common node -> distance is three
                    if(np.sum(A[_u, :] * A[_v, :]) > 0):
                        dist[i, j] = 2
                    else:
                        dist[i, j] = 3      
        return dist.astype(np.float16)

    def _transport_plan_uv(self, u, v, method = 'OTD'):
        # Extract N_u, N_v, |N_u|, |N_v|
        u_neighbors = [p for p in range(self.V) if self.adjacency_matrix[u][p] == 1]
        v_neighbors = [q for q in range(self.V) if self.adjacency_matrix[v][q] == 1]
        u_deg = len(u_neighbors)
        v_deg = len(v_neighbors)

        # Instead of using fractions [1/n,...,1/n], [1/m,...,1/m], we use [m,...,m], [n,...,n] and then divides by mn
        mu = self._to_tensor(np.full(u_deg, v_deg))
        mv = self._to_tensor(np.full(v_deg, u_deg))
        dist_matrix = self._to_tensor(self._dist_matrix_uv(u, v, u_neighbors, v_neighbors))
        
        # Update distance matrix
        if method == 'OTD':
            optimal_plan = self._to_numpy(ot.emd(mu, mv, dist_matrix))
        else:
            raise NotImplemented

        # Find pi, pi * d, sum(pi * d)
        optimal_plan = optimal_plan / (u_deg * v_deg) # PI
        optimal_cost = optimal_plan * self._to_numpy(dist_matrix) # optimal_plan = $\pi(p, q)$, optimal_cost = $\pi(p, q) * d(p, q)$
        optimal_total_cost = np.sum(optimal_cost)
        optimal_cost = pd.DataFrame(optimal_cost, columns=v_neighbors, index=u_neighbors)

        # Returns sum(pi * d) and pi * d matrix for (u, v)
        return optimal_total_cost, optimal_cost

    def curvature_uv(self, u, v, method = 'OTD'):
        optimal_total_cost, optimal_cost = self._transport_plan_uv(u, v, method)
        return 1 - optimal_total_cost, optimal_cost

    def curvature_uv_wrapper(self, curvature_dict, transport_plan_dict, u, v, method):
        curvature_dict[(u, v)], transport_plan_dict[(u, v)] = self.curvature_uv(u, v, method)

    def _edge_curvatures_multi_threaded(self, method='OTD'):
        # Curvatures + transport plans
        edge_curvature_dict = {}
        transport_plan_dict = {}

        for chunk_id in range(0, len(self.E), self.chunk_size): 
            processes = []
            chunk = self.E[chunk_id:chunk_id + self.chunk_size]

            # Run curvature computation in chunks
            for (u, v) in chunk:
                t = threading.Thread(target=lambda : self.curvature_uv_wrapper(edge_curvature_dict, transport_plan_dict, u, v, method))
                t.start()
                processes.append(t)
                
            # Wait for current chunk to finish
            for process in processes:
                process.join()
              
        return edge_curvature_dict, transport_plan_dict

    def _edge_curvatures_single_threaded(self, method='OTD'):
        # Curvatures + transport plans
        edge_curvature_dict = {}
        transport_plan_dict = {}
              
        for edge in self.E:
            edge_curvature_dict[edge], transport_plan_dict[edge] = self.curvature_uv(edge[0], edge[1], method)
        return edge_curvature_dict, transport_plan_dict

    def edge_curvatures(self, method = 'OTD'):
        if(self.chunk_size > 1):
            edge_curvature_dict, transport_plan_dict = self._edge_curvatures_multi_threaded(method=method)
        else:
            edge_curvature_dict, transport_plan_dict = self._edge_curvatures_single_threaded(method=method)
        return edge_curvature_dict, transport_plan_dict
