import os
import random
import sys
import networkx as nx
from scipy.stats import entropy
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import deque
import random
from networkx.algorithms import approximation
from shapely.geometry import LineString
import random
from collections import deque, defaultdict
import math

def get_ideal(G, V_, u):

    if V_[u]:
        return list()

    V = V_.copy()
    s = [u];  V[u] = 1
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v]:
                s.append(v)
                V[v] = 1; Q.append(v)

    return s

def deactivate_robust(G, V_, node):
    N = len(V_)
    V = {u:0 for u in G.nodes()}
    V[root] = 1; Q = deque([root]);
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                V[v] = 1
                Q.append(v)

    return {u:0 if V_[u] + V[u] == 0 else 1 for u in G.nodes()}

def get_size_precalc_robust(G, W, V_, node):

    N = len(V_)

    if V_[node]:
        return 0

    if W[node] <= 200:
        return 1

    V = {u:0 for u in G.nodes()}

    s = 1; V[node] = 1
    Q = deque([]); Q.append(root)
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                s += 1; V[v] = 1
                Q.append(v)

    return sum([1 for i in G.nodes() if (V_[i] == 0 and V[i] == 0)])

def get_ideal_robust(G, V_, root, node):

    if root == node:
        return get_ideal(G, V_, root)

    V = {u:0 for u in G.nodes()}
    V[root] = 1
    Q = deque([root])
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                V[v] = 1
                Q.append(v)

    return [i for i in G.nodes() if (V_[i] == 0 and V[i] == 0)]

def visit_robust(G, V, root, u):

    I = get_ideal_robust(G, V, root, u)

    V_ = V.copy()
    for u in I:
        V_[u] = 1

    return V_, len(I)

def get_size_robust(G, V_, root, node):

    N = len(V_)

    if V_[node]:
        return 0

    if node == root:
        return N - sum(V_)

    V = {u:0 for u in G.nodes()}

    V[root] = 1; Q = deque([root])
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                V[v] = 1
                Q.append(v)

    return sum([1 for u in G.nodes() if V_[u] + V[u] == 0])

def get_size_weight(G, W, V_, u):

    if V_[u]:
        return 0, 0

    V = V_.copy()

    s = 1; w = W[u]; V[u] = 1
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v]:
                s += 1; w += W[v]; V[v] = 1
                Q.append(v)

    return s, w

def get_size_weight_robust(G, W, V_, root, node):
    N = len(G.nodes())

    if V_[node]:
        return 0 , 0

    if node == root:
        return N - sum(V_), sum(W) - sum([W[i] for i in G.nodes() if V_[i]])

    V = {u:0 for u in G.nodes()}

    V[root] = 1; Q = deque([root])
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v] and v != node:
                V[v] = 1
                Q.append(v)

    I = [i for i in G.nodes() if (V_[i] == 0 and V[i] == 0)]
    return sum([1 for i in I]), sum([W[i] for i in I])

def add_visit_robust(G, V, root, node):
    V[node] = 1
    I = get_ideal_robust(G, V, root, node)
    return {i:1 if (i in I or V[i]) else 0 for i in G.nodes() }

def find_order(G, V, sample, s):
    ans = []; Q = deque([]); Q.append(s)
    while Q:
        u = Q.popleft()
        if u in sample:
            ans.append(u)
        for v in G.predecessors(u):
            if not V[v]:
                Q.append(v)
    return ans

def find_paths(G, root, V):
    path_counts = {n: 0 for n in G.nodes()}
    total_paths = {n: 0 for n in G.nodes()}
    root_node_paths = {n:0 for n in G.nodes()}
    list_paths = []

    # Recursive function to generate the paths
    def dfs(node, path):
        # Add the current node to the path
        total_paths[node] += 1
        root_node_paths[node] +=1

        path.append(node)
        list_paths.append(path)
        for v in path:
            path_counts[v] += 1

        # Base case: if the node has no outgoing edges, it is a leaf node
        # so we have found a complete path
        if G.in_degree(node) == 0:
            return

        # Recursive case: visit all the neighbors of the current node
        for neighbor in G.predecessors(node):
            if not V[neighbor]:
                dfs(neighbor, path.copy()) # use path[:] to create a copy of the path

    # Start the recursive function at the root node
    dfs(root, [])

    paths = sum(total_paths.values())
    path_counts = {node: max(count/root_node_paths[node], paths-count) if root_node_paths[node] else 1e6 for node, count in path_counts.items()}
    return path_counts, list_paths

def find_paths(G, root, V):
    path_counts = {n: 0 for n in G.nodes()}
    total_paths = {n: 0 for n in G.nodes()}
    root_node_paths = {n:0 for n in G.nodes()}
    list_paths = []
    stack = [(root, [])]

    while stack:
        node, path = stack.pop()
        # Add the current node to the path
        total_paths[node] += 1
        root_node_paths[node] +=1

        path.append(node)
        list_paths.append(path)
        for v in path:
            path_counts[v] += 1

        # Base case: if the node has no outgoing edges, it is a leaf node
        # so we have found a complete path
        if G.in_degree(node) == 0:
            continue

        # Recursive case: visit all the neighbors of the current node
        for neighbor in G.predecessors(node):
            if not V[neighbor]:
                stack.append((neighbor, path.copy())) # use path[:] to create a copy of the path

    paths = sum(total_paths.values())
    path_counts = {node: max(count/root_node_paths[node], paths-count) if root_node_paths[node] else 1e6 for node, count in path_counts.items()}
    return path_counts, list_paths

def entropy2(labels, paths_to_root):
  value,counts = np.unique(labels, return_counts=True)
  updated_counts = [c/paths_to_root[v] for v,c in zip(value, counts)]
  return entropy(updated_counts, base=2)

def weight_(end_nodes, paths_to_root):
    w = 0
    for node in end_nodes:
        w += 1/paths_to_root[node]
    return w

def best_node(path_matrix, paths_to_root):
    entropies = dict()
    for node,_ in enumerate(path_matrix):
        Ideal = path_matrix[:, path_matrix[node,:] >= 1]
        Ideal_end_nodes =  np.argwhere((Ideal == 2))[:, 0].flatten()
        not_robust = path_matrix[:, path_matrix[node,:] == 0]
        not_robust_end_nodes = np.argwhere((not_robust == 2))[:, 0].flatten()
        entropy_ = (entropy2(Ideal_end_nodes, paths_to_root)*weight_(Ideal_end_nodes, paths_to_root), entropy2(not_robust_end_nodes, paths_to_root)*weight_(not_robust_end_nodes, paths_to_root))
        #entropy_ = entropy2(Ideal_end_nodes, paths_to_root)*weight_(Ideal_end_nodes, paths_to_root) + entropy2(not_robust_end_nodes, paths_to_root)*weight_(not_robust_end_nodes, paths_to_root)
        entropy_ = (entropy2(Ideal_end_nodes, paths_to_root), entropy2(not_robust_end_nodes, paths_to_root))
        entropies[node] = entropy_
    #print(entropies)
    return min(entropies, key=lambda x: ( max(entropies[x][0], entropies[x][1]), min(entropies[x][0], entropies[x][1]) ) )

def count_paths(G, root, V):
    path_counts = {n: 0 for n in G.nodes()}
    total_paths = {n: 0 for n in G.nodes()}
    root_node_paths = {n:0 for n in G.nodes()}

    # Recursive function to generate the paths
    def dfs(node, path):
        # Add the current node to the path
        total_paths[node] += 1
        root_node_paths[node] +=1

        path.append(node)
        for v in path:
            path_counts[v] += 1

        # Base case: if the node has no outgoing edges, it is a leaf node
        # so we have found a complete path
        if G.in_degree(node) == 0:
            return

        # Recursive case: visit all the neighbors of the current node
        for neighbor in G.predecessors(node):
            if not V[neighbor]:
                dfs(neighbor, path.copy()) # use path[:] to create a copy of the path

    # Start the recursive function at the root node
    dfs(root, [])

    paths = sum(total_paths.values())
    path_counts = {node: max(count/root_node_paths[node], paths-count) if root_node_paths[node] else 1e6 for node, count in path_counts.items()}
    return path_counts

def min_value(d):
    return min(d, key=d.get)

def path_matrix(paths, nodes):
    # Create a dictionary with the row index as the key and the node as the value
    node_dict = {node: i for i, node in enumerate(nodes)}

    # Create an empty NumPy array with the same number of rows as the number of nodes
    matrix = np.empty((len(nodes), 0)).astype(bool)

    # Iterate over the paths
    for path in paths:
        # Create a row for the current path
        row = np.zeros(len(nodes))
        # Set the value to 1 for each node in the path
        row[np.array([node_dict[node] for node in path])] = 1
        row[node_dict[path[-1]]] = 2
        # Append the row to the matrix
        matrix = np.append(matrix, row[:, np.newaxis], axis=1)

    index_node = {i: node for node, i in node_dict.items()}
    return matrix, index_node

def root_graph(G, s):
    V = {u: 0 for u in G.nodes()};
    V[s] = 1
    Q = deque([]);
    Q.append(s)
    D = nx.DiGraph()
    while Q:
        #u = Q.popleft()
        u = Q.pop()
        D.add_node(u)
        for v in G.neighbors(u):
            D.add_node(v)
            if not V[v]:
                D.add_edge(v, u)
                Q.append(v)
                V[v] = 1
    return D

def select_separator(T):
    sep = None
    if len(T.nodes()) == 1:
        return list(T.nodes())[0]
    for node in T.nodes():
        # make a copy of the original graph
        T_ = T.copy()
        # remove the node from the copy
        T_.remove_node(node)
        # get the sizes of the connected components
        component_sizes = [len(component) for component in nx.connected_components(T_)]
        # check if the size of the largest component is within the desired range
        if max(component_sizes) <= len(T.nodes()) / 2:
            sep = node
            break
    if not sep:
        print('separator not found')
        return list(random.choice(T.nodes()))
    return list(sep)

def select_component(G, P, CV, s):
    G_removed_nodes = G.copy()
    G_removed_nodes.remove_nodes_from(P)
    components = [component for component in nx.connected_components(G_removed_nodes.to_undirected())]
    neighbors = [G.predecessors(u) for u in P]
    neighbors_set = set([item for sublist in neighbors for item in sublist])
    it = 0
    tested = []
    # Primero chequeamos que todos los hijos no estén en la misma componente
    for c in components:
        if neighbors_set.issubset(set(c)):
            for u in P:
                tested.append(u)
                it += 1
                if CV[u]:
                    return c, u, it

            # Si no hay muestras positivas retornamos componente que contiene a la raíz
            for comp in components:
                if s in comp:
                    return comp, s, it

    #En caso de que no apunten hacia la misma direccion también testearemos a los vecinos
    for u in P:
        it += 1
        tested.append(u)
        if CV[u]:
            for neigh in G.predecessors(u):
                if neigh not in tested:
                    it += 1
                    tested.append(neigh)
                    if CV[neigh]:
                        for c in components:
                            if neigh in c:
                                return c, neigh, it
            # Si los vecinos no testean positivo retornamos contagio
            return [u], u, it

    # Si no hay muestras positivas retornamos componente que contiene a la raíz
    for c in components:
        if s in c:
            return c, s, it

def numberofPaths(G, source, destination):
    dp = {u:0 for u in G.nodes()}
    dp[destination] = 1

    for i in reversed(list(nx.topological_sort(G))):
        for j in G.successors(i):
                dp[i] += dp[j]
    return dp[source] #if not V[source] else 10e2

def paths_hanging(G, source):
    dp = {u:1 for u in G.nodes() }
    #dp = {u:1 for u in G.nodes() }

    for i in list(nx.topological_sort(G)):
        for j in G.successors(i):
                dp[j] += dp[i]
    return dp[source] #if not V[source] else 10e2

def simulate_entropy_minimization(G, root):
    T = []
    path_counts, path_list = find_paths(G, root, {n:0 for n in G.nodes()})
    initial_matrix, index_node = path_matrix(path_list, G.nodes())
    value,counts = np.unique(np.argwhere((initial_matrix == 2))[:, 0], return_counts=True)
    paths_to_root = {v:c for v,c in zip(value, counts)}
    initial_p = best_node(initial_matrix, paths_to_root)
    map_prev = {}
    for cov in random.sample(list(G.nodes()), 100):
        #print(f'cov: {cov}')
        CV = {u:0 for u in G.nodes()}
        CV[cov] = 1
        Q = deque([]); Q.append(cov)
        while Q:
            u = Q.popleft()
            successors = list(G.successors(u))
            if successors:
                v = random.sample(successors, 1)[0]
                CV[v] = 1
                Q.append(v)


        V = {u:0 for u in G.nodes()}
        R = N; s = root
        W = {node: 1 for node in G.nodes()}
        matrix = initial_matrix
        for t in range(100):
            G_it = nx.induced_subgraph(G, [n for n in G.nodes() if not V[n]])
            if t ==0:
                p = initial_p

            else:
                nn = 0
                for i in G.nodes():
                    if V[i]:
                        nn += 2**i

                if nn in map_prev.keys():
                    p = map_prev[nn]
                else:
                    path_counts, path_list = find_paths(G_it, s, V)
                    matrix, index_node = path_matrix(path_list, G.nodes())
                    value,counts = np.unique(np.argwhere((matrix == 2))[:, 0], return_counts=True)
                    paths_to_root = {v:c for v,c in zip(value, counts)}; paths_to_root[s] = 1
                    p = best_node(matrix, paths_to_root)
                    map_prev[nn] = p

            P = [index_node[p]]
            #print(p)
            if s not in P:
                P.append(s)

            P = find_order(G, V, P, s)

            for u in P[::-1]:
                if CV[u]:
                    I = set(get_ideal(G, V, u))
                    V = {v: not (v in I) for v in G.nodes()}
                    s = u; break
                else:
                    V = add_visit_robust(G, V, s, u)

            size, weight = get_size_weight(G, W, V, s)

            #print(size, weight, p)


            if size == 1:
                T.append(t + 1)
                print(f"Result {cov}: {T[-1]}    acc: {sum(T) / len(T)}")
                break
            if t == 99:
                T.append(100);
                print(f"Result {cov}: {T[-1]}    acc: {sum(T) / len(T)}")
    return T

def simulate_path_separation(G, root):
    map_prev = {}
    V = {u:0 for u in G.nodes()}
    count = paths_hanging(G, root)
    path_counts = {u:max(paths_hanging(G, u),   count - paths_hanging(G, u)) for u in G.nodes()}
    initial_p = min_value(path_counts)
    T = []
    for cov in random.sample(list(G.nodes()), 100):
        CV = {u:0 for u in G.nodes()}
        CV[cov] = 1
        Q = deque([]); Q.append(cov)
        while Q:
            u = Q.popleft()
            successors = list(G.successors(u))
            if successors:
                v = random.sample(successors, 1)[0]
                CV[v] = 1
                Q.append(v)


        V = {u:0 for u in G.nodes()}
        R = N; s = root
        W = {node: 1 for node in G.nodes()}
        G_it = G
        for t in range(100):
            G_it = nx.induced_subgraph(G, [n for n in G_it.nodes() if not V[n]])
            if t ==0:
                p = initial_p
            else:
                nn = 0
                for i in G_it.nodes():
                    if not V[i]:
                        nn += 2**i

                if nn in map_prev.keys():
                    p = map_prev[nn]
                else:
                    count = paths_hanging(G_it, s)
                    path_counts = {u:max(paths_hanging(G_it, u),  count - paths_hanging(G_it, u)) for u in G_it.nodes()}
                    p = min_value(path_counts)
                    map_prev[nn] = p
            P = [p]
            #print(p)
            if s not in P:
                P.append(s)

            P = find_order(G_it, V, P, s)

            for u in P[::-1]:
                if CV[u]:
                    I = set(get_ideal(G_it, V, u))
                    V = {v: not (v in I) for v in G_it.nodes()}
                    s = u; break
                else:
                    V = add_visit_robust(G_it, V, s, u)

            size, weight = get_size_weight(G_it, W, V, s)


            if size == 1:
                T.append(t + 1)
                print(f"Result {cov}: {T[-1]}    acc: {sum(T) / len(T)}")
                break
            if t == 99:
                T.append(100);
                print(f"Result {cov}: {T[-1]}    acc: {sum(T) / len(T)}")
    return T

def simulate_treewidth_algorithm(G, root):
    N = len(G.nodes())
    map_prev = {}
    V = {u:0 for u in G.nodes()}
    _, tree_decomp = approximation.treewidth_min_fill_in(G.to_undirected())
    initial_p =  select_separator(tree_decomp)
    T = []
    for cov in random.sample(list(G.nodes()), 100):
        CV = {u:0 for u in G.nodes()}
        CV[cov] = 1
        Q = deque([]); Q.append(cov)
        while Q:
            u = Q.popleft()
            successors = list(G.successors(u))
            if successors:
                v = random.sample(successors, 1)[0]
                CV[v] = 1
                Q.append(v)

        V = {u:0 for u in G.nodes()}
        R = N; s = root
        W = {node: 1 for node in G.nodes()}
        G_it = G
        it = 0
        for t in range(100):
            G_it = nx.induced_subgraph(G, [n for n in G_it.nodes() if not V[n]])
            size = (len(G_it.nodes()))

            if t ==0:
                P_ = initial_p
            elif size > 2:
                nn = 0
                for i in G.nodes():
                    if V[i]:
                        nn += 2**i

                if nn in map_prev.keys():
                    P_ = map_prev[nn]
                else:
                    _, tree_decomp = approximation.treewidth_min_fill_in(G_it.to_undirected())
                    P_ =  select_separator(tree_decomp)
                    map_prev[nn] = P_


            P = find_order(G_it, {u:0 for u in G.nodes()}, P_, s)
            selected_component, s, extra_it = select_component(G_it, P[::-1], CV, s)
            I = get_ideal(G_it,V,s)
            V = {v: not (v in selected_component and v in I) for v in G.nodes()}

            it += extra_it

            for u in P[::-1]:
                # if u == s:
                #     break
                if not CV[u]:
                    #I = get_ideal_robust(G_it, V, s, u)
                    I = get_ideal_robust(G, V, s, u)
                    for v in I:
                        V[v] = 1

            #size, weight = get_size_weight(G_it, W, V, s)
            size = N - sum(V.values())

            if size <= 1:
                T.append(it+1)
                print(f"Result {cov}: {T[-1]}    acc: {sum(T) / len(T)}")
                break

            if size == 2:
                T.append(it + 2)
                print(f"Result {cov}: {T[-1]}    acc: {sum(T) / len(T)}")
                break
            # if t == 99:
            #     T.append(it)
            #     print(f"Result {cov}: {T[-1]}   acc: {sum(T) / len(T)}")
    return T

def greedyAppReduceAll(G, W, V_, root, K, slim, wlim, plim=0, calcRobust=True, visitRobust=False):

    V = V_.copy(); N = int(1e6); N_ = int(1e6)

    ans = []; tot = 0
    while len(ans) < K:

        E = []

        S = [0 for u in range(N_)]
        SS = [0 for u in range(N_)]
        for v in G.nodes():
            if not V[v]:
                ss = get_size(G, V, v)
                sr = get_size_robust(G, V, root, v)
                if calcRobust:
                    SS[v], S[v] = get_size_weight_robust(G, W, V, root, v)
                else:
                    SS[v], S[v] = get_size_weight(G, W, V, v)
                if (sr / ss) >= plim:
                    E.append(v)

        maxv = 0; u = -1
        for v in E:
            s = SS[v]
            if s > maxv and s <= slim and S[v] <= wlim:
                u = v
                maxv = s

        if u == -1:
            break

        x = None
        if visitRobust:
            V, x = visit_robust(G, V, root, u)
        else:
            V, x = visit(G, V, u)
        ans.append(u); tot += x

    return ans, tot

def simulate_robust_randtree(G, root, map_prev={}, k2=1, plim=0, calcRobust=True, visitRobust=True, verbose=False):
    iters = []
    N = len(G.nodes())
    N_ = int(1e6)
    node_it = 0
    W = [int(1e6) for i in range(N_) ]
    for r in random.sample(list(G.nodes()), 100):

        node_it += 1

        if verbose:
            print(r)

        CV = [0 for u in range(N_)]

        CV[r] = 1
        Q = deque([]); Q.append(r)
        while Q:
            u = Q.popleft()
            successors = list(G.successors(u))
            if successors:
                v = random.sample(successors, 1)[0]
                CV[v] = 1
                Q.append(v)

        V = {u:0 for u in G.nodes()}
        for u in G.nodes():
            V[u] = 0

        s = root
        for t in range(100):
            # G_it = nx.induced_subgraph(G, [n for n in G.nodes() if not V[n]])
            # components = nx.connected_components(G_it.to_undirected())
            # if len(list(components)) > 1:
            #     print('Error!!!',  N - sum(V.values()))
            #     break

            R = N - sum(V.values()); P = []; sP = 0

            if not verbose:
                print("                                                           ", end="\r")
                print(f"Search {r}: {t + 1}    now: {R}", end="\r")

            if verbose:
                print("it,", t, R)

            nn = 0
            for i in G.nodes():
                if V[i]:
                    nn += 2**i

            if nn in map_prev.keys():
                P = map_prev[nn]
            else:
                if R > N / 10:
                    P, sP = greedyAppReduceAll(G, W, V, root, k2, R / 3, 1e8, plim, calcRobust, visitRobust)
                elif R >= 10:
                    low = 0; high = R
                    while low != high:
                        mid = (low + high) // 2
                        P, sP = greedyAppReduceAll(G, W, V, root, k2, mid, 1e8, plim, calcRobust, visitRobust)
                        if R - sP < mid:
                            high = mid
                        else:
                            low = mid + 1
                    P, sP = greedyAppReduceAll(G, W, V, root, k2, low, 1e8, plim, calcRobust, visitRobust)

                    if verbose:
                        print("pre low:", low, R - sP)

                    if low > 1:
                        P_, sP_ = greedyAppReduceAll(G, W, V, root, k2, low - 1, 1e8, plim, calcRobust, visitRobust)

                        if verbose:
                            print("low:", low, R - sP, R - sP_)

                        if abs((R - sP_) - (low - 1)) < abs((R - sP) - low) or (len(P) == 1 and sP == R):
                            P = P_; sP = sP_
                else:
                    P, sP = greedyAppReduceAll(G, W, V, root, k2, 1, 1e8, plim, calcRobust, visitRobust)

                map_prev[nn] = P

            # if s not in P:
            #     P.append(s)

            if verbose:
                print("nx", len(P), sum([CV[u] for u in P]), N - sum(V))
                print("P: ", ' '.join([str(p) for p in P]))

            if sum([CV[u] for u in P]):
                V_ = {u: (-1 * V[u]) for u in G.nodes()}
                for u in P:
                    if CV[u]:
                        if u != s:
                            s = u
                        I = get_ideal(G, V, u)
                        for v in I:
                            V_[v] += 1

                V = {u: (V_[u] != sum([CV[u] for u in P])) for u in G.nodes()}

                if verbose:
                    print("if: ", N_ - sum(V))

            for u in P:
                if not CV[u]:
                    I = get_ideal_robust(G, V, root, u)
                    for v in I:
                        V[v] = 1


            if N - sum(V.values()) == R:

                    print("REPEATING")
                    POS = []
                    for i in G.nodes():
                        if not V[i]:
                            POS.append(i)
                    P = random.sample(POS, k2)
                    map_prev[nn] = P
                    for u in P:
                        if not CV[u]:
                            I = get_ideal_robust(G, V, s, u)
                            for v in I:
                                V[v] = 1
                        if CV[u]:
                            s = u
                            I = get_ideal(G, V, u)
                            # for v in I:
                            #     V_[v] += 1
                            V = {u: (u not in I) for u in G.nodes()}


            size = N - sum(V.values())

            if verbose:
                print(size, weight)

            if size == 1:
                iters.append(t + 1)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")
                break
            elif size <= k2:
                iters.append(t + 2)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")
                break
            if t == 99:
                iters.append(100)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")

    print(sum(iters) / len(iters), max(iters), "\n\n\n")
    return iters, map_prev

def get_location_node(locations, node):
    x = locations[locations['ID'] == node]['Longitude'].iloc[0]
    y = locations[locations['ID'] == node]['Latitude'].iloc[0]
    return x,y

def create_linestring_coord(x_1, y_1, x_2, y_2):
    return LineString([(x_1, y_1) , (x_2, y_2)])

def get_random_tree_divide(G, nodes, mxl, nodes_location, edges_location, WTP=None):

    new_edges = edges_location.copy()
    new_nodes = nodes_location.copy()

    last_edge_id = int(new_edges.tail(1)['edge_ID'])
    last_node_id = max(max(new_edges['ID_1']), max(new_edges['ID_2']))

    new_G = nx.DiGraph()
    N = len(G.nodes())

    if WTP is None:
        WTP = random.choice(list(G.nodes()))

    V = [0] * N;  V[WTP] = 1
    Q = deque([]); Q.append(WTP)
    while Q:
        if len(new_G.nodes()) < nodes:

            random.shuffle(Q)

            u = Q.popleft()
            xu, yu = get_location_node(nodes_location, u)

            for v in G.neighbors(u):
                if not V[v]:

                    Q.append(v); V[v] = 1

                    xv, yv = get_location_node(nodes_location, v)
                    dist = np.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)

                    xd, yd = (xu - xv) / dist, (yu - yv) / dist

                    while dist > mxl:

                        xvv, yvv = xv + xd * mxl, yv + yd * mxl
                        last_edge_id += 1
                        last_node_id += 1
                        new_G.add_edge(v, last_node_id)
                        new_edges = new_edges.append({'edge_ID': last_edge_id, 'ID_1': v, 'ID_2': last_node_id, 'Distance': mxl,
                                                      'geometry': create_linestring_coord(xv, yv, xvv, yvv)}, ignore_index=True)
                        new_nodes = new_nodes.append({'ID': last_node_id, "Longitude": xvv, "Latitude": yvv}, ignore_index=True)

                        xv, yv = xvv, yvv
                        dist = np.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)
                        v = last_node_id

                    last_edge_id += 1
                    new_G.add_edge(v, u)
                    new_edges = new_edges.append({'edge_ID': last_edge_id, 'ID_1': v, 'ID_2': u, 'Distance': dist,
                                                  'geometry': create_linestring_coord(xv, yv, xu, yu)}, ignore_index=True)

        else:
            break

    return new_G, WTP, new_nodes, new_edges

def get_size(G, V_, u):

    if V_[u]:
        return 0

    V = V_.copy()

    ans = 1;  V[u] = 1
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft()
        for v in G.predecessors(u):
            if not V[v]:
                ans += 1; V[v] = 1
                Q.append(v)

    return ans\

class DLNode:
    def __init__(self, v, nxt, prv):
        self.v = v
        self.next = nxt
        self.back = prv

class DList:

    MXNODES = 10000000
    nodes = [None for i in range(MXNODES)]
    cnt = 0

    def __init__(self, arr):

        self.first = None
        self.last  = None

        for v in arr:
            DList.nodes[DList.cnt] = DLNode(v, None, self.last)
            if self.last:
                DList.nodes[self.last].next = DList.cnt
            self.last = DList.cnt; DList.cnt += 1
            if not self.first:
                self.first = self.last

    def reset(self):
        DList.cnt = 0
        DList.nodes = [None for i in range(MXNODES)]

    def append_right(self, v):

        DList.nodes[DList.cnt] = DLNode(v, None, self.last)
        if self.first:
            DList.nodes[self.last].next = DList.cnt
        else:
            self.first = DList.cnt
        self.last = DList.cnt; DList.cnt += 1

    def append_left(self, v):

        DList.nodes[DList.cnt] = DLNode(v, self.first, None)
        if self.first:
            DList.nodes[self.first].back = DList.cnt
        else:
            self.last = DList.cnt
        self.first = DList.cnt; DList.cnt += 1

    def extend_right(self, first, last):

        DList.nodes[self.last].next = first
        DList.nodes[first].back = self.last
        self.last = last

    def extend_left(self, first, last):

        DList.nodes[self.first].back = last
        DList.nodes[last].next = self.first
        self.first = first

    def remove_right(self, pos):

        self.last = pos
        DList.nodes[self.last].next = None

    def remove_left(self, pos):

        self.first = pos
        DList.nodes[self.first].back = None

    def remove_pos(self, pos):

        if self.first == pos:
            self.first = DList.nodes[pos].next
        if self.last == pos:
            self.last = DList.nodes[pos].back
        l = DList.nodes[pos].back; r = DList.nodes[pos].next
        if l is not None:
            DList.nodes[l].next = r
        if r is not None:
            DList.nodes[r].back = l

    def pop_left(self):

        if self.first is not None:
            self.first = DList.nodes[self.first].next
        if self.first is not None:
            DList.nodes[self.first].back = None

    def pop_right(self):

        if self.last is not None:
            self.last = DList.nodes[self.last].back
        if self.last is not None:
            DList.nodes[self.last].next = None

def get_optimal_K_function(T, K=1, N=int(1e6), debug=False):

    order = [u for u in nx.topological_sort(T)][::-1]
    S = [DList([]) for i in range(N)]
    P = [0 for i in range(N)]

    F = {e: None for e in T.edges()}

    for u in order:  ## Generamos la extensión en orden buttom-up

        if debug:
            print(f"Node {u}")

        children = [v for v in T.successors(u)]

        if debug:
            print(f"children: {children}")
        if len(children) == 0:
            continue

        if len(children) == 1:  ## En caso de un sólo hijo

            v = children[0]

            # Buscamos el menor valor libre
            i = S[v].first
            free = 1
            while i:
                if DList.nodes[i].v[0] > free:
                    break
                elif DList.nodes[i].v[1] < K:
                    break
                free = DList.nodes[i].v[0] + 1
                i = DList.nodes[i].next

            if i and DList.nodes[i].v[0] == free:
                DList.nodes[i].v = (DList.nodes[i].v[0], DList.nodes[i].v[1] + 1)
                S[u].append_right(DList.nodes[i].v)
                i = DList.nodes[i].next
            else:
                S[u].append_right((free, 1))

            F[(u, v)] = free
            F[(v, u)] = free

            if i is not None:  # Sólo nos quedamos con los valores expuestos
                S[u].extend_right(i, S[v].last)
            continue

        for v in children:
            P[v] = S[v].first

        # Buscamos l2 iterando coordinadamente

        l2 = -1; active = set(children); last_erased = None
        while len(active) > 1:
            l2 += 1

            to_erase = []
            for v in active:
                while P[v] is not None and DList.nodes[P[v]].v[0] <= l2:
                    P[v] = DList.nodes[P[v]].next
                if P[v] is None:
                    to_erase.append(v)

            for v in to_erase:
                last_erased = v
                active.remove(v)

        i1 = None  # hijo de secuencia más grande
        if len(active) == 0:
            i1 = last_erased
        else:
            i1 = next(iter(active))

        children = [i1] + [ch for ch in children if ch != i1]

        # Generamos las listas ordenadas L[1:l2 + 1]

        L = [DList(children.copy())] + [None for i in range(l2)]
        L_ = {v: [] for v in children}

        it = L[0].first
        while it is not None:
            v = DList.nodes[it].v
            L_[v].append(it)
            it = DList.nodes[it].next

        last = {v: 0 for v in children}
        M = [{v: 0 for v in children}] + [None for i in range(l2)]
        C = [len(children)] + [0 for i in range(l2)]

        for v in children:
            P[v] = S[v].first

        for i in range(1, l2 + 1):
            to_erase = set(); has_i = set()
            it = L[i - 1].first
            while it is not None:
                v = DList.nodes[it].v
                while P[v] is not None and DList.nodes[P[v]].v[0] <= i:
                    if DList.nodes[P[v]].v[0] == i:
                        has_i.add(v)
                    P[v] = DList.nodes[P[v]].next
                if P[v] is None:
                    to_erase.add(v)
                it = DList.nodes[it].next

            L_p = []; L_m = []
            it = L[i - 1].first
            while it is not None:
                v = DList.nodes[it].v
                if v in has_i:
                    L_p.append(v)
                elif v not in to_erase:
                    L_m.append(v)
                it = DList.nodes[it].next

            C[i] = 0
            L[i] = DList(L_p + L_m)
            it = L[i].first
            while it is not None:
                v = DList.nodes[it].v
                L_[v].append(it)
                it = DList.nodes[it].next
                C[i] += 1

            M[i] = {v: 0 for v in children}
            it = L[i].first
            while it is not None:
                v = DList.nodes[it].v
                M[i][v] = last[v]
                if v in has_i:
                    last[v] = i
                it = DList.nodes[it].next

        for v in children:
            P[v] = S[v].first
            nxt = None if P[v] is None else DList.nodes[P[v]].next
            while (P[v] is not None) and (nxt is not None) and DList.nodes[nxt].v[0] <= l2:
                P[v] = nxt
                nxt = None if P[v] is None else DList.nodes[P[v]].next

        if debug:
            print(f"l2: {l2}")

        U = deque([])

        G = {v: 0 for v in children}

        p_i1 = P[i1]
        while (p_i1 is not None) and DList.nodes[p_i1].v[0] <= l2:
            p_i1 = DList.nodes[p_i1].next

        curr = l2; lst = l2; last_i1 = l2 + 1; last_i1_cnt = 0; cnt_0 = len(children)
        active = set()
        for v in children:
            if P[v] is not None:
                active.add(v)

        while curr > 0 or cnt_0 > 0:

            cnt = 0
            for v in active:
                if DList.nodes[P[v]].v[0] == curr:
                    cnt += DList.nodes[P[v]].v[1]

            if debug:
                print(f"curr: {curr}, cnt: {cnt}, cnt_0: {cnt_0}")

            if cnt < K and lst > curr:
                for x in range(max(curr, 1), lst):
                    if x == curr:
                        U.append((x, K - cnt))
                    else:
                        U.append((x, K))
            lst = curr
            if cnt <= K and curr != 0:
                to_remove = []
                for v in active:
                    if DList.nodes[P[v]].v[0] == curr:
                        P[v] = DList.nodes[P[v]].back
                        if P[v] is None:
                            to_remove.append(v)
                for v in to_remove:
                    active.remove(v)
                curr -= 1
                continue

            w = None

            if U:
                w, cc = U.pop()
                if cc > 1:
                    U.append((w, cc - 1))
            else:
                while p_i1 is not None:
                    if last_i1 == DList.nodes[p_i1].v[0] and DList.nodes[p_i1].v[1] < K:
                        w = last_i1
                        last_i1_cnt += 1
                        if last_i1_cnt + DList.nodes[p_i1].v[1] == K:
                            last_i1 += 1
                            last_i1_cnt = 0
                            p_i1 = DList.nodes[p_i1].next
                        break
                    elif last_i1 == DList.nodes[p_i1].v[0] and DList.nodes[p_i1].v[1] == K:
                        last_i1 += 1
                        last_i1_cnt = 0
                        p_i1 = DList.nodes[p_i1].next
                    elif last_i1 < DList.nodes[p_i1].v[0] and last_i1_cnt < K:
                        w = last_i1
                        last_i1_cnt += 1
                        break
                    elif last_i1 < DList.nodes[p_i1].v[0] and last_i1_cnt == K:
                        last_i1 += 1
                        last_i1_cnt = 0
                if w is None:
                    w = last_i1
                    last_i1_cnt += 1
                    if last_i1_cnt == K:
                        last_i1 += 1
                        last_i1_cnt = 0

            best_j = None
            if w <= l2:
                flag = False
                if L[w].first is not None:
                    m = M[w][DList.nodes[L[w].first].v]
                    if C[w] == C[m]:
                        best_j = DList.nodes[L[w].first].v
                    elif C[m + 1] == C[w]:
                        best_j = DList.nodes[L[m].first].v
                    else:
                        flag = True

                if L[w].first is None or flag == True:
                    for i in range(w):
                        if C[i] > C[w] and C[i + 1] == C[w]:
                            best_j = DList.nodes[L[i].first].v
            else:
                if S[i1].first is not None and DList.nodes[S[i1].first].v[0] < w:
                    best_j = i1
                else:
                    flag = False
                    if L[l2].first is not None:
                        m = M[l2][DList.nodes[L[l2].first].v]
                        if C[l2] == C[m]:
                            best_j = DList.nodes[L[l2].first].v
                        elif C[m + 1] == C[l2]:
                            best_j = DList.nodes[L[m].first].v
                        else:
                            flag = True

                    if L[l2].first is None or flag == True:
                        for i in range(l2):
                            if C[i] > C[l2] and C[i + 1] == C[l2]:
                                best_j = DList.nodes[L[i].first].v

            if debug:
                print(f"best_j: {best_j}, w: {w}, active: {len(active)}")

            to_add = deque([])
            if P[best_j] is None:
                P[best_j] = S[best_j].first
            while P[best_j] is not None and DList.nodes[P[best_j]].v[0] < w:
                v_ = DList.nodes[P[best_j]].v
                if v_[0] > curr:
                    to_add.append(v_)
                P[best_j] = DList.nodes[P[best_j]].next
            while to_add:
                U.append(to_add.pop())

            mxx = None
            if P[best_j] is None:
                if best_j != i1:
                    for i in range(last[best_j] + 1):
                        C[i] -= 1
                else:
                    for i in range(l2 + 1):
                        C[i] -= 1
                S[best_j] = DList([])
                mxx = l2
            else:
                S[best_j].remove_left(P[best_j])
                mxx = min(l2, DList.nodes[P[best_j]].v[0] - 1)

            ll = len(L_[best_j]); mxx = min(mxx, ll - 1)
            while mxx >= 0 and L_[best_j][mxx] != -1:
                L[mxx].remove_pos(L_[best_j][mxx])
                L_[best_j][mxx] = -1
                M[mxx][best_j] = -1
                mxx -= 1

            if G[best_j] == 0:
                cnt_0 -= 1
            G[best_j] = w
            if best_j in active:
                active.remove(best_j)

        for k, v in G.items():
            F[(u, k)] = v
            F[(k, u)] = v

        for v in children:
            if S[v].first is not None and DList.nodes[S[v].first].v[0] == G[v]:
                DList.nodes[S[v].first].v = (DList.nodes[S[v].first].v[0], DList.nodes[S[v].first].v[1] + 1)
            else:
                S[v].append_left((G[v], 1))

        active = set(children)
        while len(active) > 1:

            id_ = set(); mn = 1e9; cntt = 0
            for v in active:
                if debug:
                    print(v, DList.nodes[S[v].first].v)
                if DList.nodes[S[v].first].v[0] < mn:
                    mn = DList.nodes[S[v].first].v[0]
                    cntt = DList.nodes[S[v].first].v[1]
                    id_ = set([v])
                elif DList.nodes[S[v].first].v[0] == mn:
                    cntt += DList.nodes[S[v].first].v[1]
                    id_.add(v)

            if debug:
                print(mn, cntt)

            S[u].append_right((mn, cntt))
            if cntt > K:
                print("FAILED", K, cntt)
                return 0
            for v in id_:
                S[v].pop_left()
                if S[v].first is None:
                    active.remove(v)

        if active:
            i1 = next(iter(active))
            S[u].extend_right(S[i1].first, S[i1].last)

    print("Optimal Steps:", max(F.values()))

    return F

def simulate_optimal(G, N, F, s_init, nodes):

    GR = G.reverse()

    E = G.edges()

    k_sum = 0
    T = []
    N_ = int(1e6)

    for r in nodes:
        CV = [0 for u in range(N_)]

        CV[r] = 1
        Q = deque([]); Q.append(r)
        while Q:
            u = Q.popleft()
            for v in G.predecessors(u):
                if not CV[v]:
                    CV[v] = 1
                    Q.append(v)

        V = [0 for u in range(N_)]

        s = s_init; ks = []
        for t in range(100):

            P = []

            mx = 0
            Q = deque([s]); visited = set([s])
            while Q:
                u = Q.popleft()
                for v in G.successors(u): #
                    if v not in visited and not V[v]:
                        if F[(u, v)] == mx:
                            if (u, v) in E:
                                P.append(v)
                            else:
                                P.append(u)
                        if F[(u, v)] > mx:
                            mx = F[(u, v)]
                            if (u, v) in E:
                                P = [v]
                            else:
                                P = [u]
                        visited.add(v)
                        Q.append(v)

            ks.append(len(P)); k_sum += len(P)

            for u in P[::-1]:
                if CV[u]:
                    I = set(get_ideal(GR, V, u)[::-1])
                    V = [not (v in I) for v in range(N_)]
                    s = u; break
                else:
                    V, _ = visit(GR, V, u)

            size = N - sum(V)

            if size == 1:
                T.append(t + 1)
                print(f"Result {r}: {T[-1]}    acc: {sum(T) / (len(T))}    ks: {ks}")
                break
            if t == 99:
                print(f"{r}!!!")

    print(sum(T) / len(T), max(T), k_sum)

    return T

def get_optimal_partition(T, root, K):
    V = list(T.nodes())
    I = dict()
    Path = dict()
    N = max(list(T.nodes())) + 1
    for u in V:
        I[u] = get_ideal(T, [0] * N, u)
        for v in V:
            if v not in I[u]:
                Path[v,u] = list()
            else:
                Path[v,u] = nx.shortest_path(T, source = v, target = u)

    model = gurobipy.Model()
    model.Params.OutputFlag = 0
    x = model.addVars(V, vtype = GRB.BINARY, name ="Nodos raices")
    Z = model.addVar(vtype = GRB.CONTINUOUS , obj=1, name ="Ideal de mayor tamano")
    y = model.addVars(V, V, vtype = GRB.BINARY, name ="Asignacion de nodo a muestra")


    model.addConstr((quicksum(x[i] for i in V) == K), name = "R1")
    model.addConstrs((y[j,i] <= x[i] for i in V for j in I[i] ), name = "R2")
    model.addConstrs((quicksum(y[i,j] for j in Path[i, root]) == 1 for i in V), name = "R3")
    model.addConstrs((x[l] <= 1 - y[j,i] for j in V for i in V for l in Path[j,i] if l != i), name = "R4")
    model.addConstrs((y[j,i]  <= y[l, i] for j in V for i in V for l in Path[j,i] if l != i), name = "R4b")
    model.addConstrs((quicksum(y[j,i] for j in I[i]) <= Z for i in V), name = "R5")
    model.addConstr((x[root] == 1), name = "Root")


    model.setObjective(Z, GRB.MINIMIZE)
    model.update()
    model.optimize()

    all_vars = model.getVars()
    try:
        values = model.getAttr("x", all_vars)
    except:
        nx.draw(T, with_labels=True)
    names = model.getAttr("VarName", all_vars)
    P = list()
    for name, val in zip(names, values):
        if "Nodos raices" in name and val == 1:
            P.append(int(name.strip("Nodos raices[")[:-1]))
            #print(f"{name} = {val}")
    #model.printAttr('x')
    P.remove(root)
    return P

def simulate_optimal_partition(G, root):
    iters = []
    map_prev = {}
    N = len(G.nodes())
    node_it = 0
    W = {u:1e6 for u in G.nodes()}
    for r in random.sample(list(G.nodes()), 100):

        node_it += 1

        CV = {u:0 for u in G.nodes()}

        CV[r] = 1
        Q = deque([]);
        Q.append(r)
        while Q:
            u = Q.popleft()
            successors = list(G.successors(u))
            if successors:
                v = random.sample(successors, 1)[0]
                CV[v] = 1
                Q.append(v)

        V = {u:0 for u in G.nodes()}

        s = root
        G_it = G
        for t in range(100):
            G_it = nx.induced_subgraph(G_it, [n for n in G.nodes() if not V[n]])

            R = N - sum(V.values())
            P = [];
            sP = 0

            print(f"Search {r}: {t + 1}    now: {R}", end="\r")

            nn = 0
            for i in G.nodes():
                if not V[i]:
                    nn += 2 ** i

            if nn in map_prev.keys():
                P = map_prev[nn]
            else:
                P = get_optimal_partition(G_it, s, 2)

                map_prev[nn] = P

            #print(P, s, size)

            V_ = V
            for u in P:
                if CV[u]:
                    s = u
                    I = get_ideal(G, V, u)
                    # for v in I:
                    #     V_[v] += 1
                    V = {u: (u not in I) for u in G.nodes()}
            #V = {u: (V_[u] != sum([CV[u] for u in P])) for u in G.nodes()}


            for u in P:
                if not CV[u]:
                    I = get_ideal_robust(G, V, s, u)
                    for v in I:
                        V[v] = 1

            if N - sum(V.values()) == R:

                    print("REPEATING")
                    POS = []
                    for i in G.nodes():
                        if not V[i]:
                            POS.append(i)
                    P = random.sample(POS, 1)
                    for u in P:
                        if not CV[u]:
                            I = get_ideal_robust(G, V, s, u)
                            for v in I:
                                V[v] = 1
                        if CV[u]:
                            s = u
                            I = get_ideal(G, V, u)
                            # for v in I:
                            #     V_[v] += 1
                            V = {u: (u not in I) for u in G.nodes()}



            size = get_size(G,V, s)
            #size =  R - sum(V.values())


            # if tt == 0 and size == R:
            #     iters.append(min(100, t + size))
            #     print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")
            #     break
            if size == 1:
                iters.append(t + 1)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")
                break
            if size <= 2:
                iters.append(t + 2)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")
                break
            if t == 99:
                iters.append(100)
                print(f"[{node_it}] Result {r}: {iters[-1]}    acc: {sum(iters) / len(iters)}")

    print(sum(iters) / len(iters), max(iters), "\n\n\n")

    return iters, map_prev

def visit(G, V, u):

    if V[u]:
        return V, 0

    V[u] = 1; ans = 0
    Q = deque([]); Q.append(u)
    while Q:
        u = Q.popleft(); ans += 1
        for v in G.predecessors(u):
            if not V[v]:
                V[v] = 1
                Q.append(v)

    return V, ans

def by_distance(element):
    return element[1]

def get_closest(G, locations, node, q):
    # get position of node
    x = locations[locations['ID'] == _id[node]]['Longitude'].iloc[0]
    y = locations[locations['ID'] == _id[node]]['Latitude'].iloc[0]

    #Neighbors
    ng = list(G.neighbors(node))

    # Calculate distances
    distances = list()
    for n in ng:
        n_id = _id[n]
        x_row = locations[locations['ID'] == n_id]['Longitude'].iloc[0]
        y_row = locations[locations['ID'] == n_id]['Latitude'].iloc[0]
        d = (x - x_row)**2 + (y - y_row)**2
        distances.append((n, d))

    # Order by distance and return q closest
    distances.sort(key = by_distance)
    return [i[0] for i in distances[:q]]

def get_random_tree(G, nodes, WTP=None):
    new_G = nx.DiGraph()

    if WTP is None:
        WTP = random.choice(list(G.nodes()))

    V = {u:0 for u in G.nodes()};  V[WTP] = 1
    Q = deque([]); Q.append(WTP)
    while Q:
        if len(new_G.nodes()) < nodes:
            random.shuffle(Q)
            u = Q.popleft()
            for v in G.predecessors(u):
                if not V[v]:
                    Q.append(v)
                    new_G.add_edge(v,u)
                    V[v] = 1
        else:
            break
    return new_G, WTP

def get_random_tree(G, nodes, WTP=None):
    new_G = nx.DiGraph()

    if WTP is None:
        WTP = random.choice(list(G.nodes()))

    V = {u:0 for u in G.nodes()};  V[WTP] = 1
    Q = deque([]); Q.append(WTP)
    while Q:
        if len(new_G.nodes()) < nodes:
            random.shuffle(Q)
            u = Q.popleft()
            for v in G.neighbors(u):
                if not V[v]:
                    Q.append(v)
                    new_G.add_edge(v,u)
                    V[v] = 1
        else:
            break
    return new_G, WTP

def add_extra_edges(T, G, locations, ratio, edge_limit):
    nodes = []
    for u in T.nodes():
        if u in G.nodes():
            nodes.append(u)
    counter = 0
    perturbed = random.choices(nodes, k = math.ceil(len(nodes) * ratio))
    for node in perturbed:
        added = False
        if len(list(T.predecessors(node))) < edge_limit:
            closest = [i for i in get_closest(G, locations, node, 7) if i in nodes]
            if closest:
                closest = closest[0]
                T_neigbors = list(T.predecessors(node)) + list(T.successors(node))
                if len(list(T.predecessors(closest))) < edge_limit and  closest not in T_neigbors:
                    T.add_edge(node, closest)
                    added = True
                    #print('added')
        if not added and counter < len(nodes):
            perturbed.append(random.choice(nodes))
            counter += 1
    return T

def get_degree(G, node):
    return len(list(T.predecessors(node))) + len(list(G.successors(node)))

def creates_cycle(G, e):
    if e[1] in nx.ancestors(G, e[0]):
        return True
    return False

def add_extra_edges2(T, G, locations, extra, edge_limit=1000000, replace=False):

    order = list(nx.topological_sort(T))
    I = defaultdict(int)
    for i in range(len(I)):
        I[order[i]] = i

    edges = []
    for e in G.edges():
        if e not in T.edges() and e not in T.to_undirected().edges() and e[0] in T.nodes() and e[1] in T.nodes():
            if I[e[0]] < I[e[1]]:
                edges.append(e)
            else:
                edges.append((e[1], e[0]))

    sample = np.random.choice(range(len(edges)), min(extra, len(edges)), replace=replace)

    for i in sample:
        e = edges[i]
        if get_degree(T, e[0]) < edge_limit and get_degree(T, e[1]) < edge_limit and not creates_cycle(T, e):
            T.add_edge(*e)

    return T

def generate_random_graph(G, nodes, locations, ratio, edge_limit):
    T, WTP = get_random_tree(G, nodes)
    NG = add_extra_edges(T,G, locations, ratio, edge_limit)
    return NG, WTP

if __name__ == "__main__":
    sys.setrecursionlimit(25000)
    path_nodes = '../data_random/TG.txt'
    nodes_location = pd.read_csv(path_nodes, sep=" ", header=None, names=['ID', 'Longitude', 'Latitude'])
    path_edges = '../data_random/TG_edge.txt'
    edges_location = pd.read_csv(path_edges, sep=" ", header=None, names=['edge_ID', 'ID_1', 'ID_2', 'Distance'])
    edges_location.head()

    S = set()
    for index, row in edges_location.iterrows():
        origin = row['ID_1']
        dest = row['ID_2']
        S.add(origin)
        S.add(dest)

    S = list(S)
    id_ = {};
    _id = {};
    l = 0
    for u in S:
        id_[u] = l;
        _id[l] = u
        l += 1

    G_SJ = nx.Graph()
    for index, row in edges_location.iterrows():
        origin = row['ID_1']
        dest = row['ID_2']
        G_SJ.add_edge(id_[origin], id_[dest])

    N_ = l

    ratios = [0.2]
    for ratio in ratios:
        print(f'\n\n RATIO = {ratio} \n\n')
        for _ in range(100):
            print(f"\n RUN: {ratio, _+1} \n")
            N = 500
            WTP = 7139

            T, WTP = get_random_tree(G_SJ, N, None)
            g = add_extra_edges2(T, G_SJ, None, math.ceil(N * ratio))

            spanning = nx.algorithms.minimum_spanning_tree(T.to_undirected());
            Tree = root_graph(spanning, WTP)

            nE = len(list(g.edges()))
            print(N, nE)


            # print("\n Simulando: Optimal function \n")
            # T_optimal = get_optimal_K_function(Tree.reverse(), K=1)
            # print("\n Simulando: Entropy \n")
            #T_entropy = simulate_entropy_minimization(g, WTP)
            #print(list(nx.simple_cycles(g)))

            print("\n Simulando: Path separation \n")
            T_paths = simulate_path_separation(g, WTP)
            print("\n Simulando: Greedy \n")
            T_greedy, map_prev = simulate_robust_randtree(g, WTP)
            print("\n Simulando: Treewidth \n")
            T_treewidth = simulate_treewidth_algorithm(g, WTP)


            with open(f'results/Random_results_{ratio}_greedy.txt', 'a+') as myfile:
                myfile.write(' '.join([str(i) for i in T_greedy]) + '\n')
            # with open(f'results/Random_results_{ratio}_entropy.txt', 'a+') as myfile:
            #     myfile.write(' '.join([str(i) for i in T_entropy]) + '\n')
            with open(f'results/Random_results_{ratio}_paths.txt', 'a+') as myfile:
                myfile.write(' '.join([str(i) for i in T_paths]) + '\n')
            with open(f'results/Random_results_{ratio}_treewidth.txt', 'a+') as myfile:
                myfile.write(' '.join([str(i) for i in T_treewidth]) + '\n')
            # with open(f'results/Random_results_optimal_in_tree.txt', 'a+') as myfile:
            #     myfile.write(' '.join([str(max(T_optimal.values()))]) + '\n')