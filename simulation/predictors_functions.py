import numpy as np
import networkx as nx


def count(l):
    if len(l) == 0:
        return []

    l.sort()
    result = [0] * (l[-1] + 1)
    for e in l:
        result[e] += 1

    return result


def get_unconnected_nodes(G):
    adjacency_matrix = nx.to_numpy_array(G, dtype=bool)
    n = adjacency_matrix.shape[0]
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adjacency_matrix[u, v]:
                result.append((u, v))

    return result


def modular_predictor(c, a, predictor, min_threshold=0, max_fraction=1.0, max_total=float("inf")):
    G = nx.from_numpy_matrix(c).to_undirected()
    result = nx.to_numpy_array(G, dtype=bool)

    # compute scores
    scores = list(predictor(G))
    # sort by score
    scores.sort(key=lambda x: x[2], reverse=True)

    limit = int(max_fraction * len(scores))
    limit = min(limit, max_total)
    for i in range(limit):
        u, v, p = scores[i]
        if p < min_threshold:
            break

        result[u, v] = True
        result[v, u] = True

    return result


def common_neighbors_lp(G):
    for u, v in get_unconnected_nodes(G):
        yield u, v, len(list(nx.common_neighbors(G, u, v)))


def katz_lp(G, beta=0.1, max_l=None):
    for u, v in get_unconnected_nodes(G):
        all_path_num = count([len(p)
                             for p in nx.all_simple_paths(G, u, v, max_l)])
        katz = 0.0
        for l, p in enumerate(all_path_num):
            katz += (beta ** l) + p
        yield u, v, katz
