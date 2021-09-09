import numpy as np
import networkx as nx

import simulation.predictors_functions as pf


def all_agents(c, a):
    return 1 - np.identity(c.shape[0], dtype=bool)


def no_agents(c, a):
    return np.zeros(shape=c.shape, dtype=bool)


def remembered_agents(c, a):
    return c


def undirected(c, a):
    return np.logical_or(c, c.T)


def create_jaccard(min_threshold=0, max_fraction=1.0, max_total=float('inf')):
    def p(c, a): return pf.modular_predictor(
        c, a, nx.jaccard_coefficient, min_threshold, max_fraction, max_total)
    return p


def create_common_neighbors(min_threshold=0, max_fraction=1.0, max_total=float('inf')):
    def p(c, a): return pf.modular_predictor(
        c, a, pf.common_neighbors_lp, min_threshold, max_fraction, max_total)
    return p


def create_adamic_adar(min_threshold=0, max_fraction=1.0, max_total=float('inf')):
    def p(c, a): return pf.modular_predictor(
        c, a, nx.adamic_adar_index, min_threshold, max_fraction, max_total)
    return p


def create_preferential_attachment(min_threshold=0, max_fraction=1.0, max_total=float('inf')):
    def p(c, a): return pf.modular_predictor(
        c, a, nx.preferential_attachment, min_threshold, max_fraction, max_total)
    return p


def create_katz(beta=0.1, max_l=None, min_threshold=0, max_fraction=1.0, max_total=float('inf')):
    def predictor(G): return pf.katz_lp(G, beta=beta, max_l=max_l)

    def p(c, a): return pf.modular_predictor(
        c, a, predictor, min_threshold, max_fraction, max_total)
    return p
