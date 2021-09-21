from enum import auto
import numpy as np
import networkx as nx
import os
import scipy

import scipy.io as sio
from scipy import sparse

import simulation.predictors_functions as pf


def all_agents(c, a, i):
    return 1 - np.identity(c.shape[0], dtype=bool)


def no_agents(c, a, i):
    return np.zeros(shape=c.shape, dtype=bool)


def remembered_agents(c, a, i):
    return c


def undirected(c, a, i):
    return np.logical_or(c, c.T)


def create_jaccard(min_threshold=0, max_fraction=1.0, max_total=float('inf'), max_total_per_iter=float('inf')):
    def p(c, a, i): return pf.modular_predictor(
        c, a, nx.jaccard_coefficient, min_threshold, max_fraction, min(max_total, max_total_per_iter * i))
    return p


def create_common_neighbors(min_threshold=0, max_fraction=1.0, max_total=float('inf'), max_total_per_iter=float('inf')):
    def p(c, a, i): return pf.modular_predictor(
        c, a, pf.common_neighbors_lp, min_threshold, max_fraction, min(max_total, max_total_per_iter * i))
    return p


def create_adamic_adar(min_threshold=0, max_fraction=1.0, max_total=float('inf'), max_total_per_iter=float('inf')):
    def p(c, a, i): return pf.modular_predictor(
        c, a, nx.adamic_adar_index, min_threshold, max_fraction, min(max_total, max_total_per_iter * i))
    return p


def create_preferential_attachment(min_threshold=0, max_fraction=1.0, max_total=float('inf'), max_total_per_iter=float('inf')):
    def p(c, a, i): return pf.modular_predictor(
        c, a, nx.preferential_attachment, min_threshold, max_fraction, min(max_total, max_total_per_iter * i))
    return p


def create_katz(beta=0.1, max_l=None, min_threshold=0, max_fraction=1.0, max_total=float('inf'), max_total_per_iter=float('inf')):
    def predictor(G): return pf.katz_lp(G, beta=beta, max_l=max_l)

    def p(c, a, i): return pf.modular_predictor(
        c, a, predictor, min_threshold, max_fraction, min(max_total, max_total_per_iter * i))
    return p


def create_seal(seal_dir_path='./lib/SEAL/', test_split=0.5, batch_size=50, pred_batch_size=-1, pred_only_cn=False, h_hop=1, min_threshold=0.5, max_fraction=1.0, max_total=float('inf'), max_total_per_iter=float('inf')):
    # locate seal python directory
    seal_py = os.path.join(seal_dir_path, 'Python/')
    if not os.path.isfile(os.path.join(seal_py, 'Main.py')):
        raise RuntimeError(
            f"SEAL main script for python could not be located at '{seal_py}'")

    # locate seal data directory
    seal_data = os.path.join(seal_py, 'data/')
    if not os.path.isdir(seal_data):
        raise RuntimeError(
            f"SEAL data directory could not be located at '{seal_data}'")

    # create seal training command
    cmd_train = f"cd {seal_py} ; python Main.py --data-name TEMP "
    cmd_train += f"--test-ratio {test_split} "
    if isinstance(h_hop, int):
        cmd_train += f"--hop {h_hop} "
    else:
        cmd_train += "--hop 'auto' "
    cmd_train += f"--batch-size {batch_size} "
    cmd_train += "--use-embedding --use-attribute --save-model"

    # create seal prediction command
    cmd_pred = f"cd {seal_py} ; python Main.py --data-name TEMP --test-name TEMP_test.txt "
    if isinstance(h_hop, int):
        cmd_pred += f"--hop {h_hop} "
    else:
        cmd_pred += "--hop 'auto' "

    cmd_pred += "--use-embedding --use-attribute --only-predict"

    def p(c, a, i):
        # create undirected adjacency matrix
        G = nx.from_numpy_array(c).to_undirected()
        result = nx.to_numpy_array(G, dtype=bool)

        # extract agent features
        features = np.array(
            [[agent.risk, agent.age, agent.mobility] for agent in a])

        # save data for seal
        mdic = {
            'group': sparse.csr_matrix(features),
            'net': sparse.csr_matrix(result),
            }
        sio.savemat(os.path.join(seal_data, 'TEMP.mat'), mdic)

        # run seal training
        print('Training SEAL')
        os.system(cmd_train)

        # make file of links to predict
        lines = []
        if pred_only_cn:
            lines = [f"{u} {v}\n" for u, v in pf.get_unconnected_nodes(G) if len(list(nx.common_neighbors(G, u, v))) > 0]
        else:
            lines = [f"{u} {v}\n" for u, v in pf.get_unconnected_nodes(G)]
        
        # make batched predictions
        batches = []
        if pred_batch_size > 0:
            while len(lines) > pred_batch_size:
                batches.append(lines[:pred_batch_size])
                lines = lines[pred_batch_size:]
            
            batches.append(lines)
        else:
            batches = [lines]

        scores = []
        for b in batches:
            with open(os.path.join(seal_data, 'TEMP_test.txt'), 'w') as f:
                f.writelines(b)

            # make predictions
            print('Predicting links using the trained SEAL')
            os.system(cmd_pred)

            # load prediction
            predictions = []
            with open(os.path.join(seal_data, 'TEMP_test_pred.txt'), 'r') as f:
                predictions += f.readlines()
            
            # process predictions into scored tripeles (u, v, p) with u, v nodes idx and p predicted value
            predictions = [p.split(' ') for p in predictions]
            scores += [(int(u), int(v), float(p)) for u, v, p in predictions]

        scores.sort(key=lambda x: x[2], reverse=True)

        # make predictions
        limit = int(max_fraction * len(scores))
        limit = min(limit, min(max_total, max_total_per_iter * i))
        for i in range(limit):
            u, v, p = scores[i]
            if p < min_threshold:
                break

            result[u, v] = True
            result[v, u] = True

        return result

    return p
