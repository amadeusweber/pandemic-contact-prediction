from simulation.map import Map
from simulation.simulation import Simulation

import simulation.predictors as predictors
import simulation.strategies as strategies

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os


def main():
    #### config ####
    # storage
    map_pkl = './data/maps/1_map_rnd_20x20_200H_75L.pkl'
    metrics_dir = './data/metrics/map_1_total/'

    #### Creating the map ####

    # load or generate map
    if os.path.isfile(map_pkl):
        print("Loading stored map")
        m = Map.load(map_pkl)
    else:
        print("Generating random map")
        m = Map.generate_random(
            size=(20, 20),
            n_homes=200,
            n_locations=75)
        m.save(map_pkl)

    ##### Specify predictors #####
    preds = [
        ('All agents', predictors.all_agents),
        ('No agents', predictors.no_agents),
        ('Remembered', predictors.remembered_agents),
        ('Undirected', predictors.undirected),
        ('Common Neighbors', predictors.create_common_neighbors(min_threshold=0, max_total_per_iter=1000)),
        ('Jaccard', predictors.create_jaccard(min_threshold=0, max_total_per_iter=1000)),
        ('Adamic Adar', predictors.create_adamic_adar(min_threshold=0, max_total_per_iter=1000)),
        ('Preferential Attachment', predictors.create_preferential_attachment(min_threshold=0, max_total_per_iter=1000)),
        ('Katz (b=0.01, l=2)', predictors.create_katz(0.01, 2, min_threshold=0, max_total_per_iter=1000)),
        ('SEAL', predictors.create_seal(min_threshold=0.5, h_hop=1, test_split=0.2, pred_batch_size=10000, pred_only_cn=True)),
    ]

    ##### simulating for every predictor #####
    metrics = []
    for name, predictor in preds:
        print(f"[{name}] running simulation")

        s = Simulation(m=m,
                       locations_per_iter=3,
                       exposed_time=1,
                       outbreak_time=3,
                       recovery_time=14,
                       p_inf_init=.05,
                       p_contact_forget=0.2,
                       agent_outbrak_strategy=strategies.set_mobility(0.025),
                       agent_recovered_strategy=strategies.set_mobility(1.0),
                       agent_notified_strategy=strategies.adapt_mobility(-0.2),
                       agent_iteration_strategy=strategies.adapt_mobility(
                           0.05),
                       predictor=predictor,
                       random_state=42,
                       )

        data, err = s.full_run(verbose=False, intermediate_results_name=name + '_pre', intermediate_results_path=os.path.join(metrics_dir, name + '_pre.pkl'))
        metrics.append((name, data, err))
        print(f"[{name}] simulation finished after {s.iter} iteration(s)")

        #### save metrics ####
        print(f"[{name}] saving metrics")
        with open(os.path.join(metrics_dir, name+'.pkl'), 'wb') as f:
            pickle.dump((name, data, err), f)
        
        os.remove(os.path.join(metrics_dir, name + '_pre.pkl'))


    print("Done :)")


if __name__ == "__main__":
    main()
