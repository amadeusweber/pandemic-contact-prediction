import pickle
from matplotlib import lines
import matplotlib.pyplot as plt
import os
import numpy as np

def flatten(vals):
    result = [np.mean([vals[0], vals[0], vals[1]])]
    for i in range(len(vals) - 2):
        result.append(np.mean(vals[i:i+3]))
    
    result.append(np.mean([vals[-2], vals[-1], vals[-1]]))
    return result

def main():
    #### config ####
    # storage
    metrics_dir = './data/metrics/map_1_total/'
    plots_dir = './data/plots/map_1_total/'

    # maximum iterations to plot
    max_iter = 40

    # metrics to show
    to_show = [
        'No agents',
        'Remembered',
        'Undirected',
        'All agents',
        'Common Neighbors',
        'Jaccard',
        'Preferential Attachment',
        'Adamic Adar',
        'Katz (b=0.01, l=2)',
        'SEAL',
    ]

    # flatten error curves
    to_flatten = [
        #'SEAL',
    ]

    #### load data ####
    metrics = []

    for name in to_show:
        with open(os.path.join(metrics_dir, f"{name}.pkl"), 'rb') as f:
            metrics.append(pickle.load(f))

        # flatten SEAL error curve
        if name in to_flatten:
            flat_err = flatten(metrics[-1][2])
            metrics.append((f"{name} (flat)", metrics[-1][1], flat_err))

    #### plot results ####
    plt.figure(figsize=(16, 10))
    for name, data, err in metrics:
        plt.plot(err, label=name)

    plt.legend()
    plt.ylim((0, 1))
    plt.xlim(0, max_iter)

    # save figure
    plt.savefig(os.path.join(plots_dir, 'errors.png'), bbox_inches='tight')
    plt.cla()

    # plot courses of disease
    plt.figure(figsize=(4, 4))
    lines = []
    for name, data, err in metrics:
        # print key values
        n_agents = np.sum([d[0] for d in data])
        lines.append(f"<section name=\"{name}\">\n")
        lines.append(f"    Infectious curve height:{np.max(data[2])}({round(np.max(data[2]) / n_agents, 3)})\n")
        lines.append(f"    Avg infectious per iter:{round(np.average(data[2]))}({round(np.average(data[2]) / n_agents, 3)})\n")
        lines.append(f"    Total infections       :{data[3, -1]}({round(data[3, -1] / n_agents, 3)})\n")
        lines.append(f"    R0                     :{round(1 / (1 - (data[3, -1] / n_agents)), 3)}\n")
        lines.append(f"    Avg recreation error   :{round(np.average(err), 3)}\n")
        lines.append(f"</section>\n")
        lines.append('\n')

        plt.plot(data[0], color='#1f77b4', label="Susceptible")
        plt.plot(data[1], color='#ff7f0e', label="Exposed")
        plt.plot(data[2], color='#d62728', label="Infectious")
        plt.plot(data[3], color='#2ca02c', label="Recovered")
        plt.ylim(0)
        plt.xlim(0, max_iter)
        plt.legend()
        plt.title(f"{name}: Course of diease")

        # save figure
        plt.savefig(os.path.join(plots_dir, f"course_{name}.png"))
        plt.cla()
    
    # write indicators in text file
    with open(os.path.join(plots_dir, 'indicators.txt'), 'w') as f:
        f.writelines(lines)



if __name__ == '__main__':
    main()
