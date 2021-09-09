import pickle
import matplotlib.pyplot as plt
import os


def main():
    #### config ####
    # storage
    metrics_dir = './data/metrics/map_1/'

    #### load data ####
    metrics = []
    for f in os.listdir(metrics_dir):
        if f.endswith('.pkl'):
            with open(os.path.join(metrics_dir, f), 'rb') as pkl:
                metrics.append(pickle.load(pkl))

    #### plotting results ####
    for name, data, err in metrics:
        plt.plot(err, label=name)

    plt.legend()
    plt.ylim((0, 1))
    plt.xlim(0, 40)

    plt.show()


if __name__ == '__main__':
    main()
