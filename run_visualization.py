import pickle
import matplotlib.pyplot as plt

def main():
    #### config ####
    # storage
    metrics_pkl = './data/metrics/map_1.pkl'

    #### load data ####
    metrics = []
    with open(metrics_pkl, 'rb') as f:
            metrics = pickle.load(f)

    #### plotting results ####
    for name, data, err in metrics:
        plt.plot(err, label=name)

    plt.legend()
    plt.ylim((0, 1))
    plt.xlim(0, 30)

    plt.show()

if __name__ == '__main__':
    main()