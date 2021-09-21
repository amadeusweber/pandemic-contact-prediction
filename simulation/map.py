import pickle
from simulation.location import Location
import numpy as np
import matplotlib.pyplot as plt


class Map:
    @staticmethod
    def load(filename):
        locs, homes = [], []
        with open(filename, 'rb') as f:
            locs, homes = pickle.load(f)

        m = Map()
        for h in homes:
            m.add_home(h.coords, h.capacity, h.risk)

        for l in locs:
            m.add_location(l.coords, l.capacity, l.risk)

        return m

    @staticmethod
    def generate_random(size,
                        n_homes: int,
                        n_locations: int,
                        min_per_house: int = 3,
                        max_per_house: int = 3,
                        min_per_loc: int = 10,
                        max_per_loc: int = 10,
                        random_state: float = None):
        assert n_homes + n_locations <= size[0] * size[1]
        rnd = np.random.RandomState(random_state)

        map = Map()

        # add houses
        for _ in range(n_homes):
            capacity = rnd.randint(min_per_house, max_per_house + 1)
            p_inf = rnd.random()
            coords = rnd.randint(0, size, len(size))
            added = map.add_home(coords, capacity, p_inf)

        # add locations
        for _ in range(n_locations):
            capacity = rnd.randint(min_per_loc, max_per_loc + 1)
            p_inf = rnd.random()
            coords = rnd.randint(0, size, len(size))
            added = map.add_location(coords, capacity, p_inf)

        return map

    def __init__(self, distance=lambda a, b: np.linalg.norm(a-b)):
        self.distance = distance

        self.locations = []
        self.homes = []

        self.dist_home_loc = np.zeros(shape=(0, 0), dtype=float)
        self.dist_loc_loc = np.zeros(shape=(0, 0), dtype=float)

        self.map = dict()

    def add_location(self, coords, capacity, p_inf):
        # create location
        loc = Location(coords, capacity, p_inf)

        # compute new loc-loc distances
        d = [self.distance(coords, l.coords) for l in self.locations]
        self.dist_loc_loc = np.vstack((self.dist_loc_loc, d))
        d.append(0.0)
        self.dist_loc_loc = np.vstack((self.dist_loc_loc.T, d))

        # compute new home-loc distances
        d = [self.distance(coords, h.coords) for h in self.homes]
        self.dist_home_loc = np.vstack((self.dist_home_loc.T, d)).T

        # add location to list
        self.locations.append(loc)

    def add_home(self, coords, capacity, p_inf):
        # create home
        home = Location(coords, capacity, p_inf)

        # compute new home-loc distances
        d = [self.distance(coords, l.coords) for l in self.locations]
        self.dist_home_loc = np.vstack((self.dist_home_loc, d))

        # add location to list
        self.homes.append(home)

    def save(self, filename):
        data = [self.locations, self.homes]
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def plot(self, file=None):
        # determine map size
        coords = np.array([l.coords for l in self.locations] +
                          [h.coords for h in self.homes])
        min_pos = np.min(coords, axis=0)
        max_pos = np.max(coords, axis=0)
        map_shape = max_pos - min_pos + 1

        # plot locations and homes using heatmaps
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
        for label, ax, cmap, data in [('Locations', ax0, plt.cm.Blues, self.locations), ('Homes', ax1, plt.cm.Oranges, self.homes)]:
            m = np.zeros(shape=(map_shape), dtype=int)

            for d in data:
                m[tuple(d.coords - min_pos)] += 1

            pos = ax.imshow(m, cmap=cmap, interpolation='none')
            ax.set_title(label)
            fig.colorbar(pos, ax=ax, shrink=0.75, ticks=range(np.max(m) + 1))

        # display or save the plot
        if file == None:
            plt.show()
        else:
            plt.savefig(file, bbox_inches='tight')
            plt.cla()
