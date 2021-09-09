import pickle
from simulation.location import Location
import numpy as np


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

    def print(self):
        coords = np.array([l.coords for l in self.locations] +
                          [h.coords for h in self.homes])
        min_coords = np.min(coords, axis=0)
        size = np.max(coords, axis=0) + 1

        m = np.empty(size, dtype=object)
        m.fill("")

        max_len = 0
        for l in self.locations:
            m[tuple(l.coords)] += f"L({l.capacity})"
            l = len(m[tuple(l.coords)])
            if l > max_len:
                max_len = l

        for h in self.homes:
            m[tuple(h.coords)] += f"H({h.capacity})"
            l = len(m[tuple(h.coords)])
            if l > max_len:
                max_len = l

        if len(size) == 2:
            for x in range(size[0]):
                print(('+' + ('-' * max_len)) * size[0] + '+')
                row = '|'
                for y in range(size[1]):
                    row += m[x, y].ljust(max_len) + '|'
                print(row)

            print(('+' + ('-' * max_len)) * size[0] + '+')
        else:
            print(m)
