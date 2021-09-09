import numpy as np


class Location:
    def __init__(self, coords: np.ndarray, capacity: int, risk: float):
        self.coords = coords
        self.capacity = capacity
        self.risk = risk
