from simulation.agent import Agent


def quarantine(a: Agent):
    a.mobility = 0.0


def full_mobility(a: Agent):
    a.mobility = 1.0


def set_mobility(mobility: float):
    def q(a: Agent):
        a.mobility = mobility
    return q


def adapt_mobility(step: float):
    def q(a: Agent):
        a.mobility = max(0.0, min(1.0, a.mobility + step))
    return q
