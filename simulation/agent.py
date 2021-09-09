from simulation.health_state import HealthState


class Agent:
    def __init__(self, home: int, risk: float, infected: bool, exposed_time: int, outbreak_time: int, recovery_time: int):
        self.home = home

        # properties
        self.age = 20
        self.risk = risk
        self.exposed_time = exposed_time
        self.outbreak_time = outbreak_time
        self.recovery_time = recovery_time

        # variables
        self.infection_duration = 0
        self.mobility = 1.0
        self.state = HealthState.SUSCEPTIBLE
        self.infected = infected
        if infected:
            self.state = HealthState.EXPOSED
