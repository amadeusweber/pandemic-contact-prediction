from simulation.agent import Agent
from simulation.health_state import HealthState
from simulation.map import Map

import numpy as np
import networkx as nx

import os
import pickle


class Simulation:
    def __init__(self,
                 m: Map,
                 locations_per_iter: int = 3,
                 exposed_time=0,
                 outbreak_time=2,
                 recovery_time=7,
                 p_inf_init: float = .1,
                 p_contact_forget=.5,
                 agent_outbrak_strategy=lambda a: None,
                 agent_recovered_strategy=lambda a: None,
                 agent_notified_strategy=lambda a: None,
                 agent_iteration_strategy=lambda a: None,
                 predictor=lambda c, a, i: c,
                 random_state: int = None):
        # assertions
        assert outbreak_time >= exposed_time
        assert recovery_time >= outbreak_time

        self.map = m
        self.locations_per_iter = locations_per_iter
        self.p_contact_forget = p_contact_forget

        self.agent_outbrak_strategy = agent_outbrak_strategy
        self.agent_recovered_strategy = agent_recovered_strategy
        self.agent_notified_strategy = agent_notified_strategy
        self.agent_iteration_strategy = agent_iteration_strategy

        self.predictor = predictor

        self.rnd_ = np.random.RandomState(random_state)

        self.agents = []
        for h, home in enumerate(self.map.homes):
            for _ in range(home.capacity):
                a = Agent(home=h,
                          risk=self.rnd_.random(),
                          infected=self.rnd_.random() < p_inf_init,
                          exposed_time=exposed_time,
                          outbreak_time=outbreak_time,
                          recovery_time=recovery_time)
                self.agents.append(a)

        self.iter = 0

        # helper values
        # number of agents, homes and locations
        self.n_agents_ = len(self.agents)
        self.n_homes_ = len(self.map.homes)
        self.n_locs_ = len(self.map.locations)

        # assignement matrix for agents to their homes
        self.agt_hme_ = np.zeros(
            shape=(self.n_agents_, self.n_homes_), dtype=bool)
        # agents infection risk
        self.agt_rsk_ = np.zeros(shape=(self.n_agents_), dtype=float)
        for i, a in enumerate(self.agents):
            self.agt_hme_[i, a.home] = True
            self.agt_rsk_[i] = a.risk

        # distances from agents at home to locations
        self.dist_agt_loc_ = np.array(
            [self.map.dist_home_loc[i.home] for i in self.agents])

        # n_agents x n_agents identity matrix
        self.agt_id_mat_ = np.identity(self.n_agents_, dtype=int)

        # contact monitoring
        self.contacts_ = np.dot(
            self.agt_hme_, self.agt_hme_.T) - self.agt_id_mat_
        self.remembered_contacts_ = self.contacts_.copy()

    def get_contacts_as_graph(self) -> nx.Graph:
        return nx.from_numpy_matrix(self.contacts_, create_using=nx.MultiDiGraph)

    def get_remembered_contacts_as_graph(self) -> nx.Graph:
        return nx.from_numpy_matrix(self.remembered_contacts_, create_using=nx.MultiDiGraph)

    def p_inf(self, state, locations):
        # infection state
        inf = np.array(
            [a.state == HealthState.INFECTIOUS for a in self.agents])

        # get agents properties
        agent_risk = np.array([a.risk for a in self.agents])

        # get locations properties
        loc_caps = np.array([l.capacity for l in locations])
        loc_risk = np.array([l.risk for l in locations])

        # infection prob by location
        p_inf_loc = loc_risk * np.sum(state * inf[:, None], axis=0) / loc_caps

        # return agents infection prob
        return (state * agent_risk[:, None]).dot(p_inf_loc)

    def handle_state(self, state):
        ######## INFECT OTHERS ########
        # p_inf is computed by p_inf_home + p_inf_loc
        # compute infection risk for homes
        homes_state = self.agt_hme_ * state[:, -1][:, None]
        p_inf_home = self.p_inf(homes_state, self.map.homes)

        # compute infection risk for locations
        loc_state = state[:, 0:-1]
        p_inf_loc = self.p_inf(loc_state, self.map.locations)

        p_inf = p_inf_home + p_inf_loc

        choice = self.rnd_.random(self.n_agents_)
        for i, infected in enumerate(choice < p_inf):
            if infected and self.agents[i].state == HealthState.SUSCEPTIBLE:
                self.agents[i].infected = True
                self.agents[i].state = HealthState.EXPOSED

        ######## Contact monitoring ########
        new_contacts = np.dot(homes_state, homes_state.T) + \
            np.dot(loc_state, loc_state.T) - self.agt_id_mat_
        to_forget = self.rnd_.rand(self.n_agents_, self.n_agents_) > (
            1 - self.p_contact_forget)
        self.contacts_ += new_contacts
        self.remembered_contacts_ = np.clip(
            self.remembered_contacts_ + new_contacts - to_forget, a_min=0, a_max=None)

    def next_iter(self, verbose=False):
        self.iter += 1
        if verbose:
            print(f"Iteration {self.iter}")

        # create state matrix (individual x location + 1)
        # the last col represents an individual being at home
        state = np.zeros(shape=(self.n_agents_, self.n_locs_ + 1), dtype=bool)

        # collect agents mobility values
        mob = np.array([i.mobility for i in self.agents])

        # create a transition matix to crete a move
        trans = self.dist_agt_loc_.copy()
        for _ in range(self.locations_per_iter):
            # Invert distances if not 0
            trans = np.divide(1, trans, where=trans != 0)
            # normalize distances to get distance probs
            trans /= np.sum(trans, axis=1)[:, None]

            # multiply by participants mobility values
            trans *= mob[:, None]

            # create moving thresholds for each individual
            # to a column j add the values from c_0 to c_{j-1}
            for i in range(trans.shape[1] - 1):
                trans[:, i + 1] += trans[:, i]

            # generate random values to sample moves
            choice = self.rnd_.random(self.n_agents_)
            moves = np.sum(trans < choice[:, None], axis=1)

            # perform all moves for individuals not being at home
            # if a locations capacity level is reached, the individual has to wait
            loc_fill = np.zeros(shape=(self.n_locs_), dtype=int)
            ind_mov = list(enumerate(moves))
            self.rnd_.shuffle(ind_mov)
            for i, destination in ind_mov:
                if not state[i, -1]:
                    state[i] = False
                    if destination < self.n_locs_:
                        if loc_fill[destination] < self.map.locations[destination].capacity:
                            state[i, destination] = True
                            loc_fill[destination] += 1
                    else:
                        state[i, destination] = True

            # handle the intermediate state
            self.handle_state(state)

            # compute new transition matrix
            trans = np.array([
                self.map.dist_loc_loc[i] if i < self.n_locs_
                else self.map.dist_loc_loc[0]
                for i in moves])

        # all agents return to their homes
        state.fill(False)
        state[:, -1] = True
        self.handle_state(state)

        ####### RECONSTRUCT CONTACTS ########
        contacts = self.predictor(self.remembered_contacts_ > 0, self.agents, self.iter)
        error = np.sum(np.logical_xor(
            contacts, (self.contacts_ > 0))) / (self.n_agents_ ** 2)

        # HANDLE INFECTION STAGES ########:
        for i, a in enumerate(self.agents):
            if a.infected:
                if a.infection_duration == a.exposed_time:
                    a.state = HealthState.INFECTIOUS

                if a.infection_duration == a.outbreak_time:
                    self.agent_outbrak_strategy(a)
                    # Notify contacts on outbrak
                    for j, had_contact in enumerate(contacts[i]):
                        if had_contact:
                            self.agent_notified_strategy(self.agents[j])

                if a.infection_duration == a.recovery_time:
                    a.state = HealthState.RECOVERED
                    a.infected = False
                    self.agent_recovered_strategy(a)

                a.infection_duration += 1

            if (not a.infected) or (a.infection_duration < a.outbreak_time):
                self.agent_iteration_strategy(a)

        return error

    def current_data(self):
        n_sus = 0
        n_exp = 0
        n_inf = 0
        n_rec = 0
        for a in self.agents:
            if a.state == HealthState.SUSCEPTIBLE:
                n_sus += 1
            elif a.state == HealthState.EXPOSED:
                n_exp += 1
            elif a.state == HealthState.INFECTIOUS:
                n_inf += 1
            elif a.state == HealthState.RECOVERED:
                n_rec += 1

        return n_sus, n_exp, n_inf, n_rec

    def full_run(self, max_iter=-1, intermediate_results_path=None, intermediate_results_name=None, verbose=False):
        # run simulation
        data = [[*self.current_data()]]
        errors = []
        while (max_iter != 0) and (data[-1][1] + data[-1][2] > 0):
            errors.append(self.next_iter(verbose=verbose))
            data.append([*self.current_data()])
            #print(f"Infected: {data[-1][2]}")
            max_iter -= 1

            # save intermediate results
            if intermediate_results_path != None:
                with open(intermediate_results_path, 'wb') as f:
                    pickle.dump((intermediate_results_name, np.array(data).T, errors), f)

        # transform data into np array and transpose
        return np.array(data).T, errors
