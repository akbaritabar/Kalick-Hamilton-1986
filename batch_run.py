"""
The following code was adapted and modified from different Mesa examples 
(from: https://github.com/projectmesa/mesa/blob/master/examples/), 
Accessed on: November 7, 2020
"""

"""
This script is to perform parameter sweeps instead of run.py.
For details see below.
"""

import math
import itertools
import networkx as nx
import random
import numpy as np
import pandas as pd

from mesa import Agent, Model
from mesa.batchrunner import BatchRunner
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

def number_state(model, R_state=""):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.R == R_state])

def number_sex(model, sex=""):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.S == sex])

def number_single(model):
    return number_state(model, "SINGLE")

def number_female(model):
    return number_sex(model, "FEMALE")

def number_union(model):
    return number_state(model, "UNION")

def mean_attractiveness(model):
    """mean attractiveness of all agents who are in a union divided by 10 for better plotting"""
    agent_attrs = []
    agents_in_union = sum([1 for a in model.schedule.agents if a.R == 'UNION'])

    if agents_in_union > 1:
        for a in model.schedule.agents:
            if a.R == 'UNION':
                agent_attrs.append(a.A)
        # return the mean over 10 (based on Grow's tutorial)
        return str(round(np.mean(agent_attrs), ndigits = 2) / 10)

def calculate_correlations(model):
    if model.corr_results.shape[0] >= 2:
        return model.corr_results.corr().iloc[0, 1]

# ============================
#### Batch run tracks ####
# ============================
def track_params(model):
    return (model.num_nodes,
            model.preference,
            model.mean_male,
            model.sd_male,
            model.mean_female,
            model.sd_female)

def track_run(model):
    return model.uid

# ============================
#### Kalick Hamilton model ####
# ============================
class KalickHamilton(Model):
    """A model following Andre Grow's Netlogo tutorial of Kalick Hamilton 1986
    replicated using Mesa"""

    # id generator to track run number in batch run data
    id_gen = itertools.count(1)

    def __init__(
        self,
        seed=None,
        num_nodes=50,
        preference='attractiveness',
        mean_male=5,
        sd_male=1,
        mean_female=5,
        sd_female=1,
        corr_results=pd.DataFrame()
    ):

        self.uid = next(self.id_gen)
        self.num_nodes = num_nodes
        self.preference = preference
        self.mean_male = mean_male
        self.sd_male = sd_male
        self.mean_female = mean_female
        self.sd_female = sd_female
        self.corr_results = corr_results
        self.step_count = 0
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=0)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                "number_single": number_single,
                "number_female": number_female,
                "number_union": number_union,
                "mean_attractiveness":mean_attractiveness,
                "corr_results":calculate_correlations,
                "Model Params": track_params,
                "Run": track_run
            },
            agent_reporters={
                "name": lambda x: x.name,
                "sex": lambda x: x.S,
                "attractiveness": lambda x: x.A,
                "relationship": lambda x: x.R,
                "Model Params": lambda x: track_params(x.model),
                "Run": lambda x: track_run(x.model)
                })


        # Create agents
        for i, node in enumerate(self.G.nodes()):
            person = Human(
                i,
                self,
                "MALE",
                0,
                "SINGLE",
                )
            self.schedule.add(person)
            # Add the agent to the node
            self.grid.place_agent(person, node)
        # convert half of agents to "FEMALE"
        # this need number of agents to always be dividable by 2
        # as set in the user-settable slider
        female_nodes = self.random.sample(self.G.nodes(), (int(self.num_nodes / 2)))
        for a in self.grid.get_cell_list_contents(female_nodes):
            a.S = "FEMALE"
        # here assign attractiveness based on normal distributions
        for a in self.schedule.agents:
            if a.S == 'MALE':
                A2use = np.random.normal(self.mean_male, self.sd_male, 1)[0]
                while A2use < 1 or A2use > 10:
                    A2use = np.random.normal(self.mean_male, self.sd_male, 1)[0]
                a.A = A2use
            else:
                A2use = np.random.normal(self.mean_female, self.sd_female, 1)[0]
                while A2use < 1 or A2use > 10:
                    A2use = np.random.normal(self.mean_female, self.sd_female, 1)[0]
                a.A = A2use

        self.running = True
        self.datacollector.collect(self)

    def single_union_ratio(self):
        try:
            return number_state(self, "SINGLE") / number_state(self, "UNION")
        except ZeroDivisionError:
            return math.inf

    def do_match_singles(self):
        for a in self.schedule.agents:
            if a.S == 'MALE' and a.R == 'SINGLE':
                a.date_someone()

    def do_calculate_decision_probabilities(self):
        for a in self.schedule.agents:
            if a.R == 'SINGLE':
                a.calculate_decision_probabilities()

    def do_union_decisions(self):
        for a in self.schedule.agents:
            if a.S == 'MALE' and a.R == 'SINGLE':
                a.take_union_decision()

    def step(self):
        # add to step counter
        self.step_count += 1
        self.schedule.step()
        self.do_match_singles()
        self.do_calculate_decision_probabilities()
        self.do_union_decisions()
        # collect data
        self.datacollector.collect(self)
        # if all agents in union or 51 steps past, stop.
        if number_single(self) == 0 or self.step_count == 51:
            self.running = False

    def run_model(self, n):
        for i in range(n):
            self.step()


class Human(Agent):
    def __init__(
        self,
        unique_id,
        model,
        S,
        A,
        R,
    ):
        super().__init__(unique_id, model)

        self.S = S
        self.A = A
        self.R = R
        self.name = unique_id


    def date_someone(self):
        all_nodes = self.model.random.sample(self.model.G.nodes(), (int(self.model.num_nodes)))
        for a in self.model.grid.get_cell_list_contents(all_nodes):
            if a != self and a.S != self.S and a.R == 'SINGLE' and self.model.G.degree[self.unique_id] == 0 and a.model.G.degree[a.unique_id] == 0:
                self.model.G.add_edge(self.unique_id, a.unique_id)
            break

    def calculate_decision_probabilities(self):
        neighbor_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        # since we allow one to have one date per step so one has one neighbor only
        possible_dates = self.model.grid.get_cell_list_contents(neighbor_nodes)
        for this_date in possible_dates:        
            self.P1 = ((this_date.A ** 3) / 1000)
            self.P1C = (self.P1 ** ((51 - self.model.step_count) / 50))
            self.P2 = (((10 - abs(self.A - this_date.A)) ** 3) / 1000)
            self.P2C = (self.P2 ** ((51 - self.model.step_count) / 50))
            self.P3 = ((self.P1 + self.P2) / 2)
            self.P3C = (self.P3 ** ((51 - self.model.step_count) / 50))
            if self.model.preference == "attractiveness":
                self.focal_P = self.P1C
            elif self.model.preference == "matching":
                self.focal_P = self.P2C
            elif self.model.preference == "mixed":
                self.focal_P = self.P3C

    def take_union_decision(self):
        neighbor_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        # since we allow one to have one date per step so one has one neighbor only
        possible_dates = self.model.grid.get_cell_list_contents(neighbor_nodes)
        for this_date in possible_dates:
            if random.uniform(0, 1) < self.focal_P and random.uniform(0, 1) < this_date.focal_P:
                self.R = "UNION"
                this_date.R = "UNION"
                if self.S == "MALE":
                    union_data = pd.DataFrame({"M_A": [self.A], "F_A": [this_date.A]})
                else:
                    union_data = pd.DataFrame({"M_A": [this_date.A], "F_A": [self.A]})
                self.model.corr_results = self.model.corr_results.append(union_data)
            # do_unmatch_singles from Grow's Netlogo model happens here
            else:
                # remove ties that don't result in UNION
                self.model.G.remove_edge(self.unique_id, this_date.unique_id)

    def step(self):
        pass


# ============================
#### Bach run parameter swipes/scenarios ####
# ============================
# parameter lists for each parameter to be tested in batch run
br_params = {"num_nodes": [100],
             "preference": ["attractiveness", "matching", "mixed"],
             "mean_male": [5, 10],
             "sd_male": [1],
             "mean_female": [5, 10],
             "sd_female": [1]
             }

br = BatchRunner(KalickHamilton,
                 br_params,
                 iterations=3,
                 max_steps=1000,
                 model_reporters={"Data Collector": lambda m: m.datacollector},
                 agent_reporters={"name": "name"})
# ============================
#### Run & export data ####
# ============================
if __name__ == '__main__':
    br.run_all()
    br_df = br.get_model_vars_dataframe()
    br_a_df = br.get_agent_vars_dataframe()
    br_step_data = pd.DataFrame()
    br_step_agent_data = pd.DataFrame()
    for i in range(len(br_df["Data Collector"])):
        if isinstance(br_df["Data Collector"][i], DataCollector):
            i_run_data = br_df["Data Collector"][i].get_model_vars_dataframe()
            i_run_adata = br_df["Data Collector"][i].get_agent_vars_dataframe()
            br_step_data = br_step_data.append(i_run_data, ignore_index=True)
            br_step_agent_data = br_step_agent_data.append(i_run_adata, ignore_index=True)
    br_step_data.to_csv("./data/KHG_model_data.csv")
    br_step_agent_data.to_csv("./data/KHG_agents_data.csv")