"""
The following code was adapted and modified from different Mesa examples 
(from: https://github.com/projectmesa/mesa/blob/master/examples/), 
Accessed on: November 7, 2020
"""

import math

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from .model import KalickHamilton, number_single, number_female


def network_portrayal(G):

    def node_color(agent):
        return {"MALE": "#002FFF", "FEMALE": "#FF0000"}.get(
            agent.S, "#808080"
        )

    def edge_color(agent1, agent2):
        if agent1.R == "UNION" and agent2.R == "UNION":
            return "#008000"
        elif agent1.R == "SINGLE" or agent2.R == "SINGLE":
            return "#000000"


    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            "tooltip": "id: {}<br>Sex: {}<br>Relationship: {}<br>Attractiveness: {}".format(
                agents[0].unique_id, agents[0].S, agents[0].R, agents[0].A
            ),
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": 1,
        }
        for (source, target) in G.edges
    ]

    return portrayal


network = NetworkModule(network_portrayal, 500, 500, library="d3")
chart = ChartModule(
    [
        {"Label": "mean_attractiveness", "Color": "#FF0000"},
        {"Label": "corr_results", "Color": "#008080"},
    ]
)

class MyTextElement(TextElement):
    def render(self, model):
        ratio = model.single_union_ratio()
        ratio_text = "&infin;" if ratio is math.inf else "{0:.2f}".format(ratio)
        singles_text = str(number_single(model))
        female_count = str(number_female(model))

        return "Single/Union Ratio: {}<br>Singles Remaining: {}<br>Females: {}".format(
            ratio_text, singles_text, female_count
        )


model_params = {
    "num_nodes": UserSettableParameter(
        "slider",
        "Number of agents (both sexes)",
        10,
        10,
        300,
        2,
        description="Choose how many agents to include in the model",
    ),
    "preference": UserSettableParameter("choice", "Preference",
                                                       value="attractiveness", 
                                                       choices=["attractiveness", "matching", "mixed"]),
    "mean_male": UserSettableParameter(
        "slider",
        "Average Attractiveness (males)",
        5,
        1,
        10,
        1,
    ),
    "sd_male": UserSettableParameter(
        "slider",
        "SD Attractiveness (males)",
        1,
        1,
        4,
        1,
    ),
    "mean_female": UserSettableParameter(
        "slider",
        "Average Attractiveness (females)",
        5,
        1,
        10,
        1,
    ),
    "sd_female": UserSettableParameter(
        "slider",
        "SD Attractiveness (females)",
        1,
        1,
        4,
        1,
    ),

}

server = ModularServer(
    KalickHamilton, [network, MyTextElement(), chart], "Kalick Hamilton 1986 model", model_params
)
server.port = 8521
