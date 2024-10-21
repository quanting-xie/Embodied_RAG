import networkx as nx
from .llm import LLMInterface

class GraphBuilder:
    def __init__(self):
        self.G = nx.Graph()
        self.llm = LLMInterface()

    def update_graph(self, new_graph):
        self.G = nx.compose(self.G, new_graph)

    def generate_relationships(self):
        nodes = list(self.G.nodes(data=True))
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                relationship = self.llm.generate_relationship(node1, node2)
                if relationship:
                    self.G.add_edge(node1[0], node2[0], relationship=relationship)

    def save_graph(self, filename):
        nx.write_gml(self.G, filename)

    def load_graph(self, filename):
        self.G = nx.read_gml(filename)
