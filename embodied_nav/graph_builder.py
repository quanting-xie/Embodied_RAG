import networkx as nx
from llm import LLMInterface
import matplotlib.pyplot as plt
import time
import logging

class GraphBuilder:
    def __init__(self):
        self.G = nx.Graph()
        self.llm = LLMInterface()
        self.last_drone_node = None
        self.drone_node_distance = 3.0  # Minimum distance threshold in meters

    def update_graph(self, new_graph):
        self.G = nx.compose(self.G, new_graph)

    def add_drone_node(self, position, yaw):
        if not self.last_drone_node:
            # Always add the first node
            node_id = f"drone_0"
            self.G.add_node(node_id, position=position, yaw=yaw, type='drone', timestamp=time.time())
            self.last_drone_node = node_id
            logging.info(f"Added initial drone node {node_id} at position {position}")
            return node_id

        # Calculate distance from last node
        last_position = self.G.nodes[self.last_drone_node]['position']
        distance = self.calculate_distance(last_position, position)

        if distance >= self.drone_node_distance:
            # Only add a new node if the distance exceeds the threshold
            node_id = f"drone_{len([n for n in self.G.nodes() if n.startswith('drone')])}"
            self.G.add_node(node_id, position=position, yaw=yaw, type='drone', timestamp=time.time())
            self.G.add_edge(self.last_drone_node, node_id, type='path', distance=distance)
            logging.info(f"Added drone node {node_id} at position {position}")
            logging.info(f"Added edge between {self.last_drone_node} and {node_id} with distance {distance}")
            self.last_drone_node = node_id
            return node_id
        else:
            logging.debug(f"Drone movement too small ({distance} m). Node not added.")
            return self.last_drone_node

    def add_object_node(self, object_name, object_data):
        if object_name not in self.G:
            self.G.add_node(object_name, **object_data, type='object', timestamp=time.time())
            logging.info(f"Added object node {object_name}")
            if self.last_drone_node:
                self.G.add_edge(self.last_drone_node, object_name, type='observed')
                logging.info(f"Added edge between {self.last_drone_node} and {object_name}")

    def generate_relationships(self):
        nodes = list(self.G.nodes(data=True))
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                relationship = self.llm.generate_relationship(node1, node2)
                if relationship:
                    self.G.add_edge(node1[0], node2[0], relationship=relationship)

    def prune_old_nodes(self, age_threshold=300):  # 5 minutes
        current_time = time.time()
        nodes_to_remove = [node for node, data in self.G.nodes(data=True) 
                           if data['type'] == 'object' and current_time - data['timestamp'] > age_threshold]
        self.G.remove_nodes_from(nodes_to_remove)

    def save_graph(self, filename):
        nx.write_gml(self.G, filename)

    def load_graph(self, filename):
        self.G = nx.read_gml(filename)

    def visualize_graph(self):
        pos = nx.spring_layout(self.G)
        drone_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == 'drone']
        object_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == 'object']
        
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(self.G, pos, nodelist=drone_nodes, node_color='r', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(self.G, pos, nodelist=object_nodes, node_color='b', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_labels(self.G, pos)
        plt.title("Drone Path and Object Graph")
        plt.axis('off')
        plt.show()

    @staticmethod
    def calculate_distance(pos1, pos2):
        return ((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2 + (pos1['z'] - pos2['z'])**2)**0.5
