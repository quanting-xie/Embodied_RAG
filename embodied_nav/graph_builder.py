import networkx as nx
from .llm import LLMInterface 
import matplotlib.pyplot as plt
import time
import logging
import asyncio
from .config import Config

class GraphBuilder:
    def __init__(self):
        self.G = nx.Graph()
        self.llm = LLMInterface()
        self.last_drone_node = None
        self.drone_node_distance = Config.GRAPH['drone_node_distance']

    def update_graph(self, new_graph):
        self.G = nx.compose(self.G, new_graph)

    def add_drone_node(self, position, yaw):
        if not self.last_drone_node:
            # Always add the first node
            node_id = f"drone_0"
            node_data = {'id': node_id, 'position': position, 'yaw': yaw, 'type': 'drone', 'timestamp': time.time()}
            self.G.add_node(node_id, **node_data)
            self.last_drone_node = node_id
            logging.info(f"Added initial drone node {node_id} at position {position}")
            return node_data

        # Calculate distance from last node
        last_position = self.G.nodes[self.last_drone_node]['position']
        distance = self.calculate_distance(last_position, position)

        if distance >= self.drone_node_distance:
            # Only add a new node if the distance exceeds the threshold
            node_id = f"drone_{len([n for n in self.G.nodes() if n.startswith('drone')])}"
            node_data = {'id': node_id, 'position': position, 'yaw': yaw, 'type': 'drone', 'timestamp': time.time()}
            self.G.add_node(node_id, **node_data)
            self.G.add_edge(self.last_drone_node, node_id, type='path', distance=distance)
            logging.info(f"Added drone node {node_id} at position {position}")
            logging.info(f"Added edge between {self.last_drone_node} and {node_id} with distance {distance}")
            self.last_drone_node = node_id
            return node_data
        else:
            logging.debug(f"Drone movement too small ({distance} m). Node not added.")
            return self.G.nodes[self.last_drone_node]

    def add_object_node(self, object_name, object_data):
        if object_name not in self.G:
            node_data = {'id': object_name, **object_data, 'type': 'object', 'timestamp': time.time()}
            self.G.add_node(object_name, **node_data)
            logging.info(f"Added object node {object_name}")
            if self.last_drone_node:
                self.G.add_edge(self.last_drone_node, object_name, type='observed')
                logging.info(f"Added edge between {self.last_drone_node} and {object_name}")
            return node_data
        return self.G.nodes[object_name]

    async def generate_relationships(self):
        nodes = list(self.G.nodes(data=True))
        for i, (node1_id, node1_data) in enumerate(nodes):
            for node2_id, node2_data in nodes[i+1:]:
                relationship = await self.llm.generate_relationship(
                    {'id': node1_id, **node1_data},
                    {'id': node2_id, **node2_data}
                )
                if relationship:
                    # Ensure the relationship is a string
                    self.G.add_edge(node1_id, node2_id, relationship=str(relationship))

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

    def get_objects(self):
        return [{'id': node, **data} for node, data in self.G.nodes(data=True)]

    # Add a new method to run generate_relationships
    def run_generate_relationships(self):
        asyncio.run(self.generate_relationships())
