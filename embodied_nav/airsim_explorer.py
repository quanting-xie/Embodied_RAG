import airsim
import networkx as nx
import time
import matplotlib.pyplot as plt
from .graph_builder import GraphBuilder

# Hyperparameters
DETECTION_RADIUS = 10  # Distance threshold in meters

class AirSimExplorer:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.graph_builder = GraphBuilder()
        self.G = nx.Graph()

    def take_off(self):
        self.client.takeoffAsync().join()

    def get_semantic_data(self):
        drone_pose = self.client.simGetVehiclePose()
        drone_position = drone_pose.position
        all_objects = self.client.simListSceneObjects()

        semantic_labels = {}
        for obj_name in all_objects:
            object_pose = self.client.simGetObjectPose(obj_name)
            object_position = object_pose.position
            
            distance = ((object_position.x_val - drone_position.x_val) ** 2 +
                        (object_position.y_val - drone_position.y_val) ** 2 +
                        (object_position.z_val - drone_position.z_val) ** 2) ** 0.5
            
            if distance < DETECTION_RADIUS:
                object_id = self.client.simGetSegmentationObjectID(obj_name)
                semantic_labels[obj_name] = {
                    "id": object_id,
                    "position": (object_position.x_val, object_position.y_val, object_position.z_val)
                }

        return semantic_labels

    def log_to_graph(self, semantic_labels):
        for label, data in semantic_labels.items():
            self.G.add_node(label, id=data["id"], position=data["position"])
        self.graph_builder.update_graph(self.G)

    def keyboard_control(self):
        print("Use WASD keys to move the drone. Press 'q' to quit.")
        while True:
            key = input("Enter command: ")
            if key == 'w':
                self.client.moveByVelocityAsync(1, 0, 0, 1).join()
            elif key == 's':
                self.client.moveByVelocityAsync(-1, 0, 0, 1).join()
            elif key == 'a':
                self.client.moveByVelocityAsync(0, -1, 0, 1).join()
            elif key == 'd':
                self.client.moveByVelocityAsync(0, 1, 0, 1).join()
            elif key == 'q':
                break

            semantic_labels = self.get_semantic_data()
            self.log_to_graph(semantic_labels)

            time.sleep(0.1)

        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def save_graph(self):
        nx.write_gml(self.G, "semantic_graph.gml")
        print("Graph saved to semantic_graph.gml")

    def visualize_graph(self):
        pos = {node: (data['position'][0], data['position'][1]) for node, data in self.G.nodes(data=True)}
        labels = {node: f"{node}\nID: {data['id']}" for node, data in self.G.nodes(data=True)}
        
        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos, labels=labels, with_labels=True, node_size=500, node_color='lightblue', font_size=8)
        plt.title("Semantic Graph Visualization")
        plt.show()

    def run(self):
        self.take_off()
        self.keyboard_control()
        self.save_graph()
        self.visualize_graph()
        self.graph_builder.generate_relationships()

if __name__ == "__main__":
    explorer = AirSimExplorer()
    explorer.run()
