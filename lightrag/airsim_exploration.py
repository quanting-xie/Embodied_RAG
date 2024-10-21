import airsim
import networkx as nx
import time
import matplotlib.pyplot as plt

# Hyperparameters
DETECTION_RADIUS = 10  # Distance threshold in meters

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Initialize a graph to store semantic labels and positions
G = nx.Graph()

def get_semantic_data():
    # Get the current position of the drone
    drone_pose = client.simGetVehiclePose()
    drone_position = drone_pose.position

    # List all scene objects
    all_objects = client.simListSceneObjects()

    semantic_labels = {}
    for obj_name in all_objects:
        # Get object position
        object_pose = client.simGetObjectPose(obj_name)
        object_position = object_pose.position

        # Calculate distance from the drone
        distance = ((object_position.x_val - drone_position.x_val) ** 2 +
                    (object_position.y_val - drone_position.y_val) ** 2 +
                    (object_position.z_val - drone_position.z_val) ** 2) ** 0.5

        # Consider objects within the specified detection radius
        if distance < DETECTION_RADIUS:
            # Get semantic label (object ID)
            object_id = client.simGetSegmentationObjectID(obj_name)
            
            # Store the semantic label and position
            semantic_labels[obj_name] = {
                "id": object_id,
                "position": (object_position.x_val, object_position.y_val, object_position.z_val)
            }

    return semantic_labels

def log_to_graph(semantic_labels):
    for label, data in semantic_labels.items():
        G.add_node(label, id=data["id"], position=data["position"])

def keyboard_control():
    print("Use WASD keys to move the drone. Press 'q' to quit.")
    while True:
        key = input("Enter command: ")
        if key == 'w':
            client.moveByVelocityAsync(1, 0, 0, 1).join()
        elif key == 's':
            client.moveByVelocityAsync(-1, 0, 0, 1).join()
        elif key == 'a':
            client.moveByVelocityAsync(0, -1, 0, 1).join()
        elif key == 'd':
            client.moveByVelocityAsync(0, 1, 0, 1).join()
        elif key == 'q':
            break

        # Get semantic data and log it
        semantic_labels = get_semantic_data()
        log_to_graph(semantic_labels)

        # Sleep for a short duration to prevent overwhelming the simulator
        time.sleep(0.1)

    # Land the drone
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

def save_graph():
    # Save the graph to a file
    nx.write_gml(G, "semantic_graph.gml")
    print("Graph saved to semantic_graph.gml")

def visualize_graph():
    # Visualize the graph
    pos = {node: (data['position'][0], data['position'][1]) for node, data in G.nodes(data=True)}
    labels = {node: f"{node}\nID: {data['id']}" for node, data in G.nodes(data=True)}
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color='lightblue', font_size=8)
    plt.title("Semantic Graph Visualization")
    plt.show()

if __name__ == "__main__":
    keyboard_control()

    # Save and visualize the graph
    save_graph()
    visualize_graph()

    # Print the graph nodes
    print("Graph nodes with semantic labels and positions:")
    for node in G.nodes(data=True):
        print(node)
