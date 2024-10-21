import airsim
import time
import sys
import signal
import logging
import threading
from graph_builder import GraphBuilder
from airsim_utils import AirSimUtils, DroneController, DetectionVisualizer

# Hyperparameters
DETECTION_RADIUS = 5  # Distance threshold in meters

class AirSimExplorer:
    def __init__(self):
        self.initialize_airsim_client()
        self.graph_builder = GraphBuilder()
        self.is_running = True
        self.drone_controller = DroneController(self.client)
        self.detection_visualizer = DetectionVisualizer(self.client)
        self.visualization_thread = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def initialize_airsim_client(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def take_off(self):
        self.client.takeoffAsync().join()

    def get_semantic_data(self):
        try:
            camera_name = "0"
            image_type = airsim.ImageType.Scene

            self.client.simClearDetectionMeshNames(camera_name, image_type)
            self.client.simSetDetectionFilterRadius(camera_name, image_type, DETECTION_RADIUS * 100)
            self.client.simAddDetectionFilterMeshName(camera_name, image_type, "*")

            detections = self.client.simGetDetections(camera_name, image_type)

            print(f"Number of detections: {len(detections)}")

            semantic_labels = {}
            drone_pose = self.client.simGetVehiclePose()

            filter_words = ["floor", "ceiling"]

            for detection in detections:
                object_name = detection.name
                
                if any(word in object_name.lower() for word in filter_words):
                    continue

                relative_position = detection.relative_pose.position
                
                print(f"Detection: {object_name}, Position: {relative_position}")
                
                global_position = AirSimUtils.local_to_global_position(drone_pose, relative_position)
                
                semantic_labels[object_name] = {
                    "id": object_name,
                    "position": (global_position.x_val, global_position.y_val, global_position.z_val),
                    "box2D": (detection.box2D.min.x_val, detection.box2D.min.y_val, 
                              detection.box2D.max.x_val, detection.box2D.max.y_val),
                    "box3D": (detection.box3D.min.x_val, detection.box3D.min.y_val, detection.box3D.min.z_val,
                              detection.box3D.max.x_val, detection.box3D.max.y_val, detection.box3D.max.z_val)
                }
                logging.info(f"Detected object: {object_name} at global position {global_position}")

            return semantic_labels
        except Exception as e:
            logging.error(f"Error in get_semantic_data: {str(e)}")
            return {}

    def update_graph_and_semantics(self):
        while self.is_running:
            try:
                vehicle_pose = self.client.simGetVehiclePose()
                position = AirSimUtils.vector3r_to_dict(vehicle_pose.position)
                yaw = airsim.to_eularian_angles(vehicle_pose.orientation)[2]
                node_id = self.graph_builder.add_drone_node(position, yaw)
                logging.info(f"Added drone node: {node_id}")

                semantic_labels = self.get_semantic_data()
                for label, data in semantic_labels.items():
                    self.graph_builder.add_object_node(label, data)
                    logging.info(f"Added object node: {label}")

                logging.info(f"Current graph has {len(self.graph_builder.G.nodes)} nodes and {len(self.graph_builder.G.edges)} edges")
                time.sleep(1.0)
            except Exception as e:
                logging.error(f"Error in update_graph_and_semantics: {str(e)}")
                time.sleep(1)

    def run(self):
        self.take_off()
        
        update_thread = threading.Thread(target=self.update_graph_and_semantics)
        update_thread.start()

        self.detection_visualizer.start()

        self.drone_controller.keyboard_control()

        self.is_running = False
        self.drone_controller.is_running = False
        self.detection_visualizer.stop()

        update_thread.join()

        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

        self.save_graph()
        self.visualize_graph()
        self.graph_builder.generate_relationships()

    def save_graph(self):
        logging.info(f"Saving graph with {len(self.graph_builder.G.nodes)} nodes and {len(self.graph_builder.G.edges)} edges")
        self.graph_builder.save_graph("semantic_graph.gml")
        print("Graph saved to semantic_graph.gml")

    def visualize_graph(self):
        self.graph_builder.visualize_graph()

    def __del__(self):
        if hasattr(self, 'detection_visualizer'):
            self.detection_visualizer.stop()

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Saving graph and exiting...")
    explorer.is_running = False
    explorer.save_graph()
    sys.exit(0)

if __name__ == "__main__":
    explorer = AirSimExplorer()
    signal.signal(signal.SIGINT, signal_handler)
    explorer.run()
