import airsim
import time
import sys
import signal
import logging
import threading
import os
import networkx as nx
from datetime import datetime
import keyboard
import cv2

from .airsim_utils import AirSimUtils, DroneController, DetectionVisualizer, AirSimClientWrapper

# Hyperparameters
DETECTION_RADIUS = 5  # Distance threshold in meters

# Add environment name constant
ENVIRONMENT_NAME = "Building99"  # Change this based on your environment

class AirSimExplorer:
    def __init__(self):
        self.initialize_airsim_client()
        self.G = nx.Graph()  # Using NetworkX graph directly
        self.is_running = True
        self.drone_controller = DroneController(self.client)
        self.detection_visualizer = DetectionVisualizer(self.client)
        self.visualization_thread = None
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "semantic_graphs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"exploration_{ENVIRONMENT_NAME}_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Starting exploration of {ENVIRONMENT_NAME}")

        self._shutdown_event = threading.Event()
        self._threads = []
        self._update_lock = threading.Lock()
        self._visualization_lock = threading.Lock()
        self._shutdown_timeout = 5.0  # 5 seconds timeout for shutdown

        # Add node logging parameters
        self.min_distance_between_nodes = 0.5  # Minimum 0.5 meters between nodes
        self.min_time_between_nodes = 0.5      # Minimum 0.5 seconds between nodes
        self.last_node_position = None
        self.last_node_time = time.time()

    def initialize_airsim_client(self):
        self.client = AirSimClientWrapper()
        self.client.confirmConnection()
        print("Connected!")
        
        # Reset and set initial position
        self.client.reset()
        
        # Define minimum height
        self.MIN_HEIGHT = -1.0  # 2 meters above ground
        
        # Set initial pose with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                initial_pose = airsim.Pose(
                    position_val=airsim.Vector3r(0, 0, self.MIN_HEIGHT),
                    orientation_val=airsim.Quaternionr()
                )
                self.client.simSetVehiclePose(initial_pose, True)
                
                # Enable API control and arm
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                
                # Take off and maintain minimum height
                print("Taking off...")
                self.client.takeoffAsync().join()
                self.client.moveToZAsync(self.MIN_HEIGHT, 1).join()
                print("Ready to fly!")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Initialization attempt {attempt + 1} failed, retrying...")
                time.sleep(1)

    def configure_lidar(self):
        """Configure LiDAR settings"""
        try:
            # Enable LiDAR
            self.client.enableApiControl(True)
            lidar_data = self.client.getLidarData(lidar_name="LidarSensor1")
            if len(lidar_data.point_cloud) < 3:
                print("No LiDAR data received")
            else:
                print("LiDAR is functioning")
        except Exception as e:
            logging.error(f"Error configuring LiDAR: {e}")

    def maintain_minimum_height(self):
        """Thread function to maintain minimum height"""
        while self.is_running:
            try:
                state = self.client.getMultirotorState()
                current_height = state.kinematics_estimated.position.z_val
                
                if current_height > self.MIN_HEIGHT + 0.1:  # If too low
                    self.client.moveToZAsync(self.MIN_HEIGHT, 1).join()
                
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error in height maintenance: {e}")
                time.sleep(0.1)

    def take_off(self):
        self.client.takeoffAsync().join()

    def add_drone_node(self, position, yaw):
        node_id = f"drone_{len([n for n in self.G.nodes() if 'drone' in str(n)])}"
        self.G.add_node(node_id, 
                       position=position,
                       yaw=yaw,
                       type='drone',
                       level=0)
        return node_id

    def add_object_node(self, label, data):
        """Add object node to graph with proper position formatting"""
        if label not in self.G:
            print(f"Adding object node: {label}")
            try:
                self.G.add_node(label, 
                              position=data['position'],
                              type='object',
                              box2D=data['box2D'],
                              box3D=data['box3D'],
                              level=0)
                print(f"Successfully added object node: {label} at position {data['position']}")
            except Exception as e:
                print(f"Error adding object node {label}: {e}")

    def get_semantic_data(self):
        try:
            camera_name = "0"
            image_type = airsim.ImageType.Scene

            self.client.simClearDetectionMeshNames(camera_name, image_type)
            self.client.simSetDetectionFilterRadius(camera_name, image_type, DETECTION_RADIUS * 100)
            self.client.simAddDetectionFilterMeshName(camera_name, image_type, "*")

            detections = self.client.simGetDetections(camera_name, image_type)
            print(f"\nNumber of detections: {len(detections)}")

            semantic_labels = {}
            drone_pose = self.client.simGetVehiclePose()
            filter_words = ["floor", "ceiling"]

            for detection in detections:
                try:
                    object_name = detection.name
                    if any(word in object_name.lower() for word in filter_words):
                        continue

                    print(f"\nProcessing detection: {object_name}")
                    
                    # Get object position directly from relative_pose
                    relative_position = detection.relative_pose.position
                    print(f"Object relative position: {relative_position}")
                    
                    # Convert to global position
                    global_position = AirSimUtils.local_to_global_position(drone_pose, relative_position)
                    print(f"Object global position: {global_position}")
                    
                    # Convert position to dictionary format
                    position_dict = {
                        'x': float(global_position.x_val),
                        'y': float(global_position.y_val),
                        'z': float(global_position.z_val)
                    }
                    print(f"Position dictionary: {position_dict}")
                    
                    semantic_labels[object_name] = {
                        "id": object_name,
                        "position": position_dict,
                        "box2D": {
                            'min': {'x': float(detection.box2D.min.x_val), 'y': float(detection.box2D.min.y_val)},
                            'max': {'x': float(detection.box2D.max.x_val), 'y': float(detection.box2D.max.y_val)}
                        },
                        "box3D": {
                            'min': {'x': float(detection.box3D.min.x_val), 
                                   'y': float(detection.box3D.min.y_val),
                                   'z': float(detection.box3D.min.z_val)},
                            'max': {'x': float(detection.box3D.max.x_val),
                                   'y': float(detection.box3D.max.y_val),
                                   'z': float(detection.box3D.max.z_val)}
                        }
                    }
                    print(f"Successfully added semantic data for {object_name}")
                    
                except Exception as e:
                    print(f"Error processing detection {object_name}: {e}")
                    continue

            print(f"\nTotal semantic labels collected: {len(semantic_labels)}")
            return semantic_labels
            
        except Exception as e:
            logging.error(f"Error in get_semantic_data: {str(e)}")
            return {}

    def update_graph_and_semantics(self):
        """Update graph with drone positions and semantic information"""
        last_save_time = time.time()
        checkpoint_interval = 300  # Save every 5 minutes
        last_drone_node = None  # Keep track of the last drone node
        
        while self.is_running:
            try:
                with self._update_lock:
                    current_time = time.time()
                    vehicle_pose = self.client.simGetVehiclePose()
                    current_position = vehicle_pose.position
                    
                    # Check if we should add a new node
                    should_add_node = False
                    time_since_last_node = current_time - self.last_node_time
                    
                    if self.last_node_position is not None:
                        distance_moved = self._calculate_distance(
                            current_position,
                            self.last_node_position
                        )
                        print(f"Distance moved: {distance_moved:.2f}m, Time since last node: {time_since_last_node:.2f}s")
                        
                        if (distance_moved >= self.min_distance_between_nodes and 
                            time_since_last_node >= self.min_time_between_nodes):
                            should_add_node = True
                    else:
                        # First node
                        should_add_node = True
                    
                    if should_add_node:
                        position = AirSimUtils.vector3r_to_dict(current_position)
                        yaw = airsim.to_eularian_angles(vehicle_pose.orientation)[2]
                        
                        # Add new drone node
                        current_drone_node = self.add_drone_node(position, yaw)
                        print(f"Added drone node {current_drone_node} at position: {position}")
                        
                        # Connect to previous drone node if it exists
                        if last_drone_node is not None:
                            distance = self._calculate_distance(
                                current_position,
                                self.last_node_position
                            )
                            self.G.add_edge(
                                last_drone_node, 
                                current_drone_node,
                                distance=distance,
                                type='drone_path'
                            )
                            print(f"Added drone path edge between {last_drone_node} and {current_drone_node}")
                        
                        # Update tracking variables
                        last_drone_node = current_drone_node
                        self.last_node_position = current_position
                        self.last_node_time = current_time
                        
                        # Get semantic data at new node position
                        semantic_labels = self.get_semantic_data()
                        for label, data in semantic_labels.items():
                            self.add_object_node(label, data)

                    # Check for checkpoint save
                    if current_time - last_save_time >= checkpoint_interval:
                        self.save_graph(final=False)
                        last_save_time = current_time

                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error in update_graph_and_semantics: {str(e)}")
                time.sleep(0.1)

    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return ((pos1.x_val - pos2.x_val) ** 2 + 
                (pos1.y_val - pos2.y_val) ** 2 + 
                (pos1.z_val - pos2.z_val) ** 2) ** 0.5

    def shutdown(self):
        """Quick and forceful shutdown"""
        print("\nInitiating shutdown sequence...")
        
        try:
            # Set all shutdown flags immediately
            self.is_running = False
            self._shutdown_event.set()
            if hasattr(self, 'drone_controller'):
                self.drone_controller.is_running = False
            if hasattr(self, 'detection_visualizer'):
                self.detection_visualizer.is_running = False

            # Quick cleanup of visualization
            cv2.destroyAllWindows()
            
            # Save final graph
            try:
                print("Saving final graph...")
                with self._update_lock:
                    self.save_graph(final=True)
            except Exception as e:
                logging.error(f"Error saving final graph: {e}")

            # Quick landing sequence
            try:
                print("Landing drone...")
                self.client.landAsync()
                time.sleep(1)  # Brief wait
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except Exception as e:
                logging.error(f"Error during landing: {e}")
                # Emergency disarm
                try:
                    self.client.armDisarm(False)
                except:
                    pass

            print("Shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            # Force exit immediately
            os._exit(0)

    def run(self):
        try:
            # Start height maintenance thread
            height_thread = threading.Thread(target=self.maintain_minimum_height)
            height_thread.daemon = True
            self._threads.append(height_thread)
            height_thread.start()
            
            # Create and start update thread
            update_thread = threading.Thread(target=self.update_graph_and_semantics)
            update_thread.daemon = True
            self._threads.append(update_thread)
            update_thread.start()
            
            # Start visualization
            self.detection_visualizer.start()
            
            # Give threads time to start
            time.sleep(1)
            
            # Start drone control (main thread)
            try:
                self.drone_controller.keyboard_control()
            except Exception as e:
                logging.error(f"Error in keyboard control: {e}")
                raise
            
        except Exception as e:
            logging.error(f"Error during run: {e}")
            self.shutdown()

    def save_graph(self, final=False):
        """
        Save the graph with timestamp and environment name
        Args:
            final (bool): If True, indicates this is the final save on program exit
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "final" if final else "checkpoint"
        filename = f"{prefix}_semantic_graph_{ENVIRONMENT_NAME}_{timestamp}.gml"
        filepath = os.path.join(self.output_dir, filename)
        
        logging.info(f"Saving graph with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
        logging.info(f"Saving to: {filepath}")
        
        # Save graph attributes
        self.G.graph['environment'] = ENVIRONMENT_NAME
        self.G.graph['timestamp'] = timestamp
        self.G.graph['detection_radius'] = DETECTION_RADIUS
        
        nx.write_gml(self.G, filepath)
        print(f"Graph saved to {filepath}")

    def __del__(self):
        """Cleanup method"""
        try:
            if hasattr(self, '_shutdown_event') and not self._shutdown_event.is_set():
                self.shutdown()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C more forcefully"""
    print("\nCtrl+C detected. Starting graceful shutdown...")
    try:
        if explorer:
            explorer.shutdown()
    except Exception as e:
        logging.error(f"Error during signal handling: {e}")
    finally:
        # Force exit after timeout
        time.sleep(2)  # Give a chance for logs to be written
        os._exit(0)

if __name__ == "__main__":
    # Allow environment name override from command line
    if len(sys.argv) > 1:
        ENVIRONMENT_NAME = sys.argv[1]
    
    print(f"Starting exploration of {ENVIRONMENT_NAME}")
    print(f"Graphs will be saved in: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'semantic_graphs'))}")
    
    explorer = None
    try:
        explorer = AirSimExplorer()
        # Register both SIGINT and SIGTERM handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        explorer.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if explorer:
            explorer.shutdown()
    finally:
        # Force exit if we get here
        os._exit(0)
