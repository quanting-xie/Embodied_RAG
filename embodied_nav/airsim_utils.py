import airsim
import time
import threading
from pynput import keyboard
import logging
import math
import cv2
import numpy as np
from queue import Queue, Empty
import networkx as nx

class AirSimUtils:
    def __init__(self, client, graph=None):
        self.client = client
        self.graph = graph
        self.safe_distance = 2.0  # meters
        
    @staticmethod
    def vector3r_to_dict(vector3r):
        """Convert AirSim Vector3r to dictionary"""
        return {
            'x': float(vector3r.x_val),
            'y': float(vector3r.y_val),
            'z': float(vector3r.z_val)
        }

    @staticmethod
    def dict_to_vector3r(pos_dict):
        """Convert dictionary to AirSim Vector3r"""
        return airsim.Vector3r(
            float(pos_dict['x']),
            float(pos_dict['y']),
            float(pos_dict['z'])
        )

    @staticmethod
    def local_to_global_position(drone_pose, local_pos):
        """Convert local position to global position"""
        try:
            # Get drone's position
            drone_pos = drone_pose.position
            
            # Get drone's orientation quaternion
            q = drone_pose.orientation
            # Create rotation matrix from quaternion
            # Using simplified rotation for now
            yaw = np.arctan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                            1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
            
            # Create rotation matrix for yaw
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            # Convert local position to numpy array
            local_point = np.array([
                local_pos.x_val,
                local_pos.y_val,
                local_pos.z_val
            ])
            
            # Apply rotation and translation
            global_point = np.dot(R, local_point) + np.array([
                drone_pos.x_val,
                drone_pos.y_val,
                drone_pos.z_val
            ])
            
            return airsim.Vector3r(global_point[0], global_point[1], global_point[2])
        except Exception as e:
            print(f"Error in local_to_global_position: {e}")
            # Return original position if conversion fails
            return local_pos

    def find_path_through_drone_nodes(self, start_pos, target_pos):
        """Find a path through previously explored drone nodes"""
        if not self.graph:
            print("No graph available for path planning")
            return None
            
        # Convert positions to numpy arrays for distance calculations
        start_np = np.array([start_pos.x_val, start_pos.y_val, start_pos.z_val])
        target_np = np.array([target_pos.x_val, target_pos.y_val, target_pos.z_val])
        
        # Find nearest drone nodes to start and target
        start_node = None
        end_node = None
        min_start_dist = float('inf')
        min_end_dist = float('inf')
        
        drone_nodes = [(node, data) for node, data in self.graph.nodes(data=True) 
                      if data.get('type') == 'drone']
        print(f"Found {len(drone_nodes)} drone nodes in graph")
        
        for node, data in drone_nodes:
            if 'position' in data:
                pos = data['position']
                if isinstance(pos, dict):
                    node_pos = np.array([pos['x'], pos['y'], pos['z']])
                    
                    dist_to_start = np.linalg.norm(node_pos - start_np)
                    dist_to_end = np.linalg.norm(node_pos - target_np)
                    
                    if dist_to_start < min_start_dist:
                        min_start_dist = dist_to_start
                        start_node = node
                        
                    if dist_to_end < min_end_dist:
                        min_end_dist = dist_to_end
                        end_node = node
        
        if not (start_node and end_node):
            print("Could not find suitable start/end nodes in graph")
            return None
            
        print(f"Selected start node: {start_node}, end node: {end_node}")
        
        try:
            # Find shortest path through drone nodes
            path = nx.shortest_path(self.graph, start_node, end_node, weight='distance')
            return path
        except nx.NetworkXNoPath:
            return None
            
    def direct_to_waypoint(self, target_position, velocity=5):
        """Move to target position using single intermediate waypoint for safety"""
        print("\n=== Starting Movement Process ===")
        
        # Convert target position to Vector3r if needed
        if isinstance(target_position, (list, tuple)):
            target_position = {
                'x': float(target_position[0]),
                'y': float(target_position[1]),
                'z': -float(target_position[2])  # Invert Z for AirSim
            }
        if isinstance(target_position, dict):
            target_position = {
                'x': float(target_position['x']),
                'y': float(target_position['y']),
                'z': -float(target_position['z'])  # Invert Z for AirSim
            }
            target_position = self.dict_to_vector3r(target_position)

        print(f"\nTarget destination: ({target_position.x_val:.2f}, {target_position.y_val:.2f}, {target_position.z_val:.2f})")
        
        try:
            # Get current position
            current_pose = self.client.simGetVehiclePose()
            print(f"\nCurrent position: ({current_pose.position.x_val:.2f}, {current_pose.position.y_val:.2f}, {current_pose.position.z_val:.2f})")
            
            # Use single intermediate waypoint at safe height
            safe_height = -2.0  # Negative for up in AirSim
            print(f"\nMoving to safe height: {safe_height}")
            
            # First move to safe height above current position
            self.client.moveToPositionAsync(
                current_pose.position.x_val,
                current_pose.position.y_val,
                safe_height,
                velocity
            ).join()
            
            # Then move to target position
            print("\n=== Moving to target position ===")
            print(f"Final position: ({target_position.x_val:.2f}, {target_position.y_val:.2f}, {target_position.z_val:.2f})")
            
            self.client.moveToPositionAsync(
                target_position.x_val,
                target_position.y_val,
                target_position.z_val,
                velocity
            ).join()
            
            print("\nMovement completed successfully!")
            return True
            
        except Exception as e:
            if "IOLoop" not in str(e):
                print(f"Error during movement: {e}")
            return False

    def get_position_from_node(self, node_data):
        """Extract position from node data"""
        if 'position' in node_data:
            pos = node_data['position']
            if isinstance(pos, dict):
                return pos
            elif isinstance(pos, (list, tuple)):
                return {
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'z': float(pos[2])
                }
            elif isinstance(pos, airsim.Vector3r):
                return self.vector3r_to_dict(pos)
        print(f"Warning: Could not extract position from node data: {node_data}")
        return None

    def direct_to_position(self, target_position, velocity=5):
        """Directly move to target position in a single step"""
        print("\n=== Starting Direct Movement ===")
        
        # Convert target position to Vector3r if needed
        if isinstance(target_position, (list, tuple)):
            target_position = {
                'x': float(target_position[0]),
                'y': float(target_position[1]),
                'z': -float(target_position[2])  # Invert Z for AirSim
            }
        if isinstance(target_position, dict):
            target_position = {
                'x': float(target_position['x']),
                'y': float(target_position['y']),
                # 'z': -float(target_position['z'])  # Invert Z for AirSim
                'z': 0.2 # Safe hover height
            }
            target_position = self.dict_to_vector3r(target_position)

        print(f"\nTarget destination: ({target_position.x_val:.2f}, {target_position.y_val:.2f}, {target_position.z_val:.2f})")
        
        try:
            # Single step: Move directly to target position
            self.client.moveToPositionAsync(
                target_position.x_val,
                target_position.y_val,
                target_position.z_val,
                velocity
            ).join()
            
            print("\nMovement completed successfully!")
            return True
                
        except Exception as e:
            if "IOLoop" not in str(e):
                print(f"Error during movement: {e}")
            return False

class AirSimClientWrapper:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self._lock = threading.Lock()
        
    def __getattr__(self, name):
        original_attr = getattr(self.client, name)
        if callable(original_attr):
            def wrapped_function(*args, **kwargs):
                with self._lock:
                    try:
                        return original_attr(*args, **kwargs)
                    except Exception as e:
                        if "IOLoop" not in str(e):
                            logging.error(f"AirSim client error in {name}: {str(e)}")
                        # Attempt to reconnect if connection lost
                        self.reconnect()
                        # Retry once after reconnection
                        return original_attr(*args, **kwargs)
            return wrapped_function
        return original_attr
    
    def reconnect(self):
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
        except Exception as e:
            if "IOLoop" not in str(e):
                logging.error(f"Failed to reconnect: {e}")

class DroneController:
    def __init__(self, client, speed=2, yaw_rate=45, vertical_speed=2):
        self.client = client
        self.speed = speed
        self.yaw_rate = yaw_rate
        self.vertical_speed = vertical_speed
        self.current_keys = set()
        self.is_running = True
        self._move_lock = threading.Lock()

    def on_press(self, key):
        try:
            self.current_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.current_keys.remove(key.char)
        except KeyError:
            pass
        except AttributeError:
            if key == keyboard.Key.esc:
                self.is_running = False

    def move_drone(self):
        last_command_time = time.time()
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - last_command_time < 0.05:  # Limit command rate
                    time.sleep(0.01)
                    continue

                vx = vy = vz = 0
                yaw_rate = 0

                with self._move_lock:
                    if 'w' in self.current_keys:
                        vx = self.speed
                    if 's' in self.current_keys:
                        vx = -self.speed
                    if 'q' in self.current_keys:
                        vz = -self.vertical_speed
                    if 'e' in self.current_keys:
                        vz = self.vertical_speed
                    if 'a' in self.current_keys:
                        yaw_rate = -self.yaw_rate
                    if 'd' in self.current_keys:
                        yaw_rate = self.yaw_rate

                    # Add slight upward force for stability
                    if vz == 0:
                        vz = -0.1

                    self.client.moveByVelocityBodyFrameAsync(
                        vx, vy, vz, 0.1,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
                    )
                    last_command_time = current_time

                time.sleep(0.01)
            except Exception as e:
                if "IOLoop" not in str(e):
                    logging.error(f"Error in move_drone: {str(e)}")
                time.sleep(0.1)

    def keyboard_control(self):
        print("Tele-operation started. Use WASD to move, QE to ascend/descend. Press ESC to exit.")
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        move_thread = threading.Thread(target=self.move_drone)
        move_thread.start()

        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping the drone...")
        finally:
            self.is_running = False
            listener.stop()
            move_thread.join()

class DetectionVisualizer:
    def __init__(self, client, camera_name="0", image_type=airsim.ImageType.Scene):
        self.client = client
        self.camera_name = camera_name
        self.image_type = image_type
        self.filter_words = ["floor", "ceiling"]
        self.is_running = True
        self.image_queue = Queue(maxsize=1)
        self.capture_thread = None
        self.display_thread = None

    def capture_images(self):
        while self.is_running:
            try:
                rawImage = self.client.simGetImage(self.camera_name, self.image_type)
                if rawImage:
                    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
                    detections = self.client.simGetDetections(self.camera_name, self.image_type)
                    
                    # Put the new image and detections in the queue, replacing old ones if necessary
                    if self.image_queue.full():
                        self.image_queue.get()
                    self.image_queue.put((png, detections))
                
                time.sleep(0.1)  # Capture at 10 Hz
            except Exception as e:
                logging.error(f"Error in capture_images: {str(e)}")
                time.sleep(1)

    def display_images(self):
        while self.is_running:
            try:
                png, detections = self.image_queue.get(timeout=1)  # Wait up to 1 second for a new image
                
                if detections:
                    for detection in detections:
                        if any(word in detection.name.lower() for word in self.filter_words):
                            continue

                        cv2.rectangle(png,
                                      (int(detection.box2D.min.x_val), int(detection.box2D.min.y_val)),
                                      (int(detection.box2D.max.x_val), int(detection.box2D.max.y_val)),
                                      (255, 0, 0), 2)
                        
                        cv2.putText(png, detection.name,
                                    (int(detection.box2D.min.x_val), int(detection.box2D.min.y_val - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

                cv2.imshow("AirSim Detections", png)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
            except Empty:
                continue  # If no image is available, just continue the loop
            except Exception as e:
                logging.error(f"Error in display_images: {str(e)}")
                time.sleep(1)

    def start(self):
        self.capture_thread = threading.Thread(target=self.capture_images)
        self.display_thread = threading.Thread(target=self.display_images)
        self.capture_thread.start()
        self.display_thread.start()

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.display_thread:
            self.display_thread.join()
        cv2.destroyAllWindows()