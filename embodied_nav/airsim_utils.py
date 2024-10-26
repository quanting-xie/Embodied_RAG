import airsim
import time
import threading
from pynput import keyboard
import logging
import math
import cv2
import numpy as np
from queue import Queue, Empty

class AirSimUtils:
    def __init__(self, client):
        self.client = client

    @staticmethod
    def vector3r_to_dict(vector):
        return {"x": vector.x_val, "y": vector.y_val, "z": vector.z_val}

    @staticmethod
    def local_to_global_position(drone_pose, local_position):
        q = drone_pose.orientation
        R = airsim.to_eularian_angles(q)
        
        rotated_position = airsim.Vector3r(
            local_position.x_val * math.cos(R[2]) - local_position.y_val * math.sin(R[2]),
            local_position.x_val * math.sin(R[2]) + local_position.y_val * math.cos(R[2]),
            local_position.z_val
        )
        
        global_position = airsim.Vector3r(
            rotated_position.x_val + drone_pose.position.x_val,
            rotated_position.y_val + drone_pose.position.y_val,
            rotated_position.z_val + drone_pose.position.z_val
        )
        
        return global_position

    def generate_waypoints(self, start_position, target_position, planning_mode="direct"):

        # Convert dict positions to Vector3r if necessary
        if isinstance(start_position, dict):
            start_position = airsim.Vector3r(
                start_position['x'],
                start_position['y'],
                start_position['z']
            )
        if isinstance(target_position, dict):
            target_position = airsim.Vector3r(
                target_position['x'],
                target_position['y'],
                target_position['z']
            )

        if planning_mode == "direct":
            # Simple direct path - just start and end points
            return [start_position, target_position]
        
        elif planning_mode == "astar":
            try:
                # Use AirSim's path planner
                path = self.client.moveOnPathAsync(
                    [start_position, target_position],
                    velocity=2,
                    timeout_sec=30,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(False, 0),
                    lookahead=5,
                    adaptive_lookahead=True
                ).join()
                
                return path
            except Exception as e:
                logging.error(f"Path planning failed: {str(e)}")
                # Fallback to direct path if planning fails
                return [start_position, target_position]
        
        else:
            raise ValueError(f"Unknown planning mode: {planning_mode}")

class DroneController:
    def __init__(self, client, speed=2, yaw_rate=45, vertical_speed=2):
        self.client = client
        self.speed = speed
        self.yaw_rate = yaw_rate
        self.vertical_speed = vertical_speed
        self.current_keys = set()
        self.is_running = True

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
        while self.is_running:
            try:
                vx = vy = vz = 0
                yaw_rate = 0

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

                self.client.moveByVelocityBodyFrameAsync(
                    vx, vy, vz, 0.1,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
                )

                time.sleep(0.05)
            except Exception as e:
                logging.error(f"Error in move_drone: {str(e)}")
                time.sleep(1)

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
