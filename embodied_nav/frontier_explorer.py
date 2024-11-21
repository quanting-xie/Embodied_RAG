import airsim
import numpy as np
import threading
import time
import asyncio
from queue import Queue
import logging
from .config import Config
from .airsim_utils import AirSimClientWrapper

class FrontierExplorer:
    def __init__(self, client):
        self.client = client
        self.is_exploring = False
        self.frontier_points = []
        self.explored_areas = set()
        self.current_path = []
        self.safe_height = -2.0  # 2 meters above ground
        self.grid_resolution = 1.0  # 1 meter grid cells
        self.min_frontier_size = 3  # Minimum points to consider a frontier
        self.exploration_radius = 50  # meters
        self.obstacle_threshold = 1.0  # meters
        self.visualization_queue = Queue()
        
        # Configure LiDAR
        self.setup_lidar()
        
        # Start visualization thread
        self.visualization_thread = threading.Thread(target=self.visualize_exploration)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()

    def setup_lidar(self):
        """Configure LiDAR settings"""
        try:
            lidar_settings = airsim.LidarSettings()
            lidar_settings.number_of_channels = 16
            lidar_settings.points_per_second = 100000
            lidar_settings.horizontal_FOV_degrees = 360
            lidar_settings.vertical_FOV_degrees = 45
            lidar_settings.range = self.exploration_radius
            lidar_settings.rotation_frequency = 10
            
            self.client.simAddLidarSensor("Lidar1", lidar_settings, airsim.Pose(airsim.Vector3r(0, 0, 0)))
            print("LiDAR sensor configured successfully")
        except Exception as e:
            print(f"Error setting up LiDAR: {str(e)}")

    def get_lidar_data(self):
        """Get and process LiDAR data"""
        try:
            lidar_data = self.client.getLidarData(lidar_name="Lidar1")
            
            if len(lidar_data.point_cloud) < 3:
                return None
                
            # Convert point cloud to numpy array
            points = np.array(lidar_data.point_cloud).reshape((-1, 3))
            
            # Transform points to world frame
            drone_pose = self.client.simGetVehiclePose()
            points = self.transform_points(points, drone_pose)
            
            return points
        except Exception as e:
            print(f"Error getting LiDAR data: {str(e)}")
            return None

    def transform_points(self, points, drone_pose):
        """Transform points from LiDAR frame to world frame"""
        # Create rotation matrix from quaternion
        q = drone_pose.orientation
        R = np.array([
            [1 - 2*(q.y_val**2 + q.z_val**2), 2*(q.x_val*q.y_val - q.z_val*q.w_val), 2*(q.x_val*q.z_val + q.y_val*q.w_val)],
            [2*(q.x_val*q.y_val + q.z_val*q.w_val), 1 - 2*(q.x_val**2 + q.z_val**2), 2*(q.y_val*q.z_val - q.x_val*q.w_val)],
            [2*(q.x_val*q.z_val - q.y_val*q.w_val), 2*(q.y_val*q.z_val + q.x_val*q.w_val), 1 - 2*(q.x_val**2 + q.y_val**2)]
        ])
        
        # Apply rotation and translation
        points = np.dot(points, R.T)
        points += np.array([
            drone_pose.position.x_val,
            drone_pose.position.y_val,
            drone_pose.position.z_val
        ])
        
        return points

    def update_frontiers(self, points):
        """Update frontier points based on LiDAR data"""
        try:
            # Convert points to grid cells
            grid_cells = set(map(self.discretize_position, points))
            
            # Find frontier candidates
            frontiers = []
            for cell in grid_cells:
                # Check if cell is at the boundary of explored space
                neighbors = self.get_neighbors(cell)
                unexplored_neighbors = [n for n in neighbors if n not in grid_cells]
                
                if len(unexplored_neighbors) > 0 and cell not in self.explored_areas:
                    frontiers.append(cell)
            
            # Cluster frontiers and filter small clusters
            if frontiers:
                clusters = self.cluster_frontiers(frontiers)
                valid_clusters = [c for c in clusters if len(c) >= self.min_frontier_size]
                
                # Update frontier points with cluster centers
                self.frontier_points = [self.get_cluster_center(c) for c in valid_clusters]
                
            # Update visualization
            self.visualization_queue.put({
                'points': points,
                'frontiers': self.frontier_points,
                'explored': list(self.explored_areas)
            })
            
        except Exception as e:
            print(f"Error updating frontiers: {str(e)}")

    def cluster_frontiers(self, frontiers, cluster_threshold=2.0):
        """Cluster frontier points"""
        clusters = []
        visited = set()
        
        for point in frontiers:
            if point in visited:
                continue
                
            # Start new cluster
            cluster = {point}
            queue = [point]
            visited.add(point)
            
            # Grow cluster
            while queue:
                current = queue.pop(0)
                neighbors = self.get_neighbors(current)
                
                for neighbor in neighbors:
                    if neighbor in frontiers and neighbor not in visited:
                        cluster.add(neighbor)
                        queue.append(neighbor)
                        visited.add(neighbor)
            
            clusters.append(cluster)
        
        return clusters

    def get_cluster_center(self, cluster):
        """Calculate center point of a cluster"""
        points = np.array(list(cluster))
        return tuple(np.mean(points, axis=0))

    async def explore(self):
        """Main exploration loop"""
        self.is_exploring = True
        print("\nStarting frontier exploration...")
        
        try:
            while self.is_exploring:
                # Get current position
                drone_pose = self.client.simGetVehiclePose()
                current_pos = (
                    drone_pose.position.x_val,
                    drone_pose.position.y_val,
                    drone_pose.position.z_val
                )
                
                # Get and process LiDAR data
                points = self.get_lidar_data()
                if points is not None:
                    self.update_frontiers(points)
                    
                    # Mark current area as explored
                    self.explored_areas.add(self.discretize_position(current_pos))
                    
                    # Choose and move to next frontier
                    if self.frontier_points:
                        next_frontier = self.choose_next_frontier(current_pos)
                        if next_frontier:
                            print(f"\nMoving to frontier: {next_frontier}")
                            await self.move_to_frontier(next_frontier)
                    else:
                        print("\nNo more frontiers to explore!")
                        self.is_exploring = False
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"Error during exploration: {str(e)}")
            self.is_exploring = False

    def choose_next_frontier(self, current_pos):
        """Choose next frontier point based on distance and information gain"""
        if not self.frontier_points:
            return None
            
        # Find closest unexplored frontier
        distances = [self.calculate_distance(current_pos, f) for f in self.frontier_points]
        closest_idx = np.argmin(distances)
        
        return self.frontier_points[closest_idx]

    async def move_to_frontier(self, frontier):
        """Move drone to frontier point"""
        try:
            # Always maintain safe height
            target_pos = (frontier[0], frontier[1], self.safe_height)
            
            print(f"Moving to position: {target_pos}")
            
            # Move to position
            self.client.moveToPositionAsync(
                target_pos[0], target_pos[1], target_pos[2],
                velocity=2,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
            ).join()
            
            return True
        except Exception as e:
            print(f"Error moving to frontier: {str(e)}")
            return False

    @staticmethod
    def discretize_position(pos):
        """Convert continuous position to discrete grid cell"""
        if isinstance(pos, (tuple, list, np.ndarray)):
            return (int(pos[0]), int(pos[1]), int(pos[2]))
        return (int(pos.x_val), int(pos.y_val), int(pos.z_val))

    @staticmethod
    def get_neighbors(cell):
        """Get neighboring grid cells"""
        x, y, z = cell
        return [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z)
        ]

    @staticmethod
    def calculate_distance(pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def stop_exploration(self):
        """Stop the exploration process"""
        self.is_exploring = False
        print("\nStopping exploration...")

    def visualize_exploration(self):
        """Visualize exploration progress (implement visualization as needed)"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            while True:
                try:
                    data = self.visualization_queue.get(timeout=1)
                    
                    ax.clear()
                    
                    # Plot LiDAR points
                    if 'points' in data and len(data['points']) > 0:
                        points = np.array(data['points'])
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)
                    
                    # Plot frontiers
                    if 'frontiers' in data and len(data['frontiers']) > 0:
                        frontiers = np.array(data['frontiers'])
                        ax.scatter(frontiers[:, 0], frontiers[:, 1], frontiers[:, 2], c='r', s=100)
                    
                    # Plot explored areas
                    if 'explored' in data and len(data['explored']) > 0:
                        explored = np.array(data['explored'])
                        ax.scatter(explored[:, 0], explored[:, 1], explored[:, 2], c='g', s=10)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.draw()
                    plt.pause(0.01)
                    
                except Queue.Empty:
                    continue
                
        except Exception as e:
            print(f"Error in visualization: {str(e)}")