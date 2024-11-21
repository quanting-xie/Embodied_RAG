import airsim
import networkx as nx
from datetime import datetime
from pathlib import Path

class DirectSceneLogger:
    def __init__(self, environment_name="Building99"):
        self.environment_name = environment_name
        self.G = nx.Graph()
        self.filter_words = [
            "floor", "ceiling", "ground",
            "SpotLight", "spotlight", "PointLight",
            "camera", "Camera", "cam", "Cam",
            "light_cam", "camLight", "CameraRig",
            "CameraSystem", "CameraMount"
        ]
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
    def get_scene_objects(self):
        """Get all objects in the scene using AirSim's object API"""
        try:
            # Get all object names in the scene
            object_names = self.client.simListSceneObjects()
            print(f"\nTotal objects in scene before filtering: {len(object_names)}")
            
            # Filter objects
            filtered_objects = []
            filtered_out = {}  # Dictionary to count filtered objects by type
            
            for obj_name in object_names:
                # Check which filter word matched
                filtered = False
                for filter_word in self.filter_words:
                    if filter_word.lower() in obj_name.lower():
                        filtered_out[filter_word] = filtered_out.get(filter_word, 0) + 1
                        filtered = True
                        break
                
                if filtered:
                    continue
                    
                # Get object pose to verify it exists and has a valid position
                pose = self.client.simGetObjectPose(obj_name)
                if pose and pose.position != airsim.Vector3r():
                    filtered_objects.append(obj_name)
            
            # Print statistics
            print("\nFiltering Statistics:")
            print("-" * 50)
            for filter_word, count in filtered_out.items():
                print(f"Filtered out {count} objects containing '{filter_word}'")
            print(f"Objects remaining after filtering: {len(filtered_objects)}")
            print("-" * 50)
            
            return filtered_objects
            
        except Exception as e:
            print(f"Error getting scene objects: {e}")
            return []
            
    def get_object_position(self, object_name):
        """Get object position using AirSim's pose API with inverted z-axis"""
        try:
            pose = self.client.simGetObjectPose(object_name)
            if pose and pose.position != airsim.Vector3r():
                position = {
                    'x': float(pose.position.x_val),
                    'y': float(pose.position.y_val),
                    'z': float(-pose.position.z_val)  # Invert z-axis
                }
                return position
            return None
        except Exception as e:
            print(f"Error getting position for {object_name}: {e}")
            return None
            
    def build_topological_graph(self):
        """Build graph of objects"""
        # Get all objects
        objects = self.get_scene_objects()
        
        # Add each object to the graph
        for obj in objects:
            position = self.get_object_position(obj)
            if position:
                self.G.add_node(obj,
                              position=position,
                              type='object',
                              level=1,
                              name=obj,
                              label=obj,
                              summary=f"Object: {obj}")
        
        return self.G
    
    def save_graph(self):
        """Save the graph to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"direct_semantic_graph_{self.environment_name}_{timestamp}.gml"
        filepath = Path("semantic_graphs") / filename
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(exist_ok=True)
        
        # Add graph attributes
        self.G.graph['environment'] = self.environment_name
        self.G.graph['timestamp'] = timestamp
        self.G.graph['creation_method'] = 'direct_scene_logger'
        
        # Save graph
        nx.write_gml(self.G, filepath)
        return filepath

def main():
    """Main function to run the direct scene logger"""
    import argparse
    parser = argparse.ArgumentParser(description='Direct Scene Logger for AirSim')
    parser.add_argument('--env', type=str, default="Building99",
                      help='Environment name')
    args = parser.parse_args()
    
    try:
        logger = DirectSceneLogger(environment_name=args.env)
        logger.build_topological_graph()
        graph_path = logger.save_graph()
        print(f"\nGraph saved to:\n{graph_path}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()