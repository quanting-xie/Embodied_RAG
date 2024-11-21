import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
import asyncio
from tqdm import tqdm
from .config import Config
import re
from sklearn.neighbors import NearestNeighbors
from .llm import LLMInterface
import traceback
from sklearn.cluster import AgglomerativeClustering

class SpatialRelationshipExtractor:
    def __init__(self, llm_interface):
        """Initialize the SpatialRelationshipExtractor with config parameters
        
        Args:
            llm_interface: LLMInterface instance for generating summaries
        """
        self.llm = llm_interface  # Store the whole interface
        
        # Get parameters from config
        self.cluster_distance_threshold = Config.SPATIAL['cluster_distance_threshold']
        self.spatial_threshold = Config.SPATIAL['spatial_threshold']
        self.vertical_threshold = Config.SPATIAL['vertical_threshold']
        self.cardinal_directions = Config.CARDINAL_DIRECTIONS

    def _get_leaf_positions(self, members, G=None, visited=None):
        """Recursively get positions of all leaf nodes in a cluster"""
        if visited is None:
            visited = set()
            
        positions = []
        print("\nDebug _get_leaf_positions:")
        print(f"Processing {len(members)} members")
        
        # Group similar objects
        member_groups = {}
        for member in members:
            if isinstance(member, dict):
                member_id = member.get('id')
                if member_id:
                    # Extract base name without numbers
                    base_name = re.sub(r'\d+$', '', member_id)
                    if base_name not in member_groups:
                        member_groups[base_name] = []
                    member_groups[base_name].append(member)
        
        print(f"Found {len(member_groups)} object groups:")
        for base_name, group in member_groups.items():
            print(f"- {base_name}: {len(group)} objects")
            
        # Process each group
        for base_name, group in member_groups.items():
            print(f"\nProcessing group: {base_name}")
            for member in group:
                member_id = member.get('id')
                member_type = member.get('type')
                
                if member_id is None or member_id in visited:
                    continue
                    
                visited.add(member_id)
                
                if member_type == 'cluster':
                    print(f"Found cluster node: {member_id}")
                    if G and member_id in G:
                        neighbors = list(G.neighbors(member_id))
                        print(f"Cluster {member_id} has neighbors: {neighbors}")
                        
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                node_data = G.nodes[neighbor]
                                child_positions = self._get_leaf_positions(
                                    [{'id': neighbor, **node_data}], 
                                    G, 
                                    visited
                                )
                                positions.extend(child_positions)
                                print(f"Added {len(child_positions)} positions from {neighbor}")
                else:
                    pos = self._get_position(member)
                    if not np.all(pos == 0):
                        positions.append(pos)
                        print(f"Added position for leaf node: {member_id}")
        
        print(f"Total positions found: {len(positions)}")
        return positions

    async def extract_relationships(self, objects):
        print("Extracting relationships...")
        G = nx.Graph()
        
        # Filter and add base objects
        filtered_objects = [
            obj for obj in objects 
            if not (isinstance(obj.get('id', ''), str) and 'drone' in obj.get('id', '').lower())
        ]
        
        print(f"\nProcessing {len(filtered_objects)} objects")
        
        # Get valid positions and add to graph
        valid_positions = []
        valid_objects = []
        
        # Group similar objects by name pattern
        object_groups = {}
        for obj in filtered_objects:
            pos = self._get_position(obj)
            if not np.all(pos == 0):
                # Extract base name without numbers
                base_name = re.sub(r'\d+$', '', obj['id'])
                if base_name not in object_groups:
                    object_groups[base_name] = []
                object_groups[base_name].append(obj)
                valid_positions.append(pos)
                valid_objects.append(obj)
                G.add_node(obj['id'], **obj, level=0)
        
        print("\nObject Groups Found:")
        for base_name, group in object_groups.items():
            print(f"{base_name}: {len(group)} objects")
        
        if len(valid_positions) < 2:
            print("Not enough objects with valid positions for clustering")
            return G
            
        positions_array = np.array(valid_positions)
        current_nodes = valid_objects.copy()
        current_level = 0
        
        # Use exponential scaling for threshold
        base_threshold = self.cluster_distance_threshold
        max_level = 5  # Maximum levels to prevent infinite loops
        
        while len(current_nodes) > 1 and current_level < max_level:  # Continue until single top node
            level = current_level + 1
            print(f"\n=== Processing Level {level} ===")
            
            # Get current positions and IDs for clustering
            current_positions = []
            current_ids = []  # Add this
            position_map = {}  # Add this for tracking positions
            
            for node in current_nodes:
                pos = self._get_position(node)
                if not np.all(pos == 0):
                    current_positions.append(pos)
                    current_ids.append(node['id'])  # Store ID
                    position_map[node['id']] = pos  # Store position mapping
            
            positions_array = np.array(current_positions)
            
            # Calculate level threshold
            level_threshold = self.cluster_distance_threshold * (1 + (level * 1.0))
            print(f"Distance threshold for level {level}: {level_threshold}")
            
            try:
                # Use for clustering
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=level_threshold,
                    metric='euclidean',
                    linkage='complete'
                )
                
                labels = clustering.fit_predict(positions_array)
                
                # Debug clustering results
                unique_labels = set(labels)
                print(f"Found {len(unique_labels)} natural clusters at threshold {level_threshold}")
                
                # Group nodes by cluster
                clusters = {}
                single_nodes = []
                
                # First pass - identify clusters and single nodes
                for idx, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(current_ids[idx])
                
                # Handle single-member clusters
                final_clusters = {}
                for label, members in clusters.items():
                    if len(members) == 1:
                        single_nodes.extend(members)
                    else:
                        final_clusters[label] = members
                
                # If we have single nodes, assign them to nearest clusters
                if single_nodes:
                    print(f"\nAssigning {len(single_nodes)} single nodes to nearest clusters")
                    for node_id in single_nodes:
                        node_pos = position_map[node_id]
                        min_dist = float('inf')
                        best_cluster = None
                        
                        # Find nearest cluster by average distance to members
                        for label, members in final_clusters.items():
                            cluster_positions = [position_map[m] for m in members]
                            avg_pos = np.mean(cluster_positions, axis=0)
                            dist = np.linalg.norm(node_pos - avg_pos)
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_cluster = label
                        
                        if best_cluster is not None and min_dist <= level_threshold:
                            print(f"Adding {node_id} to cluster {best_cluster} (distance: {min_dist:.2f})")
                            final_clusters[best_cluster].append(node_id)
                        else:
                            print(f"Creating new cluster for {node_id} (no nearby clusters found)")
                            new_label = max(final_clusters.keys()) + 1 if final_clusters else 0
                            final_clusters[new_label] = [node_id]
                
                # Process final clusters
                new_nodes = []
                cluster_info = []
                
                print("\nValidating clusters and creating nodes:")
                for label, members in final_clusters.items():
                    print(f"\nProcessing cluster {label}:")
                    print(f"Members: {members}")
                    
                    member_objects = []
                    for member_id in members:
                        if G.has_node(member_id):
                            node_data = G.nodes[member_id]
                            member_objects.append({'id': member_id, **node_data})
                        else:
                            print(f"Warning: Node {member_id} not found in graph")
                    
                    print(f"Found {len(member_objects)} valid member objects")
                    
                    leaf_positions = self._get_leaf_positions(member_objects, G)
                    print(f"Found {len(leaf_positions)} leaf positions")
                    
                    if leaf_positions:
                        center = np.mean(leaf_positions, axis=0)
                        center_dict = {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])}
                        
                        # Create temporary node name
                        temp_name = f"area_{level}_cluster_{label}"
                        print(f"Creating node: {temp_name}")
                        
                        # Create node immediately
                        G.add_node(temp_name,
                                  type='cluster',
                                  level=level,
                                  position=center_dict)
                        
                        # Add edges to members
                        for member_id in members:
                            G.add_edge(member_id, temp_name, relationship='part_of')
                        
                        new_nodes.append({
                            'id': temp_name,
                            'type': 'cluster',
                            'level': level,
                            'position': center_dict
                        })
                        
                        # Save info for summary generation
                        cluster_info.append({
                            'node_name': temp_name,
                            'member_data': member_objects,
                        })
                else:
                    print("No valid leaf positions found - skipping cluster")

                print(f"\nCreated {len(new_nodes)} new nodes")
                
                # Add spatial relationships for base objects (level 0) first time
                if level == 1:
                    await self._add_positional_relationships(G, 0, self.spatial_threshold)
                
                # Process clusters and add their relationships
                if new_nodes:
                    # Add spatial relationships for the current level
                    await self._add_positional_relationships(G, level, self.spatial_threshold)
                    
                    # Generate summaries after relationships are added
                    if cluster_info:
                        summary_tasks = [
                            self.llm.generate_community_summary(info['member_data'])
                            for info in cluster_info
                        ]
                        area_infos = await asyncio.gather(*summary_tasks)
                        # Update existing nodes with summaries
                        for info, area_info in zip(cluster_info, area_infos):
                            G.nodes[info['node_name']].update({
                                'name': area_info['name'],
                                'summary': area_info['summary']
                            })
                
                if len(new_nodes) == 0:
                    print("No valid clusters formed")
                    break
                
                current_nodes = new_nodes
                current_level = level
                print(f"Created {len(new_nodes)} clusters at level {level}")
            except Exception as e:
                print(f"Clustering failed at level {level}: {str(e)}")
                traceback.print_exc()
                break
        
        return G

    def _get_position(self, obj):
        """Extract position from object data with better error handling"""
        try:
            if isinstance(obj, dict):
                pos = obj.get('position')
                if pos is None:
                    print(f"Warning: No position found for object {obj.get('id')}")
                    return np.array([0, 0, 0])
                
                # Handle dictionary format {'x': val, 'y': val, 'z': val}
                if isinstance(pos, dict) and all(k in pos for k in ['x', 'y', 'z']):
                    return np.array([pos['x'], pos['y'], pos['z']])
                
                # Handle string format
                if isinstance(pos, str):
                    try:
                        # Try to parse string format "[x, y, z]"
                        pos = eval(pos.replace('[', '').replace(']', ''))
                    except:
                        print(f"Warning: Could not parse position string {pos} for object {obj.get('id')}")
                        return np.array([0, 0, 0])
                
                # Handle list/tuple format
                if isinstance(pos, (list, tuple)) and len(pos) == 3:
                    return np.array(pos)
                
                print(f"Warning: Invalid position format {pos} for object {obj.get('id')}")
                return np.array([0, 0, 0])
                
            return np.array([0, 0, 0])
            
        except Exception as e:
            print(f"Error extracting position for object {obj.get('id', 'unknown')}: {str(e)}")
            return np.array([0, 0, 0])

    def _get_cardinal_direction(self, pos1, pos2):
        """Convert relative positions to cardinal directions"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        
        angle = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle)
        
        if -45 <= angle_deg <= 45:
            horizontal = "east"
        elif 45 < angle_deg <= 135:
            horizontal = "north"
        elif -135 <= angle_deg < -45:
            horizontal = "south"
        else:
            horizontal = "west"
            
        vertical = ""
        if abs(dz) > self.vertical_threshold:
            vertical = "above" if dz > 0 else "below"
            
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        return horizontal, vertical, distance

    async def _add_positional_relationships(self, G, current_level, level_threshold):
        """Add spatial relationships based on provided threshold"""
        print(f"Adding spatial relationships for level {current_level}")
        
        # Get ONLY nodes at the current level (remove children)
        level_nodes = []
        for node, data in G.nodes(data=True):
            if data.get('level') == current_level:  # Only include nodes at current level
                level_nodes.append((node, data))
        
        print(f"Processing {len(level_nodes)} nodes at level {current_level} with spatial threshold {level_threshold:.2f}")
        
        # Track added relationships for debugging
        added_relationships = 0
        
        for i, (node1, data1) in enumerate(level_nodes):
            pos1 = self._get_position(data1)
            if np.all(pos1 == 0):
                continue
                
            for node2, data2 in level_nodes[i+1:]:  # Start from i+1 to avoid duplicates
                if node1 != node2:
                    pos2 = self._get_position(data2)
                    if np.all(pos2 == 0):
                        continue
                        
                    # Calculate actual distance
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    # Use spatial_threshold from config instead of level_threshold
                    if distance <= Config.SPATIAL['spatial_threshold']:
                        horizontal, vertical, _ = self._get_cardinal_direction(pos1, pos2)
                        
                        relationship = horizontal
                        if vertical:
                            relationship += f"_{vertical}"
                        
                        # Only add if not already connected
                        if not G.has_edge(node1, node2):
                            G.add_edge(node1, node2, 
                                     relationship=relationship,
                                     type='spatial',
                                     cardinal_direction=relationship,
                                     distance=float(distance))  # Convert to float for serialization
                            added_relationships += 1
                            print(f"Added spatial relationship: {node1} is {relationship} of {node2} ({distance:.2f}m)")
        
        print(f"Added {added_relationships} spatial relationships at level {current_level}")

