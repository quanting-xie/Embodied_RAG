import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
import asyncio
from tqdm import tqdm
from .config import Config
import re
from sklearn.neighbors import NearestNeighbors
from .llm import LLMInterface

class SpatialRelationshipExtractor:
    def __init__(self, llm_interface):
        """Initialize the SpatialRelationshipExtractor with config parameters
        
        Args:
            llm_interface: LLMInterface instance for generating summaries
        """
        self.llm = llm_interface  # Store the whole interface
        
        # Get parameters from config
        self.cluster_distance_threshold = Config.SPATIAL['cluster_distance_threshold']
        self.proximity_threshold = Config.SPATIAL['proximity_threshold']
        self.spatial_threshold = Config.SPATIAL['spatial_threshold']
        self.vertical_threshold = Config.SPATIAL['vertical_threshold']
        self.cardinal_directions = Config.CARDINAL_DIRECTIONS

    async def extract_relationships(self, objects):
        print("Extracting relationships...")
        
        # Filter out drone nodes BEFORE creating the graph
        filtered_objects = [
            obj for obj in objects 
            if not (isinstance(obj.get('id', ''), str) and 'drone' in obj.get('id', '').lower())
        ]
        
        # Create a graph with only non-drone nodes
        G = nx.Graph()
        
        # Add filtered nodes to graph
        for obj in tqdm(filtered_objects, desc="Adding object nodes"):
            G.add_node(obj['id'], **obj, level=0)
            
        # Add drone nodes separately (they will be isolated)
        drone_objects = [obj for obj in objects if isinstance(obj.get('id', ''), str) and 'drone' in obj.get('id', '').lower()]
        for drone in drone_objects:
            G.add_node(drone['id'], **drone, level=0)
        
        print(f"Added {len(filtered_objects)} regular nodes and {len(drone_objects)} drone nodes")
        
        # Process relationships using filtered objects only
        positions = []
        for obj in filtered_objects:
            pos = self._get_position(obj)
            positions.append(pos)
        
        positions = np.array(positions)
        Z = linkage(positions, 'ward')
        base_clusters = fcluster(Z, self.cluster_distance_threshold, criterion='distance')
        
        await self._process_base_clusters(G, filtered_objects, base_clusters)
        await self._process_higher_level_clusters(G)
        await self._add_positional_relationships(G, filtered_objects)
        self._add_proximity_relationships(G, filtered_objects)

        return G

    async def _process_base_clusters(self, G, objects, clusters):
        """Process first level clusters based on spatial proximity"""
        print("\nProcessing base level clusters...")
        cluster_members = {}
        
        # Group objects by cluster
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_members:
                cluster_members[cluster_id] = []
            cluster_members[cluster_id].append(objects[i])
        
        # Process each cluster
        for cluster_id, members in cluster_members.items():
            if len(members) > 1:  # Only create area nodes for clusters with multiple members
                # Calculate average position
                member_positions = [self._get_position(obj) for obj in members]
                avg_position = np.mean(member_positions, axis=0)

                # Generate area summary
                area_info = await self._create_community_summary(members)
                node_name = f"area_1_{area_info['name'].lower().replace(' ', '_')}"

                # Add area node
                G.add_node(node_name,
                          position=avg_position.tolist(),
                          level=1,
                          name=area_info['name'],
                          summary=area_info['summary'])

                # Add edges to members
                for member in members:
                    G.add_edge(member['id'], node_name, relationship="part_of")

    async def _process_higher_level_clusters(self, G):
        """Process higher level clusters using KNN"""
        print("\nProcessing higher level clusters...")
        current_level = 1
        
        while True:
            # Get nodes from current level
            current_nodes = [
                (node, data) for node, data in G.nodes(data=True)
                if data.get('level', 0) == current_level
            ]
            
            if len(current_nodes) < 2:  # Stop if less than 2 nodes at current level
                break
                
            # Get positions for current level nodes
            positions = []
            node_ids = []
            for node, data in current_nodes:
                if isinstance(data['position'], list):
                    pos = data['position']
                else:
                    pos = [data['position'].get('x', 0), 
                          data['position'].get('y', 0), 
                          data['position'].get('z', 0)]
                positions.append(pos)
                node_ids.append(node)
            
            positions = np.array(positions)
            
            # Use KNN to find clusters (minimum 2 nodes per cluster)
            n_neighbors = min(3, len(positions))  # Use at most 3 neighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(positions[:, :2])  # Only use x,y for clustering
            distances, indices = nbrs.kneighbors(positions[:, :2])
            
            # Create clusters based on mutual nearest neighbors
            clusters = []
            used_nodes = set()
            
            for i in range(len(positions)):
                if i in used_nodes:
                    continue
                    
                cluster = {i}
                for j in indices[i][1:]:  # Skip first index (self)
                    if i in indices[j][1:]:  # Mutual nearest neighbor
                        cluster.add(j)
                        used_nodes.add(j)
                
                if len(cluster) > 1:  # Only keep clusters with multiple nodes
                    clusters.append(cluster)
                    used_nodes.add(i)
            
            if not clusters:  # No more clusters formed
                break
            
            # Create next level nodes
            next_level = current_level + 1
            for cluster in clusters:
                member_nodes = [node_ids[i] for i in cluster]
                member_data = [G.nodes[node] for node in member_nodes]
                
                # Calculate average position from original member positions
                member_positions = []
                for data in member_data:
                    if isinstance(data['position'], list):
                        pos = data['position']
                    else:
                        pos = [data['position'].get('x', 0), 
                              data['position'].get('y', 0), 
                              data['position'].get('z', 0)]
                    member_positions.append(pos)
                
                # Calculate average position WITHOUT adding height increment
                avg_position = np.mean(member_positions, axis=0)
                
                # Generate area summary
                area_info = await self._create_community_summary(member_data)
                node_name = f"area_{next_level}_{area_info['name'].lower().replace(' ', '_')}"
                
                print(f"Creating level {next_level} node '{node_name}' at position {avg_position.tolist()}")
                print(f"Members: {member_nodes}")
                
                # Add area node
                G.add_node(node_name,
                          position=avg_position.tolist(),
                          level=next_level,
                          name=area_info['name'],
                          summary=area_info['summary'])
                
                # Add edges to members
                for member in member_nodes:
                    G.add_edge(member, node_name, relationship="part_of")
            
            current_level = next_level

    def _add_proximity_relationships(self, G, objects):
        print("Adding proximity relationships...")
        # Filter out drone nodes first
        filtered_objects = [
            obj for obj in objects 
            if not (isinstance(obj.get('id', ''), str) and 
                   ('drone' in obj.get('id', '').lower() or 'area_' in str(obj.get('id', ''))))
        ]
        
        for i, obj1 in tqdm(enumerate(filtered_objects), total=len(filtered_objects), desc="Processing object pairs"):
            for j, obj2 in enumerate(filtered_objects[i+1:], start=i+1):
                pos1 = self._get_position(obj1)
                pos2 = self._get_position(obj2)
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if distance <= self.proximity_threshold:
                    G.add_edge(obj1['id'], obj2['id'], relationship="near", distance=distance)

    def _get_position(self, obj):
        pos = obj.get('position', {})
        if isinstance(pos, dict) and 'x' in pos and 'y' in pos and 'z' in pos:
            return [pos['x'], pos['y'], pos['z']]
        elif isinstance(pos, (list, tuple)) and len(pos) == 3:
            return pos
        else:
            print(f"Warning: Unexpected position format for object: {obj['id']}")
            return [0, 0, 0]  # Default position

    async def _add_hierarchical_relationships(self, G, Z, filtered_objects):
        print("\nAdding hierarchical relationships...")
        n = len(filtered_objects)
        clusters = fcluster(Z, self.cluster_distance_threshold, criterion='distance')
        max_level = max(clusters)
        print(f"Found {max_level} hierarchical levels")
        
        # Track cluster summaries for higher-level abstraction
        cluster_summaries = {}
        processed_clusters = set()  # Add this to track which clusters we've processed

        for level in tqdm(range(1, max_level + 1), desc="Processing levels"):
            print(f"\n=== Processing Level {level} ===")
            cluster_members = {}
            
            # Modified cluster assignment logic
            for i, cluster_id in enumerate(clusters):
                if cluster_id >= level:
                    normalized_cluster_id = cluster_id - level + 1  # Normalize cluster IDs
                    if normalized_cluster_id not in cluster_members:
                        cluster_members[normalized_cluster_id] = []
                    cluster_members[normalized_cluster_id].append(i)

            print(f"Found {len(cluster_members)} clusters at level {level}")
            
            for cluster_id, members in cluster_members.items():
                cluster_key = f"level_{level}_{cluster_id}"
                
                # Skip if we've already processed this cluster
                if cluster_key in processed_clusters:
                    continue
                processed_clusters.add(cluster_key)
                
                print(f"\nProcessing cluster {cluster_id} with {len(members)} members:")
                # Get member objects or their summaries
                member_objects = []
                for i in members:
                    if i < len(filtered_objects):
                        obj = filtered_objects[i]
                        member_objects.append(obj)
                        print(f"  - Object: {obj.get('id', 'Unknown')}")
                    else:
                        prev_cluster_key = f"level_{level-1}_{i}"
                        if prev_cluster_key in cluster_summaries:
                            summary = cluster_summaries[prev_cluster_key]
                            member_objects.append(summary)
                            print(f"  - Subcluster: {summary.get('label', 'Unknown')}")

                if not member_objects:  # Skip empty clusters
                    continue

                # Calculate average position
                member_positions = [self._get_position(obj) for obj in member_objects]
                avg_position = np.mean(member_positions, axis=0)

                # Generate area summary
                print("\nGenerating area summary...")
                area_info = await self._create_community_summary(member_objects)
                print(f"Area Name: {area_info['name']}")
                print(f"Summary: {area_info['summary']}")
                print("-" * 50)
                
                # Create node name based on functional area
                node_name = f"area_{level}_{area_info['name'].lower().replace(' ', '_')}"

                # Store summary for higher-level clustering
                cluster_summaries[cluster_key] = {
                    'id': node_name,
                    'label': area_info['name'],
                    'summary': area_info['summary'],
                    'position': avg_position.tolist()
                }

                # Add node to graph
                G.add_node(node_name, 
                          position=avg_position.tolist(), 
                          level=level,
                          name=area_info['name'],
                          summary=area_info['summary'])

                # Add edges to members
                for member in members:
                    if member < len(filtered_objects):
                        member_node = filtered_objects[member]['id']
                    else:
                        member_node = f"area_{level-1}_{clusters[member-len(filtered_objects)]}"
                    
                    if G.has_node(member_node):  # Only add edge if member node exists
                        G.add_edge(member_node, node_name, relationship="part_of")

    async def _create_community_summary(self, objects):
        """Create a summary for a group of objects using LLM interface"""
        return await self.llm.generate_community_summary(objects)

    def _get_cardinal_direction(self, pos1, pos2):
        """Convert relative positions to cardinal directions"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        
        # Determine primary horizontal direction using angle
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
            
        # Add vertical component if significant
        vertical = ""
        if abs(dz) > self.vertical_threshold:
            vertical = "above" if dz > 0 else "below"
            
        # Calculate distance
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return horizontal, vertical, distance

    async def _add_positional_relationships(self, G, objects):
        """Add spatial relationships based on distance threshold"""
        filtered_objects = [
            obj for obj in objects 
            if not (isinstance(obj.get('id', ''), str) and 
                   ('drone' in obj.get('id', '').lower() or 'area_' in str(obj.get('id', ''))))
        ]
        
        print(f"Adding spatial relationships for {len(filtered_objects)} base objects...")
        
        for obj1 in filtered_objects:
            node1 = obj1.get('id', obj1.get('name', str(obj1)))
            if isinstance(node1, str) and 'drone' in node1.lower():
                continue
                
            pos1 = self._get_position(obj1)
            level1 = obj1.get('level', 0)

            for obj2 in filtered_objects:
                if obj1 != obj2:
                    node2 = obj2.get('id', obj2.get('name', str(obj2)))
                    if isinstance(node2, str) and 'drone' in node2.lower():
                        continue
                    
                    pos2 = self._get_position(obj2)
                    level2 = obj2.get('level', 0)
                    
                    if level1 == level2:
                        horizontal, vertical, distance = self._get_cardinal_direction(pos1, pos2)
                        
                        # Use a separate threshold for spatial relationships
                        if distance <= self.spatial_threshold:
                            relationship = horizontal
                            if vertical:
                                relationship += f"_{vertical}"
                            
                            # Add both relationships in one edge
                            G.add_edge(node1, node2, 
                                     relationship=relationship,
                                     cardinal_direction=relationship,  # Add explicit cardinal direction
                                     distance=distance)

