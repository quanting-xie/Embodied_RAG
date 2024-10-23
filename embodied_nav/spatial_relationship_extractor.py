import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
import asyncio
from tqdm import tqdm

class SpatialRelationshipExtractor:
    def __init__(self, llm_func, cluster_distance_threshold=5.0, proximity_threshold=10.0, vertical_threshold=2.0):
        self.llm_func = llm_func
        self.cluster_distance_threshold = cluster_distance_threshold
        self.proximity_threshold = proximity_threshold
        self.vertical_threshold = vertical_threshold

    async def extract_relationships(self, objects):
        print("Extracting relationships...")
        
        # Extract positions from objects
        positions = []
        for obj in tqdm(objects, desc="Processing objects"):
            pos = obj.get('position', {})
            if isinstance(pos, dict) and 'x' in pos and 'y' in pos and 'z' in pos:
                positions.append([pos['x'], pos['y'], pos['z']])
            elif isinstance(pos, (list, tuple)) and len(pos) == 3:
                positions.append(pos)
            else:
                print(f"Warning: Unexpected position format for object: {obj['id']}")
                positions.append([0, 0, 0])  # Default position
        
        # Convert to numpy array and ensure it's 2D
        positions = np.array(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        
        print("Performing hierarchical clustering...")
        Z = linkage(positions, 'ward')

        # Create a graph
        G = nx.Graph()

        # Add object nodes
        for obj in tqdm(objects, desc="Adding object nodes"):
            G.add_node(obj['id'], **obj, level=0)

        # Add hierarchical relationships and community summary nodes
        await self._add_hierarchical_relationships(G, Z, objects)

        # Add proximity relationships
        await self._add_positional_relationships(G, objects)

        return G

    def _add_proximity_relationships(self, G, objects):
        print("Adding proximity relationships...")
        for i, obj1 in tqdm(enumerate(objects), total=len(objects), desc="Processing object pairs"):
            for j, obj2 in enumerate(objects[i+1:], start=i+1):
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

    async def _add_hierarchical_relationships(self, G, Z, objects):
        print("Adding hierarchical relationships...")
        n = len(objects)
        clusters = fcluster(Z, self.cluster_distance_threshold, criterion='distance')
        max_level = max(clusters)

        for level in tqdm(range(1, max_level + 1), desc="Processing levels"):
            cluster_members = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id >= level:
                    if cluster_id not in cluster_members:
                        cluster_members[cluster_id] = []
                    cluster_members[cluster_id].append(i)

            for cluster_id, members in cluster_members.items():
                cluster_node = f"cluster_{level}_{cluster_id}"
                
                member_positions = []
                for i in members:
                    pos = objects[i]['position']
                    if isinstance(pos, dict):
                        pos = [pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)]
                    elif isinstance(pos, (list, tuple)) and len(pos) == 3:
                        pos = list(pos)
                    else:
                        print(f"Warning: Unexpected position format for object {i}: {pos}")
                        pos = [0, 0, 0]  # Default position if format is unexpected
                    member_positions.append(pos)

                avg_position = np.mean(member_positions, axis=0)
                member_objects = [objects[i] for i in members]
                
                # Create community summary
                summary = await self._create_community_summary(member_objects)

                G.add_node(cluster_node, position=avg_position.tolist(), level=level, summary=summary)

                for member in members:
                    member_node = objects[member]['id'] if level == 1 else f"cluster_{level-1}_{clusters[member]}"
                    G.add_edge(member_node, cluster_node, relationship="part_of")

    async def _create_community_summary(self, objects):
        object_descriptions = []
        for obj in objects:
            try:
                label = obj.get('label', obj.get('id', 'Unknown object'))
                position = obj.get('position', 'Unknown position')
                object_descriptions.append(f"{label} at position {position}")
            except Exception as e:
                print(f"Warning: Error processing object: {obj}. Error: {e}")
                object_descriptions.append("Unprocessable object")

        prompt = f"Create a brief summary of the following group of objects: {', '.join(object_descriptions)}"
        return await self.llm_func(prompt)

    async def _add_positional_relationships(self, G, objects):
        k_nearest = 5  # Number of nearest neighbors to consider

        for obj1 in objects:
            node1 = obj1.get('id', obj1.get('name', str(obj1)))  # Use 'id' or 'name' as node identifier
            pos1 = np.array(obj1.get('position', [0, 0, 0]))
            level1 = obj1.get('level', 0)

            # Calculate distances to all other objects
            distances = []
            for obj2 in objects:
                if obj1 != obj2:
                    node2 = obj2.get('id', obj2.get('name', str(obj2)))
                    pos2 = np.array(obj2.get('position', [0, 0, 0]))
                    level2 = obj2.get('level', 0)
                    
                    if level1 == level2:
                        diff = pos2 - pos1
                        distance = np.linalg.norm(diff)
                        distances.append((node2, obj2, diff, distance))
            
            # Sort by distance and get the k nearest neighbors
            nearest_neighbors = sorted(distances, key=lambda x: x[3])[:k_nearest]

            for node2, obj2, diff, distance in nearest_neighbors:
                # Check vertical relationships
                if abs(diff[2]) > self.vertical_threshold:
                    if diff[2] > 0:
                        G.add_edge(node1, node2, relationship="below")
                        G.add_edge(node2, node1, relationship="above")
                    else:
                        G.add_edge(node1, node2, relationship="above")
                        G.add_edge(node2, node1, relationship="below")

                # Check horizontal relationships
                if abs(diff[0]) > self.horizontal_threshold:
                    if diff[0] > 0:
                        G.add_edge(node1, node2, relationship="in_front_of")
                        G.add_edge(node2, node1, relationship="behind")
                    else:
                        G.add_edge(node1, node2, relationship="behind")
                        G.add_edge(node2, node1, relationship="in_front_of")

                if abs(diff[1]) > self.horizontal_threshold:
                    if diff[1] > 0:
                        G.add_edge(node1, node2, relationship="to_the_left_of")
                        G.add_edge(node2, node1, relationship="to_the_right_of")
                    else:
                        G.add_edge(node1, node2, relationship="to_the_right_of")
                        G.add_edge(node2, node1, relationship="to_the_left_of")

    # def _add_functional_relationships(self, G, nodes):
    #     for node1, data1 in nodes:
    #         for node2, data2 in nodes:
    #             if node1 != node2 and data1['level'] == data2['level']:
    #                 relationship = self.llm.generate_relationship(
    #                     (node1, data1),
    #                     (node2, data2)
    #                 )
    #                 if relationship:
    #                     G.add_edge(node1, node2, relationship=relationship)
