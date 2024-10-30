import networkx as nx
from scipy.spatial.distance import cosine
import numpy as np
from .llm import LLMInterface
import re
from .config import Config

class EmbodiedRetriever:
    def __init__(self, graph, embedding_func):
        self.graph = graph
        self.embedding_func = embedding_func
        self.llm = LLMInterface()
        self.top_k_default = Config.RETRIEVAL['top_k_default']
        self.semantic_threshold = Config.RETRIEVAL['semantic_similarity_threshold']
        self.max_hierarchical_level = Config.RETRIEVAL['max_hierarchical_level']

    async def retrieve(self, query, query_type="global", top_k=None):
        """Main retrieval method for all query types"""
        top_k = top_k or self.top_k_default
        query_embedding = await self.embedding_func([query])
        
        # Get semantically similar nodes
        semantic_results = await self._semantic_retrieval(query_embedding, top_k)
        
        # Get spatial and hierarchical context
        expanded_nodes = set(semantic_results)
        for node in semantic_results:
            # Add hierarchical chain
            chain = self._get_hierarchical_chain(node)
            expanded_nodes.update(chain)
            
            # Add spatially related nodes
            spatial_neighbors = self._get_spatial_neighbors(node)
            expanded_nodes.update(spatial_neighbors)
        
        return list(expanded_nodes)

    def _get_spatial_neighbors(self, node):
        """Get nodes with spatial relationships to the given node"""
        neighbors = []
        for neighbor, edge_data in self.graph[node].items():
            if ('relationship' in edge_data and 
                edge_data['relationship'] not in ['part_of']):
                neighbors.append(neighbor)
        return neighbors

    async def _semantic_retrieval(self, query_embedding, top_k):
        """Get semantically similar nodes"""
        node_similarities = []
        for node, data in self.graph.nodes(data=True):
            if 'embedding' in data:
                similarity = 1 - cosine(query_embedding[0], data['embedding'])
                node_similarities.append((node, similarity))
        
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        return [
            node for node, sim in node_similarities[:top_k] 
            if sim > self.semantic_threshold
        ]

    async def generate_response(self, query, retrieved_nodes, query_type):
        """Generate response using context from retrieved nodes"""
        context = self._build_context(retrieved_nodes)
        return await self.llm.generate_navigation_response(query, context, query_type)

    def extract_target_object(self, response):
        """Extract the target object from the LLM response."""
        import re
        match = re.search(r'<<(.+?)>>', response)
        if match:
            return match.group(1) #
        return None

    def _get_hierarchical_chain(self, node):
        """Get the full hierarchical chain from node to top-level cluster"""
        chain = []
        current = node
        visited = set()  # Prevent infinite loops
        level = 0
        
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            
            # Find parent cluster with highest level
            parent = None
            max_level = -1
            for neighbor in self.graph.neighbors(current):
                neighbor_data = self.graph.nodes[neighbor]
                if ('cluster' in neighbor and 
                    neighbor_data.get('level', 0) > max_level):
                    parent = neighbor
                    max_level = neighbor_data.get('level', 0)
            
            current = parent
            level += 1
            if level > self.max_hierarchical_level:
                break
                
        return chain

    def _build_context(self, nodes):
        context = []
        
        # 1. Hierarchical Structure Overview
        context.append("=== Hierarchical Structure ===")
        
        # Group nodes by level
        level_nodes = {}
        for node in nodes:
            node_data = self.graph.nodes[node]
            level = node_data.get('level', 0)
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        # Display hierarchy from top down
        for level in sorted(level_nodes.keys(), reverse=True):
            context.append(f"\nLevel {level}:")
            for node in level_nodes[level]:
                node_data = self.graph.nodes[node]
                indent = "  " * (3 - level)
                context.append(f"{indent}{node}")
                if 'summary' in node_data:
                    context.append(f"{indent}Summary: {node_data['summary']}")
                
                # Show children for clusters
                children = [n for n, _ in self.graph.edges(node) 
                          if self.graph.nodes[n].get('level', 0) < level]
                if children:
                    context.append(f"{indent}Contains: {', '.join(children)}")
        
        # 2. Object Information
        context.append("\n=== Object Information ===")
        for node in nodes:
            if 'cluster' not in node:  # Only show details for non-cluster objects
                node_data = self.graph.nodes[node]
                context.append(f"\nObject: {node}")
                
                # Show hierarchical path
                chain = self._get_hierarchical_chain(node)
                if len(chain) > 1:
                    path = " → ".join(reversed(chain))
                    context.append(f"Located in: {path}")
                
                # Position
                if 'position' in node_data:
                    pos = node_data['position']
                    context.append(f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                
                # Spatial relationships from graph edges
                spatial_relations = []
                for neighbor, edge_data in self.graph[node].items():
                    if 'relationship' in edge_data and edge_data['relationship'] != 'part_of':
                        distance = edge_data.get('distance', 'unknown')
                        distance_str = f" ({distance:.1f}m away)" if isinstance(distance, (int, float)) else ""
                        spatial_relations.append(f"{neighbor} is {edge_data['relationship']}{distance_str}")
                
                if spatial_relations:
                    context.append("Spatial relationships:")
                    for rel in spatial_relations:
                        context.append(f"- {rel}")
                
                # Other properties
                props = {k: v for k, v in node_data.items() 
                        if k not in ['embedding', 'position', 'label']}
                if props:
                    context.append(f"Properties: {props}")
        
        final_context = "\n".join(context)
        print("\nGenerated Context for LLM:")
        print("="*50)
        print(final_context)
        print("="*50)
        return final_context

    def extract_target_position(self, response):
        # Extract the object name from the response
        object_name_match = re.search(r'<<(.+?)>>', response)
        if object_name_match:
            object_name = object_name_match.group(1)
            
            # Find the node in the graph corresponding to the object
            for node, data in self.graph.nodes(data=True):
                if node == object_name:
                    if 'position' in data:
                        position = data['position']
                        print(f"\nTarget object '{object_name}' found at position: {position}")
                        return position
                    else:
                        print(f"\nNo position data found for '{object_name}'")
                        return None
            
            print(f"\nNo node found for '{object_name}'")
            return None
        
        print("\nNo target object found in response")
        return None
