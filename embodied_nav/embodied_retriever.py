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
        top_k = top_k or self.top_k_default
        query_embedding = await self.embedding_func([query])
        
        # Get initial nodes based on query type
        initial_nodes = []
        if query_type == "explicit":
            initial_nodes = await self._explicit_retrieval(query, query_embedding, top_k)
        elif query_type == "implicit":
            initial_nodes = await self._implicit_retrieval(query, query_embedding, top_k)
        elif query_type == "global":
            initial_nodes = await self._global_retrieval(query, query_embedding, top_k)
        
        # Expand nodes to include their full hierarchical chains
        expanded_nodes = set()
        for node in initial_nodes:
            chain = self._get_hierarchical_chain(node)
            expanded_nodes.update(chain)
        
        return list(expanded_nodes)

    async def _explicit_retrieval(self, query, query_embedding, top_k):
        # Get directly matching objects
        matching_objects = [node for node, data in self.graph.nodes(data=True) 
                          if query.lower() in data.get('label', '').lower()]
        
        # Get semantic matches if needed
        if len(matching_objects) < top_k:
            semantic_results = await self._semantic_retrieval(query_embedding, top_k)
            matching_objects.extend(semantic_results)
        
        return matching_objects

    async def _implicit_retrieval(self, query, query_embedding, top_k):
        # Get semantic matches
        semantic_results = await self._semantic_retrieval(query_embedding, top_k)
        
        # Get all relevant information
        return semantic_results

    async def _global_retrieval(self, query, query_embedding, top_k):
        # Combine semantic, hierarchical, spatial, and functional retrieval
        semantic_results = await self._semantic_retrieval(query_embedding, top_k)
        hierarchical_results = self._hierarchical_retrieval(top_k)
        spatial_results = self._spatial_retrieval(top_k)
        # functional_results = self._functional_retrieval(top_k)
        
        combined_results = list(set(semantic_results + hierarchical_results + spatial_results))
        return combined_results[:top_k]

    async def _semantic_retrieval(self, query_embedding, top_k):
        print("\nDebug: Starting semantic retrieval")
        node_similarities = []
        
        for node, data in self.graph.nodes(data=True):
            if 'embedding' in data:
                similarity = 1 - cosine(query_embedding[0], data['embedding'])  # Note: using query_embedding[0]
                node_similarities.append((node, similarity))
                print(f"Debug: Node {node} similarity: {similarity}")
        
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        result = [node for node, sim in node_similarities[:top_k] if sim > 0.25]  # Add similarity threshold
        print(f"Debug: Semantic retrieval results: {result}")
        return result

    def _hierarchical_retrieval(self, top_k):
        hierarchical_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('level', 0) > 0]
        return sorted(hierarchical_nodes, key=lambda n: self.graph.nodes[n]['level'], reverse=True)[:top_k]

    def _spatial_retrieval(self, top_k):
        # Simplified to use relationship types from graph edges
        spatial_edges = [
            (u, v) for (u, v, d) in self.graph.edges(data=True) 
            if 'relationship' in d and d['relationship'] not in ['part_of']  # Exclude hierarchical relationships
        ]
        return list(set([node for edge in spatial_edges for node in edge]))[:top_k]

    async def generate_response(self, query, retrieved_nodes, query_type):
        context = self._build_context(retrieved_nodes)
        
        if query_type in ["explicit", "implicit"]:
            prompt = f"""Given the following context about objects in a 3D environment:

            {context}

            For the query: '{query}', provide the following:
            1. A brief description of the most relevant object(s)
            2. An explanation of why these objects are relevant to the query, considering their semantic properties, hierarchical relationships, spatial context.
            3. Choose the best object to answer the query and output in this exact format at the end: <<object_name>>
            
            Make sure the object_name matches exactly with one of the objects in the context.
            """
        else:  # global query
            prompt = f"Given the following context about objects in a 3D environment:\n\n{context}\n\nAnswer the following query: {query}"
        
        response = await self.llm.generate_response(prompt)
        return response

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
            if level > 10:  # Safety check
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
