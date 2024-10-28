import networkx as nx
from scipy.spatial.distance import cosine
import numpy as np
from .llm import LLMInterface
import re

class EmbodiedRetriever:
    def __init__(self, graph, embedding_func):
        self.graph = graph
        self.embedding_func = embedding_func
        self.llm = LLMInterface()

    async def retrieve(self, query, query_type="global", top_k=5):
        query_embedding = await self.embedding_func([query])
        
        if query_type == "explicit":
            return await self._explicit_retrieval(query, query_embedding, top_k)
        elif query_type == "implicit":
            return await self._implicit_retrieval(query, query_embedding, top_k)
        elif query_type == "global":
            return await self._global_retrieval(query, query_embedding, top_k)
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    async def _explicit_retrieval(self, query, query_embedding, top_k):
        # 1. Get directly matching objects
        matching_objects = [node for node, data in self.graph.nodes(data=True) 
                          if query.lower() in data.get('label', '').lower()]
        
        # 2. Get semantic matches if needed
        if len(matching_objects) < top_k:
            semantic_results = await self._semantic_retrieval(query_embedding, top_k)
            matching_objects.extend(semantic_results)
        
        # 3. Get all relevant information (objects, clusters, relationships)
        return self._gather_comprehensive_context(matching_objects)

    async def _implicit_retrieval(self, query, query_embedding, top_k):
        # 1. Get semantic matches
        semantic_results = await self._semantic_retrieval(query_embedding, top_k)
        
        # 2. Get all relevant information
        return self._gather_comprehensive_context(semantic_results)

    def _gather_comprehensive_context(self, seed_objects):
        relevant_nodes = set(seed_objects)
        
        # 1. Get parent clusters
        for obj in seed_objects:
            clusters = [n for n in self.graph.neighbors(obj) 
                       if 'cluster' in self.graph.nodes[n].get('label', '')]
            relevant_nodes.update(clusters)
        
        # 2. Get spatially related objects
        for obj in seed_objects:
            spatial_neighbors = [n for n in self.graph.neighbors(obj)
                               if self.graph.get_edge_data(obj, n).get('relationship') 
                               in ['near', 'above', 'below', 'in_front_of', 'behind', 
                                   'to_the_left_of', 'to_the_right_of']]
            relevant_nodes.update(spatial_neighbors)
        
        return list(relevant_nodes)

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
        result = [node for node, sim in node_similarities[:top_k] if sim > 0.3]  # Add similarity threshold
        print(f"Debug: Semantic retrieval results: {result}")
        return result

    def _hierarchical_retrieval(self, top_k):
        hierarchical_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('level', 0) > 0]
        return sorted(hierarchical_nodes, key=lambda n: self.graph.nodes[n]['level'], reverse=True)[:top_k]

    def _spatial_retrieval(self, top_k):
        spatial_edges = [
            (u, v) for (u, v, d) in self.graph.edges(data=True) 
            if d.get('relationship') in ['near', 'above', 'below', 'in_front_of', 'behind', 'to_the_left_of', 'to_the_right_of']
        ]
        return list(set([node for edge in spatial_edges for node in edge]))[:top_k]

    def _functional_retrieval(self, top_k):
        functional_edges = [
            (u, v) for (u, v, d) in self.graph.edges(data=True) 
            if d.get('relationship') not in ['near', 'above', 'below', 'in_front_of', 'behind', 'to_the_left_of', 'to_the_right_of', 'part_of']
        ]
        return list(set([node for edge in functional_edges for node in edge]))[:top_k]

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

    def _build_context(self, nodes):
        context = []
        
        # 1. First, add cluster information
        clusters = [n for n in nodes if 'cluster' in self.graph.nodes[n].get('label', '')]
        if clusters:
            context.append("=== Cluster Information ===")
            for cluster in clusters:
                cluster_data = self.graph.nodes[cluster]
                context.append(f"Cluster: {cluster}")
                if 'level' in cluster_data:
                    context.append(f"Level: {cluster_data['level']}")
                if 'summary' in cluster_data:
                    context.append(f"Summary: {cluster_data['summary']}")
                context.append("")
        
        # 2. Add object information with their relationships
        context.append("=== Object Information ===")
        for node in nodes:
            if 'cluster' not in self.graph.nodes[node].get('label', ''):
                node_data = self.graph.nodes[node]
                context.append(f"Object: {node}")
                
                # Basic properties
                if 'position' in node_data:
                    pos = node_data['position']
                    if isinstance(pos, (list, tuple)):
                        context.append(f"Position: [{pos[0]}, {pos[1]}, {pos[2]}]")
                    else:
                        context.append(f"Position: {pos}")
                
                # Other properties (excluding embedding and already handled properties)
                other_props = {k: v for k, v in node_data.items() 
                             if k not in ['embedding', 'position', 'label']}
                if other_props:
                    context.append(f"Properties: {other_props}")
                
                # Hierarchical relationships
                parent_clusters = [n for n in self.graph.neighbors(node) 
                                 if 'cluster' in self.graph.nodes[n].get('label', '')]
                if parent_clusters:
                    context.append("Belongs to clusters: " + ", ".join(parent_clusters))
                
                # Spatial relationships
                spatial_relations = []
                for neighbor in self.graph.neighbors(node):
                    if 'cluster' not in self.graph.nodes[neighbor].get('label', ''):
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        rel = edge_data.get('relationship', 'connected')
                        spatial_relations.append(f"{neighbor} ({rel})")
                if spatial_relations:
                    context.append("Spatial relationships: " + ", ".join(spatial_relations))
                
                context.append("")  # Add blank line between objects
        
        # 3. Add overall spatial layout information
        context.append("=== Spatial Layout ===")
        spatial_edges = [(u, v, d) for (u, v, d) in self.graph.edges(data=True)
                        if d.get('relationship') in ['near', 'above', 'below', 
                            'in_front_of', 'behind', 'to_the_left_of', 'to_the_right_of']]
        for u, v, data in spatial_edges:
            if u in nodes and v in nodes:
                context.append(f"{u} is {data['relationship']} {v}")
        
        return "\n".join(context)

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
