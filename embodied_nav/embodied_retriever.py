import networkx as nx
from scipy.spatial.distance import cosine
import numpy as np
from .llm import LLMInterface

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
        # Find objects that match the query directly
        matching_objects = [node for node, data in self.graph.nodes(data=True) 
                            if query.lower() in data.get('label', '').lower()]
        
        # Combine with semantic, hierarchical, spatial, and functional retrieval
        semantic_results = await self._semantic_retrieval(query_embedding, top_k * 2)
        hierarchical_results = self._hierarchical_retrieval(top_k * 2)
        spatial_results = self._spatial_retrieval(top_k * 2)
        # functional_results = self._functional_retrieval(top_k * 2)
        
        combined_results = list(set(matching_objects + semantic_results + hierarchical_results + spatial_results + functional_results))
        
        # Use LLM to rank the relevance of the combined results
        context = self._build_context(combined_results)
        ranked_results = await self.llm.rank_results(query, combined_results, context)
        
        return ranked_results[:top_k]

    async def _implicit_retrieval(self, query, query_embedding, top_k):
        semantic_results = await self._semantic_retrieval(query_embedding, top_k * 2)
        hierarchical_results = self._hierarchical_retrieval(top_k * 2)
        spatial_results = self._spatial_retrieval(top_k * 2)
        # functional_results = self._functional_retrieval(top_k * 2)
        
        combined_results = list(set(semantic_results + hierarchical_results + spatial_results + functional_results))
        
        context = self._build_context(combined_results)
        ranked_results = await self.llm.rank_results(query, combined_results, context)
        
        return ranked_results[:top_k]

    async def _global_retrieval(self, query, query_embedding, top_k):
        # Combine semantic, hierarchical, spatial, and functional retrieval
        semantic_results = await self._semantic_retrieval(query_embedding, top_k)
        hierarchical_results = self._hierarchical_retrieval(top_k)
        spatial_results = self._spatial_retrieval(top_k)
        # functional_results = self._functional_retrieval(top_k)
        
        combined_results = list(set(semantic_results + hierarchical_results + spatial_results + functional_results))
        return combined_results[:top_k]

    async def _semantic_retrieval(self, query_embedding, top_k):
        node_similarities = []
        for node, data in self.graph.nodes(data=True):
            if 'embedding' in data:
                similarity = 1 - cosine(query_embedding, data['embedding'])
                node_similarities.append((node, similarity))
        
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in node_similarities[:top_k]]

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
2. The position of the object(s)
3. A suggested path to reach the object(s)
4. An explanation of why these objects are relevant to the query, considering their semantic properties, hierarchical relationships, spatial context, and functional relevance."""
        else:  # global query
            prompt = f"Given the following context about objects in a 3D environment:\n\n{context}\n\nAnswer the following query: {query}"
        
        response = await self.llm.generate_response(prompt)
        return response

    def generate_waypoints(self, start_position, target_nodes):
        waypoints = []
        current_position = start_position

        for target in target_nodes:
            target_position = self.graph.nodes[target]['position']
            # Simple straight-line path (you might want to implement more sophisticated pathfinding)
            waypoints.append(target_position)
            current_position = target_position

        return waypoints

    def _build_context(self, nodes):
        context = []
        for node in nodes:
            node_data = self.graph.nodes[node]
            context.append(f"Object: {node}")
            context.append(f"Properties: {', '.join([f'{k}: {v}' for k, v in node_data.items() if k != 'embedding'])}")
            
            neighbors = list(self.graph.neighbors(node))
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(node, neighbor)
                context.append(f"Relationship to {neighbor}: {edge_data.get('relationship', 'connected')}")
            
            context.append("")  # Add a blank line between objects
        
        return "\n".join(context)
