import networkx as nx
from scipy.spatial.distance import cosine
import numpy as np
from .llm import LLMInterface
import re
from .config import Config

class RetrievalMethod:
    SEMANTIC = "semantic"  # Fast retrieval method
    LLM_HIERARCHICAL = "llm_hierarchical"  # Original semantic-based method in paper
    HYBRID = "hybrid"  # Combination of both methods

class EmbodiedRetriever:
    def __init__(self, graph, embedding_func, retrieval_method=RetrievalMethod.SEMANTIC):
        self.graph = graph
        self.embedding_func = embedding_func
        self.llm = LLMInterface()
        self.top_k_default = Config.RETRIEVAL['top_k_default']
        self.semantic_threshold = Config.RETRIEVAL['semantic_similarity_threshold']
        self.max_hierarchical_level = Config.RETRIEVAL['max_hierarchical_level']
        self.retrieval_method = retrieval_method
        
        # Initialize LLM hierarchical retriever if needed
        if retrieval_method in [RetrievalMethod.LLM_HIERARCHICAL, RetrievalMethod.HYBRID]:
            from .use_llm_selection import LLMHierarchicalRetriever
            self.llm_retriever = LLMHierarchicalRetriever(self.graph, self.llm)

    async def retrieve(self, query, query_type="global", top_k=None):
        """Main retrieval method with support for different retrieval strategies"""
        if self.retrieval_method == RetrievalMethod.SEMANTIC:
            return await self._semantic_based_retrieval(query, query_type, top_k)
        elif self.retrieval_method == RetrievalMethod.LLM_HIERARCHICAL:
            return await self._llm_based_retrieval(query, query_type)
        else:  # HYBRID
            return await self._hybrid_retrieval(query, query_type, top_k)

    async def _semantic_based_retrieval(self, query, query_type="global", top_k=None):
        """Enhanced semantic retrieval with hierarchical and spatial boosting"""
        top_k = top_k or self.top_k_default
        query_embedding = await self.embedding_func([query])
        
        # 1. Get initial semantic matches (get more candidates)
        initial_nodes, initial_scores = await self._semantic_retrieval(query_embedding, top_k * 2)
        
        print("\nInitial semantic scores:")
        for node, score in zip(initial_nodes, initial_scores):
            print(f"- {node}: {score:.3f}")
        
        # 2. Apply hierarchical boosting
        hierarchical_scores = self._apply_hierarchical_boost(initial_nodes, initial_scores, query_embedding)
        
        print("\nAfter hierarchical boost:")
        for node, score in zip(initial_nodes, hierarchical_scores):
            print(f"- {node}: {score:.3f}")
        
        # 3. Apply spatial boosting
        final_scores = self._apply_spatial_boost(initial_nodes, hierarchical_scores)
        
        print("\nAfter spatial boost:")
        for node, score in zip(initial_nodes, final_scores):
            print(f"- {node}: {score:.3f}")
        
        # 4. Normalize scores
        normalized_scores = self._normalize_scores(final_scores)
        
        print("\nFinal normalized scores:")
        for node, score in zip(initial_nodes, normalized_scores):
            print(f"- {node}: {score:.3f}")
        
        # 5. Re-rank and select top K
        node_scores = list(zip(initial_nodes, normalized_scores))
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nFinal top K nodes:")
        for node, score in node_scores[:top_k]:
            print(f"- {node}: {score:.3f}")
        
        # Store top K nodes for context building
        self.top_k_nodes = [node for node, _ in node_scores[:top_k]]
        
        # 6. Get context for final nodes
        context_nodes = set()
        for node in self.top_k_nodes:
            context_nodes.add(node)
            chain = self._get_hierarchical_chain(node)
            context_nodes.update(chain)
            spatial_neighbors = self._get_spatial_neighbors(node)
            context_nodes.update(spatial_neighbors)
        
        return list(context_nodes)

    async def _llm_based_retrieval(self, query, query_type):
        """LLM-guided hierarchical retrieval method"""
        # Just use the results directly from LLM hierarchical retriever
        results = await self.llm_retriever.retrieve(query)
        
        # Optionally, only get immediate spatial neighbors for the leaf nodes
        if query_type != "global":
            expanded_nodes = set(results)
            leaf_nodes = [node for node in results 
                         if not any(self.graph.nodes[n].get('level', 0) < 
                                  self.graph.nodes[node].get('level', 0)
                                  for n in self.graph.neighbors(node))]
            
            for node in leaf_nodes:
                spatial_neighbors = self._get_spatial_neighbors(node)
                expanded_nodes.update(spatial_neighbors)
            return list(expanded_nodes)
        
        return results

    async def _hybrid_retrieval(self, query, query_type="global", top_k=None):
        """Combine both semantic and LLM-guided approaches"""
        # Get results from both methods
        semantic_results = await self._semantic_based_retrieval(query, query_type, top_k)
        llm_results = await self._llm_based_retrieval(query, query_type)
        
        # Combine and deduplicate results
        combined_results = list(set(semantic_results + llm_results))
        
        # If we have too many results, use LLM to rank and filter them
        if len(combined_results) > self.top_k_default:
            context = self._build_context(combined_results)
            ranked_results = await self.llm.rank_results(query, combined_results, context)
            return ranked_results[:self.top_k_default]
        
        return combined_results

    def _get_spatial_neighbors(self, node):
        """Get nodes with spatial relationships to the given node"""
        neighbors = []
        for neighbor, edge_data in self.graph[node].items():
            if ('relationship' in edge_data and 
                edge_data['relationship'] not in ['part_of']):
                neighbors.append(neighbor)
        return neighbors

    async def _semantic_retrieval(self, query_embedding, top_k):
        """Get semantically similar nodes with their scores"""
        node_similarities = []
        for node, data in self.graph.nodes(data=True):
            if ('embedding' in data and 
                data.get('type') not in ['drone', 'structural']):
                similarity = 1 - cosine(query_embedding[0], data['embedding'])
                node_similarities.append((node, similarity))
        
        # Sort by similarity
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K nodes and scores
        top_nodes = []
        top_scores = []
        for node, sim in node_similarities[:top_k]:
            if sim > self.semantic_threshold:
                print(f"- {node} (similarity: {sim:.3f})")
                top_nodes.append(node)
                top_scores.append(sim)
        
        return top_nodes, top_scores

    async def generate_response(self, query, retrieved_nodes, query_type):
        """Generate response using context from retrieved nodes"""
        # Build and print context
        context = self._build_context(retrieved_nodes[:self.top_k_default])  # Only use top K nodes
        print("\nGenerated Context for LLM:")
        print("="*50)
        print(context)
        print("="*50)

        # Generate LLM response
        prompt = await self.llm.generate_navigation_response(query, context, query_type)
        return prompt

    def extract_target_object(self, response):
        """Extract the target object from the LLM response."""
        import re
        match = re.search(r'<<(.+?)>>', response)
        if match:
            return match.group(1) #
        return None

    def _get_hierarchical_chain(self, node):
        """Get the full hierarchical chain from node to top-level area"""
        chain = []
        current = node
        visited = set()  # Prevent infinite loops
        
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            
            # Find parent with higher level
            parent = None
            max_level = self.graph.nodes[current].get('level', 0)
            
            for neighbor in self.graph.neighbors(current):
                edge = self.graph[current][neighbor]
                neighbor_level = self.graph.nodes[neighbor].get('level', 0)
                
                if (edge.get('relationship') == 'part_of' and 
                    neighbor_level > max_level):
                    parent = neighbor
                    max_level = neighbor_level
            
            if not parent:  # No higher-level parent found
                break
                
            current = parent
            
            if len(chain) > self.max_hierarchical_level:  # Prevent infinite loops
                break
        
        return chain

    def _build_context(self, nodes):
        """Build rich context for retrieved nodes"""
        context = []
        
        # 1. Hierarchical Structure
        context.append("=== Hierarchical Structure ===")
        
        # Get all hierarchical parents for top K nodes
        hierarchy = {}
        for node in self.top_k_nodes:  # Use stored top K nodes
            chain = self._get_hierarchical_chain(node)
            for area in chain:
                level = self.graph.nodes[area].get('level', 0)
                if level > 0:  # Only include areas/clusters
                    if level not in hierarchy:
                        hierarchy[level] = set()
                    hierarchy[level].add(area)
        
        # Display hierarchy from top down
        for level in sorted(hierarchy.keys(), reverse=True):
            context.append(f"\nLevel {level}:")
            indent = "   " * level
            for area in sorted(hierarchy[level]):
                data = self.graph.nodes[area]
                context.append(f"{indent}{area}")
                if 'summary' in data:
                    context.append(f"{indent}Summary: {data['summary']}")
                
                # Show contained objects that are in our top K
                contained = [n for n in self.graph.neighbors(area)
                           if n in self.top_k_nodes]
                if contained:
                    context.append(f"{indent}Contains: {', '.join(contained)}")
        
        # 2. Object Information
        context.append("\n=== Object Information ===")
        for node in self.top_k_nodes:  # Only show info for top K nodes
            data = self.graph.nodes[node]
            context.append(f"\nObject: {node}")
            
            # Show hierarchical path
            chain = self._get_hierarchical_chain(node)
            if len(chain) > 1:
                path = " → ".join(reversed(chain))
                context.append(f"Located in: {path}")
            
            # Position
            if 'position' in data:
                pos = data['position']
                if isinstance(pos, dict):
                    context.append(f"Position: [x:{pos['x']:.2f}, y:{pos['y']:.2f}, z:{pos['z']:.2f}]")
                elif isinstance(pos, (list, tuple)):
                    context.append(f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            # Spatial relationships
            spatial_relations = []
            for neighbor, edge_data in self.graph[node].items():
                if ('relationship' in edge_data and 
                    edge_data['relationship'] not in ['part_of', 'contains']):
                    distance = edge_data.get('distance', 'unknown')
                    direction = edge_data.get('direction', '')
                    rel_str = f"{neighbor} is {direction}"
                    if isinstance(distance, (int, float)):
                        rel_str += f" ({distance:.1f}m away)"
                    spatial_relations.append(rel_str)
            
            if spatial_relations:
                context.append("Spatial relationships:")
                for rel in spatial_relations:
                    context.append(f"- {rel}")
        
        final_context = "\n".join(context)
        return final_context

    def extract_target_position(self, response):
        """Extract target position from LLM response"""
        try:
            # Extract the object name from the response
            object_name_match = re.search(r'<<(.+?)>>', response)
            if not object_name_match:
                print("No target object found in response")
                return None
                
            object_name = object_name_match.group(1)
            print(f"Found target object: {object_name}")
            
            # Find the node in the graph
            if object_name not in self.graph:
                print(f"Object {object_name} not found in graph")
                return None
                
            node_data = self.graph.nodes[object_name]
            if 'position' not in node_data:
                print(f"No position data for object {object_name}")
                return None
                
            position = node_data['position']
            print(f"Found position: {position}")
            
            # Convert position to proper format if needed
            if isinstance(position, dict):
                return position
            elif isinstance(position, (list, tuple)):
                return {
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': float(position[2])
                }
            
            print(f"Invalid position format: {position}")
            return None
            
        except Exception as e:
            print(f"Error extracting target position: {str(e)}")
            return None

    def _apply_hierarchical_boost(self, nodes, similarities, query_embedding):
        """Boost scores based on hierarchical relationships"""
        boosted_scores = similarities.copy()
        
        for i, node in enumerate(nodes):
            # Get hierarchical chain
            chain = self._get_hierarchical_chain(node)
            
            # Check if node is in a relevant area
            for area in chain:
                area_data = self.graph.nodes[area]
                if 'summary' in area_data and 'embedding' in area_data:
                    # Calculate semantic similarity between query and area summary
                    area_similarity = 1 - cosine(query_embedding[0], area_data['embedding'])
                    
                    # Boost score if area is relevant
                    if area_similarity > self.semantic_threshold:
                        boost_factor = 1.2  # Can be adjusted
                        original_score = boosted_scores[i]
                        boosted_scores[i] *= boost_factor
                        print(f"Boosting {node} (in {area}) from {original_score:.3f} to {boosted_scores[i]:.3f}")
        
        return boosted_scores

    def _apply_spatial_boost(self, nodes, similarities):
        """Boost scores based on spatial relationships"""
        boosted_scores = similarities.copy()
        
        # Find clusters of related objects
        for i, node in enumerate(nodes):
            neighbors = self._get_spatial_neighbors(node)
            neighbor_scores = []
            relevant_neighbors = []
            
            for neighbor in neighbors:
                if neighbor in nodes:
                    neighbor_idx = nodes.index(neighbor)
                    neighbor_scores.append(similarities[neighbor_idx])
                    relevant_neighbors.append(neighbor)
            
            if neighbor_scores:
                # Boost score if nearby objects are also relevant
                avg_neighbor_score = sum(neighbor_scores) / len(neighbor_scores)
                boost_factor = 1 + (avg_neighbor_score * 0.2)  # Adjustable
                original_score = boosted_scores[i]
                boosted_scores[i] *= boost_factor
                print(f"Boosting {node} (near {', '.join(relevant_neighbors)}) from {original_score:.3f} to {boosted_scores[i]:.3f}")
        
        return boosted_scores

    def _normalize_scores(self, scores):
        """Normalize scores to [0,1] range"""
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
