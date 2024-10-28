from lightrag import LightRAG, QueryParam
from .graph_builder import GraphBuilder
from .spatial_relationship_extractor import SpatialRelationshipExtractor
from .embodied_retriever import EmbodiedRetriever
import networkx as nx
import logging
import airsim
import numpy as np

logger = logging.getLogger(__name__)

class EmbodiedRAG:
    def __init__(self, working_dir, airsim_utils=None):
        self.graph_builder = GraphBuilder()
        self.rag = LightRAG(working_dir=working_dir)
        self.relationship_extractor = SpatialRelationshipExtractor(
            llm_func=self.rag.llm_model_func,  # Changed from llm to llm_func
            cluster_distance_threshold=5.0,
            proximity_threshold=10.0,
            vertical_threshold=2.0
        )
        self.retriever = None
        self.airsim_utils = airsim_utils  # Add AirSimUtils instance

    async def load_graph_to_rag(self, enhanced_graph_file):
        # Load the enhanced graph
        enhanced_graph = nx.read_gml(enhanced_graph_file)
        print("\nDebug: Graph Loading Details:")
        print(f"Number of nodes: {len(enhanced_graph.nodes())}")
        
        # Check if we need to generate embeddings
        needs_embeddings = False
        for node, data in enhanced_graph.nodes(data=True):
            if 'embedding' not in data:
                needs_embeddings = True
                break
        
        # Generate embeddings only if needed
        if needs_embeddings:
            print("\nGenerating embeddings for nodes...")
            for node, data in enhanced_graph.nodes(data=True):
                if 'embedding' not in data:
                    node_text = f"{data.get('label', node)} "
                    if 'summary' in data:
                        node_text += data['summary']
                    
                    embedding = await self.rag.embedding_func([node_text])
                    # Convert numpy array to list before saving
                    enhanced_graph.nodes[node]['embedding'] = embedding[0].tolist()
                    print(f"Generated embedding for {node}")
            
            # Save the graph with embeddings
            nx.write_gml(enhanced_graph, enhanced_graph_file)
            print("Saved graph with embeddings")
        else:
            print("Loading existing embeddings from graph file")
            # Convert loaded embeddings back to numpy arrays if needed
            for node in enhanced_graph.nodes():
                if 'embedding' in enhanced_graph.nodes[node]:
                    enhanced_graph.nodes[node]['embedding'] = np.array(enhanced_graph.nodes[node]['embedding'])

        # Initialize the retriever with the enhanced graph
        self.retriever = EmbodiedRetriever(enhanced_graph, self.rag.embedding_func)
        return enhanced_graph
    
    def _convert_graph_to_rag_format(self, graph):
        rag_data = []
        
        for node, data in graph.nodes(data=True):
            # Use summary as the node name if it exists, otherwise use the original node id
            node_name = data.get('summary', node)
            node_text = f"Node: {node_name}, "
            node_text += f"Level: {data.get('level', 'N/A')}, "
            
            if 'position' in data:
                pos = data['position']
                if isinstance(pos, dict):
                    # If pos is a dictionary, extract x, y, z values
                    node_text += f"Position: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f}), "
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    # If pos is a list or tuple, use the first three elements
                    node_text += f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                else:
                    # If pos is in an unexpected format, add a note about it
                    node_text += f"Position: (format unknown), "
            
            # Add other attributes, excluding those we've already handled
            node_text += ", ".join([f"{k}: {v}" for k, v in data.items() 
                                    if k not in ['id', 'level', 'position', 'summary']])
            
            rag_data.append(node_text)
        
        for u, v, data in graph.edges(data=True):
            # Use summaries as node names if they exist
            u_name = graph.nodes[u].get('summary', u)
            v_name = graph.nodes[v].get('summary', v)
            
            edge_text = f"Relationship: {u_name} to {v_name}, "
            edge_text += f"Type: {data.get('relationship', 'Unknown')}, "
            
            # Add other edge attributes
            edge_text += ", ".join([f"{k}: {v}" for k, v in data.items() 
                                    if k != 'relationship'])
            
            rag_data.append(edge_text)
        
        return rag_data

    async def query(self, query_text, query_type="explicit", start_position=None):
        print(f"\n==== Processing Query: '{query_text}' ====")
        
        # Get retrieved nodes
        retrieved_nodes = await self.retriever.retrieve(query_text, query_type=query_type)
        
        print("\nRetrieved Objects:")
        for node in retrieved_nodes:
            print(f"- {node}")

        # Generate response
        response = await self.retriever.generate_response(query_text, retrieved_nodes, query_type)
        print("\nGenerated Response:")
        print(response)
        
        # Move to target for explicit and implicit queries
        if query_type in ["explicit", "implicit"] and self.airsim_utils is not None:
            # Extract target position from response
            target_position = self.retriever.extract_target_position(response)
            print (f"\nTarget position: {target_position}")
            if target_position:
                success = self.airsim_utils.direct_to_waypoint(target_position, velocity=5)
                return response, success
            else:
                print("\nNo target position found.")
                return response, False
        
        return response

    def visualize_graph(self):
        self.graph_builder.visualize_graph()
