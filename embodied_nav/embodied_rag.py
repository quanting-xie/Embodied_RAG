from lightrag import LightRAG, QueryParam
from .graph_builder import GraphBuilder
from .spatial_relationship_extractor import SpatialRelationshipExtractor
from .embodied_retriever import EmbodiedRetriever
import networkx as nx

class EmbodiedRAG:
    def __init__(self, working_dir):
        self.graph_builder = GraphBuilder()
        self.rag = LightRAG(working_dir=working_dir)
        self.relationship_extractor = SpatialRelationshipExtractor(
            llm_func=self.rag.llm_model_func,  # Changed from llm to llm_func
            cluster_distance_threshold=5.0,
            proximity_threshold=10.0,
            vertical_threshold=2.0
        )
        self.retriever = None

    async def load_graph_to_rag(self, enhanced_graph_file):
        # Load the enhanced graph
        enhanced_graph = nx.read_gml(enhanced_graph_file)
        print(f"Enhanced graph loaded with {len(enhanced_graph.nodes)} nodes and {len(enhanced_graph.edges)} edges")

        # Initialize the retriever with the enhanced graph
        self.retriever = EmbodiedRetriever(enhanced_graph, self.rag.embedding_func)
        
        # Convert graph data to a format suitable for LightRAG insertion
        rag_data = self._convert_graph_to_rag_format(enhanced_graph)
        await self.rag.ainsert(rag_data)

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

    async def query(self, query_text, query_type="global", start_position=None):
        if self.retriever is None:
            raise ValueError("Graph not loaded. Call load_graph_to_rag() first.")
        
        retrieved_nodes = await self.retriever.retrieve(query_text, query_type=query_type)
        response = await self.retriever.generate_response(query_text, retrieved_nodes, query_type)
        
        if query_type in ["explicit", "implicit"] and start_position:
            waypoints = self.retriever.generate_waypoints(start_position, retrieved_nodes)
            return response, waypoints
        else:
            return response

    def visualize_graph(self):
        self.graph_builder.visualize_graph()
