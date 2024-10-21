from lightrag import LightRAG, QueryParam
from .graph_builder import GraphBuilder
import networkx as nx

class EmbodiedRAG:
    def __init__(self, working_dir):
        self.graph_builder = GraphBuilder()
        self.rag = LightRAG(working_dir=working_dir)

    def load_graph_to_rag(self, graph_file="semantic_graph.gml"):
        # Load the graph from the GML file
        self.graph_builder.load_graph(graph_file)
        graph_data = self.graph_builder.G
        # Convert graph data to a format suitable for LightRAG insertion
        rag_data = self._convert_graph_to_rag_format(graph_data)
        self.rag.insert(rag_data)

    def _convert_graph_to_rag_format(self, graph):
        # Convert NetworkX graph to a format suitable for LightRAG
        rag_data = []
        for node, data in graph.nodes(data=True):
            node_text = f"Object: {node}, "
            node_text += ", ".join([f"{k}: {v}" for k, v in data.items() if k != 'id'])
            rag_data.append(node_text)
        for u, v, data in graph.edges(data=True):
            edge_text = f"Relationship: {u} to {v}, "
            edge_text += ", ".join([f"{k}: {v}" for k, v in data.items()])
            rag_data.append(edge_text)
        return rag_data

    def query(self, query_text, mode="hybrid"):
        return self.rag.query(query_text, param=QueryParam(mode=mode))

    def visualize_graph(self):
        self.graph_builder.visualize_graph()
