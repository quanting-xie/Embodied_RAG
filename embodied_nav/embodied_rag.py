from lightrag import LightRAG, QueryParam
from .airsim_explorer import AirSimExplorer
from .graph_builder import GraphBuilder

class EmbodiedRAG:
    def __init__(self, working_dir):
        self.explorer = AirSimExplorer()
        self.graph_builder = GraphBuilder()
        self.rag = LightRAG(working_dir=working_dir)

    def explore_and_build_graph(self):
        self.explorer.take_off()
        self.explorer.keyboard_control()
        self.graph_builder.generate_relationships()
        self.graph_builder.save_graph("semantic_graph.gml")

    def load_graph_to_rag(self):
        self.graph_builder.load_graph("semantic_graph.gml")
        graph_data = self.graph_builder.G
        # Convert graph data to a format suitable for LightRAG insertion
        rag_data = self._convert_graph_to_rag_format(graph_data)
        self.rag.insert(rag_data)

    def _convert_graph_to_rag_format(self, graph):
        # Convert NetworkX graph to a format suitable for LightRAG
        # This might involve creating text representations of nodes and edges
        rag_data = []
        for node, data in graph.nodes(data=True):
            node_text = f"Object: {node}, ID: {data['id']}, Position: {data['position']}"
            rag_data.append(node_text)
        for edge in graph.edges(data=True):
            edge_text = f"Relationship: {edge[0]} to {edge[1]}, {edge[2].get('relationship', '')}"
            rag_data.append(edge_text)
        return rag_data

    def query(self, query_text, mode="hybrid"):
        return self.rag.query(query_text, param=QueryParam(mode=mode))

