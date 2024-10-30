import networkx as nx
from embodied_nav.spatial_relationship_extractor import SpatialRelationshipExtractor
from embodied_nav.llm import LLMInterface
import logging
import asyncio

class SemanticGraphBuilder:
    """Simple utility class for loading and processing semantic graphs"""
    def __init__(self):
        self.G = nx.Graph()

    def load_graph(self, filename):
        """Load graph from GML file"""
        self.G = nx.read_gml(filename)
        logging.info(f"Loaded graph with {len(self.G.nodes())} nodes")

    def get_objects(self):
        """Get all non-drone objects from the graph"""
        return [
            {'id': node, **data} 
            for node, data in self.G.nodes(data=True)
            if data.get('type') != 'drone'  # Filter out drone nodes
        ]

async def generate_semantic_forest(initial_graph_file, enhanced_graph_file):
    """Generate enhanced semantic forest with spatial relationships"""
    # Initialize components
    graph = SemanticGraphBuilder()
    llm_interface = LLMInterface()
    relationship_extractor = SpatialRelationshipExtractor(llm_interface)

    # Load and process graph
    graph.load_graph(initial_graph_file)
    objects = graph.get_objects()
    
    # Extract spatial relationships
    enhanced_graph = await relationship_extractor.extract_relationships(objects)
    
    # Merge graphs and save
    merged_graph = nx.compose(graph.G, enhanced_graph)
    nx.write_gml(merged_graph, enhanced_graph_file)
    print(f"Enhanced graph saved to {enhanced_graph_file}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # File paths
    initial_graph_file = "/home/quanting/Embodied_RAG/embodied_nav/semantic_graph.gml"
    enhanced_graph_file = "/home/quanting/Embodied_RAG/embodied_nav/enhanced_semantic_graph.gml"
    
    # Run generation
    asyncio.run(generate_semantic_forest(initial_graph_file, enhanced_graph_file))
