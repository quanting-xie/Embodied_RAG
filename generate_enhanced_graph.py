import networkx as nx
from embodied_nav.graph_builder import GraphBuilder
from embodied_nav.spatial_relationship_extractor import SpatialRelationshipExtractor
from embodied_nav.llm import LLMInterface
import asyncio

async def generate_enhanced_graph(initial_graph_file, enhanced_graph_file):
    graph_builder = GraphBuilder()
    graph_builder.load_graph(initial_graph_file)
    objects = graph_builder.get_objects()

    llm_interface = LLMInterface()
    relationship_extractor = SpatialRelationshipExtractor(llm_interface.generate_response)

    enhanced_graph = await relationship_extractor.extract_relationships(objects)
    
    # Merge the original graph with the spatial relationship graph
    merged_graph = nx.compose(graph_builder.G, enhanced_graph)
    
    # Save the enhanced graph
    nx.write_gml(merged_graph, enhanced_graph_file)
    print(f"Enhanced graph saved to {enhanced_graph_file}")

if __name__ == "__main__":
    initial_graph_file = "/home/quanting/Embodied_RAG/embodied_nav/semantic_graph.gml"
    enhanced_graph_file = "/home/quanting/Embodied_RAG/embodied_nav/enhanced_semantic_graph.gml"
    asyncio.run(generate_enhanced_graph(initial_graph_file, enhanced_graph_file))
