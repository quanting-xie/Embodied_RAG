import networkx as nx
from embodied_nav.spatial_relationship_extractor import SpatialRelationshipExtractor
from embodied_nav.llm import LLMInterface
import logging
import asyncio
import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import argparse

# Define paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_GRAPHS_DIR = os.path.join(PROJECT_ROOT, "semantic_graphs")

class SemanticGraphBuilder:
    """Simple utility class for loading and processing semantic graphs"""
    def __init__(self):
        self.G = nx.Graph()
        self.spatial_clusters = {}  # Store spatial clusters
        self.cluster_level = 1  # Start cluster levels from 1

    def load_graph(self, filename):
        """Load graph from GML file"""
        self.G = nx.read_gml(filename)
        logging.info(f"Loaded graph with {len(self.G.nodes())} nodes")

    def get_objects(self):
        """Get all non-drone objects from the graph"""
        objects = [
            {'id': node, **{k:v for k,v in data.items() if k != 'level'}}
            for node, data in self.G.nodes(data=True)
            if data.get('type') != 'drone'
        ]
        logging.info(f"Found {len(objects)} objects to process")
        return objects

async def generate_semantic_forest(initial_graph_file, enhanced_graph_file):
    """Generate enhanced semantic forest with spatial relationships"""
    print("\n=== Starting Semantic Forest Generation ===")
    
    # Initialize components
    graph = SemanticGraphBuilder()
    llm_interface = LLMInterface()
    relationship_extractor = SpatialRelationshipExtractor(llm_interface)

    # Load and process graph
    print("\nLoading initial graph...")
    graph.load_graph(initial_graph_file)
    objects = graph.get_objects()
    
    # Extract spatial relationships and create hierarchical clusters
    print("\nExtracting spatial relationships and creating clusters...")
    print(f"Processing {len(objects)} objects. This may take a while...")
    
    # Process all objects at once for proper clustering
    enhanced_graph = await relationship_extractor.extract_relationships(objects)
    
    print("\nMerging graphs...")
    # Merge original and enhanced graphs
    merged_graph = nx.Graph()
    
    # Add nodes from both graphs
    print("Adding nodes...")
    for node, data in tqdm(graph.G.nodes(data=True), desc="Original nodes"):
        merged_graph.add_node(node, **data)
    
    for node, data in tqdm(enhanced_graph.nodes(data=True), desc="Enhanced nodes"):
        if node in merged_graph:
            merged_graph.nodes[node].update(data)
        else:
            merged_graph.add_node(node, **data)
    
    # Add edges from both graphs
    print("Adding edges...")
    total_edges = len(graph.G.edges()) + len(enhanced_graph.edges())
    with tqdm(total=total_edges, desc="Adding edges") as pbar:
        for u, v, data in graph.G.edges(data=True):
            merged_graph.add_edge(u, v, **data)
            pbar.update(1)
        
        for u, v, data in enhanced_graph.edges(data=True):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v, **data)
            pbar.update(1)
    
    # Save merged graph
    print("\nSaving enhanced graph...")
    nx.write_gml(merged_graph, enhanced_graph_file)
    
    print("\n=== Semantic Forest Generation Complete ===")
    print(f"Enhanced graph saved to: {enhanced_graph_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Semantic Forest')
    parser.add_argument('--input', type=str, required=True, help='Input GML file path')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        initial_graph_file = args.input
        # Generate enhanced filename from input filename
        enhanced_graph_file = initial_graph_file.replace('direct_semantic_graph', 'enhanced_semantic_graph')
        
        print(f"Using graph file: {initial_graph_file}")
        print(f"Will save enhanced graph to: {enhanced_graph_file}")
        asyncio.run(generate_semantic_forest(initial_graph_file, enhanced_graph_file))
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
