import os
import sys
sys.path.append('.')
from embodied_nav.embodied_rag import EmbodiedRAG
from embodied_nav.embodied_retriever import RetrievalMethod
import asyncio
import time
import ipdb
import logging
import argparse
import json
from datetime import datetime
from embodied_nav.config import Config
from pathlib import Path

_cached_rag = None

method_map = {
    'semantic': RetrievalMethod.SEMANTIC,
    'llm_hierarchical': RetrievalMethod.LLM_HIERARCHICAL
}


def setup_logging(method_name, query_type):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('experiment_logs', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'experiment_logs/{method_name}_{query_type}_{timestamp}.log'
    
    # Get the logger
    logger = logging.getLogger('experiment')
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


async def interactive_session(embodied_rag, data):
    """Handle the interactive query session"""
    for item in data:
        retries = 10 
        while retries > 0:
            try:
                await embodied_rag.query(
                    item['query'],
                    query_type=item['type'],
                    start_position=None,
                    data_construction=True,
                )
                break 
            except Exception as e:
                retries -= 1
                if retries == 0:
                    print(f"Query failed after 3 attempts: {item['query']} | Error: {e}")
                else:
                    print(f"Query failed, retrying... ({3 - retries} retries left)")


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive Embodied Navigation System')
    parser.add_argument('--method', type=str, required=True, 
                       choices=list(Config.RETRIEVAL_METHODS.keys()),
                       help='Retrieval method to use')
    parser.add_argument('--query-type', type=str, required=True,
                       choices=list(Config.QUERIES.keys()),
                       help='Type of queries to handle')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.method, args.query_type)
    
    try:
        # Initialize EmbodiedRAG
        global _cached_rag
        if _cached_rag is None:
            logger.info("Creating new EmbodiedRAG instance...")
            _cached_rag = EmbodiedRAG(
                working_dir="./embodied_nav_cache",
                airsim_utils=None,
                retrieval_method=method_map[args.method]
            )
        
        # Load graph
        project_root = Path(__file__).resolve().parent.parent
        semantic_graphs_dir = project_root / "semantic_graphs"
        
        # List all graph files and get the latest one
        graph_files = list(semantic_graphs_dir.glob("*.gml"))
        if not graph_files:
            raise FileNotFoundError(f"No .gml files found in {semantic_graphs_dir}")
        
        # latest_graph = max(graph_files, key=lambda x: x.stat().st_mtime)
        # logger.info(f"Found latest graph: {latest_graph.name}")
        latest_graph = 'semantic_graphs/enhanced_semantic_graph_semantic_graph_Building99_20241118_160313.gml'

        # Load the graph
        graph = await _cached_rag.load_graph_to_rag(str(latest_graph))
        if not graph:
            raise RuntimeError("Failed to load graph")
        
        with open('benchmark/data/query.jsonl', 'r') as file:
            data = [json.loads(line) for line in file]

        await interactive_session(_cached_rag, data)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())