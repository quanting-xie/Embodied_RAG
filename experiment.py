from embodied_nav.embodied_rag import EmbodiedRAG
from embodied_nav.embodied_retriever import RetrievalMethod
import asyncio
import airsim
from embodied_nav.airsim_utils import AirSimUtils, AirSimClientWrapper
import time
import logging
import argparse
from datetime import datetime
from embodied_nav.config import Config
import os
from pathlib import Path
import threading

_cached_rag = None

method_map = {
    'semantic': RetrievalMethod.SEMANTIC,
    'llm_hierarchical': RetrievalMethod.LLM_HIERARCHICAL,
    'hybrid': RetrievalMethod.HYBRID
}


def setup_logging(method_name, query_type):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'experiment_logs/{method_name}_{query_type}_{timestamp}.log'
    
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
    
    # Setup logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


async def interactive_session(embodied_rag, airsim_utils, client, query_type, logger):
    """Handle the interactive query session"""
    print("\nInteractive Mode Started!")
    print(f"Query Type: {query_type}")
    print("\nAvailable example queries:")
    for example in Config.QUERIES[query_type]['examples']:
        print(f"- {example}")
    print("\nType 'exit' to end the session")
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'exit':
                break
                
            current_pose = client.simGetVehiclePose()
            current_position = airsim_utils.vector3r_to_dict(current_pose.position)
            
            # Run the query
            start_time = time.time()
            retrieved_nodes = await embodied_rag.retriever.retrieve(query, query_type=query_type)
            
            # Log retrieval results
            logger.info("\nRetrieval Results:")
            logger.info(f"Number of retrieved nodes: {len(retrieved_nodes)}")
            for node in retrieved_nodes:
                logger.info(f"- {node}")
            
            # Generate and execute response
            response, waypoints = await embodied_rag.query(
                query,
                query_type=query_type,
                start_position=current_position if query_type != "global" else None
            )
            
            # Log results
            logger.info(f"\nResponse: {response}")
            if waypoints:
                if isinstance(waypoints, bool):
                    logger.info(f"Navigation success: {waypoints}")
                else:
                    logger.info(f"Waypoints: {waypoints}")
            logger.info(f"Query completed in {time.time() - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            logger.info("\nSession interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            continue

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
        # Initialize AirSim with wrapper
        client = AirSimClientWrapper()
        client.confirmConnection()
        print("Connected!")
        
        # Reset and set initial position
        client.reset()
        
        # Define minimum height
        MIN_HEIGHT = -2.0  # 2 meters above ground
        
        # Set initial pose with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                initial_pose = airsim.Pose(
                    position_val=airsim.Vector3r(0, 0, MIN_HEIGHT),
                    orientation_val=airsim.Quaternionr()
                )
                client.simSetVehiclePose(initial_pose, True)
                
                # Enable API control and arm
                client.enableApiControl(True)
                client.armDisarm(True)
                
                # Take off and maintain minimum height
                logger.info("Taking off...")
                client.takeoffAsync().join()
                client.moveToZAsync(MIN_HEIGHT, 2).join()
                logger.info("Ready to fly!")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Initialization attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
        
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
        project_root = Path(__file__).resolve().parent
        semantic_graphs_dir = project_root / "semantic_graphs"
        
        # List all graph files and get the latest one
        graph_files = list(semantic_graphs_dir.glob("*.gml"))
        if not graph_files:
            raise FileNotFoundError(f"No .gml files found in {semantic_graphs_dir}")
        
        latest_graph = max(graph_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found latest graph: {latest_graph.name}")
        
        # Load the graph
        graph = await _cached_rag.load_graph_to_rag(str(latest_graph))
        if not graph:
            raise RuntimeError("Failed to load graph")
        
        # Initialize AirSimUtils with graph and client
        airsim_utils = AirSimUtils(client, graph=graph)
        _cached_rag.airsim_utils = airsim_utils
        
        # Test navigation capability
        print("\nTesting navigation capability...")
        try:
            # Get a sample object node from the graph
            object_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'object']
            if object_nodes:
                test_node = object_nodes[0]
                test_pos = graph.nodes[test_node]['position']
                print(f"Testing navigation to {test_node} at position {test_pos}")
                
                # Try to navigate
                success = airsim_utils.direct_to_waypoint(test_pos)
                print(f"Navigation test {'successful' if success else 'failed'}")
            else:
                print("No object nodes found for navigation test")
        except Exception as e:
            print(f"Navigation test failed: {str(e)}")
        
        # Start height maintenance thread
        def maintain_height():
            while True:
                try:
                    state = client.getMultirotorState()
                    current_height = state.kinematics_estimated.position.z_val
                    if current_height > MIN_HEIGHT + 0.1:  # If too high
                        client.moveToZAsync(MIN_HEIGHT, 1).join()
                    time.sleep(0.1)
                except Exception as e:
                    # Only log non-IOLoop errors
                    if "IOLoop" not in str(e):
                        logger.error(f"Error in height maintenance: {str(e)}")
                    time.sleep(0.1)
        
        # Start height maintenance thread as daemon
        height_thread = threading.Thread(target=maintain_height, daemon=True)
        height_thread.start()
        
        # Start interactive session
        await interactive_session(_cached_rag, airsim_utils, client, args.query_type, logger)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        try:
            # Ensure proper landing
            logger.info("Landing...")
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception as e:
            logger.error(f"Error during landing: {str(e)}")
        logger.info("Session ended")

if __name__ == "__main__":
    asyncio.run(main())