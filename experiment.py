from embodied_nav.embodied_rag import EmbodiedRAG
import asyncio
import airsim
from embodied_nav.airsim_utils import AirSimUtils

async def main():
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    airsim_utils = AirSimUtils(client)

    # Take off
    print("\nInitializing drone...")
    print("="*50)
    print("Attempting takeoff...")
    client.takeoffAsync().join()
    print("Takeoff successful!")
    print("="*50)

    # Initialize EmbodiedRAG with AirSimUtils
    embodied_rag = EmbodiedRAG(working_dir="./embodied_nav_cache", airsim_utils=airsim_utils)
    
    # Load graph
    await embodied_rag.load_graph_to_rag("/home/quanting/Embodied_RAG/embodied_nav/enhanced_semantic_graph.gml")
    
    # Get current drone position
    current_pose = client.simGetVehiclePose()
    current_position = airsim_utils.vector3r_to_dict(current_pose.position)
    
    # Example 1: Implicit Query
    print("\n" + "="*50)
    print("IMPLICIT QUERY TEST")
    print("Query: 'Where can I eat my lunch?'")
    print("="*50)
    
    response, waypoints = await embodied_rag.query(
        "Where can I eat my lunch?",
        query_type="implicit",
        start_position=current_position
    )
    print("\nResponse:")
    print(response)
    print("\nWaypoints:")
    print(waypoints)

    # Example 2: Explicit Query
    print("\n" + "="*50)
    print("EXPLICIT QUERY TEST")
    print("Query: 'Find the dining table'")
    print("="*50)
    
    response, waypoints = await embodied_rag.query(
        "Find the dining table",
        query_type="explicit",
        start_position=current_position
    )
    print("\nResponse:")
    print(response)
    print("\nWaypoints:")
    print(waypoints)

    # Example 3: Global Query
    print("\n" + "="*50)
    print("GLOBAL QUERY TEST")
    print("Query: 'What are the main types of furniture in this environment?'")
    print("="*50)
    
    response, _ = await embodied_rag.query(
        "What are the main types of furniture in this environment?",
        query_type="global"
    )
    print("\nResponse:")
    print(response)

    # Landing sequence
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("Initiating landing sequence...")
    client.landAsync().join()
    print("Landing successful!")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
