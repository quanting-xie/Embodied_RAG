from embodied_nav.embodied_rag import EmbodiedRAG
import asyncio
import airsim
from embodied_nav.airsim_utils import AirSimUtils

async def main():
    # Initialize AirSim client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Initialize AirSimUtils
    airsim_utils = AirSimUtils(client)
    
    # Initialize EmbodiedRAG with AirSimUtils
    embodied_rag = EmbodiedRAG(working_dir="./embodied_nav_cache", airsim_utils=airsim_utils)
    
    # Load graph
    await embodied_rag.load_graph_to_rag("/Users/danielxie/Embodied_RAG/embodied_nav/enhanced_semantic_graph.gml")
    
    # Get current drone position
    current_pose = client.simGetVehiclePose()
    current_position = AirSimUtils.vector3r_to_dict(current_pose.position)
    
    print("\n" + "="*50)
    print("IMPLICIT QUERY: 'Where can I sit and eat?'")
    print("="*50)
    # Implicit query
    response, waypoints = await embodied_rag.query(
        "Where can I sit and eat?",
        query_type="implicit",
        start_position=current_position
    )
    print("\nResponse:")
    print(response)
    print("\nWaypoints:")
    print(waypoints)

    print("\n" + "="*50)
    print("EXPLICIT QUERY: 'Find the red chair'")
    print("="*50)
    # Explicit query
    response, waypoints = await embodied_rag.query(
        "Find the red chair",
        query_type="explicit",
        start_position=current_position
    )
    print("\nResponse:")
    print(response)
    print("\nWaypoints:")
    print(waypoints)

    print("\n" + "="*50)
    print("GLOBAL QUERY: 'What are the main types of furniture in this environment?'")
    print("="*50)
    # Global query
    response, _ = await embodied_rag.query(
        "What are the main types of furniture in this environment?",
        query_type="global"
    )
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
