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
    print("Attempting takeoff...")

    client.takeoffAsync().join()

    # Initialize EmbodiedRAG with AirSimUtils
    embodied_rag = EmbodiedRAG(working_dir="./embodied_nav_cache", airsim_utils=airsim_utils)
    
    # Load graph
    await embodied_rag.load_graph_to_rag("/home/quanting/Embodied_RAG/embodied_nav/enhanced_semantic_graph.gml")
    

    # Get current drone position
    current_pose = client.simGetVehiclePose()
    current_position = airsim_utils.vector3r_to_dict(current_pose.position)
    
    # print("\n" + "="*50)
    # print("IMPLICIT QUERY: 'Where can I sit and eat?'")
    # print("="*50)
    # # Implicit query
    # response, waypoints = await embodied_rag.query(
    #     "Where can I sit and eat?",
    #     query_type="implicit",
    #     start_position=current_position
    # )
    # print("\nResponse:")
    # print(response)
    # print("\nWaypoints:")
    # print(waypoints)

    print("\n" + "="*50)
    print("EXPLICIT QUERY: 'Find the chair'")
    print("="*50)
    # Explicit query
    response, waypoints = await embodied_rag.query(
        "Find the chair",
        query_type="explicit",
        start_position=current_position
    )
    print("\nResponse:")
    print(response)
    print("\nWaypoints:")
    print(waypoints)

    # Add more debugging information
    print("\nDebugging query results:")
    print(f"Response type: {type(response)}")
    print(f"Waypoints type: {type(waypoints)}")
    print(f"Waypoints content: {waypoints}")

    # Add debugging information
    print("\nDebugging generate_waypoints:")
    print(f"Start position: {current_position}")
    target_position = waypoints[0] if waypoints else None
    print(f"Target position: {target_position}")
    
    if target_position:
        debug_waypoints = airsim_utils.generate_waypoints(current_position, target_position, planning_mode="direct")
        print(f"Debug waypoints: {debug_waypoints}")

    # print("\n" + "="*50)
    # print("GLOBAL QUERY: 'What are the main types of furniture in this environment?'")
    # print("="*50)
    # # Global query
    # response, _ = await embodied_rag.query(
    #     "What are the main types of furniture in this environment?",
    #     query_type="global"
    # )
    # print("\nResponse:")
    # print(response)

if __name__ == "__main__":
    asyncio.run(main())
