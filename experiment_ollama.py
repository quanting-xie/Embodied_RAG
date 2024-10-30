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

    # Initialize EmbodiedRAG with Ollama
    embodied_rag = EmbodiedRAG(
        working_dir="./embodied_nav_cache", 
        airsim_utils=airsim_utils,
        use_ollama=True
    )
    
    # Load graph
    await embodied_rag.load_graph_to_rag("/home/quanting/Embodied_RAG/embodied_nav/enhanced_semantic_graph.gml")
    
    # Get current drone position
    current_pose = client.simGetVehiclePose()
    current_position = airsim_utils.vector3r_to_dict(current_pose.position)
    
    # Run the same experiments but with Ollama
    print("\n" + "="*50)
    print("TESTING OLLAMA-POWERED QUERIES")
    print("="*50)

    queries = [
        ("Where can I eat my lunch?", "implicit"),
        ("Find the dining table", "explicit"),
        ("What are the main types of furniture in this environment?", "global")
    ]

    for query_text, query_type in queries:
        print(f"\n{'='*50}")
        print(f"{query_type.upper()} QUERY TEST")
        print(f"Query: '{query_text}'")
        print("="*50)

        response, waypoints = await embodied_rag.query(
            query_text,
            query_type=query_type,
            start_position=current_position if query_type != "global" else None
        )
        print("\nResponse:")
        print(response)
        if query_type != "global":
            print("\nWaypoints:")
            print(waypoints)

    # Landing sequence
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("Initiating landing sequence...")
    client.landAsync().join()
    print("Landing successful!")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main()) 