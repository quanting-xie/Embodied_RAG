from embodied_nav.embodied_rag import EmbodiedRAG
import asyncio

async def main():
    embodied_rag = EmbodiedRAG("./embodied_nav_cache")
    await embodied_rag.load_graph_to_rag(
        "/home/quanting/Embodied_RAG/embodied_nav/initial_semantic_graph.gml",
        "/home/quanting/Embodied_RAG/embodied_nav/enhanced_semantic_graph.gml"
    )

    # Explicit query
    response, waypoints = await embodied_rag.query("Find the red chair", query_type="explicit", start_position=(0, 0, 0))
    print(response)
    print("Waypoints:", waypoints)

    # Implicit query
    response, waypoints = await embodied_rag.query("Where can I sit and eat?", query_type="implicit", start_position=(0, 0, 0))
    print(response)
    print("Waypoints:", waypoints)

    # Global query
    response = await embodied_rag.query("What are the main types of furniture in this environment?", query_type="global")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
