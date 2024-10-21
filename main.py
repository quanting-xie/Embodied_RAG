from embodied_nav.embodied_rag import EmbodiedRAG

def main():
    embodied_rag = EmbodiedRAG(working_dir="./embodied_nav_cache")
    
    # Load the existing graph into LightRAG
    embodied_rag.load_graph_to_rag()
    
    # Perform queries
    query = "What objects are near the center of the room?"
    result = embodied_rag.query(query)
    print(result)

if __name__ == "__main__":
    main()
