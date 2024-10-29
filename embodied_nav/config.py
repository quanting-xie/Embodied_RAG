class Config:
    # Spatial Relationship Parameters
    SPATIAL = {
        'cluster_distance_threshold': 10.0,  # meters, for hierarchical clustering
        'proximity_threshold': 3.0,        # meters, for spatial relationships
        'spatial_threshold': 5.0,          # meters, for cardinal directions
        'vertical_threshold': 1.0,          # meters, for above/below relationships
    }

    # Retrieval Parameters
    RETRIEVAL = {
        'semantic_similarity_threshold': 0.25,  # threshold for semantic matching
        'top_k_default': 5,                    # default number of results to return
        'max_hierarchical_level': 10,          # maximum levels in hierarchy
    }

    # LLM Parameters
    LLM = {
        'model': 'gpt-4o',
        'temperature': 0.7,
        'max_tokens': 500,
    }

    # Graph Parameters
    GRAPH = {
        'drone_node_distance': 3.0,  # minimum distance between drone nodes
        'edge_types': {
            'spatial': ['north', 'south', 'east', 'west', 'above', 'below'],
            'hierarchical': ['part_of'],
        }
    }

    # Cardinal Directions
    CARDINAL_DIRECTIONS = {
        'north': (0, 1),   # +y
        'south': (0, -1),  # -y
        'east': (1, 0),    # +x
        'west': (-1, 0)    # -x
    } 