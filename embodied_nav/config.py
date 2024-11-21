from enum import Enum

class Config:

    # Path configurations
    PATHS = {
        'semantic_graphs_dir': 'semantic_graphs',  # Directory containing semantic graphs
        'latest_graph': 'enhanced_semantic_graph_semantic_graph_Building99_20241103_193232.gml',  # Your latest graph file
        'experiment_logs_dir': 'experiment_logs'  # Directory for experiment logs
    }
    
    # Spatial Relationship Parameters
    SPATIAL = {
        'cluster_distance_threshold': 5.0,  # Base clustering distance
        'proximity_threshold': 3.0,  # Distance threshold for proximity relationships
        'spatial_threshold': 3.0,  # Distance threshold for spatial relationships (reduced from 5.0)
        'vertical_threshold': 1.0,  # Threshold for vertical relationships
        'level_multiplier': 3.0,  # Multiplier for clustering threshold per level
    }

    # Retrieval Parameters
    RETRIEVAL = {
        'semantic_similarity_threshold': 0.6,  # threshold for semantic matching
        'top_k_default': 5,                    # default number of results to return
        'max_hierarchical_level': 3,          # maximum levels in hierarchy
        'max_parallel_paths': 3,              # maximum number of parallel hierarchical chains
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

    # Cardinal Directions(Don't need to change unless you want to change the cardinal directions)
    CARDINAL_DIRECTIONS = {
        'north': (0, 1),   # +y
        'south': (0, -1),  # -y
        'east': (1, 0),    # +x
        'west': (-1, 0)    # -x
    }

    # Query Configuration
    QUERIES = {
        'implicit': {
            'default': "Where can I eat my lunch?",
            'examples': [
                "I need a place to work",
                "Where can I relax?",
                "I'm looking for a place to have a meeting"
            ]
        },
        'explicit': {
            'default': "Find the dining table",
            'examples': [
                "Navigate to the nearest chair",
                "Find the coffee table",
                "Locate the bookshelf"
            ]
        },
        'global': {
            'default': "What are the main types of furniture in this environment?",
            'examples': [
                "Describe the layout of this space",
                "What are the different functional areas?",
                "Give me an overview of this environment"
            ]
        }
    }

    # Retrieval Method Configuration
    RETRIEVAL_METHODS = {
        'semantic': 'Semantic-based retrieval',
        'llm_hierarchical': 'LLM-guided hierarchical retrieval'
    }

    # Online Semantic Forest Parameters
    ONLINE_SEMANTIC = {
        'forest_update_interval': 10,  # seconds between semantic forest updates
        'max_update_timeout': 60,      # maximum seconds to wait for final update
        'min_objects_for_update': 5    # minimum number of objects needed before updating
    }

