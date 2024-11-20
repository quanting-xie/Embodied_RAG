from .spatial_relationship_extractor import SpatialRelationshipExtractor
from .embodied_retriever import EmbodiedRetriever
import networkx as nx
import logging
import numpy as np
from openai import AsyncOpenAI
import json
import os
from .embodied_retriever import EmbodiedRetriever, RetrievalMethod
from .config import Config
import time
import pickle
from tqdm import tqdm
import asyncio
from itertools import islice
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbodiedRAG:
    # Class-level cache for graph and embeddings
    _cached_graph = None
    _cached_graph_path = None
    _cached_embeddings = {}  # Store numpy embeddings separately

    def __init__(self, working_dir, airsim_utils=None, retrieval_method=RetrievalMethod.SEMANTIC):
        self.working_dir = working_dir
        self.logger = logging.getLogger('experiment')
        self.cache_file = os.path.join(working_dir, "llm_response_cache.json")
        self.client = AsyncOpenAI()
        
        if retrieval_method == RetrievalMethod.SEMANTIC:
            from .ollama_llm import OllamaInterface
            self.llm = OllamaInterface()
        else:
            from .llm import LLMInterface
            self.llm = LLMInterface()
        
        self.relationship_extractor = SpatialRelationshipExtractor(self)
        self.retriever = None
        self.airsim_utils = airsim_utils
        self.retrieval_method = retrieval_method
        self.graph = None
        self._load_cache()

    def _load_cache(self):
        """Load LLM response cache from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                self.llm_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.llm_cache = {}
            os.makedirs(self.working_dir, exist_ok=True)

    async def embedding_func(self, texts):
        """Generate embeddings using OpenAI API"""
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [np.array(embedding.embedding) for embedding in response.data]


    def _normalize_graph_heights(self, graph, safe_height=-0.8):
        """Normalize Z coordinates and invert for AirSim's coordinate system"""
        print("\nAnalyzing and inverting node heights for AirSim:")
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'drone' and 'position' in data:
                if isinstance(data['position'], dict):
                    original_height = data['position'].get('z')
                    # Invert Z for AirSim
                    data['position']['z'] = -original_height
                    print(f"Node height: {original_height} -> {data['position']['z']}")
                elif isinstance(data['position'], (list, tuple)):
                    x, y, original_height = data['position']
                    # Invert Z for AirSim
                    data['position'] = (x, y, -original_height)
                    print(f"Node height: {original_height} -> {-original_height}")

    async def load_graph_to_rag(self, enhanced_graph_file):
        """Load and enhance graph with embeddings"""
        start_time = time.time()
                
        # Verify file exists
        if not os.path.exists(enhanced_graph_file):
            raise FileNotFoundError(f"Graph file not found: {enhanced_graph_file}")
        
        # First check if exact same graph is already cached in memory
        if (EmbodiedRAG._cached_graph is not None and 
            EmbodiedRAG._cached_graph_path == enhanced_graph_file):
            self.graph = EmbodiedRAG._cached_graph
            return self.graph

        
        try:
            self.graph = nx.read_gml(enhanced_graph_file)

                
            # Generate embeddings if needed
            await self._ensure_embeddings(self.graph, enhanced_graph_file)
            
            # Cache the loaded graph
            EmbodiedRAG._cached_graph = self.graph
            EmbodiedRAG._cached_graph_path = enhanced_graph_file
            
            # Initialize retriever
            self._initialize_retriever()
            return self.graph
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return None

    def _initialize_retriever(self):
        """Initialize the appropriate retriever based on method"""
        if self.retrieval_method == RetrievalMethod.SEMANTIC:
            self.retriever = EmbodiedRetriever(
                self.graph, 
                self.embedding_func,
                retrieval_method=self.retrieval_method
            )
        elif self.retrieval_method == RetrievalMethod.LLM_HIERARCHICAL:
            from .use_llm_selection import LLMHierarchicalRetriever
            self.retriever = LLMHierarchicalRetriever(
                self.graph,
                self.llm,
                max_parallel_paths=Config.RETRIEVAL['max_parallel_paths']
            )
        else:  # HYBRID
            self.retriever = EmbodiedRetriever(
                self.graph,
                self.embedding_func,
                retrieval_method=self.retrieval_method
            )

    async def _ensure_embeddings(self, graph, graph_file):
        """Ensure all nodes have embeddings"""
        # Check if we have cached embeddings
        if graph_file in EmbodiedRAG._cached_embeddings:
            print("Using cached embeddings")
            for node, embedding in EmbodiedRAG._cached_embeddings[graph_file].items():
                if node in graph.nodes:
                    graph.nodes[node]['embedding'] = embedding
            return

        needs_embeddings = any('embedding' not in data for _, data in graph.nodes(data=True))
        
        if needs_embeddings:
            print("Generating missing embeddings...")
            await self._generate_node_embeddings(graph)
            nx.write_gml(graph, graph_file)
            print("Saved graph with embeddings")
        
        # Convert and cache embeddings
        if not hasattr(graph, '_embeddings_converted'):
            print("Converting embeddings to numpy arrays...")
            self._convert_embeddings_to_numpy(graph)
            graph._embeddings_converted = True
            
            # Cache the numpy embeddings
            EmbodiedRAG._cached_embeddings[graph_file] = {
                node: data['embedding'] 
                for node, data in graph.nodes(data=True) 
                if 'embedding' in data
            }
            print("Embeddings cached")

    async def _generate_node_embeddings(self, graph):
        """Generate embeddings for nodes in parallel batches"""
        logger = logging.getLogger('EmbodiedRAG')
        nodes_needing_embeddings = [
            (node, data) for node, data in graph.nodes(data=True)
            if 'embedding' not in data
        ]
        
        logger.info(f"Generating embeddings for {len(nodes_needing_embeddings)} nodes...")
        
        # Configure batch size based on API limits
        batch_size = 100
        
        async def process_batch(batch):
            texts = [self._get_node_text(node, data) for node, data in batch]
            try:
                embeddings = await self.embedding_func(texts)
                return list(zip([node for node, _ in batch], embeddings))
            except Exception as e:
                logger.error(f"Error in batch: {str(e)}")
                return []

        try:
            with tqdm(total=len(nodes_needing_embeddings), desc="Generating embeddings", unit="node") as pbar:
                for i in range(0, len(nodes_needing_embeddings), batch_size):
                    batch = nodes_needing_embeddings[i:i + batch_size]
                    
                    results = await process_batch(batch)
                    for node, embedding in results:
                        graph.nodes[node]['embedding'] = embedding.tolist()
                    
                    pbar.update(len(batch))
                    
                    if i % (batch_size * 5) == 0 and EmbodiedRAG._cached_graph_path:
                        try:
                            nx.write_gml(graph, EmbodiedRAG._cached_graph_path)
                            logger.info(f"Saved progress after {i} nodes")
                        except Exception as save_error:
                            logger.warning(f"Could not save progress - {str(save_error)}")

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
        
        logger.info("Finished generating embeddings")

    def _convert_embeddings_to_numpy(self, graph):
        """Convert stored embeddings to numpy arrays"""
        for node in graph.nodes():
            if 'embedding' in graph.nodes[node]:
                if not isinstance(graph.nodes[node]['embedding'], np.ndarray):
                    graph.nodes[node]['embedding'] = np.array(graph.nodes[node]['embedding'])

    def _get_node_text(self, node, data):
        """Generate text representation of a node"""
        return f"{data.get('label', node)} {data.get('summary', '')}"

    async def query(self, query_text, query_type="explicit", start_position=None, use_topological=False):
        """Process queries and handle navigation"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"New Query: '{query_text}'")
        self.logger.info(f"Type: {query_type}")
        self.logger.info(f"{'='*50}\n")
        
        try:
            retrieved_nodes = await self.retriever.retrieve(query_text, query_type=query_type)
            self.logger.info("\nRetrieved Nodes:")
            self.logger.info("-" * 30)
            for node in retrieved_nodes:
                self.logger.info(f"- {node}")

            response = await self.retriever.generate_response(query_text, retrieved_nodes, query_type)
            self.logger.info("\nGenerated Response:")
            self.logger.info("-" * 30)
            self.logger.info(response)
            
            if query_type in ["explicit", "implicit"] and self.airsim_utils:
                target_position = self.retriever.extract_target_position(response)
                
                if target_position:
                    print(f"\nNavigating to position: {target_position}")
                    if use_topological:
                        success = self.airsim_utils.direct_to_waypoint(target_position, velocity=5)
                    else:
                        success = self.airsim_utils.direct_to_position(
                            target_position, 
                            velocity=5
                        )
                    return response, success
                else:
                    print("\nNo valid target position found in response")
                    return response, False
            
            return response, None
            
        except Exception as e:
            import traceback
            print(f"Error in query: {str(e)}")
            print(traceback.format_exc())
            return None, False
