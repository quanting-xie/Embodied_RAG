from .spatial_relationship_extractor import SpatialRelationshipExtractor
from .embodied_retriever import EmbodiedRetriever
import networkx as nx
import logging
import numpy as np
from openai import AsyncOpenAI
import json
import os

logger = logging.getLogger(__name__)

class EmbodiedRAG:
    def __init__(self, working_dir, airsim_utils=None, use_ollama=False):
        self.working_dir = working_dir
        self.cache_file = os.path.join(working_dir, "llm_response_cache.json")
        if use_ollama:
            from .ollama_llm import OllamaInterface
            self.llm = OllamaInterface()
        else:
            self.client = AsyncOpenAI()
        self.relationship_extractor = SpatialRelationshipExtractor(self)
        self.retriever = None
        self.airsim_utils = airsim_utils
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


    async def load_graph_to_rag(self, enhanced_graph_file):
        """Load and enhance graph with embeddings"""
        enhanced_graph = nx.read_gml(enhanced_graph_file)
        print(f"Loaded graph with {len(enhanced_graph.nodes())} nodes")
        
        # Generate embeddings if needed
        await self._ensure_embeddings(enhanced_graph, enhanced_graph_file)
        
        # Initialize retriever with enhanced graph
        self.retriever = EmbodiedRetriever(enhanced_graph, self.embedding_func)
        return enhanced_graph

    async def _ensure_embeddings(self, graph, graph_file):
        """Ensure all nodes have embeddings"""
        needs_embeddings = any('embedding' not in data for _, data in graph.nodes(data=True))
        
        if needs_embeddings:
            print("Generating missing embeddings...")
            await self._generate_node_embeddings(graph)
            nx.write_gml(graph, graph_file)
            print("Saved graph with embeddings")
        else:
            print("Using existing embeddings")
            self._convert_embeddings_to_numpy(graph)

    async def _generate_node_embeddings(self, graph):
        """Generate embeddings for nodes"""
        for node, data in graph.nodes(data=True):
            if 'embedding' not in data:
                node_text = self._get_node_text(node, data)
                embedding = await self.embedding_func([node_text])
                graph.nodes[node]['embedding'] = embedding[0].tolist()

    def _convert_embeddings_to_numpy(self, graph):
        """Convert stored embeddings to numpy arrays"""
        for node in graph.nodes():
            if 'embedding' in graph.nodes[node]:
                graph.nodes[node]['embedding'] = np.array(graph.nodes[node]['embedding'])

    def _get_node_text(self, node, data):
        """Generate text representation of a node"""
        return f"{data.get('label', node)} {data.get('summary', '')}"

    async def query(self, query_text, query_type="explicit", start_position=None):
        """Process queries and handle navigation"""
        print(f"\n==== Processing Query: '{query_text}' ====")
        
        retrieved_nodes = await self.retriever.retrieve(query_text, query_type=query_type)
        print("\nRetrieved Objects:", *[f"- {node}" for node in retrieved_nodes], sep="\n")

        response = await self.retriever.generate_response(query_text, retrieved_nodes, query_type)
        print("\nGenerated Response:", response)
        
        if query_type in ["explicit", "implicit"] and self.airsim_utils:
            return await self._handle_navigation(response)
        
        return response

    async def _handle_navigation(self, response):
        """Handle navigation to target position"""
        target_position = self.retriever.extract_target_position(response)
        print(f"\nTarget position: {target_position}")
        
        if target_position:
            success = self.airsim_utils.direct_to_waypoint(target_position, velocity=5)
            return response, success
        
        print("\nNo target position found.")
        return response, False


