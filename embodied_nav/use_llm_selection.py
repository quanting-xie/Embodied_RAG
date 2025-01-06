import asyncio
from typing import List, Dict, Any
import networkx as nx
from .llm import LLMInterface
from .config import Config
import time
import logging
import ipdb
import json

class LLMHierarchicalRetriever:
    def __init__(self, graph, llm_interface, max_parallel_paths=None):
        self.graph = graph
        self.llm = llm_interface
        self.max_parallel_paths = (max_parallel_paths if max_parallel_paths is not None 
                                 else Config.RETRIEVAL['max_parallel_paths'])
        print(f"Initialized LLMHierarchicalRetriever with {self.max_parallel_paths} parallel paths")

    async def retrieve(self, query: str, query_type="global", top_k=None, data_construction=False) -> List[str]:
        """Retrieve relevant nodes using parallel LLM-guided hierarchical traversal."""
        logger = logging.getLogger('experiment')
        start_time = time.time()
        
        logger.info("\n=== Starting LLM Hierarchical Traversal ===")
        logger.info(f"Query: '{query}'")
        
        # Track timing for each level
        level_times = {}
        
        self._chains = []
        top_k = top_k or Config.RETRIEVAL['top_k_default']
        
        # Get all nodes by level and print debug info
        level_nodes = {}
        for node, data in self.graph.nodes(data=True):
            level = data.get('level', 0)
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append((node, data))
            
        max_level = max(level_nodes.keys())
        print(f"\nFound nodes at {len(level_nodes)} levels (max level: {max_level})")
        
        # Debug print for top level nodes
        print("\nTop level nodes with full details:")
        for node, data in level_nodes[max_level]:
            print(f"\nNode ID: {node}")
            print(f"Name: {data.get('name', 'unnamed')}")
            print(f"Type: {data.get('type', 'unknown')}")
            print(f"Level: {data.get('level', 'unknown')}")
        
        # Start traversal
        expanded_results = set()
        current_level = max_level
        chain = []
        
        while current_level >= 0:
            level_start_time = time.time()
            
            available_nodes = level_nodes[current_level]
            if not available_nodes:
                print(f"No nodes available at level {current_level}")
                break
            
            # Filter nodes based on previous selection
            if chain:
                previous_node = chain[-1]
                filtered_nodes = [
                    (n, d) for n, d in available_nodes
                    if self.graph.has_edge(previous_node, n) and
                    self.graph.edges[previous_node, n].get('relationship') == 'part_of'
                ]
                print(f"\nFiltered nodes under {previous_node}: {len(filtered_nodes)}")
                for n, d in filtered_nodes:
                    print(f"- {n} (Type: {d.get('type')}, Level: {d.get('level')})")
                available_nodes = filtered_nodes
            
            if not available_nodes:
                # Check if we've reached an object node
                if chain and self.graph.nodes[chain[-1]].get('type') == 'object':
                    print(f"Reached object node: {chain[-1]}")
                    break
                print(f"No valid nodes after filtering at level {current_level}")
                break
            
            # Format nodes for LLM selection
            nodes_for_selection = []
            for node, data in available_nodes:
                node_info = {
                    'id': node,
                    'summary': data.get('concise_summary', 'No summary') if current_level else data.get('summary', 'No summary'),
                    'level': current_level,
                    'type': data.get('type', 'unknown'),
                    'name': data.get('name', node)
                }
                nodes_for_selection.append(node_info)
            
            # Select best node
            selection_start = time.time()
            selected_node = await self.llm.select_best_node(
                query=query,
                nodes=nodes_for_selection,
                context=await self.llm.generate_hierarchical_context(nodes_for_selection) if current_level else await self.llm.generate_object_context(nodes_for_selection),
                data_construction=data_construction
            )
            selection_time = time.time() - selection_start
            logger.info(f"LLM Selection Time at Level {current_level}: {selection_time:.2f} seconds")
            
            if selected_node:
                chain.append(selected_node)
                node_data = self.graph.nodes[selected_node]
                print(f"\nSelected node: {selected_node}")
                print(f"Type: {node_data.get('type')}")
                print(f"Level: {node_data.get('level')}")
                expanded_results.add(selected_node)
                
                # If we've reached an object node, we can stop
                if node_data.get('type') == 'object':
                    print(f"Reached object node, stopping traversal")
                    break
            else:
                raise ValueError("No valid node selected for traversal")
            
            level_end_time = time.time()
            level_time = level_end_time - level_start_time
            level_times[current_level] = level_time
            logger.info(f"Level {current_level} processing time: {level_time:.2f} seconds")
            
            current_level -= 1
            
        # Verify we reached an object
        if chain:
            final_node = chain[-1]
            final_node_type = self.graph.nodes[final_node].get('type')
            if final_node_type != 'object':
                print(f"\nWarning: Traversal ended at non-object node of type {final_node_type}")
                # Try to find connected object nodes
                for neighbor in self.graph.neighbors(final_node):
                    if self.graph.nodes[neighbor].get('type') == 'object':
                        print(f"Found connected object node: {neighbor}")
                        chain.append(neighbor)
                        expanded_results.add(neighbor)
                        break
        
        logger.info("\n=== Retrieval Complete ===")
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total Retrieval Time: {total_time:.2f} seconds")
        
        if chain:
            print("\nFinal path:")
            for node in chain:
                node_data = self.graph.nodes[node]
                print(f"- {node} (Type: {node_data.get('type')}, Level: {node_data.get('level')})")
        
        # Log detailed timing statistics
        logger.info("\n=== LLM Hierarchical Retrieval Statistics ===")
        logger.info(f"Total Retrieval Time: {total_time:.2f} seconds")
        logger.info("Time breakdown by level:")
        for level, time_taken in level_times.items():
            logger.info(f"- Level {level}: {time_taken:.2f} seconds ({(time_taken/total_time)*100:.1f}%)")
        logger.info(f"Number of Retrieved Nodes: {len(expanded_results)}")
        logger.info("Retrieved Nodes:")
        for node in expanded_results:
            logger.info(f"- {node}")
        
        return list(expanded_results)[:top_k]

    def get_hierarchical_chains(self):
        """Return the stored hierarchical chains from the last retrieval."""
        return self._chains

    def _build_context(self, nodes):
        """Build context from retrieved nodes"""
        if not nodes:
            return "No relevant nodes found in the environment."
            
        context = []
        context.append("=== Hierarchical Structure ===")
        
        # Group nodes by level
        level_nodes = {}
        for node in nodes:
            node_data = self.graph.nodes[node]
            level = node_data.get('level', 0)
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        # Display hierarchy from top down
        nodes = []
        for level in sorted(level_nodes.keys(), reverse=True):
            context.append(f"\nLevel {level}:")
            for node in level_nodes[level]:
                node_data = self.graph.nodes[node]
                indent = "  " * (3 - level)
                context.append(f"{node}")
                # if 'concise_summary' in node_data:
                #     context.append(f"{indent}Summary: {node_data['concise_summary']}")
                nodes.append(node)
                
        # Object Information
        context.append("\n=== Object Information ===")
        for node in nodes:
            node_data = self.graph.nodes[node]
            context.append(f"\nObject: {node}")
            
            # Position handling with proper type checking
            if 'position' in node_data:
                pos = node_data['position']
                if isinstance(pos, (list, tuple)) and len(pos) == 3:
                    context.append(f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                elif isinstance(pos, dict) and all(k in pos for k in ['x', 'y', 'z']):
                    context.append(f"Position: [{pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}]")
                else:
                    context.append(f"Position: {pos}")  # Fallback for other formats
            
            # Other properties
            props = {k: v for k, v in node_data.items() 
                    if k not in ['embedding', 'position', 'label', 'summary']} # use concise summary as replace
            if props:
                context.append(f"Properties: {props}")
        
        return "\n".join(context)

    async def generate_response(self, query, retrieved_nodes, query_type, data_construction=False):
        """Generate response using context from retrieved nodes"""
        try:
            context = self._build_context(retrieved_nodes)
            print("\nContext for response generation:")
            print(context)
            response = await self.llm.generate_navigation_response(query, context, query_type)
            
            if data_construction:
                log_entry = {"query": query, "context": context, "response": response}
                with open("benchmark/data/generation_qa.jsonl", "a") as file:
                    file.write(json.dumps(log_entry) + "\n")

            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"Error generating response: {str(e)}"

    def extract_target_position(self, response):
        """Extract target position from response"""
        import re
        object_name_match = re.search(r'<<(.+?)>>', response)
        if object_name_match:
            target_name = object_name_match.group(1)
            # First try to find by name
            for node, data in self.graph.nodes(data=True):
                if data.get('name') == target_name and 'position' in data:
                    print(f"Found position for {target_name}: {data['position']}")
                    return data['position']
                
            # If not found by name, try to find by node ID
            if target_name in self.graph.nodes and 'position' in self.graph.nodes[target_name]:
                print(f"Found position for node {target_name}: {self.graph.nodes[target_name]['position']}")
                return self.graph.nodes[target_name]['position']
                
            print(f"No position found for {target_name}")
        return None