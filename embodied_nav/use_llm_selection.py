import asyncio
from typing import List, Dict, Any
import networkx as nx
from .llm import LLMInterface
from .config import Config

class LLMHierarchicalRetriever:
    def __init__(self, graph, llm_interface, max_parallel_paths=None):
        self.graph = graph
        self.llm = llm_interface
        self.max_parallel_paths = (max_parallel_paths if max_parallel_paths is not None 
                                 else Config.RETRIEVAL['max_parallel_paths'])
        print(f"Initialized LLMHierarchicalRetriever with {self.max_parallel_paths} parallel paths")

    async def retrieve(self, query: str, query_type="global", top_k=None) -> List[str]:
        """
        Retrieve relevant nodes using parallel LLM-guided hierarchical traversal.
        """
        print("\n=== Starting LLM Hierarchical Retrieval ===")
        self._chains = []  # Reset chains for new query
        
        # Get all top-level nodes
        max_level = max(data.get('level', 0) for _, data in self.graph.nodes(data=True))
        top_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('level', 0) == max_level
        ]
        print(f"\nFound {len(top_nodes)} top-level nodes at level {max_level}")
        
        # Format top-level nodes for LLM
        top_nodes_context = "\n".join([
            f"Area {i+1}:\n"
            f"- Name: {node}\n"
            f"- Summary: {self.graph.nodes[node].get('summary', 'No summary')}\n"
            f"- Level: {self.graph.nodes[node].get('level', 'Unknown')}"
            for i, node in enumerate(top_nodes)
        ])
        
        # LLM selection of top nodes
        system_prompt = """You are an AI assistant helping to navigate a 3D environment.
        Given a user query and available areas, select the most relevant areas.
        You MUST select exactly 3 areas (or all areas if less than 3 are available).
        Respond ONLY with the exact area names, separated by commas."""
        
        user_prompt = f"""Query: {query}

Available areas:
{top_nodes_context}

Select exactly 3 most relevant areas (or all if less than 3). Return only the area names, separated by commas:"""

        print("\n=== Top-Level Selection ===")
        print(f"Sending prompt to LLM:\n{user_prompt}")
        
        selected_areas = await self.llm.generate_response(user_prompt, system_prompt)
        print(f"\nLLM Response: {selected_areas}")
        
        selected_top_nodes = [node.strip() for node in selected_areas.split(',') if node.strip() in top_nodes]
        print(f"Validated selected nodes: {selected_top_nodes}")

        expanded_results = set()
        print("\n=== Building Hierarchical Chains ===")
        
        # Process each selected top node
        for i, top_node in enumerate(selected_top_nodes[:self.max_parallel_paths], 1):
            print(f"\nProcessing Chain {i} starting from: {top_node}")
            chain = [top_node]
            current = top_node
            
            while current:
                children = [
                    n for n in self.graph.neighbors(current)
                    if self.graph.nodes[n].get('level', 0) < self.graph.nodes[current].get('level', 0)
                ]
                
                print(f"\nNode: {current}")
                print(f"Found {len(children)} children: {children}")
                
                if children:
                    children_context = "\n".join([
                        f"Object {i+1}:\n"
                        f"- Name: {node}\n"
                        f"- Summary: {self.graph.nodes[node].get('summary', 'No summary')}\n"
                        f"- Level: {self.graph.nodes[node].get('level', 'Unknown')}"
                        for i, node in enumerate(children)
                    ])
                    
                    child_system_prompt = """You are an AI assistant helping to find relevant objects.
                    Given a user query and available objects, select the most relevant object.
                    Respond ONLY with the exact object name."""
                    
                    child_prompt = f"""Query: {query}

Available objects:
{children_context}

Select the single most relevant object. Return only the exact object name:"""

                    print(f"\nSending child selection prompt for {current}:")
                    print(child_prompt)
                    
                    selected_child = await self.llm.generate_response(child_prompt, child_system_prompt)
                    selected_child = selected_child.strip()
                    print(f"LLM selected child: '{selected_child}'")
                    
                    if selected_child in children:
                        chain.append(selected_child)
                        current = selected_child
                        print(f"Added {selected_child} to chain")
                    else:
                        print(f"Warning: Selected child '{selected_child}' not in available children")
                        current = None
                else:
                    print("No children found, ending chain")
                    current = None
            
            if chain:
                print(f"\nCompleted Chain {i}: {' -> '.join(reversed(chain))}")
                self._chains.append(chain)
                expanded_results.update(chain)

        print("\n=== Retrieval Complete ===")
        print(f"Found {len(self._chains)} chains:")
        for i, chain in enumerate(self._chains, 1):
            print(f"Chain {i}: {' -> '.join(reversed(chain))}")
        print(f"Total unique nodes: {len(expanded_results)}")
        
        return list(expanded_results)

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
        for level in sorted(level_nodes.keys(), reverse=True):
            context.append(f"\nLevel {level}:")
            for node in level_nodes[level]:
                node_data = self.graph.nodes[node]
                indent = "  " * (3 - level)
                context.append(f"{indent}{node}")
                if 'summary' in node_data:
                    context.append(f"{indent}Summary: {node_data['summary']}")
                
        # Object Information
        context.append("\n=== Object Information ===")
        for node in nodes:
            node_data = self.graph.nodes[node]
            context.append(f"\nObject: {node}")
            
            # Position
            if 'position' in node_data:
                pos = node_data['position']
                context.append(f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            # Other properties
            props = {k: v for k, v in node_data.items() 
                    if k not in ['embedding', 'position', 'label']}
            if props:
                context.append(f"Properties: {props}")
        
        return "\n".join(context)

    async def generate_response(self, query, retrieved_nodes, query_type):
        """Generate response using context from retrieved nodes"""
        context = self._build_context(retrieved_nodes)
        return await self.llm.generate_navigation_response(query, context, query_type)

    def extract_target_position(self, response):
        """Extract target position from response"""
        import re
        object_name_match = re.search(r'<<(.+?)>>', response)
        if object_name_match:
            object_name = object_name_match.group(1)
            for node, data in self.graph.nodes(data=True):
                if node == object_name and 'position' in data:
                    return data['position']
        return None