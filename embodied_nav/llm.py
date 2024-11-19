from openai import AsyncOpenAI
from .config import Config
import re
import os
import traceback

class LLMInterface:
    def __init__(self):
        self.model = Config.LLM['model']
        self.temperature = Config.LLM['temperature']
        self.max_tokens = Config.LLM['max_tokens']
        self.client = AsyncOpenAI()

    async def generate_response(self, prompt, system_prompt=None):
        """Base method for generating responses from the LLM"""
        if system_prompt is None:
            system_prompt = "You are an AI assistant specialized in spatial navigation and environment understanding."
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()

    async def generate_relationship(self, node1, node2):
        prompt = f"""
        Given two objects in a 3D environment:
        Object 1: {node1[0]} (ID: {node1[1]['id']}, Position: {node1[1]['position']})
        Object 2: {node2[0]} (ID: {node2[1]['id']}, Position: {node2[1]['position']})

        Describe the spatial relationship between these two objects. Consider their relative positions, possible interactions, and any logical connections based on their semantic labels.

        Output the relationship as a short phrase or sentence.
        """

        return await self.generate_response(
            prompt,
            system_prompt="You are an AI assistant specialized in describing spatial relationships between objects in a 3D environment."
        )

    async def rank_results(self, query, results, context):
        prompt = f"""Given the following query and list of objects in a 3D environment, rank the objects based on their relevance to the query. Consider semantic similarity, hierarchical relationships, spatial proximity, and functional relevance.

            Query: {query}

            Objects:
            {context}

            Output a ranked list of object IDs, separated by commas, from most relevant to least relevant.
            """
        ranked_ids = [id.strip() for id in self.generate_response(prompt).split(',')]
        return [id for id in ranked_ids if id in results]  # Ensure we only return valid results

    async def generate_community_summary(self, objects):
        """Create both a functional name and summary for a group of objects or areas"""
        descriptions = []
        for obj in objects:
            if obj.get('summary'):
                descriptions.append(f"Area: {obj.get('name', 'Unnamed')}\n"
                                f"Summary: {obj.get('summary')}")
            else:
                obj_id = obj.get('id', 'Unknown object')
                obj_type = obj.get('type', 'object')
                obj_label = obj.get('label', obj_id)
                descriptions.append(f"Object: {obj_label} (Type: {obj_type}, ID: {obj_id})")

        descriptions_text = '\n'.join(descriptions)
        
        prompt = (
            f"Given these objects/areas in a 3D environment:\n"
            f"{descriptions_text}\n\n"
            "Please provide:\n"
            "1. A SHORT functional name that describes this area's primary purpose\n"
            "   - Use exactly 2-3 words in snake_case format (e.g., dining_area, meeting_corner, multi_purpose_room, multi_function_building)\n"
            "   - Be specific and descriptive\n"
            "2. A detailed summary of how these elements work together\n\n"
            "Format your response EXACTLY as follows (including the <<>> markers):\n"
            "AREA_NAME: <<functional_name_in_snake_case>>\n"
            "AREA_SUMMARY: <<detailed description>>\n\n"
            "Bad name examples:\n"
            "- DiningArea (not snake_case)\n"
            "- the dining area (not snake_case)\n"
            "- dining and social space (too long)\n"
            "- area (too vague)"
        )

        response = await self.generate_response(prompt)
        
        try:
            print(f"\nRaw LLM Response:\n{response}\n")
            
            # More flexible regex patterns that handle both formats:
            # Format 1: AREA_NAME: <<name>>
            # Format 2: AREA_NAME: name
            name_pattern = r'AREA_NAME:[ \t]*(?:<<)?([^>\n]+?)(?:>>)?[ \t]*$'
            summary_pattern = r'AREA_SUMMARY:[ \t]*(?:<<)?(.+?)(?:>>)?[ \t]*$'
            
            # Find matches in multiline text
            name_match = re.search(name_pattern, response, re.MULTILINE)
            summary_match = re.search(summary_pattern, response, re.MULTILINE | re.DOTALL)
            
            if not name_match or not summary_match:
                print("Warning: Could not parse response format")
                print(f"Name match: {name_match}")
                print(f"Summary match: {summary_match}")
                return {
                    'name': 'undefined_zone',
                    'summary': 'Area containing multiple objects or spaces'
                }
            
            name = name_match.group(1).strip().lower()
            summary = summary_match.group(1).strip()
            
            # Validate name format (allow only lowercase letters, numbers, and underscores)
            if not re.match(r'^[a-z0-9_]+$', name):
                print(f"Warning: Invalid name format: {name}")
                name = 'undefined_zone'
            
            print(f"Parsed name: {name}")
            print(f"Parsed summary: {summary[:50]}...")
            
            return {
                'name': name,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Warning: Error parsing LLM response: {str(e)}")
            traceback.print_exc()  # Print full traceback
            return {
                'name': 'undefined_zone',
                'summary': 'Area containing multiple objects or spaces'
            }

    async def generate_navigation_response(self, query, context, query_type):
        """Generate response based on query type"""
        if query_type == "global":
            # For global queries, just provide information without navigation
            prompt = f"""Given the following context about objects in a 3D environment:

            {context}

            Answer the following query about the environment: {query}
            Focus on describing the spatial relationships and overall layout.
            """
        else:
            # For explicit/implicit queries, include navigation target
            prompt = f"""Given the following context about objects in a 3D environment:

            {context}

            For the query: '{query}', provide:
            1. A brief description of the most relevant object(s)
            2. An explanation of why these objects are relevant and query
            3. Choose the best target object and output it in this format: <<object_name>>
            
            Make sure the object_name matches exactly with one of the objects in the context.
            """
        
        return await self.generate_response(prompt)

    async def select_best_node(self, query, nodes, context):
        """Select the most relevant node from a list of candidates based on the query."""
        prompt = f"""Given the following query and a list of nodes in a 3D environment, select the single most relevant node that best matches the query's intent.

        Query: {query}

        Available nodes and their context:
        {context}

        Output ONLY the exact name of the chosen node, nothing else. The name must match one of the provided nodes exactly."""

        response = await self.generate_response(prompt)
        # Clean up response to ensure we get just the node name
        response = response.strip().split('\n')[0].strip()
        
        # Verify the response matches one of the provided nodes
        if response in [n['id'] for n in nodes]:
            return response
        return None

    async def generate_hierarchical_context(self, nodes):
        """Generate a readable context for a list of nodes."""
        context = []
        for node in nodes:
            node_desc = [f"Node: {node['id']}"]
            for key, value in node.items():
                if key not in ['id', 'embedding', 'position']:
                    node_desc.append(f"{key}: {value}")
            context.append("\n".join(node_desc))
        return "\n\n".join(context)

    async def select_nodes_for_query(self, query, nodes_context, system_prompt=None):
        """Helper method for selecting nodes during hierarchical traversal"""
        if system_prompt is None:
            system_prompt = "You are an AI assistant helping to select relevant nodes in a 3D environment."
        
        response = await self.generate_response(nodes_context, system_prompt)
        return response.strip()

    async def generate_hierarchical_traversal(self, query, node_options, is_top_level=False):
        """Specialized method for hierarchical graph traversal"""
        context_type = "high-level areas" if is_top_level else "objects"
        
        prompt = f"""Given the query '{query}', analyze these {context_type}:

{node_options}

{f'Select up to 3 most relevant areas (comma-separated list)' if is_top_level else 'Select the single most relevant object (exact name only)'}."""

        return await self.generate_response(
            prompt,
            system_prompt=f"You are an AI assistant specialized in navigating hierarchical spaces. Select the most relevant {context_type} based on the query."
        )
