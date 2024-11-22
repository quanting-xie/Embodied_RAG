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
        self.vllm_settings = Config.LLM['vllm_settings']
        
        if self.vllm_settings['enabled']:
            self.model = self.vllm_settings['model']
            self.client = AsyncOpenAI(
                base_url=self.vllm_settings['api_base'] + "/v1",
                api_key=self.vllm_settings['api_key'],
            )
        else:
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
            "1. A SHORT functional name that describes the COMBINED purpose of ALL areas/objects\n"
            "   - Use exactly 2-3 words in snake_case format (e.g., art_work_zone, meeting_dining_area)\n"
            "   - The name MUST reflect ALL major functions present\n"
            "   - Do not focus on just one function if multiple exist\n"
            "2. A two-sentence summary that:\n"
            "   - Describes ALL distinct functions in the space\n"
            "   - Mentions the key objects/features from EACH sub-area\n"
            "   - Captures the mixed-use nature if multiple functions exist\n"
            "\n"
            "Format your response EXACTLY as follows (including the <<>> markers):\n"
            "AREA_NAME: <<combined_functional_name_in_snake_case>>\n"
            "AREA_SUMMARY: <<single concise sentence covering ALL functions>>\n"
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
        """Generate a navigation response based on the retrieved nodes."""
        prompt = f"""Given the following navigation query and context about available locations,
        generate a response that helps navigate to the most relevant location.

        Query: {query}
        Query Type: {query_type}

        Available Context:
        {context}

        Instructions:
        1. Analyze the query and available locations
        2. Select the most specific and appropriate destination (prefer specific objects over general areas)
        3. Format your response as follows:
           - Include the exact location name in double angle brackets: <<exact_name>>
           - Provide a brief explanation of why this location is relevant
           - Include any relevant spatial relationships or navigation hints

        IMPORTANT: Use the exact name as it appears in the context, maintaining exact spelling and format.
        """

        response = await self.generate_response(prompt)
        return response.strip()

    async def select_best_node(self, query, nodes, context):
        """Select the single most relevant node from a list of candidates based on the query."""
        prompt = f"""Given the following navigation query and available nodes in a 3D environment, 
        select the SINGLE most relevant node that best matches the query's intent.

        Navigation Query: {query}

        Available Nodes:
        {context}

        Instructions:
        1. Consider each node's summary and function
        2. Evaluate relevance to the navigation query
        3. Prioritize specific object nodes over general areas when possible
        4. Select the single most specific and relevant node for navigation

        CRITICAL: You must respond with ONLY the exact Node ID from the list above.
        For example, if you see 'Node ID: cafeteria_table_1', respond with exactly 'cafeteria_table_1'.
        Do not add any explanation or additional text.

        Your response must be one of these exact Node IDs: {[n['id'] for n in nodes]}"""

        response = await self.generate_response(prompt)
        response = response.strip()
        
        # Debug print
        print(f"\nLLM Selection Process:")
        print(f"Query: {query}")
        print(f"Available Node IDs: {[n['id'] for n in nodes]}")
        print(f"Node Types: {[(n['id'], n['type']) for n in nodes]}")
        print(f"LLM Response: '{response}'")
        
        # Verify response matches an available node
        if response in [n['id'] for n in nodes]:
            print(f"✓ Valid selection: {response}")
            return response
        else:
            print(f"✗ Invalid selection: '{response}' not in available nodes")
            return None

    async def generate_hierarchical_context(self, nodes):
        """Generate a readable context for a list of nodes."""
        context_parts = ["Available Locations for Selection:"]
        
        for i, node in enumerate(nodes, 1):
            context_parts.extend([
                f"\n{i}. Location Details:",
                f"   Name: {node['name']}",
                f"   Type: {node['type']}",
                f"   Level: {node['level']}",
                f"   Summary: {node['summary']}",
                "   ---"
            ])
        return "\n".join(context_parts)

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
