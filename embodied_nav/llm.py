from lightrag.llm import openai_complete_if_cache
from .config import Config
import re

class LLMInterface:
    def __init__(self):
        self.model = Config.LLM['model']
        self.temperature = Config.LLM['temperature']
        self.max_tokens = Config.LLM['max_tokens']

    async def generate_relationship(self, node1, node2):
        prompt = f"""
        Given two objects in a 3D environment:
        Object 1: {node1[0]} (ID: {node1[1]['id']}, Position: {node1[1]['position']})
        Object 2: {node2[0]} (ID: {node2[1]['id']}, Position: {node2[1]['position']})

        Describe the spatial relationship between these two objects. Consider their relative positions, possible interactions, and any logical connections based on their semantic labels.

        Output the relationship as a short phrase or sentence.
        """

        response = await openai_complete_if_cache(
            self.model,
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt="You are an AI assistant specialized in describing spatial relationships between objects in a 3D environment.",
        )

        return response.strip()

    async def generate_response(self, prompt, system_prompt=None):
        """Base method for generating responses from the LLM"""
        if system_prompt is None:
            system_prompt = "You are an AI assistant specialized in spatial navigation and environment understanding."
            
        response = await openai_complete_if_cache(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=system_prompt
        )
        return response.strip()

    async def rank_results(self, query, results, context):
        prompt = f"""Given the following query and list of objects in a 3D environment, rank the objects based on their relevance to the query. Consider semantic similarity, hierarchical relationships, spatial proximity, and functional relevance.

            Query: {query}

            Objects:
            {context}

            Output a ranked list of object IDs, separated by commas, from most relevant to least relevant.
            """
        response = await self.generate_response(prompt)
        ranked_ids = [id.strip() for id in response.split(',')]
        return [id for id in ranked_ids if id in results]  # Ensure we only return valid results

    async def generate_community_summary(self, objects):
        """Create a summary for a group of objects, returning name and description"""
        # Group objects by type/category
        object_groups = {}
        for obj in objects:
            label = obj.get('label', obj.get('id', 'Unknown object'))
            if label not in object_groups:
                object_groups[label] = 1
            else:
                object_groups[label] += 1

        # Create a structured description
        group_descriptions = [f"{count} {label}(s)" for label, count in object_groups.items()]
        
        prompt = f"""Given a group of objects in n environment: {', '.join(group_descriptions)}

        1. What would be an appropriate functional area name for this group of objects or descriptions of areas? (e.g., 'Meeting Area', 'Park Area', 'Reception Space')
        2. Provide a brief description of this area's purpose and characteristics.

        Format your response EXACTLY as follows (including the <<>> markers):
        AREA_NAME: <<functional area name>>
        AREA_SUMMARY: <<brief description>>
        """

        response = await self.generate_response(prompt)
        try:
            # Look for content between <<>> markers
            name_match = re.search(r'AREA_NAME:\s*<<(.+?)>>', response, re.DOTALL)
            summary_match = re.search(r'AREA_SUMMARY:\s*<<(.+?)>>', response, re.DOTALL)
            
            if name_match and summary_match:
                return {
                    'name': name_match.group(1).strip(),
                    'summary': summary_match.group(1).strip()
                }
            else:
                # Fallback to simple line splitting if no markers found
                lines = response.strip().split('\n')
                name = ''
                summary = ''
                for line in lines:
                    if line.startswith('AREA_NAME:'):
                        name = line.split('AREA_NAME:')[1].strip()
                    elif line.startswith('AREA_SUMMARY:'):
                        summary = line.split('AREA_SUMMARY:')[1].strip()
                
                if name and summary:
                    return {
                        'name': name,
                        'summary': summary
                    }
                else:
                    raise ValueError("Could not parse response format")
        except Exception as e:
            print(f"Warning: Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            # Provide a fallback name and summary
            return {
                'name': 'Unnamed Area',
                'summary': response.strip()
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
        
        system_prompt = "You are an AI assistant specialized in spatial navigation and environment understanding."
        return await self.generate_response(prompt, system_prompt=system_prompt)
