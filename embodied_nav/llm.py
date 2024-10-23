from lightrag.llm import openai_complete_if_cache

class LLMInterface:
    def __init__(self):
        self.model = "gpt-4o-mini"  # or any other model you prefer

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
            system_prompt="You are an AI assistant specialized in describing spatial relationships between objects in a 3D environment.",
        )

        return response.strip()

    async def generate_summary(self, prompt):
        response = await openai_complete_if_cache(
            self.model,
            prompt,
            system_prompt="You are specialized in summarizing groups of objects into high level concepts.",
        )
        return response.strip()

    async def generate_response(self, prompt):
        response = await openai_complete_if_cache(
            self.model,
            prompt,
            system_prompt="You are an AI assistant specialized in answering queries about objects and their relationships in a 3D environment.",
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
