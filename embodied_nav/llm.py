from lightrag.llm import openai_complete_if_cache

class LLMInterface:
    def __init__(self):
        self.model = "gpt-4"  # or any other model you prefer

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
