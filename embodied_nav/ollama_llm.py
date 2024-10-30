import ollama
import numpy as np
from .config import Config

class OllamaInterface:
    def __init__(self, model_name="gemma:2b", host="http://localhost:11434"):
        self.model = model_name
        self.host = host
        self.client = ollama.Client(host=self.host)
        self.temperature = Config.LLM['temperature']
        self.max_tokens = Config.LLM['max_tokens']

    async def generate_response(self, prompt, system_prompt=None):
        """Base method for generating responses from Ollama"""
        try:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    async def generate_embeddings(self, texts):
        """Generate embeddings using Ollama"""
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings(
                    model=self.model,
                    prompt=text
                )
                embeddings.append(np.array(response['embedding']))
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None

    async def generate_navigation_response(self, query, context, query_type):
        """Generate response based on query type using the same logic as original LLMInterface"""
        return await self.generate_response(
            self._construct_navigation_prompt(query, context, query_type),
            system_prompt="You are an AI assistant specialized in spatial navigation and environment understanding."
        )

    def _construct_navigation_prompt(self, query, context, query_type):
        if query_type == "global":
            return f"""Given the following context about objects in a 3D environment:

            {context}

            Answer the following query about the environment: {query}
            Focus on describing the spatial relationships and overall layout.
            """
        else:
            return f"""Given the following context about objects in a 3D environment:

            {context}

            For the query: '{query}', provide:
            1. A brief description of the most relevant object(s)
            2. An explanation of why these objects are relevant and query
            3. Choose the best target object and output it in this format: <<object_name>>
            
            Make sure the object_name matches exactly with one of the objects in the context.
            """ 