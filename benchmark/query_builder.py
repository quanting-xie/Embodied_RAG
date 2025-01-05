import networkx as nx
import openai
import json
import random
import os
import ipdb

gml_file_path = './semantic_graphs/enhanced_semantic_graph_semantic_graph_Building99_20241118_160313.gml'

graph = nx.read_gml(gml_file_path)
contexts = []

for node, data in graph.nodes(data=True):
    if 'level' in data and data['level']:
        contexts.append(data['summary'])

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

system_message = {
    "role": "system",
    "content": """
        You are a specialist in designing meaningful and contextually relevant queries for embodied scenarios.
        Your task is to create high-quality queries based on provided descriptions of subareas within a scenario.
        
        ### Types of Queries:
        1. **Explicit**: Direct and specific questions targeting objects or locations (e.g., "Find me the nearest water fountain").
        2. **Implicit**: Indirect questions that require reasoning or contextual understanding (e.g., "Where can I find something to drink?").
        3. **Global**: Broad questions summarizing or analyzing the overall scenario or environment (e.g., "What is the purpose of this space?").

        ### Task Instructions:
        1. Ensure that the queries are aligned with the provided context and presented concisely.
        2. Generate one explicit query, two implicit queries, and one global query relevant to the given scenario.
        3. Queries must reflect natural language and avoid using technical or internal identifiers (e.g., "Cafeteria_ColaRefrigerator_5").

        ### Output Format:
        Provide the queries in JSON format as follows:
        [
            {
                "query": "Generated query text here",
                "type": "explicit"
            },
            {
                "query": "Generated query text here",
                "type": "implicit"
            },
            {
                "query": "Generated query text here",
                "type": "implicit"
            },
            {
                "query": "Generated query text here",
                "type": "global"
            }
        ]

        ### Example Queries:
        - **Explicit**: 
          - "Find me the stairs."
          - "Locate the emergency exit."
          - "Point me to a vending machine."
        - **Implicit**: 
          - "Where can I grab a quick snack?"
          - "Where would someone looking for a comfortable place to rest go?"
          - "Where is a good spot to study quietly?"
        - **Global**: 
          - "What activities does this building support?"
          - "How is the environment designed for accessibility?"
          - "Describe the overall function of the space."
    """
}

idx = 0
query_generation_num = 125
with open("benchmark/data/query.jsonl", 'a') as outfile:
    while idx < query_generation_num:
        context_fragments = random.sample(contexts, 2)
        context_str = "\n".join([f"Subarea {i+1}:\n{frag}" for i, frag in enumerate(context_fragments)])

        messages = [
            system_message,
            {"role": "user", "content": f'Subarea descriptions of the scene:\n"""\n{context_str}\n"""'}
        ]

        try:
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=messages
            )

            if response and response.choices:
                content = response.choices[0].message.content
                content = content.split('```json', 1)[1].split('```')[0]
                parsed_data = json.loads(content)

                for item in parsed_data:
                    outfile.write(json.dumps(item) + '\n')
            print(f"Generation done: {idx}")
            idx += 1
        except Exception as e:
            print(f"Error during generation {idx}: {e}")