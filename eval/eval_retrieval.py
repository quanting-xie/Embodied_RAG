import json
import argparse
import sys
sys.path.append('.')
from embodied_nav.config import Config
from openai import AsyncOpenAI  

def generate_response(client, model, prompt, system_prompt=None):
    """Generate a response using the LLM."""
    if system_prompt is None:
        system_prompt = "You are an AI assistant specialized in spatial navigation and environment understanding."
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=Config.LLM.get("temperature", 0.7),
        max_tokens=Config.LLM.get("max_tokens", 500),
    )
    return response.choices[0].message.content.strip()

def inference(client, model, query, context, node_ids):
    """Perform inference to get the best-matched node."""
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

    Your response must be one of these exact Node IDs: {node_ids}"""

    response = generate_response(client, model, prompt)

    if response not in node_ids:
        raise ValueError(f"Invalid response: {response}. Must be one of {node_ids}.")
    
    return response

def evaluate_rag(client, model, data):
    """Evaluate the RAG system using the given dataset."""
    correct = 0
    total = 0

    for item in data:
        query = item['query']
        context = item['context']
        node_ids = item['node_ids']
        gold_response = item['response']
        try:
            result = inference(client, model, query, context, node_ids)
            if result.strip() == gold_response.strip():
                correct += 1
            total += 1
        except Exception as e:
            print(f"[ERROR] Query failed for: {query}")
            print(e)

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Exact Match Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG with a JSONL dataset.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'gpt-4').")
    args = parser.parse_args()

    # Initialize the client
    client = AsyncOpenAI(
        base_url=Config.LLM.vllm_settings["api_base"] + "/v1",
        api_key=Config.LLM.vllm_settings["api_key"],
    )

    # Load the dataset
    data = []
    with open('benchmark/data/generation_qa.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Evaluate the RAG system
    evaluate_rag(client, args.model, data)

if __name__ == "__main__":
    main()
