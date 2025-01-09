import json
import argparse
import sys
sys.path.append('.')

from embodied_nav.config import Config
from openai import OpenAI  
# from tqdm import tqdm  # Uncomment if you want to show progress

def generate_response(client, prompt, system_prompt=None):
    """Generate a response using the LLM."""
    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant specialized in spatial navigation "
            "and environment understanding."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        temperature=Config.LLM.get("temperature", 0.7),
        max_tokens=Config.LLM.get("max_tokens", 500),
    )
    return response.choices[0].message.content.strip()


def llm_as_judge(client, query, context, answer_1, answer_2):
    """
    Ask the LLM to decide which of the two answers is better, given the query and context.
    Returns 1 if Answer 1 is chosen by the LLM, else 0.
    """

    prompt = f"""Given the following navigation query and context about available locations,
which response generates a more helpful answer to navigate to the most relevant location?

Judgement based on:
i) Faithfulness: How the answer relates to the query and available locations.
ii) Specificity: More specific and appropriate destination is preferred (prefer specific objects over general areas).
iii) Comprehensiveness: Does the answer address all details of the question (location name, explanation, relevant spatial hints)?
iv) Overall: Which is the best overall answer?

Query:
{query}

Context:
{context}

Response from two agents:
Answer 1:
{answer_1}

Answer 2:
{answer_2}

Please decide which answer is better. Respond strictly with either 'Answer 1' or 'Answer 2' without any additional commentary.
"""

    response = generate_response(client, prompt, system_prompt=None)

    # A simple heuristic to check which one is chosen.
    # In practice, you'd likely parse the text more rigorously.
    final_answer = response.strip().lower()
    if "answer 1" in final_answer:
        return 1
    else:
        return 0


def inference(client, query, context, query_type):
    """Generate a navigation response based on the retrieved nodes."""
    prompt = f"""Given the following navigation query and context about available locations,
generate a response that helps navigate to the most relevant location.

Query: {query}
Query Type: {query_type}

Available Context:
{context}

Instructions:
1. Analyze the query and available locations.
2. Select the most specific and appropriate destination (prefer specific objects over general areas).
3. Format your response as follows:
    - Include the exact location name in double angle brackets: <<exact_name>>
    - Provide a brief explanation of why this location is relevant
    - Include any relevant spatial relationships or navigation hints

IMPORTANT: Use the exact name as it appears in the context, maintaining exact spelling and format.
"""

    response = generate_response(client, prompt)
    return response


def evaluate_rag(client, data):
    """Evaluate the RAG system using the given dataset."""
    win = 0
    total = 0

    # Uncomment if you want progress bars:
    # for item in tqdm(data):
    for item in data:
        query = item['query']
        context = item['context']
        query_type = item['query_type']
        gold_response = item['response']
        try:
            # The system's generated response:
            result = inference(client, query, context, query_type)

            # Compare the system's response (result) with the gold response:
            # If the LLM decides that 'result' is better, it returns 1, else 0.
            score = llm_as_judge(client, query, context, result, gold_response)
            win += score
            total += 1
        except Exception as e:
            print(e)

    win_rate = (win / total) if total > 0 else 0.0
    print(f"Valid Data: {total}")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    return win_rate


def main():
    # Initialize the client
    client = OpenAI(
        base_url=Config.LLM['vllm_settings']["api_base"] + "/v1",
        api_key=Config.LLM['vllm_settings']["api_key"],
    )

    # Load the dataset
    data = []
    with open('benchmark/data/generation_qa.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Evaluate the RAG system
    evaluate_rag(client, data)


if __name__ == "__main__":
    main()