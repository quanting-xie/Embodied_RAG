from embodied_nav.config import Config
from subprocess import Popen
import time
import os
import requests

# Load vLLM settings from the configuration
vllm_info = Config.LLM['vllm_settings']

# Extract necessary parameters from the configuration
model = vllm_info['model']
api_key = vllm_info['api_key']
tensor_parallel_size = vllm_info['tensor_parallel_size']
max_num_seqs = vllm_info['max_num_seqs']
swap_space = vllm_info['swap_space']
server_startup_timeout = vllm_info['server_startup_timeout']

docker_command = [
    "docker", "run", "--runtime", "nvidia", "--gpus", "all",
    "-v", os.path.expanduser("~/.cache/huggingface:/root/.cache/huggingface"),
    "--env", f"HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN')}",
    "--env", f"API_KEY={api_key}",
    "-p", "8000:8000",
    "--ipc=host",
    "vllm/vllm-openai:latest",
    "--model", model,
    "--tensor-parallel-size", str(tensor_parallel_size),
    "--max-num-seqs", str(max_num_seqs),
    "--swap-space", str(swap_space),
    "--trust-remote-code"
]

print("Starting the vLLM server container...")
process = Popen(docker_command)

# Wait for the server to initialize
time.sleep(server_startup_timeout)
try:
    response = requests.get(f"{api_key}/health")
    if response.status_code == 200:
        print("vLLM server is up and running.")
    else:
        print("vLLM server did not start as expected.")
except requests.exceptions.RequestException as e:
    print(f"Error connecting to the vLLM server: {e}")