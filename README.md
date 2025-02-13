# Embodied RAG: General Non-parametric Embodied Memory for Retrieval and Generation


## News 📰
- **[2024-11-20]** Released the first version of **Embodied RAG**!
- **[2024-11-21]** Added new retrieval method: **Hierarchical and Spatial Boosted Embedding Retrieval** for much faster retreival speed, from 10s to 0.7s!!
- **[2024-11-21]** Added online semantic forest building during exploration, now can build the tree progressively.

---
![image](https://github.com/user-attachments/assets/d0dd0e9b-3d97-4df8-8f8e-f2b69ff39485)

## Overview
**Embodied RAG** is a system that efficiently builds and queries spatial knowledge graphs from a hierarchical graph we call semantic forest. It uses the saved non-parametric embodied memory to boost the retrieval performance, and enableing question answering for global and implicit queries. 

## Setup

**Prerequisites:**
- Python >= 3.9
- AirSim simulator (Docker setup below)
- OpenAI API key for LLM functionality

1. Clone the repository:
```bash
git clone git@github.com:quanting-xie/Embodied_RAG.git
cd Embodied_RAG
pip install -r requirements.txt

```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

3. Configure parameters (optional):

   Edit `embodied_nav/config.py` to adjust thresholds:

4. Running AirSim with Docker:
```bash
# get docker envs
docker pull jinhuiye/airsim_binary:last


# set up docker
# PROJECT_ROOT_PATH: map your prject root to docker container
bash run_with_docker.sh $PROJECT_ROOT_PATH

# inside docker
cd /home/airsim_user/Documents/AirSim/Unreal/Environments/Building_99/LinuxNoEditor
bash ./Building_99.sh -windowed -ResX=1080 -ResY=720

# The environment inside the Docker container is now ready. You might need additional terminal windows to attach to the container's environment.
docker ps
docker attach <container-name-or-id>
cd $PROJECT_ROOT_PATH

```

4. Running AirSim without Docker:

Follow the instruction here to download UE4 Engine and the environments: https://microsoft.github.io/AirSim/build_linux/



## Usage
### 1. Data Collection (3 method)

1. Offline Tele-operation Collection
```bash
python embodied_nav/airsim_explorer.py
```
- Logs object positions and realtionships as nodes and edges to a topological graph

2. Offline All object Collection
```bash
python embodied_nav/direct_scene_logger.py
```
- Logs all objects in the environment as nodes and edges to a topological graph

3. Online Semantic Forest Collection
```bash
python embodied_nav/online_semantic_explorer.py
```
- Build the topological graph and semantic forest during exploration

### 2. Semantic Forest Building (for offline data)

```bash
python generate_semantic_forest.py
```

- Builds hierarchical structure
- Extracts spatial relationships
- Creates multi-level semantic forest with LLM summaries
- Supports both proximity and cardinal direction relationships


### 3. Graph Visualization

```bash
python graph_visualizer.py semantic_graphs/your_graph.gml
```

Above is an example of the graph visualization, change to your own semantic graph file.

### 4. Retrieval Processing
```bash
python experiment.py --method semantic --query-type implicit
```
or 
```bash
python experiment.py --method llm_hierarchical --query-type implicit
```


## Retrieval Methods Explaination:
1. **LLM-based Hierarchical Traversal(Original Method In Paper)**
   - BFS using LLM to traverse the semantic forest
   - Obtain a hierarchical chain of nodes to the context of answer generation

2. **Hierarchical and Spatial Boosted Embedding Retrieval(New and Faster)**
   - Computes semantic similarity between query and nodes
   - Updatet the scores with a hierarchy boost and a spatial boost
   - Retrieve the top k nodes with the highest scores after normalization



## Citation
If you like our work, please cite:

```bibtex
@article{xie2024embodied,
  title={Embodied-RAG: General Non-parametric Embodied Memory for Retrieval and Generation},
  author={Xie, Quanting and Min, So Yeon and Zhang, Tianyi and Xu, Kedi and Bajaj, Aarav and Salakhutdinov, Ruslan and Johnson-Roberson, Matthew and Bisk, Yonatan},
  journal={arXiv preprint arXiv:2409.18313},
  year={2024}
}

