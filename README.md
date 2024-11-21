# Embodied RAG - Spatial Knowledge Graph Builder and Query System


## News 📰
- **[2024-11-20]** Released the first version of **Embodied RAG**!
- **[2024-11-21]** Added new retrieval method: **Embedding-based Retrieval** for much faster retreival speed.
- **[2024-11-21]** Added online semantic forest building during exploration, now can build the tree progressively.

---

## Overview
**Embodied RAG** is a system that builds and queries spatial knowledge graphs by extracting relationships between objects in 3D environments. It combines spatial, hierarchical, and proximity relationships to create a rich understanding of object arrangements.

## Setup

**Prerequisites:**
- Python >= 3.9 (required by LightRAG)
- AirSim simulator
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
docker pull jinhuiye/airsim_binary:1.2
UNREAL_ROOT_PATH=your path for AirSim

# enter docker
bash run_with_docker.sh $UNREAL_ROOT_PATH

# inside docker
cd AirSim/Unreal/Environments/Building99/LinuxNoEditor
./Building99.sh -windowed -ResX=1080 -ResY=720

```
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


### Retrieval Methods Explaination:
1. **LLM-based Hierarchical Traversal(Original Method In Paper)**
   - BFS using LLM to traverse the semantic forest
   - Obtain a hierarchical chain of nodes to the context of answer generation

2. **Embedding-based Retrieval(New and Faster)**
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

