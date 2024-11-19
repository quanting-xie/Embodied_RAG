# Embodied RAG - Spatial Knowledge Graph Builder and Query System

## Overview
**Embodied RAG** is a system that builds and queries spatial knowledge graphs by extracting relationships between objects in 3D environments. It combines spatial, hierarchical, and proximity relationships to create a rich understanding of object arrangements.

## Setup

**Prerequisites:**
- Python >= 3.9 (required by LightRAG)
- AirSim simulator
- OpenAI API key for LLM functionality

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Embodied_RAG.git
cd Embodied_RAG
pip install -r requirements.txt

```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

3. Configure parameters (optional):

   Edit `embodied_nav/config.py` to adjust thresholds:

4. Running AirSim:
```bash
cd AirSim/Unreal/Environments/Building99/LinuxNoEditor
./Building99.sh

```
## Usage
### 1. Data Collection

```bash
cd embodied_rag
python airsim_explorer.py
```
- Explore the environment with a drone and object detector
- Logs object positions and realtionships as nodes and edges to NanoDB

### 2. Semantic Forest Building

```bash
cd ..
python generate_semantic_forest.py
```

- Builds hierarchical structure
- Extracts spatial relationships
- Creates multi-level semantic forest with LLM summaries
- Supports both proximity and cardinal direction relationships

### 3. Retrieval Processing
```bash
python experiment.py --method semantic --query-type implicit
# or
python experiment.py --method llm_hierarchical --query-type explicit --query "Find the nearest chair"
```
- Remember to change the retreival method in the config.py file

#### Retrieval Methods:
1. **LLM-based Hierarchical Traversal(Original Method In Paper)**
   - Intelligent node selection using LLM
   - Hierarchical traversal through semantic forest
   - Supports context-aware selection
   - Maximum 3 parallel paths for diverse results

2. **Embedding-based Retrieval(New and Faster)**
   - Computes semantic similarity between query and nodes
   - Retrieves top k nodes with highest similarity
   - Customizable thresholds in config.py





## Core Components

### 1. SpatialRelationshipExtractor
The main component for building spatial knowledge graphs. It extracts three types of relationships:

- **Spatial (Cardinal) Relationships**: Directional relationships (north, south, east, west) with vertical components (above, below)
- **Proximity Relationships**: "Near" relationships based on distance thresholds
- **Hierarchical Relationships**: Part-of relationships forming a hierarchical structure of spaces

### 2. EmbodiedRAG
The main interface for the system that:
- Manages the spatial knowledge graph
- Handles queries about spatial relationships
- Integrates with AirSim for physical navigation

