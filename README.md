# Embodied RAG - Spatial Knowledge Graph Builder and Query System

## Overview
**Embodied RAG** is a system that builds and queries spatial knowledge graphs by extracting relationships between objects in 3D environments. It combines spatial, hierarchical, and proximity relationships to create a rich understanding of object arrangements.

## Setup

**Prerequisites:**
- Python >= 3.9 (required by LightRAG)
- AirSim simulator
- OpenAI API key for LLM functionality


```bash
git clone https://github.com/yourusername/Embodied_RAG.git
cd Embodied_RAG
pip install -r requirements.txt

```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

Configure parameters (optional):
   Edit `embodied_nav/config.py` to adjust thresholds:

Running AirSim:
    ```bash
    cd AirSim/Unreal/Environments/Building99/LinuxNoEditor
    ./Building99.sh
    ```

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

## Key Features

### 1. Topological Graph Building
```bash
cd embodied_rag
python airsim_explorer.py
```
- Explore the environment with a drone and object detector
- Logs object positions and realtionships as nodes and edges to NanoDB

### 2. Semantic Forest Building
```bash
python generate_semantic_forest.py
```
- Adding hierarhical information and relative relationships between nodes.
- Three types of relationships are extracted for edges: spatial, proximity, and hierarchical
    - Spatial: Cardinal direction relationships between objects (north, south, east, west) with vertical components (above, below)
    - Proximity: Is objects A near object B (within a threshold distance)
    - Hierarchical: A part of B (forming a hierarchical structure of spaces)
- Generating summaries for clustered object groups
- Computing embeddings for nodes and clusters and relationships

### 3. Retrieval Processing
```bash
cd ..
python retrieval.py
```
- Retrieving relevant nodes and relationships based on query embeddings:
    - Compute semantic similarity between query and nodes
    - Retrieve top k nodes with highest similarity
    - Retrieve each node's spatial, proximity, and hierarchical relationships(customizable in config.py)

Supports different types of queries:
- Explicit spatial queries
- Implicit spatial queries
- Global queries
