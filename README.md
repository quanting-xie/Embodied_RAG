# Embodied RAG - Spatial Knowledge Graph Builder and Query System

## Overview
**Embodied RAG** is a system that builds and queries spatial knowledge graphs by extracting relationships between objects in 3D environments. It combines spatial, hierarchical, and proximity relationships to create a rich understanding of object arrangements.

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
- Explore the environment with a drone and object detector
- Logs object positions and realtionships as nodes and edges to NanoDB

### 2. Semantic Forest Building
- Adding hierarhical information and relative relationships between nodes.
- Three types of relationships are extracted for edges: spatial, proximity, and hierarchical
    - Spatial: Cardinal direction relationships between objects (north, south, east, west) with vertical components (above, below)
    - Proximity: Is objects A near object B (within a threshold distance)
    - Hierarchical: A part of B (forming a hierarchical structure of spaces)
- Generating summaries for clustered object groups
- Computing embeddings for nodes and clusters and relationships

### 3. Retrieval Processing
- Retrieving relevant nodes and relationships based on query embeddings:
    - Compute semantic similarity between query and nodes
    - Retrieve top k nodes with

### 3. Query Processing
Supports different types of queries:
- Explicit spatial queries
- Implicit spatial queries
- Navigation queries with AirSim integration

## Usage

### 1. Generate Enhanced Graph
Use the system to build and enhance the spatial knowledge graph, and query it to extract spatial relationships.
