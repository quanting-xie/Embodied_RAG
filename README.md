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

### 1. Graph Building
- Filters out drone nodes from relationship extraction
- Creates hierarchical clusters of spatially related objects
- Adds cardinal direction relationships between objects within threshold distances
- Adds proximity relationships for nearby objects
- Maintains separate thresholds for different relationship types

### 2. Graph Enhancement
The system enhances basic spatial graphs by:
- Adding positional relationships
- Creating hierarchical clusters
- Generating summaries for object groups
- Computing embeddings for nodes

### 3. Query Processing
Supports different types of queries:
- Explicit spatial queries
- Implicit spatial queries
- Navigation queries with AirSim integration

## Usage

### 1. Generate Enhanced Graph
Use the system to build and enhance the spatial knowledge graph, and query it to extract spatial relationships.
