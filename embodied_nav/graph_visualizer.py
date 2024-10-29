import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def hierarchical_layout(G):
    pos = {}
    levels = set(nx.get_node_attributes(G, 'level').values())
    for level in sorted(levels):
        nodes = [n for n, d in G.nodes(data=True) if d.get('level') == level]
        pos.update(nx.spring_layout(G.subgraph(nodes), dim=3, k=1/np.sqrt(len(nodes))))
    return pos

def visualize_enhanced_graph_3d(G):
    # Initialize position dictionary
    pos = {}
    z_increment = 2  # Height increment per level
    
    # First pass: Add all nodes with explicit positions
    for node, data in G.nodes(data=True):
        if 'position' in data:
            level = data.get('level', 0)
            if isinstance(data['position'], list):
                # Flip z for drone nodes, add increment for others
                if isinstance(node, str) and 'drone' in node.lower():
                    z = -data['position'][2]
                else:
                    z = data['position'][2] + (level * z_increment)
                pos[node] = [data['position'][0], data['position'][1], z]
            elif isinstance(data['position'], dict):
                if isinstance(node, str) and 'drone' in node.lower():
                    z = -data['position'].get('z', 0)
                else:
                    z = data['position'].get('z', 0) + (level * z_increment)
                pos[node] = [
                    data['position'].get('x', 0),
                    data['position'].get('y', 0),
                    z
                ]

    # Remove the circular arrangement code and just use the positions directly
    # The positions are already calculated correctly in the extractor
    
    # Create separate edge traces for different types of edges
    part_of_edge_x, part_of_edge_y, part_of_edge_z = [], [], []
    spatial_edge_x, spatial_edge_y, spatial_edge_z = [], [], []
    drone_edge_x, drone_edge_y, drone_edge_z = [], [], []
    part_of_edge_text, spatial_edge_text, drone_edge_text = [], [], []

    # Cardinal directions for identifying spatial relationships
    cardinal_directions = ['north', 'south', 'east', 'west', 
                         'north_above', 'north_below', 
                         'south_above', 'south_below',
                         'east_above', 'east_below',
                         'west_above', 'west_below']

    for edge in G.edges(data=True):
        if edge[0] in pos and edge[1] in pos:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            # Create hover text for edge with distance if available
            relationship = edge[2].get('relationship', 'connected')
            distance = edge[2].get('distance', None)
            
            # Create base hover text
            if distance is not None and isinstance(distance, (int, float)):
                hover_text = f"{edge[0]}<br>"
                
                # Add spatial relationship if it exists
                if any(direction in str(relationship) for direction in cardinal_directions):
                    hover_text += f"--{relationship}--<br>"
                
                # Add proximity information
                hover_text += f"--near ({distance:.2f}m)--<br>"
                hover_text += f"{edge[1]}"
            else:
                hover_text = f"{edge[0]}<br>--{relationship}--<br>{edge[1]}"
            
            # Check if this is a drone path edge
            is_drone_path = (
                isinstance(edge[0], str) and 'drone' in edge[0].lower() and
                isinstance(edge[1], str) and 'drone' in edge[1].lower()
            )
            
            if is_drone_path:
                drone_edge_x.extend([x0, x1, None])
                drone_edge_y.extend([y0, y1, None])
                drone_edge_z.extend([z0, z1, None])
                drone_edge_text.extend([hover_text, hover_text, None])
            elif edge[2].get('relationship') == 'part_of':
                part_of_edge_x.extend([x0, x1, None])
                part_of_edge_y.extend([y0, y1, None])
                part_of_edge_z.extend([z0, z1, None])
                part_of_edge_text.extend([hover_text, hover_text, None])
            elif edge[2].get('relationship') == 'near' or any(direction in str(relationship) for direction in cardinal_directions):
                # Combine spatial and proximity relationships
                spatial_edge_x.extend([x0, x1, None])
                spatial_edge_y.extend([y0, y1, None])
                spatial_edge_z.extend([z0, z1, None])
                spatial_edge_text.extend([hover_text, hover_text, None])

    # Create traces for different edge types
    part_of_edge_trace = go.Scatter3d(
        x=part_of_edge_x, y=part_of_edge_y, z=part_of_edge_z,
        line=dict(color='red', width=2),
        name='Hierarchical Relationship',
        hoverinfo='text',
        text=part_of_edge_text,
        mode='lines',
        showlegend=True)

    spatial_edge_trace = go.Scatter3d(
        x=spatial_edge_x, y=spatial_edge_y, z=spatial_edge_z,
        line=dict(color='rgba(0, 0, 0, 0.5)', width=1),  # Solid black, increased width
        name='Spatial/Proximity Relationship',
        hoverinfo='text',
        text=spatial_edge_text,
        mode='lines',
        showlegend=True)

    drone_edge_trace = go.Scatter3d(
        x=drone_edge_x, y=drone_edge_y, z=drone_edge_z,
        line=dict(color='blue', width=2, dash='dot'),
        name='Drone Path',
        hoverinfo='text',
        text=drone_edge_text,  # Use drone specific hover text
        mode='lines',
        showlegend=True)

    # Create separate node traces for different types of nodes
    # Regular nodes
    node_x, node_y, node_z = [], [], []
    node_text = []
    node_color = []
    node_size = []
    # Drone nodes
    drone_node_x, drone_node_y, drone_node_z = [], [], []
    drone_node_text = []

    for node, data in G.nodes(data=True):
        if node in pos:
            x, y, z = pos[node]
            is_drone = isinstance(node, str) and 'drone' in node.lower()
            
            if is_drone:
                drone_node_x.append(x)
                drone_node_y.append(y)
                drone_node_z.append(z)
                drone_node_text.append(f"Drone: {node}")
            else:
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                
                # Create hover text
                hover_text = [f"Node: {node}"]
                for key, value in data.items():
                    if key not in ['position', 'embedding']:
                        hover_text.append(f"{key}: {value}")
                node_text.append("<br>".join(hover_text))
                
                # Color by level
                node_color.append(data.get('level', 0))
                
                # Size based on level
                node_size.append(20 if data.get('level', 0) > 0 else 10)

    # Create regular node trace with smaller gradient legend
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        name='Objects',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=10,        # Reduced thickness
                title='Node Level',
                xanchor='left',
                titleside='right',
                len=0.5,            # Reduced length to 50%
                yanchor='middle',   # Center the colorbar
                y=0.5,             # Center position
                tickfont=dict(
                    size=10,  # Smaller tick labels
                    color='black'
                )
            ),
            line_width=2))

    # Create drone node trace
    drone_node_trace = go.Scatter3d(
        x=drone_node_x, y=drone_node_y, z=drone_node_z,
        mode='markers',
        hoverinfo='text',
        text=drone_node_text,
        name='Drone Positions',
        marker=dict(
            color='blue',
            size=8,
            symbol='diamond',
            line_width=1
        ))

    # Create the figure with all traces
    fig = go.Figure(data=[
        part_of_edge_trace, 
        spatial_edge_trace,
        drone_edge_trace, 
        node_trace, 
        drone_node_trace
    ])

    fig.update_layout(
        title='3D Graph Visualization with Hierarchical Layout and Drone Path',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True,
        margin=dict(b=0, l=0, r=0, t=40)
    )
    
    fig.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python graph_visualizer.py <path_to_enhanced_graph.gml>")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    G = nx.read_gml(graph_file)
    visualize_enhanced_graph_3d(G)
