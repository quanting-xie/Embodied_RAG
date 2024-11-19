import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

def hierarchical_layout(G):
    pos = {}
    levels = set(nx.get_node_attributes(G, 'level').values())
    for level in sorted(levels):
        nodes = [n for n, d in G.nodes(data=True) if d.get('level') == level]
        pos.update(nx.spring_layout(G.subgraph(nodes), dim=3, k=1/np.sqrt(len(nodes))))
    return pos

def format_hover_text(text, max_line_length=40):
    """Format hover text with line breaks and proper spacing"""
    if isinstance(text, str):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines)
    return text

def visualize_enhanced_graph_3d(G, output_path=None):
    # Initialize position dictionary
    pos = {}
    z_increment = 5  # Height increment per level
    
    # First pass: Add all nodes with explicit positions
    for node, data in G.nodes(data=True):
        if 'position' in data:
            level = data.get('level', 0)
            is_drone = isinstance(node, str) and 'drone' in node.lower()
            
            if isinstance(data['position'], list):
                # Invert Z only for drone nodes
                z = -data['position'][2] if is_drone else data['position'][2]
                if level > 0:  # Only add increment for higher levels
                    z += (level * z_increment)
                pos[node] = [data['position'][0], data['position'][1], z]
            elif isinstance(data['position'], dict):
                # Invert Z only for drone nodes
                z = -data['position'].get('z', 0) if is_drone else data['position'].get('z', 0)
                if level > 0:  # Only add increment for higher levels
                    z += (level * z_increment)
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
            
            if distance is not None and isinstance(distance, (int, float)):
                hover_text = [
                    f"From: {edge[0]}",
                    f"Relationship: {relationship}",
                    f"Distance: {distance:.2f}m",
                    f"To: {edge[1]}"
                ]
                hover_text = '<br>'.join(hover_text)
            else:
                hover_text = '<br>'.join([
                    f"From: {edge[0]}",
                    f"Relationship: {relationship}",
                    f"To: {edge[1]}"
                ])
            
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
            else:
                relationship = edge[2].get('relationship', '')
                distance = edge[2].get('distance', None)
                
                # Check relationship type
                if isinstance(relationship, str):
                    if relationship == 'part_of':
                        part_of_edge_x.extend([x0, x1, None])
                        part_of_edge_y.extend([y0, y1, None])
                        part_of_edge_z.extend([z0, z1, None])
                        part_of_edge_text.extend([hover_text, hover_text, None])
                    elif any(direction in relationship.lower() for direction in cardinal_directions):
                        spatial_edge_x.extend([x0, x1, None])
                        spatial_edge_y.extend([y0, y1, None])
                        spatial_edge_z.extend([z0, z1, None])
                        spatial_edge_text.extend([hover_text, hover_text, None])

    # Create traces for different edge types
    spatial_edge_trace = go.Scatter3d(
        x=spatial_edge_x, y=spatial_edge_y, z=spatial_edge_z,
        line=dict(
            color='rgba(200, 200, 200, 0.4)',  # Light grey with transparency
            width=1,
            dash='dot'  # Dotted line style
        ),
        name='Spatial Relationship',
        hoverinfo='text',
        text=spatial_edge_text,
        mode='lines',
        showlegend=True
    )

    # Create part-of edge trace with solid red lines
    part_of_edge_trace = go.Scatter3d(
        x=part_of_edge_x, y=part_of_edge_y, z=part_of_edge_z,
        line=dict(
            color='rgba(255, 0, 0, 0.7)',  # Red with some transparency
            width=2  # Thicker than spatial relationships
        ),
        name='Hierarchical Relationship',
        hoverinfo='text',
        text=part_of_edge_text,
        mode='lines',
        showlegend=True
    )

    drone_edge_trace = go.Scatter3d(
        x=drone_edge_x, y=drone_edge_y, z=drone_edge_z,
        line=dict(color='blue', width=2, dash='dot'),
        name='Drone Path',
        hoverinfo='text',
        text=drone_edge_text,  # Use drone specific hover text
        mode='lines',
        showlegend=True)

    # Create separate node traces for different types of nodes
    node_x, node_y, node_z = [], [], []
    node_text = []
    node_color = []
    node_size = []
    # Drone nodes
    drone_node_x, drone_node_y, drone_node_z = [], [], []
    drone_node_text = []

    # Get max level for size scaling
    max_level = max((data.get('level', 0) for _, data in G.nodes(data=True)), default=0)
    
    # Base and scale factors for node size
    base_size = 5
    size_increment = 4  # Size increase per level
    
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
                    if key not in ['box2D', 'box3D', 'embedding']:
                        if key == 'position':
                            if isinstance(value, dict):
                                hover_text.append(f"position: (x:{value['x']:.2f}, y:{value['y']:.2f}, z:{value['z']:.2f})")
                        elif key == 'summary':
                            hover_text.append(f"{key}:")
                            hover_text.append(format_hover_text(str(value)))
                        else:
                            hover_text.append(f"{key}: {value}")
                
                node_text.append("<br>".join(hover_text))
                
                # Color by level
                level = data.get('level', 0)
                node_color.append(level)
                
                # Calculate size based on level - gradually increasing
                node_size.append(base_size + (level * size_increment))

    # Create regular node trace with updated marker sizes
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
            sizemode='diameter',  # Ensure consistent size scaling
            colorbar=dict(
                thickness=10,
                title='Node Level',
                xanchor='left',
                titleside='right',
                len=0.5,
                yanchor='middle',
                y=0.5,
                tickfont=dict(
                    size=10,
                    color='black'
                )
            ),
            line_width=2))

    # Create drone node trace with smaller size
    drone_node_trace = go.Scatter3d(
        x=drone_node_x, y=drone_node_y, z=drone_node_z,
        mode='markers',
        hoverinfo='text',
        text=drone_node_text,
        name='Drone Positions',
        marker=dict(
            color='blue',
            size=1.5,
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
    
    if output_path:
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Save all files to plots directory
        output_path = os.path.join('plots', output_path)
        
        # Save as HTML (fully interactive)
        fig.write_html(f"{output_path}.html")
        
        # Save as HTML with all requirements bundled (self-contained)
        fig.write_html(f"{output_path}_standalone.html", include_plotlyjs=True)
   
        
        print(f"Saved visualizations to:")
        print(f"- {output_path}.html")
        print(f"- {output_path}_standalone.html")

    
    # Still show the interactive plot
    fig.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python graph_visualizer.py <path_to_enhanced_graph.gml> [output_path]")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else None
    
    G = nx.read_gml(graph_file)
    visualize_enhanced_graph_3d(G, output_path)
