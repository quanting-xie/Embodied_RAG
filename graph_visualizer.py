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

    # Process edges
    for edge in G.edges(data=True):
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        
        # Get edge type and relationship
        edge_type = edge[2].get('type', '')
        relationship = edge[2].get('relationship', '')
        distance = edge[2].get('distance', None)
        
        # Create edge coordinates
        edge_x = [x0, x1, None]
        edge_y = [y0, y1, None]
        edge_z = [z0, z1, None]
        
        # Create hover text
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
        
        # Add to appropriate edge collection
        if edge_type == 'drone_path':
            drone_edge_x.extend(edge_x)
            drone_edge_y.extend(edge_y)
            drone_edge_z.extend(edge_z)
            drone_edge_text.append(hover_text)
        elif relationship in ['part_of', 'contains']:
            part_of_edge_x.extend(edge_x)
            part_of_edge_y.extend(edge_y)
            part_of_edge_z.extend(edge_z)
            part_of_edge_text.append(hover_text)
        else:  # All other relationships are spatial
            spatial_edge_x.extend(edge_x)
            spatial_edge_y.extend(edge_y)
            spatial_edge_z.extend(edge_z)
            spatial_edge_text.append(hover_text)

    # Create traces for different edge types
    spatial_edge_trace = go.Scatter3d(
        x=spatial_edge_x, y=spatial_edge_y, z=spatial_edge_z,
        line=dict(
            color='rgba(150, 150, 150, 0.4)',  # Lighter grey
            width=1,
            dash='dot'
        ),
        name='Spatial Relationship',
        hoverinfo='text',
        text=spatial_edge_text,
        mode='lines',
        showlegend=True
    )

    # Update part-of edge trace with light orange lines
    part_of_edge_trace = go.Scatter3d(
        x=part_of_edge_x, y=part_of_edge_y, z=part_of_edge_z,
        line=dict(
            color='rgba(255, 165, 0, 0.7)',  # Light orange
            width=2
        ),
        name='Hierarchical Relationship',
        hoverinfo='text',
        text=part_of_edge_text,
        mode='lines',
        showlegend=True
    )

    drone_edge_trace = go.Scatter3d(
        x=drone_edge_x, y=drone_edge_y, z=drone_edge_z,
        line=dict(
            color='cyan',  # Changed to cyan
            width=2,
            dash='dot'
        ),
        name='Drone Path',
        hoverinfo='text',
        text=drone_edge_text,
        mode='lines',
        showlegend=True
    )

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
            colorscale='Plasma',  # Changed from Viridis for better contrast on dark background
            color=node_color,
            size=node_size,
            sizemode='diameter',
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
                    color='white'  # Changed to white
                ),
                title_font=dict(
                    color='white'  # Added white color for title
                )
            ),
            line=dict(
                color='rgba(255, 255, 255, 0.3)',  # Added white border
                width=1
            )
        ))

    # Update drone node trace with darker blue
    drone_node_trace = go.Scatter3d(
        x=drone_node_x, y=drone_node_y, z=drone_node_z,
        mode='markers',
        hoverinfo='text',
        text=drone_node_text,
        name='Drone Positions',
        marker=dict(
            color='rgb(0, 102, 204)',  # Darker blue
            size=1.5,
            symbol='diamond',
            line=dict(
                color='white',
                width=1
            )
        ))

    # Create the figure with dark theme
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
            xaxis=dict(
                showbackground=True,
                backgroundcolor='rgb(30, 30, 30)',
                gridcolor='white',
                title=dict(
                    text='X',
                    font=dict(color='white')
                ),
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor='rgb(30, 30, 30)',
                gridcolor='white',
                title=dict(
                    text='Y',
                    font=dict(color='white')
                ),
                tickfont=dict(color='white')
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor='rgb(30, 30, 30)',
                gridcolor='white',
                title=dict(
                    text='Z',
                    font=dict(color='white')
                ),
                tickfont=dict(color='white')
            ),
            bgcolor='rgb(10, 10, 10)'  # Dark background
        ),
        paper_bgcolor='black',  # Page background
        plot_bgcolor='black',   # Plot background
        showlegend=True,
        margin=dict(b=0, l=0, r=0, t=40),
        title_font=dict(color='white'),  # Title color
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)'  # Transparent legend background
        )
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
