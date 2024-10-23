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
    pos = {}
    for node, data in G.nodes(data=True):
        if 'position' in data:
            pos[node] = data['position']
    
    # If any node doesn't have a position, fall back to hierarchical layout
    if len(pos) != len(G.nodes()):
        pos = hierarchical_layout(G)
    
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y, node_z = [], [], []
    node_text = []
    node_color = []
    for node, data in G.nodes(data=True):
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f"Node: {node}<br>"
                         f"Level: {data.get('level', 'N/A')}<br>"
                         f"Summary: {data.get('summary', 'N/A')}<br>"
                         f"Position: {data.get('position', 'N/A')}")
        node_color.append(data.get('level', 0))

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Level',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    fig.update_layout(
        title='3D Graph Visualization (Using Object Positions)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        annotations=[dict(
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)])
    
    fig.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python graph_visualizer.py <path_to_enhanced_graph.gml>")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    G = nx.read_gml(graph_file)
    visualize_enhanced_graph_3d(G)
