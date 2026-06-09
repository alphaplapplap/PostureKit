import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create a hierarchical layout for the architecture diagram
fig = go.Figure()

# Define positions and connections for the architecture components
# Level 0: Top level
top_level = {'x': [0.5], 'y': [0.9], 'text': ['PostureKit Results View']}

# Level 1: Main components
level1_x = [0.2, 0.5, 0.8]
level1_y = [0.7, 0.7, 0.7]
level1_text = ['Thumbnail Grid<br>(LazyVGrid)', 'Selection Manager', 'File Operations<br>Manager']

# Level 2: Sub-components
level2_x = [0.05, 0.2, 0.35, 0.4, 0.5, 0.6, 0.65, 0.8, 0.95]
level2_y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
level2_text = ['Memory-Optimized<br>Image Loading', 'Lazy Loading<br>with Caching', 'Resizable<br>Thumbnails',
               'Multi-Selection<br>Support', 'Cmd+Click &<br>Shift+Click', 'Checkbox<br>Interface',
               'Context Menu<br>(Right-Click)', 'Move to Folder', 'Batch Operations']

# Level 3: Status bar
level3_x = [0.5]
level3_y = [0.3]
level3_text = ['Status Bar (File Path Display)']

# Combine all positions and texts
all_x = top_level['x'] + level1_x + level2_x + level3_x
all_y = top_level['y'] + level1_y + level2_y + level3_y
all_text = top_level['text'] + level1_text + level2_text + level3_text

# Create connections (edges)
edge_x = []
edge_y = []

# Top level to level 1
for i in range(1, 4):
    edge_x.extend([all_x[0], all_x[i], None])
    edge_y.extend([all_y[0], all_y[i], None])

# Level 1 to level 2 connections
# Thumbnail Grid to its components (indices 4, 5, 6)
for i in range(4, 7):
    edge_x.extend([all_x[1], all_x[i], None])
    edge_y.extend([all_y[1], all_y[i], None])

# Selection Manager to its components (indices 7, 8, 9)
for i in range(7, 10):
    edge_x.extend([all_x[2], all_x[i], None])
    edge_y.extend([all_y[2], all_y[i], None])

# File Operations Manager to its components (indices 10, 11, 12)
for i in range(10, 13):
    edge_x.extend([all_x[3], all_x[i], None])
    edge_y.extend([all_y[3], all_y[i], None])

# Level 1 components to status bar
for i in range(1, 4):
    edge_x.extend([all_x[i], all_x[13], None])
    edge_y.extend([all_y[i], all_y[13], None])

# Add edges
fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                         mode='lines',
                         line=dict(color='#21808d', width=2),
                         hoverinfo='none',
                         showlegend=False))

# Add nodes
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C'] * 3  # Cycle through brand colors

for i, (x, y, text) in enumerate(zip(all_x, all_y, all_text)):
    # Different sizes for different levels
    if i == 0:  # Top level
        size = 60
        color = colors[0]
    elif i <= 3:  # Level 1
        size = 50
        color = colors[i % 5]
    elif i <= 12:  # Level 2
        size = 40
        color = colors[i % 5]
    else:  # Level 3
        size = 45
        color = colors[4]
    
    fig.add_trace(go.Scatter(x=[x], y=[y],
                             mode='markers+text',
                             marker=dict(size=size, color=color, 
                                       line=dict(width=2, color='white')),
                             text=text,
                             textposition='middle center',
                             textfont=dict(size=10, color='white'),
                             hoverinfo='text',
                             hovertext=text,
                             showlegend=False))

# Update layout
fig.update_layout(
    title="PostureKit Results Architecture",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.2, 1.0]),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Save the chart as both PNG and SVG
fig.write_image('posturekit_architecture.png')
fig.write_image('posturekit_architecture.svg', format='svg')

print("Architecture diagram created successfully!")
fig.show()