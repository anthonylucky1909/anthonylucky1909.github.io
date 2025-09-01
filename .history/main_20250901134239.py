import matplotlib.pyplot as plt
import networkx as nx
import time
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.patches as patches

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Dynamic Programming Visualization: Fibonacci with DFS and Memoization', fontsize=16)

# Create a directed graph
G = nx.DiGraph()

# Track the state of the visualization
current_node = None
visited_nodes = set()
memo = {}
call_stack = []
animation_speed = 1.0  # Adjust for faster/slower animation

def fib(n):
    """Fibonacci function with memoization for visualization"""
    global current_node, visited_nodes, memo, call_stack
    
    # Base case visualization
    if n <= 1:
        if n not in visited_nodes:
            visited_nodes.add(n)
            current_node = n
            plt.pause(0.5 * animation_speed)
        return n
    
    # Check if already computed
    if n in memo:
        # Visualization for memo hit
        if n not in visited_nodes:
            visited_nodes.add(n)
        current_node = n
        plt.pause(0.3 * animation_speed)
        return memo[n]
    
    # Recursive case visualization
    if n not in visited_nodes:
        visited_nodes.add(n)
    current_node = n
    call_stack.append(n)
    plt.pause(0.7 * animation_speed)
    
    # Recursive calls
    result = fib(n-1) + fib(n-2)
    
    # Store result in memo
    memo[n] = result
    
    # Visualization for returning from recursive call
    call_stack.pop()
    if call_stack:
        current_node = call_stack[-1]
    else:
        current_node = None
    plt.pause(0.5 * animation_speed)
    
    return result

def update_visualization():
    """Update the visualization based on current state"""
    ax.clear()
    ax.set_title('Dynamic Programming Visualization: Fibonacci with DFS and Memoization', fontsize=16)
    
    # Draw the Fibonacci tree structure
    max_n = max(visited_nodes) if visited_nodes else 5
    pos = hierarchy_pos(G, 5)  # Use 5 as root for better layout
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=20)
    
    # Label nodes with their values and whether they're memoized
    labels = {}
    for node in G.nodes():
        label = f"fib({node})"
        if node in memo:
            label += f"\n={memo[node]}"
        labels[node] = label
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    
    # Highlight the current node
    if current_node is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[current_node], node_color='red', node_size=1500, ax=ax)
    
    # Highlight memoized nodes
    memoized_nodes = [node for node in visited_nodes if node in memo]
    if memoized_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=memoized_nodes, node_color='lightgreen', node_size=1500, ax=ax)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='lightblue', label='Not visited'),
        patches.Patch(facecolor='red', label='Current node'),
        patches.Patch(facecolor='lightgreen', label='Memoized'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Create a hierarchical layout for the tree"""
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                pos=pos, parent=root, parsed=parsed)
    return pos

# Build the Fibonacci call graph for visualization
def build_fib_graph(n):
    """Build the Fibonacci call graph for visualization"""
    G.clear()
    nodes_to_add = set()
    stack = [n]
    
    while stack:
        current = stack.pop()
        nodes_to_add.add(current)
        
        if current > 1:
            if current-1 not in nodes_to_add:
                stack.append(current-1)
            if current-2 not in nodes_to_add:
                stack.append(current-2)
            G.add_edge(current, current-1)
            G.add_edge(current, current-2)
    
    return G

# Initialize the graph for Fibonacci(5)
build_fib_graph(5)

# Create animation
def animate(i):
    if i == 0:
        # Start the Fibonacci calculation
        global visited_nodes, memo, call_stack, current_node
        visited_nodes = set()
        memo = {}
        call_stack = []
        current_node = None
        fib(5)
    update_visualization()

# Create the animation
ani = FuncAnimation(fig, animate, frames=2, interval=100, repeat=False)

# Display the animation
plt.tight_layout()
plt.show()

# For Jupyter notebook, you can use:
# HTML(ani.to_jshtml())