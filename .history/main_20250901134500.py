import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
from matplotlib import colors
import time

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle('Dynamic Programming Visualization: Fibonacci with DFS and Memoization', fontsize=16, fontweight='bold')

# Initialize variables
n = 6  # Calculate Fibonacci for n=6
call_stack = []
memo = {}
current_call = None
visited_nodes = set()
animation_speed = 1.0

# Set up the tree visualization
ax1.set_title('Function Call Tree')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-7, 1)
ax1.axis('off')

# Set up the memoization table visualization
ax2.set_title('Memoization Table')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, n + 2)
ax2.axis('off')

# Draw the initial memoization table
memo_table = {}
for i in range(n + 1):
    memo_table[i] = ax2.text(2.5, n - i + 0.5, f"fib({i}) = ?", fontsize=12, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax2.text(1, n - i + 0.5, f"n = {i}:", fontsize=12, fontweight='bold')

# Function to draw the call stack
def draw_call_stack():
    ax1.clear()
    ax1.set_title('Function Call Tree')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-7, 1)
    ax1.axis('off')
    
    # Draw the call stack
    for i, call in enumerate(call_stack):
        level = -i
        ax1.text(0, level, f"fib({call})", fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral" if i == len(call_stack)-1 else "lightblue"))
        if i > 0:
            ax1.annotate("", xy=(0, level), xytext=(0, level+1),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Draw visited nodes
    for i, node in enumerate(visited_nodes):
        x = -4 + (i % 3) * 3
        y = -5 + (i // 3) * 1.5
        color = "lightgreen" if node in memo else "lightblue"
        ax1.text(x, y, f"fib({node}) = {memo.get(node, '?')}", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
    
    if current_call is not None and current_call not in call_stack:
        ax1.text(3, 0, f"Computing: fib({current_call})", fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow"))

# Fibonacci function with visualization
def fib(n):
    global current_call, call_stack, visited_nodes
    
    # Add to call stack
    call_stack.append(n)
    current_call = n
    visited_nodes.add(n)
    
    # Update visualization
    draw_call_stack()
    plt.pause(0.5 * animation_speed)
    
    # Check if already computed
    if n in memo:
        # Remove from call stack
        call_stack.pop()
        current_call = call_stack[-1] if call_stack else None
        
        # Update visualization
        draw_call_stack()
        plt.pause(0.3 * animation_speed)
        return memo[n]
    
    # Base cases
    if n <= 1:
        memo[n] = n
        
        # Update memoization table
        memo_table[n].set_text(f"fib({n}) = {memo[n]}")
        memo_table[n].get_bbox_patch().set_facecolor("lightgreen")
        
        # Remove from call stack
        call_stack.pop()
        current_call = call_stack[-1] if call_stack else None
        
        # Update visualization
        draw_call_stack()
        plt.pause(0.5 * animation_speed)
        return n
    
    # Recursive case
    result = fib(n-1) + fib(n-2)
    memo[n] = result
    
    # Update memoization table
    memo_table[n].set_text(f"fib({n}) = {memo[n]}")
    memo_table[n].get_bbox_patch().set_facecolor("lightgreen")
    
    # Remove from call stack
    call_stack.pop()
    current_call = call_stack[-1] if call_stack else None
    
    # Update visualization
    draw_call_stack()
    plt.pause(0.5 * animation_speed)
    return result

# Add explanation text
explanation_text = """
Dynamic Programming with DFS and Memoization:

1. The algorithm traverses the call tree in a Depth-First manner
2. Each function call is added to the call stack
3. Base cases are solved directly (fib(0)=0, fib(1)=1)
4. Results are stored in the memoization table
5. When a previously computed value is needed, it's retrieved from the memo table
6. This avoids redundant calculations and reduces time complexity from O(2^n) to O(n)
"""

fig.text(0.02, 0.02, explanation_text, fontsize=12, verticalalignment='bottom',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

# Calculate Fibonacci sequence with visualization
draw_call_stack()
plt.pause(1)
result = fib(n)

# Add result text
result_text = f"Fibonacci({n}) = {result}"
fig.text(0.5, 0.02, result_text, fontsize=14, fontweight='bold', 
         ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.show()