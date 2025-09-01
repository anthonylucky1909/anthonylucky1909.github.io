import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Steps in the pipeline
steps = [
    "Table (row data)",
    "Serialization (Text Sequence)",
    "Tokenization",
    "GPT-2 (Auto-regressive Model)",
    "Sampling with Temperature",
    "Decoding",
    "Synthetic Table"
]

# Colors for styling
colors = [
    "#4CAF50",  # green
    "#2196F3",  # blue
    "#FFC107",  # amber
    "#9C27B0",  # purple
    "#FF5722",  # deep orange
    "#00BCD4",  # cyan
    "#8BC34A"   # light green
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

boxes = []
texts = []

# Initialize empty elements
for i, step in enumerate(steps):
    y = 9 - i * 1.2
    rect = plt.Rectangle((2, y), 6, 0.8, fill=True, color=colors[i], alpha=0.6)
    ax.add_patch(rect)
    text = ax.text(5, y + 0.4, "", ha="center", va="center", fontsize=12, color="white", weight="bold")
    boxes.append(rect)
    texts.append(text)

# Draw arrows
arrows = []
for i in range(len(steps) - 1):
    y_start = 9 - i * 1.2
    y_end = 9 - (i + 1) * 1.2
    arrow = ax.arrow(5, y_start, 0, -(y_start - y_end) + 0.4, 
                     head_width=0.3, head_length=0.2, fc="black", ec="black", alpha=0)
    arrows.append(arrow)

# Animation function
def update(frame):
    for i in range(len(steps)):
        if i <= frame:
            texts[i].set_text(steps[i])
            boxes[i].set_alpha(0.9)
        else:
            texts[i].set_text("")
            boxes[i].set_alpha(0.2)
    for j in range(len(arrows)):
        if j < frame:
            arrows[j].set_alpha(1)
        else:
            arrows[j].set_alpha(0)
    return texts + boxes + arrows

ani = FuncAnimation(fig, update, frames=len(steps), interval=1200, blit=True, repeat=True)

# Save animation as mp4 (professional look)
ani.save("/mnt/data/Tabular_Data_Pipeline.mp4", writer="ffmpeg", dpi=150)

plt.close()
