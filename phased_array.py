import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Number of frames (time steps) for the animation
frames = 50

# Redefine the figure and axis to avoid conflicts
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')

# Define positions of the four elements along the x-axis
phase_factor = 0# For best results, keep between -2 and 2

n_elements = 12
element_positions = np.linspace(-3, 3, n_elements)
delay = np.linspace(-1, 1, n_elements)
delay = delay*phase_factor
# Function to initialize the plot
def init_phased_array():
    ax.clear()
    return []

# Function to update the plot for phased array source (4 elements)
def update_phased_array(frame):
    ax.clear()  # Clear previous frame
    radius = frame * 0.2  # Wave expands by 0.2 units per frame
    
    # Create a wavefront circle for each element
    i=0
    for pos in element_positions:
        theta = np.linspace(0, 2 * np.pi, 100)
        if radius < delay[i]:
            x = 0
            y = 0
        else:
            x = pos + (radius-delay[i]) * np.cos(theta)  # X-coordinates of the wavefront for this element
            y = (radius-delay[i]) * np.sin(theta)  # Y-coordinates of the wavefront
        ax.plot(x, y, lw=2, color = 'red')
        ax.plot(element_positions, element_positions*0, lw=2, color = 'blue', marker='o')
        i = i + 1
        
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

# Create the animation for the phased array source
anim_phased = FuncAnimation(fig, update_phased_array, frames=frames, init_func=init_phased_array, blit=False)

# Display the animation
plt.show()
