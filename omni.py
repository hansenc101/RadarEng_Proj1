import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')

# Initial radar source point at the origin
source = (0, 0)

# Number of frames (time steps) for the animation
frames = 20

# List to hold the wavefront circles
wavefronts = []

# Create a circle (wavefront) that expands
circle, = ax.plot([], [], lw=2)


# Function to initialize the plot
def init():
    circle.set_data([], [])
    return circle,

# Function to update the plot for each frame
def update(frame):
    # Calculate the radius of the wavefront at each time step
    radius = frame * 0.1  # Wave expands by 0.1 units per frame
    
    # Create the wavefront circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = source[0] + radius * np.cos(theta)
    y = source[1] + radius * np.sin(theta)
    
    circle.set_data(x, y)
    return circle,

# Create the animation
anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# Display the animation
plt.show()
