import numpy as np
import matplotlib.pyplot as plt

# 1. Define the points
points = np.array([
    [-1.50, -0.63],
    [-0.44, -0.55],
    [ 1.10,  0.03],
    [ 2.02, -0.26],
    [ 1.57, -0.42],
    [ 0.56, -0.50],
    [-0.94, -0.21],
    [-1.43,  0.03],
    [-1.13,  0.30],
    [-0.12,  0.28],
    [ 2.04,  0.26]
])

# 2. Setup the plot
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot(points[:, 0], points[:, 1], 'o-', color='blue', linewidth=2, markersize=8, markerfacecolor='red')

ax.set_title('Interactive Plot: Drag and Drop Points', fontsize=16)
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# This will hold the index of the point currently being dragged
dragged_point_index = None

def on_press(event):
    """
    This function is called when a mouse button is pressed.
    It checks if the click is close to any of the plotted points.
    """
    global dragged_point_index
    
    # Check if the click is within the plot axes
    if event.inaxes != ax:
        return

    # Convert click coordinates from display to data coordinates
    x, y = event.xdata, event.ydata
    
    # Calculate the distance from the click to each point
    distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
    
    # Find the index of the closest point
    closest_point_index = np.argmin(distances)
    
    # Define a tolerance (in data units) to determine if a point is "clicked"
    tolerance = 0.05 
    
    if distances[closest_point_index] < tolerance:
        dragged_point_index = closest_point_index
        # Print the initial coordinates of the point being dragged
        print(f"Dragging Point {dragged_point_index + 1}: ({points[dragged_point_index, 0]:.2f}, {points[dragged_point_index, 1]:.2f})")

def on_motion(event):
    """
    This function is called when the mouse is moved.
    If a point is being dragged, it updates its position and redraws the plot.
    """
    global points, line
    
    if dragged_point_index is not None and event.inaxes == ax:
        # Update the coordinates of the dragged point
        points[dragged_point_index, 0] = event.xdata
        points[dragged_point_index, 1] = event.ydata
        
        # Update the plot with the new coordinates
        line.set_xdata(points[:, 0])
        line.set_ydata(points[:, 1])
        
        # Redraw the canvas
        fig.canvas.draw_idle()
        
        # Print the updated coordinates in real-time
        # print(f"  > New Location: ({points[dragged_point_index, 0]:.2f}, {points[dragged_point_index, 1]:.2f})")

def on_release(event):
    """
    This function is called when the mouse button is released.
    It "drops" the point and prints its final coordinates.
    """
    global dragged_point_index
    if dragged_point_index is not None:
        print(f"Dropped Point {dragged_point_index + 1} at final location: ({points[dragged_point_index, 0]:.2f}, {points[dragged_point_index, 1]:.2f})\n")
        print(points)
    dragged_point_index = None

# Connect the event handlers to the figure canvas
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()