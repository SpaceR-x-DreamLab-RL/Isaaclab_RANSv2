import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_trajectories_and_gates(csv_file, trajectory_ids=None):
    """
    Reads trajectory data from a CSV file, plots specified trajectories,
    and visualizes the race gates.

    Args:
        csv_file (str): Path to the CSV file containing trajectory data.
        trajectory_ids (list or None): A list of trajectory IDs to plot. 
                                      If None, all trajectories will be plotted.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Extract the gate positions. The column names follow a specific pattern:
    # 'target_positions_{gate_id * 6}', 'target_positions_{gate_id * 6 + 1}', ..., 'target_positions_{gate_id * 6 + 5}'
    # We can identify these by their name format.
    gate_columns = [col for col in df.columns if col.startswith('target_positions_')]
    num_gates = len(gate_columns) // 6
    print(f"Found {num_gates} gates in the data.")

    # Create a figure and a 3D axes object for plotting
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Drone Trajectories and Race Gates")
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    ax.set_zlabel("Z Position (meters)")
    ax.set_aspect('equal', adjustable='box')

    # Get a list of unique trajectory IDs
    unique_trajectories = df['trajectory'].unique()
    
    # Determine which trajectories to plot
    if trajectory_ids is None:
        trajectories_to_plot = unique_trajectories
    else:
        # Filter for the specified trajectory IDs
        trajectories_to_plot = [tid for tid in trajectory_ids if tid in unique_trajectories]
        if not trajectories_to_plot:
            print(f"Warning: No valid trajectories found for IDs {trajectory_ids}.")
            return

    # Plot each specified trajectory
    print("Plotting trajectories...")
    for tid in trajectories_to_plot:
        trajectory_data = df[df['trajectory'] == tid]
        x_positions = trajectory_data['step'].values
        y_positions = trajectory_data['target_positions_0'].values
        z_positions = trajectory_data['target_positions_1'].values
        
        ax.plot(x_positions, y_positions, z_positions, label=f'Trajectory {tid}', linewidth=2)
        ax.scatter(x_positions, y_positions, z_positions, s=20)
    
    # Plot the gates and their IDs
    print("Plotting gates...")
    gate_colors = plt.cm.get_cmap('viridis', num_gates)
    for i in range(num_gates):
        # The first row of the CSV seems to contain the gate positions
        gate_x = df.loc[0, f'target_positions_{i * 6}']
        gate_y = df.loc[0, f'target_positions_{i * 6 + 1}']
        gate_z = df.loc[0, f'target_positions_{i * 6 + 2}']
        
        # Plot the gate location
        ax.scatter(gate_x, gate_y, gate_z, color=gate_colors(i), marker='^', s=100, label=f'Gate {i}', edgecolor='black', linewidth=1)
        
        # Add a text label for the gate ID
        ax.text(gate_x, gate_y, gate_z, str(i), fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    # Add a legend and display the plot
    ax.legend(loc='upper right')
    plt.tight_layout()
    # plt.show()
    plt.savefig("racing_circuit_plot.png")

if __name__ == '__main__':
    # Define the path to your CSV file
    csv_file_path = 'extracted_trajectories_RaceGates.csv'
    
    plot_trajectories_and_gates(csv_file_path)
