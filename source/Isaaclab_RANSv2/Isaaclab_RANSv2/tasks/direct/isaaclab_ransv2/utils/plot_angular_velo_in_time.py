import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('2026-04-30_11-27-24_ppo_Pingu_TrackVelocities_rsl_rl_seed_1/metrics/detailed_trajectories_TrackVelocities.csv')

# Use the same 5 unique trajectory IDs for consistency
unique_trajectories = np.arange(7).tolist()

# Set the style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 8))

# Define colors to ensure matching between actual and target for each trajectory
colors = sns.color_palette("tab10", len(unique_trajectories))

# Plot each trajectory using world-frame angular velocity (w_z)
for i, t_id in enumerate(unique_trajectories):
    subset = df[df['trajectory_id'] == t_id]
    color = colors[i]
    
    # Plot Actual World Angular Velocity Z (Solid)
    plt.plot(subset['step'], subset['angular_velocity_w_z'], 
             label=f'Actual (w_z) Traj {t_id}', color=color, linestyle='-', linewidth=2)
    
    # Plot Target Angular Velocity (Discontinuous/Dashed)
    plt.plot(subset['step'], subset['angular_velocity_target'], 
             label=f'Target Traj {t_id}', color=color, linestyle='--', alpha=0.7)

plt.title('Actual (World Frame) vs Target Angular Velocity (Z-axis) for 5 Trajectories', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Angular Velocity (rad/s)', fontsize=12)
plt.legend(title='Trajectory ID & Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save the plot
plt.savefig('actual_w_z_vs_target.png')
plt.show()