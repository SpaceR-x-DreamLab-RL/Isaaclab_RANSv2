from . import BaseTaskMetrics, Registerable
import torch

class RaceGatesMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def time_to_complete_a_loop(self):
        print("[INFO][METRICS][TASK] Time to complete a loop")

        target_positions = self.trajectories['target_positions'][0][0]
        
        # Initialize with max timesteps (for environments that didn't complete)
        total_timesteps = torch.full((self.trajectories['trajectory_completed'].shape[0],), self.trajectories['trajectory_completed'].shape[-1], dtype=torch.float32, device=self.trajectories_masks.device)
        
        # Find first occurrence of trajectory completion for each environment
        trajectory_completed_env, trajectory_completed_idx = torch.where(self.trajectories['trajectory_completed'] == True)
        
        # For environments that completed, use completion timestep + 1 (since we want time including the completion step)
        total_timesteps[trajectory_completed_env] = trajectory_completed_idx.float() + 1

        # Remove trajectories that didn't complete
        faild_trajectories_mask = self.trajectories['missed_gate'].sum(dim=-1) > 0
        total_timesteps[faild_trajectories_mask] = torch.nan

        time_in_s = total_timesteps * self.step_dt
        self.metrics["time_to_complete_a_loop.s"] = time_in_s
        self.metrics["faild_trajectories_mask.u"] = faild_trajectories_mask

    @BaseTaskMetrics.register
    def average_velocity(self):
        print("[INFO][METRICS][TASK] Average velocity")

        # Get velocity magnitude (norm of x, y, z components)
        # Shape: [num_envs, timesteps, 3] -> [num_envs, timesteps]
        velocity_magnitude = torch.norm(self.trajectories['linear_velocity'], dim=-1)
        
        # Create mask for valid timesteps (until trajectory completion)
        # Get the completion timesteps for each environment
        total_timesteps = torch.full((self.trajectories['trajectory_completed'].shape[0],), self.trajectories['trajectory_completed'].shape[-1], dtype=torch.float32, device=self.trajectories_masks.device)
        trajectory_completed_env, trajectory_completed_idx = torch.where(self.trajectories['trajectory_completed'] == True)
        total_timesteps[trajectory_completed_env] = trajectory_completed_idx.float()
        # Remove trajectories that didn't complete
        faild_trajectories_mask = self.trajectories['missed_gate'].sum(dim=-1) > 0
        total_timesteps[faild_trajectories_mask] = torch.nan
        
        # Create mask for timesteps before completion
        timestep_indices = torch.arange(velocity_magnitude.shape[1], device=velocity_magnitude.device).float()
        timestep_mask = timestep_indices.unsqueeze(0) < total_timesteps.unsqueeze(1)
        
        # Apply mask to velocities
        masked_velocities = velocity_magnitude * timestep_mask
        
        # Calculate average velocity (sum of valid velocities / number of valid timesteps)
        valid_timesteps = timestep_mask.sum(dim=1)
        # Avoid division by zero
        valid_timesteps = torch.clamp(valid_timesteps, min=1)
        
        average_velocity = masked_velocities.sum(dim=1) / valid_timesteps
        self.metrics["average_velocity.mps"] = average_velocity