from . import BaseTaskMetrics, Registerable
import torch

class TrackVelocitiesMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def track_velocity_error(self):
        print("[INFO][METRICS][TASK] Track velocity error")
        masked_lin_vel_error = self.trajectories['error_linear_velocity'] * self.trajectories_masks
        masked_lat_vel_error = self.trajectories['error_lateral_velocity'] * self.trajectories_masks
        masled_ang_vel_error = self.trajectories['error_angular_velocity'] * self.trajectories_masks

        len_trajec = torch.sum(self.trajectories_masks, dim=1)
        avg_lin_vel_error = torch.sum(torch.abs(masked_lin_vel_error), dim=1) / len_trajec
        avg_lat_vel_error = torch.sum(torch.abs(masked_lat_vel_error), dim=1) / len_trajec
        avg_ang_vel_error = torch.sum(torch.abs(masled_ang_vel_error), dim=1) / len_trajec

        self.metrics["linear_velocity_error.m/s"] = avg_lin_vel_error
        self.metrics["lateral_velocity_error.m/s"] = avg_lat_vel_error
        self.metrics["angular_velocity_error.m/s"] = avg_ang_vel_error

    # @BaseTaskMetrics.register
    # def overshoot(self):
    #     print("[INFO][METRICS][TASK] Overshoot")
    #     num_env, num_steps = self.trajectories['error_linear_velocity'].shape
    #     device = self.trajectories['error_linear_velocity'].device
    #     masked_lin_vel_error = self.trajectories['error_linear_velocity'] * self.trajectories_masks
    #     masked_lat_vel_error = self.trajectories['error_lateral_velocity'] * self.trajectories_masks
    #     masled_ang_vel_error = self.trajectories['error_angular_velocity'] * self.trajectories_masks

    #     env_idx, trajectory_idx = torch.where(self.trajectories['goal_reached'] == 1)
    #     unique_envs, unique_indices = torch.unique(env_idx, return_inverse=True)
    #     unique_indices_goal_reach_idx = torch.unique(unique_indices)

    #     overshoot_lin_vel = torch.full((num_env,), -1, dtype=torch.float32, device=device)
    #     overshoot_lat_vel = torch.full((num_env,), -1, dtype=torch.float32, device=device)
    #     overshoot_ang_vel = torch.full((num_env,), -1, dtype=torch.float32, device=device)    

    #     for i in range(len(unique_indices_goal_reach_idx)):
    #         vel_matched_indx = torch.where(unique_indices == unique_indices_goal_reach_idx[i])[0][0]

    #         current_env_id = unique_envs[unique_indices_goal_reach_idx[i]]
    #         first_reach_step = trajectory_idx[vel_matched_indx]

    #         # Lin vel
    #         overshoot_linv = torch.max(masked_lin_vel_error[current_env_id, first_reach_step:]).item()
    #         overshoot_lin_vel[current_env_id] = overshoot_linv
    #         # Lat vel
    #         overshoot_latv = torch.max(masked_lat_vel_error[current_env_id, first_reach_step:]).item()
    #         overshoot_lat_vel[current_env_id] = overshoot_latv
    #         # Ang vel
    #         overshoot_angv = torch.max(masled_ang_vel_error[current_env_id, first_reach_step:]).item()
    #         overshoot_ang_vel[current_env_id] = overshoot_angv   
        
    #     self.metrics["overshoot_linear_velocity.m/s"] = overshoot_lin_vel
    #     self.metrics["overshoot_lateral_velocity.m/s"] = overshoot_lat_vel
    #     self.metrics["overshoot_angular_velocity.rad/s"] = overshoot_ang_vel