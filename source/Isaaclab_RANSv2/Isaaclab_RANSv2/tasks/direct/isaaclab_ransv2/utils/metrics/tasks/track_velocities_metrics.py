from . import BaseTaskMetrics, Registerable
import torch

class TrackVelocitiesMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def track_velocity_error(self):
        print("[INFO][METRICS][TASK] Track velocity error")
        masked_ang_vel_error = self.trajectories['error_angular_velocity'] * self.trajectories_masks

        len_trajec = torch.sum(self.trajectories_masks, dim=1)
        avg_ang_vel_error = torch.sum(torch.abs(masked_ang_vel_error), dim=1) / len_trajec

        self.metrics["angular_velocity_error.rad/s"] = avg_ang_vel_error

    def _get_success_threshold(self) -> float:
        """Read success_threshold_ang_vel from the task cfg (single- or multi-task env)."""
        if "MultiTask" in self.env.unwrapped.__class__.__name__:
            return self.env.unwrapped.tasks_apis[self.task_index]._task_cfg.success_threshold_ang_vel
        return self.env.unwrapped.task_api._task_cfg.success_threshold_ang_vel

    def _angular_velocity(self) -> torch.Tensor:
        """Robot chassis angular velocity around z (rad/s), body frame.

        Sourced from the robot's eval_data 'angular_velocity_w' (= root_com_ang_vel_b),
        which is the root rigid body (chassis) — explicitly NOT the reaction wheel
        (reaction wheel speed is exposed separately as 'omega_reaction_wheel').
        """
        return self.trajectories['angular_velocity_w'][..., 2]

    @BaseTaskMetrics.register
    def time_to_half_initial_angular_velocity(self):
        """First step where |omega| <= |omega_init|/2.

        omega_init is taken as the angular velocity at trajectory step 0 (i.e. just
        after spawn). If never reached within the episode, falls back to the episode
        length (so the metric still reflects 'didn't converge in time').
        """
        print("[INFO][METRICS][TASK] Time to half initial angular velocity")
        ang_vel_abs = torch.abs(self._angular_velocity())
        init_abs = ang_vel_abs[:, 0]                    # (num_envs,)
        half_init = (init_abs * 0.5).unsqueeze(1)       # (num_envs, 1)

        # Only count steps that are inside the episode mask.
        below = (ang_vel_abs <= half_init) & self.trajectories_masks

        len_trajec = self.trajectories_masks.sum(dim=1)
        reached_idx = torch.argmax(below.int(), dim=1)
        # argmax returns 0 when nothing is True — treat that as "never reached"
        none_reached = ~torch.any(below, dim=1)
        reached_idx[none_reached] = len_trajec[none_reached]

        self.metrics["time_to_half_initial_angular_velocity.s"] = reached_idx.float() * self.step_dt
        self.metrics["steps_to_half_initial_angular_velocity.u"] = reached_idx.float()

    @BaseTaskMetrics.register
    def time_to_angular_velocity_threshold(self):
        """First step where |omega| <= success_threshold_ang_vel.

        If never reached, falls back to the episode length (== 'episode end').
        """
        print("[INFO][METRICS][TASK] Time to angular velocity threshold")
        threshold = self._get_success_threshold()
        ang_vel_abs = torch.abs(self._angular_velocity())
        below = (ang_vel_abs <= threshold) & self.trajectories_masks

        len_trajec = self.trajectories_masks.sum(dim=1)
        reached_idx = torch.argmax(below.int(), dim=1)
        none_reached = ~torch.any(below, dim=1)
        reached_idx[none_reached] = len_trajec[none_reached]

        self.metrics["time_to_angular_velocity_threshold.s"] = reached_idx.float() * self.step_dt
        self.metrics["steps_to_angular_velocity_threshold.u"] = reached_idx.float()

    @BaseTaskMetrics.register
    def angular_velocity_first10s_stats(self):
        """Mean, SD of |ω| over the first 10 s, and time to reach half that mean.

        The 10-s window is masked so shorter episodes only use valid steps.
        The half-mean threshold is searched over the full episode.
        """
        print("[INFO][METRICS][TASK] Angular velocity first-10s mean, SD, time-to-half-mean")
        n_steps = min(int(10.0 / self.step_dt), self.trajectories_masks.shape[1])
        mask_10s = self.trajectories_masks[:, :n_steps]           # (rows, n_steps)
        ang_vel_abs = torch.abs(self._angular_velocity())         # (rows, total_steps)
        ang_vel_abs_10s = ang_vel_abs[:, :n_steps]

        count = mask_10s.sum(dim=1).clamp(min=1)
        mean_10s = (ang_vel_abs_10s * mask_10s).sum(dim=1) / count
        mean_sq  = ((ang_vel_abs_10s ** 2) * mask_10s).sum(dim=1) / count
        std_10s  = (mean_sq - mean_10s ** 2).clamp(min=0).sqrt()

        self.metrics["angular_velocity_mean_first10s.rad/s"] = mean_10s
        self.metrics["angular_velocity_std_first10s.rad/s"]  = std_10s

        # Time to reach half of the first-10s mean
        half_mean = (mean_10s * 0.5).unsqueeze(1)
        below = (ang_vel_abs <= half_mean) & self.trajectories_masks
        len_trajec = self.trajectories_masks.sum(dim=1)
        reached_idx = torch.argmax(below.int(), dim=1)
        reached_idx[~torch.any(below, dim=1)] = len_trajec[~torch.any(below, dim=1)]

        self.metrics["time_to_half_mean10s_angular_velocity.s"] = reached_idx.float() * self.step_dt
        self.metrics["steps_to_half_mean10s_angular_velocity.u"] = reached_idx.float()
        
        print(f"step_dt={self.step_dt:.4f}  n_steps={n_steps}  traj_len={self.trajectories_masks.shape[1]}")
        print(f"mean_10s[0]={mean_10s[0]:.4f}  init[0]={ang_vel_abs[0,0]:.4f}")


    @BaseTaskMetrics.register
    def angular_velocity_success(self):
        """Per-trajectory success: |omega| at the final episode step is below threshold.

        Stored as 0/1 per trajectory; the mean across trajectories in post-processing
        gives the overall success rate.
        """
        print("[INFO][METRICS][TASK] Angular velocity success (final step under threshold)")
        threshold = self._get_success_threshold()
        ang_vel = self._angular_velocity() * self.trajectories_masks
        idx = torch.arange(ang_vel.shape[0], device=ang_vel.device)
        final_ang_vel = ang_vel[idx, self.last_true_index]
        success = (torch.abs(final_ang_vel) <= threshold).float()
        self.metrics["angular_velocity_success.u"] = success

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