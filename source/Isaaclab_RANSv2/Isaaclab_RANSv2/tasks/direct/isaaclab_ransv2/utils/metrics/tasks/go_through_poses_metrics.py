from . import BaseTaskMetrics, Registerable
import torch

class GoThroughPosesMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env=env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    # @BaseTaskMetrics.register
    # def success_rate_weighted_by_path_length(self):
    #     print("[INFO][METRICS][TASK] Success rate weighted by path length")
        
    #     # Check if required data is available
    #     if 'target_positions' not in self.trajectories:
    #         print("[WARNING] target_positions not found in trajectories, skipping success_rate_weighted_by_path_length")
    #         return
            
    #     if 'position' not in self.trajectories:
    #         print("[WARNING] position not found in trajectories, skipping success_rate_weighted_by_path_length")
    #         return
            
    #     if 'trajectory_completed' not in self.trajectories:
    #         print("[WARNING] trajectory_completed not found in trajectories, skipping success_rate_weighted_by_path_length")
    #         return
        
    #     try:
    #         shortest_path_lenght = self.trajectory_shortest_distance(self.trajectories['target_positions'])
    #         robot_distance_traveled = self.robot_distance_traveled(self.trajectories['position'][..., :2] * self.trajectories_masks.unsqueeze(-1))
    #         trajectory_completed = (self.trajectories['trajectory_completed'] * self.trajectories_masks).sum(dim=1)

    #         self.metrics["spl.u"] = trajectory_completed * (shortest_path_lenght / torch.max(robot_distance_traveled, shortest_path_lenght))
    #     except Exception as e:
    #         print(f"[WARNING] Error in success_rate_weighted_by_path_length: {e}")
    #         # Set a default value
    #         self.metrics["spl.u"] = torch.zeros(self.trajectories_masks.shape[0], device=self.trajectories_masks.device)

    @BaseTaskMetrics.register
    def orientation_error_following_path(self):
        print("[INFO][METRICS][TASK] Orientation error following path")
        
        # Check if required data is available
        if 'sin_target_heading_error' not in self.trajectories or 'cos_target_heading_error' not in self.trajectories:
            print("[WARNING] sin_target_heading_error or cos_target_heading_error not found in trajectories, skipping orientation_error_following_path")
            return
        
        try:
            masked_sin = self.trajectories['sin_target_heading_error'] * self.trajectories_masks
            masked_cos = self.trajectories['cos_target_heading_error'] * self.trajectories_masks

            avg_sin = torch.sum(masked_sin, dim=1) / torch.sum(self.trajectories_masks, dim=1)
            avg_cos = torch.sum(masked_cos, dim=1) / torch.sum(self.trajectories_masks, dim=1)

            angle_error = torch.arctan2(avg_sin, avg_cos)
            self.metrics["orientation_error_following_path.rad"] = angle_error
        except Exception as e:
            print(f"[WARNING] Error in orientation_error_following_path: {e}")
            # Set a default value
            self.metrics["orientation_error_following_path.rad"] = torch.zeros(self.trajectories_masks.shape[0], device=self.trajectories_masks.device)

    # @BaseTaskMetrics.register
    # def avg_time_to_reach_goal(self):
    #     print("[INFO][METRICS][TASK] Time to reach goal")


    # @BaseTaskMetrics.register
    # def num_goals_reached_in_fixed_steps(self):
    #     print("[INFO][METRICS][TASK] Number of goals reached in fixed steps")

    #     fix_num_steps = self.trajectories['target_index'].shape[1] // 2 #TODO: Hardcoded
    #     # fix_num_steps = torch.min(torch.sum(self.trajectories_masks.int(), dim=1))
    #     masked_target_index = (self.trajectories['target_index'] * self.trajectories_masks)[:, :fix_num_steps]
    #     max_target_index_idx = torch.argmax(masked_target_index, dim=1)
        
    #     env_idx = torch.arange(masked_target_index.shape[0], device=masked_target_index.device)
    #     max_target_reached = masked_target_index[env_idx, max_target_index_idx]

    #     self.metrics[f"num_goals_reached_in_{fix_num_steps}_steps.u"] = max_target_reached

    # @BaseTaskMetrics.register
    # def time_to_reach_goals(self):
    #     print("[INFO][METRICS][TASK] Time to reach goals")

    #     masked_target_index = self.trajectories['target_index'] * self.trajectories_masks
    #     masked_num_goals = self.trajectories['num_goals'] * self.trajectories_masks

    #     # Figure out number of goals per environment
    #     max_val = max(0, masked_num_goals.max().item())
    #     num_goals = max_val + 1  # From 0 to max_val

    #     # Prepare the result tensor: one row per environment, one col per goal.
    #     steps_per_goal = torch.full((masked_target_index.shape[0], num_goals), -1, 
    #                             dtype=torch.long, device=masked_target_index.device)

    #     # Identify where changes occur along the trajectory
    #     changes = masked_target_index[:, 1:] != masked_target_index[:, :-1]

    #     # Create tensors to track for each environment the last change step (starting at 0) and the number of goal changes recorded.
    #     prev_steps   = torch.zeros(masked_target_index.shape[0], dtype=torch.long, 
    #                                 device=masked_target_index.device)
    #     goal_counter = torch.zeros(masked_target_index.shape[0], dtype=torch.long, 
    #                                 device=masked_target_index.device)

    #     # Get the indices and the time where changes occur.
    #     env_idxs, traj_idxs = torch.where(changes)

    #     env_idxs_list = env_idxs.tolist()
    #     traj_idxs_list = traj_idxs.tolist()

    #     for env_idx, traj_idx in zip(env_idxs_list, traj_idxs_list):
    #         current_step = traj_idx + 1
    #         delta = current_step - prev_steps[env_idx].item()
            
    #         # Get the next available goal index for this environment.
    #         current_goal = goal_counter[env_idx].item()
    #         if current_goal < num_goals:
    #             # Record the delta for this change.
    #             steps_per_goal[env_idx, current_goal] = delta
    #             # Update the per–environment state.
    #             goal_counter[env_idx] += 1
    #             prev_steps[env_idx] = current_step
        
    #     valid_goals_mask = steps_per_goal > 0
    #     steps_per_goal_time = steps_per_goal.clone().to(torch.float32)
    #     steps_per_goal_time[valid_goals_mask] *= self.step_dt

    #     for i in range(num_goals):
    #         self.metrics[f"time_to_reach_goal_num_{i}.s"] = steps_per_goal_time[:, i]