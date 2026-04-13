from . import BaseTaskMetrics, Registerable
import torch

class GoThroughPositionsMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env=env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    def populate_env_info(self):
        if "MultiTask" in self.env.unwrapped.__class__.__name__:
            self.env_info["max_num_goals"] = self.env.unwrapped.tasks_apis[self.task_index]._task_cfg.max_num_goals
        else:
            self.env_info["max_num_goals"] = self.env.unwrapped.task_api._task_cfg.max_num_goals
    
    @BaseTaskMetrics.register
    def avg_time_to_reach_goal(self):
        print("[INFO][METRICS][TASK] Time to reach goal")


    @BaseTaskMetrics.register
    def num_goals_reached_in_fixed_steps(self):
        print("[INFO][METRICS][TASK] Number of goals reached in fixed steps")

        fix_num_steps = self.trajectories['target_index'].shape[1] // 2 #TODO: Hardcoded
        # fix_num_steps = torch.min(torch.sum(self.trajectories_masks.int(), dim=1))
        masked_target_index = (self.trajectories['target_index'] * self.trajectories_masks)[:, :fix_num_steps]
        max_target_index_idx = torch.argmax(masked_target_index, dim=1)
        
        env_idx = torch.arange(masked_target_index.shape[0], device=masked_target_index.device)
        max_target_reached = masked_target_index[env_idx, max_target_index_idx]

        self.metrics[f"num_goals_reached_in_{fix_num_steps}_steps.u"] = max_target_reached

    @BaseTaskMetrics.register
    def time_to_reach_goals(self):
        print("[INFO][METRICS][TASK] Time to reach goals")

        masked_target_index = self.trajectories['target_index'] * self.trajectories_masks
        masked_num_goals = self.trajectories['num_goals'] * self.trajectories_masks

        # Figure out number of goals per environment
        max_val = max(0, masked_num_goals.max().item())
        num_goals = max_val + 1  # From 0 to max_val
        self.env_info["num_goals"] = num_goals

        # Prepare the result tensor: one row per environment, one col per goal.
        steps_per_goal = torch.full((masked_target_index.shape[0], num_goals), -1, 
                                dtype=torch.long, device=masked_target_index.device)

        # Identify where changes occur along the trajectory
        changes = masked_target_index[:, 1:] != masked_target_index[:, :-1]

        # Create tensors to track for each environment the last change step (starting at 0) and the number of goal changes recorded.
        prev_steps   = torch.zeros(masked_target_index.shape[0], dtype=torch.long, 
                                    device=masked_target_index.device)
        goal_counter = torch.zeros(masked_target_index.shape[0], dtype=torch.long, 
                                    device=masked_target_index.device)

        # Get the indices and the time where changes occur.
        env_idxs, traj_idxs = torch.where(changes)

        env_idxs_list = env_idxs.tolist()
        traj_idxs_list = traj_idxs.tolist()

        for env_idx, traj_idx in zip(env_idxs_list, traj_idxs_list):
            current_step = traj_idx + 1
            delta = current_step - prev_steps[env_idx].item()
            
            # Get the next available goal index for this environment.
            current_goal = goal_counter[env_idx].item()
            if current_goal < num_goals:
                # Record the delta for this change.
                steps_per_goal[env_idx, current_goal] = delta
                # Update the per–environment state.
                goal_counter[env_idx] += 1
                prev_steps[env_idx] = current_step
        
        valid_goals_mask = steps_per_goal > 0
        steps_per_goal_time = steps_per_goal.clone().to(torch.float32)
        steps_per_goal_time[valid_goals_mask] *= self.step_dt

        for i in range(num_goals):
            self.metrics[f"time_to_reach_goal_num_{i}.u"] = steps_per_goal_time[:, i]