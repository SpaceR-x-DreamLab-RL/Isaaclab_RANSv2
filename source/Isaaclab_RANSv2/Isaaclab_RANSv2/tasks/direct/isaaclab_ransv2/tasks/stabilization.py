# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.scene import InteractiveScene

from ..tasks_cfg import StabilizationCfg

from .go_to_position import GoToPositionTask


class StabilizationTask(GoToPositionTask):
    """
    Stabilization task: robot spawns close to the goal with high initial angular
    velocity and small initial linear velocity. The objective is to reduce
    angular velocity (stabilize) while staying near the goal. Rewards prioritize
    angular velocity reduction with a dense exponential reward; linear velocity
    reward is not used.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: StabilizationCfg = StabilizationCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            scene=scene,
            task_cfg=task_cfg,
            task_uid=task_uid,
            num_envs=num_envs,
            device=device,
            env_ids=env_ids,
        )

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the stabilization task.
        Uses exponential angular velocity reward (dense, high priority), position
        and heading rewards, and boundary penalty. Linear velocity reward is
        omitted (task is to remove angular velocity, not linear).
        """
        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed angular velocity (for stabilization we care about reducing this)
        angular_velocity = torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])
        # Compute the heading to the target
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )

        # Update logs
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", angular_velocity)
        self.scalar_logger.log("task_state", "AVG/target_heading_error", target_heading_error)

        # position reward
        position_rew = torch.exp(
            -self._position_dist / self._task_cfg.position_exponential_reward_coeff
        )
        # heading reward + distance scaling
        dist_scaling = (
            torch.clamp(
                self._position_dist,
                self._task_cfg.min_heading_dist_scaler,
                self._task_cfg.max_heading_dist_scaler,
            )
            - self._task_cfg.min_heading_dist_scaler
        ) / (self._task_cfg.max_heading_dist_scaler - self._task_cfg.min_heading_dist_scaler)
        heading_rew = (
            torch.exp(
                -torch.abs(target_heading_error)
                / self._task_cfg.heading_exponential_reward_coeff
            )
            * dist_scaling
        )
        # Dense exponential angular velocity reward: high when ang_vel is low
        angular_velocity_rew = torch.exp(
            -angular_velocity
            / self._task_cfg.angular_velocity_exponential_reward_coeff
        )
        # boundary rew
        boundary_rew = torch.exp(
            -boundary_dist / self._task_cfg.boundary_exponential_reward_coeff
        )

        # Goal reached check
        goal_is_reached = (self._position_dist < self._task_cfg.position_tolerance).int()
        self._goal_reached *= goal_is_reached
        self._goal_reached += goal_is_reached

        # Update logs for rewards (no linear_velocity for stabilization)
        self.scalar_logger.log("task_reward", "AVG/position", position_rew * self._task_cfg.position_weight)
        self.scalar_logger.log("task_reward", "AVG/heading", heading_rew * self._task_cfg.heading_weight)
        self.scalar_logger.log(
            "task_reward",
            "AVG/angular_velocity",
            angular_velocity_rew * self._task_cfg.angular_velocity_weight,
        )
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew * self._task_cfg.boundary_weight)

        # Combined reward: no linear velocity term; angular velocity has higher priority via config weights
        return (
            position_rew * self._task_cfg.position_weight
            + heading_rew * self._task_cfg.heading_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
        ) + self._robot.compute_rewards()
