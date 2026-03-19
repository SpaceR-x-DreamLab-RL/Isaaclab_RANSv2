# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math

from isaaclab.scene import InteractiveScene

from ..tasks_cfg import MomentumManagementCfg
from .task_core import TaskCore


class MomentumManagementTask(TaskCore):
    """
    Implements the Momentum Management task.
    Demonstrates stabilization using reaction wheels and thrusters.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: MomentumManagementCfg = MomentumManagementCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)
        self._task_cfg = task_cfg
        self.initialize_buffers()

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        super().initialize_buffers(env_ids)

    def get_observations(self) -> torch.Tensor:
        # Task observation 
        
        # 0. Angular Velocity (Z-axis)
        ang_vel = self._robot.root_com_ang_vel_w[:, 2]
        
        # 1. Reaction Wheel Velocity (if exists)
        if self._robot._robot_cfg.has_reaction_wheel:
            rw_vel = self._robot.data.joint_vel[:, self._robot._reaction_wheel_dof_idx].squeeze()
        else:
            rw_vel = torch.zeros_like(ang_vel)
            
        # 2. Arm Position (Sum of joint positions)
        arm_pos = torch.sum(torch.abs(self._robot.data.joint_pos[:, self._robot._arms_ids]), dim=1)

        # Observations buffer (size 6)
        obs = torch.zeros((self._num_envs, 6), device=self._device)
        obs[:, 0] = ang_vel
        obs[:, 1] = rw_vel
        obs[:, 2] = arm_pos
        
        # Concatenate with robot observations
        return torch.cat((obs, self._robot.get_observations()), dim=-1)

    def compute_rewards(self) -> torch.Tensor:
        # 1. Base Angular Velocity (Minimize) -> Goal: 0
        current_ang_vel = torch.abs(self._robot.root_com_ang_vel_w[:, 2])
        # Exponential reward for being close to 0
        rew_ang_vel = torch.exp(-current_ang_vel / self._task_cfg.ang_vel_exponential_coeff)

        # 2. Thruster Usage (Penalty) -> Encourage using RW
        # Assuming thruster action is stored in _thrust_action[:, :, 2] (magnitude of directed thrust)
        thrust_mag = torch.sum(self._robot._thrust_action[:, :, 2], dim=1)
        rew_thrust = thrust_mag * self._task_cfg.thruster_cost_weight

        # 3. Reaction Wheel Velocity (Penalty) -> Encourage Desaturation
        if self._robot._robot_cfg.has_reaction_wheel:
            rw_vel = torch.abs(self._robot.data.joint_vel[:, self._robot._reaction_wheel_dof_idx]).squeeze()
            rew_rw = rw_vel * self._task_cfg.reaction_wheel_saturation_weight
        else:
            rew_rw = 0.0

        # 4. Arm Extension Bonus (When spinning fast) -> Passive Braking
        # Reward: Open Arms when Spinning => (arm_pos) * (ang_vel)
        # Sum of joint positions as metric for "open"
        arm_pos = torch.sum(torch.abs(self._robot.data.joint_pos[:, self._robot._arms_ids]), dim=1)
        rew_arms = arm_pos * current_ang_vel * self._task_cfg.arm_extension_bonus_weight

        # Total
        return (
            rew_ang_vel * self._task_cfg.angular_velocity_weight
            + rew_thrust
            + rew_rw
            + rew_arms
        ) + self._robot.compute_rewards()

    def reset(self, env_ids: torch.Tensor, gen_actions: torch.Tensor | None = None, env_seeds: torch.Tensor | None = None) -> None:
        super().reset(env_ids, gen_actions, env_seeds)
        
        # Set Initial High Angular Velocity
        num_resets = len(env_ids)
        
        # Random ang vel between min and max
        ang_vel = (
            self._rng.sample_uniform_torch(
                self._task_cfg.spawn_min_ang_vel, 
                self._task_cfg.spawn_max_ang_vel, 
                1, 
                ids=env_ids
            ).squeeze() 
            * self._rng.sample_sign_torch("float", 1, ids=env_ids).squeeze()
        )

        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)
        initial_velocity[:, 5] = ang_vel
        
        self._robot.set_velocity(initial_velocity, env_ids)
