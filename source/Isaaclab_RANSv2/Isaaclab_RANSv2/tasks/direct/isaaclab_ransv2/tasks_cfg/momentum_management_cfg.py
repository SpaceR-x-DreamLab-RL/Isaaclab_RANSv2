# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .task_core_cfg import TaskCoreCfg


@configclass
class MomentumManagementCfg(TaskCoreCfg):
    """Configuration for the Momentum Management task.
    
    This task is designed to demonstrate:
    1. Stabilization from high angular velocity.
    2. Use of reaction wheel for primary stabilization.
    3. Desaturation of reaction wheel using thrusters.
    4. Use of arms to modulate moment of inertia.
    """

    # Initial conditions
    spawn_min_lin_vel: float = 0.0
    spawn_max_lin_vel: float = 0.0
    spawn_min_ang_vel: float = 2.0
    """Minimal angular velocity at spawn in rad/s (High spin)."""
    spawn_max_ang_vel: float = 5.0
    """Maximal angular velocity at spawn in rad/s (High spin)."""

    # Reward Weights
    
    # 1. Main Goal: Stop spinning
    angular_velocity_weight: float = 2.0
    ang_vel_exponential_coeff: float = 1.0

    # 2. Desaturation: Penalize high reaction wheel velocity
    reaction_wheel_saturation_weight: float = -0.05
    
    # 3. Efficiency: Penalize Thruster usage (Encourage RW use first)
    thruster_cost_weight: float = -0.5
    
    # 4. Mechanism: Bonus for arm extension when spinning (Braking)
    arm_extension_bonus_weight: float = 0.5

    # Spaces (Boilerplate)
    observation_space: int = 6
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 5
