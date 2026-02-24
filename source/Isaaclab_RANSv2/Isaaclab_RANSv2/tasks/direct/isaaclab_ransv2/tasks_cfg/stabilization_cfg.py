# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .go_to_position_cfg import GoToPositionCfg


@configclass
class StabilizationCfg(GoToPositionCfg):
    """Configuration for the Stabilization task.

    The robot spawns close to the goal with high initial angular velocity and
    small initial linear velocity. The objective is to stabilize (reduce angular
    velocity) while staying near the goal. Rewards prioritize angular velocity
    reduction over position, with no linear velocity reward.
    """

    # Initial conditions: spawn close to goal
    spawn_min_dist: float = 0.001 # 0.05
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 0.05 m."""
    spawn_max_dist: float = 0.01 # 0.5
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 0.5 m."""

    # Small initial linear velocity
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.3
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.3 m/s."""

    # High initial angular velocity
    spawn_min_ang_vel: float = 1.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.5 rad/s."""
    spawn_max_ang_vel: float = 3.0
    """Maximal angular velocity at spawn in rad/s. Defaults to 3.0 rad/s."""

    # Reward: no linear velocity term, dense exponential angular velocity reward
    linear_velocity_weight: float = 0.0
    """Linear velocity reward weight. Set to 0 for stabilization (task is to reduce angular vel)."""

    # Exponential angular velocity reward: exp(-|ang_vel| / coeff), higher coeff = softer decay
    angular_velocity_exponential_reward_coeff: float = 1 # 0.2
    """Coefficient for exponential angular velocity reward. Lower = steeper reward near zero."""
    angular_velocity_weight: float = 2.0
    """Weight for angular velocity (exponential) reward. Higher than position for priority."""

    # Reduce position priority relative to go_to_position
    position_weight: float = 0.5
    """Position reward weight. Lower than angular_velocity_weight for stabilization focus."""
