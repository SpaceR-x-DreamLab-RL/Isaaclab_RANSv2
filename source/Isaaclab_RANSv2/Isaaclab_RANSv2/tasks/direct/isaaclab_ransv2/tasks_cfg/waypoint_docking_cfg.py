# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from .go_through_poses_cfg import GoThroughPosesCfg


@configclass
class WaypointDockingCfg(GoThroughPosesCfg):
    """Configuration for the WaypointDocking task.

    A GoThroughPoses variant tuned for slow, precise navigation between closely-spaced
    waypoints (sensor-based docking curriculum). Key differences vs GoThroughPoses:
      - Waypoints are much closer together (0.3–1.5 m vs 1.0–5.0 m).
      - Strong linear and angular velocity penalties enforce slow motion.
      - Robot always spawns at rest, close to the first goal.
      - No per-step time penalty (training-friendly).
      - Arm actions should be zeroed at the robot level (use PinguNoArmsRobot).
    """

    # ── Spawn conditions ──────────────────────────────────────────────────────
    # Spawn close to the first goal, aligned with it, at rest.
    spawn_min_dist: float = 0.1
    """Minimum spawn distance from the first goal [m]."""
    spawn_max_dist: float = 1.0
    """Maximum spawn distance from the first goal [m]."""
    spawn_min_cone_spread: float = 0.0
    """Minimum half-angle of the spawn cone behind the first goal [rad]."""
    spawn_max_cone_spread: float = math.pi / 4
    """Maximum half-angle of the spawn cone behind the first goal [rad]. ±45°."""
    spawn_min_heading_dist: float = 0.0
    """Minimum heading offset at spawn [rad]."""
    spawn_max_heading_dist: float = math.pi / 4
    """Maximum heading offset at spawn [rad]. ±45°."""
    spawn_min_lin_vel: float = 0.0
    """Minimum linear speed at spawn [m/s]. Always start at rest."""
    spawn_max_lin_vel: float = 0.0
    """Maximum linear speed at spawn [m/s]. Always start at rest."""
    spawn_min_ang_vel: float = 0.0
    """Minimum angular speed at spawn [rad/s]. Always start at rest."""
    spawn_max_ang_vel: float = 0.0
    """Maximum angular speed at spawn [rad/s]. Always start at rest."""

    # ── Goal / waypoint geometry ──────────────────────────────────────────────
    # Short inter-waypoint distances for docking-regime training.
    goal_max_dist_from_origin: float = 0.3
    """First goal is placed within this radius of the environment origin [m]."""
    goal_min_dist: float = 0.3
    """Minimum distance between consecutive goals [m]."""
    goal_max_dist: float = 1.5
    """Maximum distance between consecutive goals [m]."""
    goal_min_heading_dist: float = 0.0
    """Minimum heading change between consecutive goals [rad]."""
    goal_max_heading_dist: float = math.pi / 2
    """Maximum heading change between consecutive goals [rad]."""
    goal_min_cone_spread: float = 0.0
    """Minimum cone spread for next-goal sampling [rad]."""
    goal_max_cone_spread: float = math.pi / 4
    """Maximum cone spread for next-goal sampling [rad]. ±45°."""

    # Fewer goals keeps the early curriculum simple.
    min_num_goals: int = 3
    """Minimum number of waypoints per episode."""
    max_num_goals: int = 6
    """Maximum number of waypoints per episode."""

    # ── Tolerances ────────────────────────────────────────────────────────────
    # Tighter thresholds to reward genuinely precise arrivals.
    position_tolerance: float = 0.05
    """Goal is considered reached when the robot is within this distance [m]."""
    heading_tolerance: float = math.pi * 10.0 / 180.0
    """Goal is considered reached when heading error is within this bound [rad]. 10°."""

    # ── Reward shaping ────────────────────────────────────────────────────────
    # Heavy velocity penalties push the policy toward slow, controlled motion.
    linear_velocity_weight: float = -0.5
    """Weight on the linear velocity penalty term."""
    linear_velocity_min_value: float = 0.0
    """Velocity below this threshold incurs zero linear velocity penalty [m/s]."""
    linear_velocity_max_value: float = 0.2
    """Penalty saturates at this linear velocity [m/s]. Anything faster = max cost."""

    angular_velocity_weight: float = -0.1
    """Weight on the angular velocity penalty term."""
    angular_velocity_min_value: float = 0.0
    """Velocity below this threshold incurs zero angular velocity penalty [rad/s]."""
    angular_velocity_max_value: float = 0.5
    """Penalty saturates at this angular velocity [rad/s]."""

    # No per-step time penalty — makes the early curriculum easier to solve.
    time_penalty: float = 0.0

    # Keep the strong progress reward and goal bonus for a dense training signal.
    progress_weight: float = 1.0
    reached_bonus: float = 10.0
