# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from gymnasium import spaces, vector
import math

from isaaclab.assets import Articulation
from isaaclab.markers import ARROW_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from ..robots_cfg import PinguRobotCfg

from .robot_core import RobotCore

import numpy as np
import warp as wp

from ..utils import compute_thruster_mapping


class PinguRobot(RobotCore):

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        robot_cfg: PinguRobotCfg = PinguRobotCfg(),
        robot_uid: int = 0,
        num_envs: int = 1,
        decimation: int = 6,
        device: str = "cuda",
    ):
        super().__init__(scene=scene, robot_uid=robot_uid, num_envs=num_envs, decimation=decimation, device=device)
        self._robot_cfg = robot_cfg
        # Available for use robot_cfg.is_reaction_wheel,robot_cfg.split_thrust,robot_cfg.rew_reaction_wheel_scale
        self._dim_robot_obs = self._robot_cfg.observation_space
        self._dim_robot_act = self._robot_cfg.action_space
        self._dim_gen_act = self._robot_cfg.gen_space
        
        # Physical parameters for reaction wheel dynamics
        if self._robot_cfg.has_reaction_wheel:
            self.b = self._robot_cfg.b
            self.J_rw = self._robot_cfg.J_rw

        # Buffers
        self.initialize_buffers()
        
    @property
    def eval_data_keys(self) -> list[str]:
        return [
            "position",
            "heading",
            "linear_velocity",
            "angular_velocity",
            "linear_velocity_w",
            "angular_velocity_w",
            "reaction_wheel_action",
            "omega_reaction_wheel",
            "thrust_action",
            "actions",
            "unaltered_actions",
            "left_arm_position",
            "right_arm_position",
            "left_arm_velocity",
            "right_arm_velocity",
        ]

    @property
    def eval_data_specs(self)->dict[str, list[str]]:
        num_thrusters = self._robot_cfg.num_thrusters
        return {
            "position": [".robot_pos.x.m", ".robot_pos.y.m", ".robot_pos.z.m"],
            "heading": [".robot_heading.rad"],
            "linear_velocity": [".robot_lin_vel.x.m/s", ".robot_lin_vel.y.m/s", ".robot_lin_vel.z.m/s"],
            "angular_velocity": [".robot_ang_vel.x.rad/s", ".robot_ang_vel.y.rad/s", ".robot_ang_vel.z.rad/s"],
            "linear_velocity_w": [".robot_lin_vel_w.x.m/s", ".robot_lin_vel_w.y.m/s", ".robot_lin_vel_w.z.m/s"],
            "angular_velocity_w": [".robot_ang_vel_w.x.rad/s", ".robot_ang_vel_w.y.rad/s", ".robot_ang_vel_w.z.rad/s"],
            "reaction_wheel_action": [".reaction_wheel_action.u"],
            "omega_reaction_wheel": [".omega_reaction_wheel.rad/s"],
            "thrust_action": [f".thruster{i}.force.N" for i in range(num_thrusters)],
            "actions": [f".robot_actions{i}.u" for i in range(self._robot_cfg.action_space)],
            "unaltered_actions": [f".robot_unaltered_actions{i}.u" for i in range(self._robot_cfg.action_space)],
            "left_arm_position": [".left_shoulder.pos.rad", ".left_elbow.pos.rad"],
            "right_arm_position": [".right_shoulder.pos.rad", ".right_elbow.pos.rad"],
            "left_arm_velocity": [".left_shoulder.vel.rad/s", ".left_elbow.vel.rad/s"],
            "right_arm_velocity": [".right_shoulder.vel.rad/s", ".right_elbow.vel.rad/s"],
        }

    @property
    def eval_data(self) -> dict:
        return {
            "position": self.root_pos_w,
            "heading": self.heading_w,
            # Use CoM-frame quantities so they match what the TrackVelocities task
            # tracks (root_com_lin_vel_b[:, 0/1] for lin/lat and root_com_ang_vel_w[:, 2]
            # for yaw). Using root_*_b here would differ by omega × r_com_from_actor.
            "linear_velocity": self.root_com_lin_vel_b,
            "angular_velocity": self.root_com_ang_vel_b,
            "linear_velocity_w": self.root_com_lin_vel_w,
            "angular_velocity_w": self.root_com_ang_vel_w,
            "reaction_wheel_action": self._reaction_wheel_action,
            "omega_reaction_wheel": self.omega_reaction_wheel,
            "thrust_action": self._thrust_action[..., -1],
            "actions": self._actions,
            "unaltered_actions": self._unaltered_actions,
            "left_arm_position": self._robot.data.joint_pos[:, self._left_levionarm_dof_idx],
            "right_arm_position": self._robot.data.joint_pos[:, self._right_levionarm_dof_idx],
            "left_arm_velocity": self._robot.data.joint_vel[:, self._left_levionarm_dof_idx],
            "right_arm_velocity": self._robot.data.joint_vel[:, self._right_levionarm_dof_idx],
        }

    def initialize_buffers(self, env_ids=None):
        super().initialize_buffers(env_ids)
        self._actions = torch.zeros((self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32)
        self._previous_actions = torch.zeros(
            (self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32
        )
        self._thrust_action = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3), device=self._device, dtype=torch.float32
        )
        if self._robot_cfg.has_reaction_wheel:
            self._reaction_wheel_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32) #torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32) #
            self._reaction_wheel_to_body_torque_action = torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32)
            self.omega_reaction_wheel = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

        self.arm_position_targets = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self._previous_arm_position_targets = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

        # Initialize swing state buffers
        # 0: Hold Right, 1: Swing to Left, 2: Hold Left, 3: Swing to Right
        self._swing_state = torch.zeros(self._num_envs, device=self._device, dtype=torch.int32)
        self._swing_timer = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._swing_start_angle = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._swing_target_angle = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._swing_duration = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        
        # Initialize random timer
        self._swing_timer[:] = torch.rand(self._num_envs, device=self._device) * 2.0  # Initial random wait

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._thrusters_dof_idx, _ = self._robot.find_bodies(self._robot_cfg.thrusters_dof_name)
        self._root_idx, _ = self._robot.find_bodies([self._robot_cfg.root_id_name])

        if self._robot_cfg.has_reaction_wheel:
            self._reaction_wheel_dof_idx, _ = self._robot.find_joints(self._robot_cfg.reaction_wheel_dof_name)

        self._locking_joint_dof_idx, _ = self._robot.find_joints(
            self._robot_cfg.locking_joint_dof_name
        )

        self._left_levionarm_dof_idx, _ = self._robot.find_joints(self._robot_cfg.left_levionarm_dof_name)
        self._right_levionarm_dof_idx, _ = self._robot.find_joints(self._robot_cfg.right_levionarm_dof_name)
        
        self._arms_ids = self._left_levionarm_dof_idx + self._right_levionarm_dof_idx
        
        self._shoulder_lower_limit = self._robot.data.soft_joint_pos_limits[:, self._arms_ids][:, 0, 0]
        self._shoulder_upper_limit = self._robot.data.soft_joint_pos_limits[:, self._arms_ids][:, 0, 1]
        
        self._right_elbow_lower_limit = self._robot.data.soft_joint_pos_limits[:, self._arms_ids][:, 3, 0]
        self._right_elbow_upper_limit = self._robot.data.soft_joint_pos_limits[:, self._arms_ids][:, 3, 1]
        
        self._left_elbow_lower_limit = self._robot.data.soft_joint_pos_limits[:, self._arms_ids][:, 1, 0]
        self._left_elbow_upper_limit = self._robot.data.soft_joint_pos_limits[:, self._arms_ids][:, 1, 1]

    def create_logs(self):
        super().create_logs()

        self.scalar_logger.add_log("robot_state", "AVG/thruster_norm", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_body_torque", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_omega", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/thruster_action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/thruster_effort", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_saturation", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_usage", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/arm_action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/arm_symmetry", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/thruster_action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/thruster_effort", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/rw_saturation", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/rw_usage", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/arm_action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/arm_symmetry", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/arm_contact_force", "mean")
        self.scalar_logger.add_log("robot_state", "SUM/arm_collision", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/arm_collision", "mean")

    def get_observations(self) -> torch.Tensor:
        return self._previous_actions

    def compute_rewards(self):
        #TODO: Should dt be fatored in?
        
        """
        Reward term reference for Pingu
        --------------------------------
        All terms are penalties (negative scales) applied on top of the task-level reward.
        
        THRUSTERS
          joint_acceleration    (rew_joint_accel_scale=-2.5e-6)
              Penalizes the sum of squared joint accelerations across all joints.
              Discourages high-frequency, jerky motion anywhere in the articulation.
        
          thruster_action_rate  (rew_action_rate_scale=-0.12/8, direct mode only)
              Penalizes the L1 change in thruster commands between consecutive steps.
              Sliced to thruster dims only so arms and RW do not inflate this term.
              Encourages smooth, gradual thrust transitions rather than bang-bang control.
        
          thruster_effort       (rew_thruster_effort_scale=-0.05)
              Penalizes the total normalized thrust summed over all 8 thrusters.
              Captures sustained activation cost that action_rate misses (a robot
              holding constant thrust pays zero action_rate but non-zero effort).
        
        REACTION WHEEL
          rw_saturation         (rew_reaction_wheel_saturation_scale=-2.5e-6)
              Penalizes omega_rw^2 (internal reaction wheel angular speed, squared).
              A saturated wheel cannot produce torque; this keeps speed well below the
              physical limit so heading authority is always available.
        
          rw_usage              (rew_reaction_wheel_usage_scale=-0.05)
              Penalizes |commanded_torque| — discourages spinning the wheel unnecessarily.
              Together with rw_saturation: the policy learns to use the wheel only when
              there is a heading error to correct and to desaturate it afterwards.
        
        ARMS
          arm_action_rate       (rew_arm_action_rate_scale=-0.1)
              Penalizes the L1 change in arm position targets (all 4 joints) per step.
              Rapid arm movements cause body vibrations on a floating platform; this
              encourages smooth, deliberate arm trajectories.
        
          arm_symmetry          (rew_arm_symmetry_scale=-0.05)
              Penalizes (left_shoulder - right_shoulder)^2 + (left_elbow - right_elbow)^2.
              Persistent left/right asymmetry creates a constant angular momentum bias
              that the thrusters or reaction wheel must continuously compensate for.
              This nudges the policy toward symmetric resting poses.
              
        """
        reward = torch.zeros(self._num_envs, device=self._device)

        # joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)
        # self.scalar_logger.log("robot_state", "AVG/joint_acceleration", joint_accelerations)
        # self.scalar_logger.log("robot_reward", "AVG/joint_acceleration", joint_accelerations * self._robot_cfg.rew_joint_accel_scale)

        # reward = joint_accelerations * self._robot_cfg.rew_joint_accel_scale
        # if self._robot_cfg.direct_thruster_control:
        #     # Slice only thruster dims so arm/RW changes don't pollute this term
        #     thruster_action_rate = torch.sum(
        #         torch.abs(
        #             self._unaltered_actions[:, : self._robot_cfg.num_thrusters]
        #             - self._previous_unaltered_actions[:, : self._robot_cfg.num_thrusters]
        #         ),
        #         dim=1,
        #     )
        #     self.scalar_logger.log("robot_state", "AVG/thruster_action_rate", thruster_action_rate)
        #     self.scalar_logger.log("robot_reward", "AVG/thruster_action_rate", thruster_action_rate * self._robot_cfg.rew_action_rate_scale)
        #     reward = reward + thruster_action_rate * self._robot_cfg.rew_action_rate_scale

        # --- Thruster effort: penalize sustained total thrust (fuel cost) ---
        # thruster_effort = torch.sum(torch.abs(self._thrust_action[:, :, 2]), dim=-1) / self._robot_cfg.max_thrust
        # self.scalar_logger.log("robot_state", "AVG/thruster_effort", thruster_effort)
        # self.scalar_logger.log("robot_reward", "AVG/thruster_effort", thruster_effort * self._robot_cfg.rew_thruster_effort_scale)
        # reward = reward + thruster_effort * self._robot_cfg.rew_thruster_effort_scale

        if self._robot_cfg.has_reaction_wheel:
            # --- RW saturation: penalize high internal wheel speed (quadratic) ---
            rw_saturation = torch.square(self.omega_reaction_wheel).squeeze(-1)
            self.scalar_logger.log("robot_state", "AVG/rw_saturation", rw_saturation)
            self.scalar_logger.log("robot_reward", "AVG/rw_saturation", rw_saturation * self._robot_cfg.rew_reaction_wheel_saturation_scale)
            reward = reward + rw_saturation * self._robot_cfg.rew_reaction_wheel_saturation_scale

            # --- RW usage: penalize commanded torque magnitude ---
            rw_usage = torch.abs(self._reaction_wheel_action).squeeze(-1)
            self.scalar_logger.log("robot_state", "AVG/rw_usage", rw_usage)
            self.scalar_logger.log("robot_reward", "AVG/rw_usage", rw_usage * self._robot_cfg.rew_reaction_wheel_usage_scale)
            reward = reward + rw_usage * self._robot_cfg.rew_reaction_wheel_usage_scale

        # --- Arm action rate: penalize rapid arm target changes (vibration/jerk) ---
        # arm_action_rate = torch.sum(torch.abs(self.arm_position_targets - self._previous_arm_position_targets), dim=-1)
        # self.scalar_logger.log("robot_state", "AVG/arm_action_rate", arm_action_rate)
        # self.scalar_logger.log("robot_reward", "AVG/arm_action_rate", arm_action_rate * self._robot_cfg.rew_arm_action_rate_scale)
        # reward = reward + arm_action_rate * self._robot_cfg.rew_arm_action_rate_scale

        # --- Arm collision: penalize any contact on the arm links ---
        # net_forces_w has shape (num_envs, num_bodies, 3). Take the max over bodies
        # of the force magnitude so a single colliding arm triggers the penalty.
        # arm_forces = self.arm_contacts.data.net_forces_w  # (N, B, 3)
        # if arm_forces is not None:
        #     arm_force_mag = torch.norm(arm_forces, dim=-1)  # (N, B)
        #     max_arm_force = torch.max(arm_force_mag, dim=-1)[0]  # (N,)
        #     arm_collision = (max_arm_force > self._robot_cfg.arm_collision_force_threshold).float()
        #     self.scalar_logger.log("robot_state", "AVG/arm_contact_force", max_arm_force)
        #     self.scalar_logger.log("robot_state", "SUM/arm_collision", arm_collision)
        #     self.scalar_logger.log(
        #         "robot_reward", "AVG/arm_collision", arm_collision * self._robot_cfg.rew_arm_collision_scale
        #     )
        #     reward = reward + arm_collision * self._robot_cfg.rew_arm_collision_scale

        # --- Arm symmetry: penalize left/right asymmetry (floating platform angular momentum bias) ---
        # arm_position_targets: [left_shoulder, left_elbow, right_shoulder, right_elbow]
        # arm_symmetry = (
        #     torch.square(self.arm_position_targets[:, 0] - self.arm_position_targets[:, 2])
        #     + torch.square(self.arm_position_targets[:, 1] - self.arm_position_targets[:, 3])
        # )
        # self.scalar_logger.log("robot_state", "AVG/arm_symmetry", arm_symmetry)
        # self.scalar_logger.log("robot_reward", "AVG/arm_symmetry", arm_symmetry * self._robot_cfg.rew_arm_symmetry_scale)
        # reward = reward + arm_symmetry * self._robot_cfg.rew_arm_symmetry_scale

        return reward

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        task_done = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        return task_failed, task_done

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ):
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_actions[env_ids] = 0
        self._previous_arm_position_targets[env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor):
        # thrust_reset = torch.zeros_like(self._thrust_action)
        # self._robot.set_external_force_and_torque(
        #    thrust_reset, thrust_reset, body_ids=self._thrusters_dof_idx, env_ids=env_ids
        # )
        locking_joints = torch.zeros((len(env_ids), 3), device=self._device) # [['/World/envs/env_0/Robot/joints/x_lock_joint', '/World/envs/env_0/Robot/joints/y_lock_joint', '/World/envs/env_0/Robot/joints/base_joint', '/World/envs/env_0/Robot/joints/left_shoulder_joint', '/World/envs/env_0/Robot/joints/right_shoulder_joint', '/World/envs/env_0/Robot/joints/reaction_wheel_joint', '/World/envs/env_0/Robot/joints/left_elbow_joint', '/World/envs/env_0/Robot/joints/right_elbow_joint']]
        self._robot.set_joint_velocity_target(locking_joints, joint_ids=self._locking_joint_dof_idx, env_ids=env_ids)
        self._robot.set_joint_position_target(locking_joints, joint_ids=self._locking_joint_dof_idx, env_ids=env_ids)

        # Reset arms to zero effort (this is to let them settle naturally, can change this later)
        levionarms_reset = torch.zeros((len(env_ids), 2), device=self._device)
        self._robot.set_joint_effort_target(levionarms_reset, joint_ids=self._left_levionarm_dof_idx, env_ids=env_ids)
        self._robot.set_joint_effort_target(levionarms_reset, joint_ids=self._right_levionarm_dof_idx, env_ids=env_ids)

        if self._robot_cfg.has_reaction_wheel:
            rw_reset = torch.zeros((len(env_ids), 1), device=self._device, dtype=torch.float32)
            self._robot.set_joint_velocity_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)
            self._robot.set_joint_effort_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)
            self.omega_reaction_wheel[env_ids] = 0
            self._reaction_wheel_to_body_torque_action[env_ids] = 0


    def process_actions(self, actions: torch.Tensor):
        """Process the actions for the robot.

        Continuous array with values in [-1, 1]. Layout: [thrust_dims..., arm_dims, reaction_wheel].
        - Thrust: first 3 (body-frame / position-heading) or first num_thrusters (direct). Then arms and rw follow.

        Position-heading (body-frame) thrust mapping (when direct_thruster_control is False):
        - actions[:, 0]: Forward/Backward. +X: thrusters 4 and 7 on, -X: thrusters 3 and 8 on.
        - actions[:, 1]: Left/Right. +Y: thrusters 2 and 5 on, -Y: thrusters 1 and 6 on.
        - actions[:, 2]: Rotate CW/CCW. CW: thrusters 2, 4, 6, 8 on, CCW: thrusters 1, 3, 5, 7 on.

        After thrust dims (3 or num_thrusters):
        - actions[:, thrust_start+0:thrust_start+2]: Left Arm joints (shoulder, elbow).
        - actions[:, thrust_start+2:thrust_start+4]: Right Arm joints (shoulder, elbow).
        - actions[:, -1]: Reaction wheel speed control.

        Logged robot_state terms (all logged here unless noted):
        - AVG/thruster_norm:         L2 norm of the 8 thruster force magnitudes [N]. Reflects overall
                                     thrust intensity; high values mean the robot is pushing hard.
        - AVG/rw_body_torque:        Z-torque transferred to the body from the reaction wheel [N·m].
                                     The equal-and-opposite reaction of the wheel's angular acceleration.
        - AVG/rw_omega:              Internal reaction wheel angular velocity [rad/s] from the dynamics
                                     model (not the Isaac joint sensor). Tracks saturation risk.
        - AVG/thruster_action_rate:  L1 change in thruster commands (direct mode only) across the
                                     num_thrusters dims only. High values = bang-bang switching.
                                     (logged in compute_rewards)
        - AVG/thruster_effort:       Sum of all 8 normalized thrust forces [0, 1]. Captures sustained
                                     activation cost independent of how fast commands change.
                                     (logged in compute_rewards)
        - AVG/joint_acceleration:    Sum of squared accelerations across all articulation joints.
                                     Proxy for mechanical stress and vibration.
                                     (logged in compute_rewards)
        - AVG/rw_saturation:         omega_rw^2. Grows sharply as the wheel approaches its speed limit.
                                     (logged in compute_rewards)
        - AVG/rw_usage:              |commanded_torque| to the reaction wheel. Non-zero even when the
                                     wheel is not saturating; shows how actively it is being driven.
                                     (logged in compute_rewards)
        - AVG/arm_action_rate:       L1 change in arm position targets (all 4 joints) per step.
                                     (logged in compute_rewards)
        - AVG/arm_symmetry:          (left_shoulder - right_shoulder)^2 + (left_elbow - right_elbow)^2.
                                     Zero when both arms are mirrored; grows with asymmetric poses.
                                     (logged in compute_rewards)

        Args:
            actions (torch.Tensor): The actions to process.
        """
        # Enforce action limits at the robot level
        actions = actions[:, : self._dim_robot_act].float()
        # Store the unaltered actions, by default the robot should only observe the unaltered actions.
        self._previous_unaltered_actions = self._unaltered_actions.clone()
        self._unaltered_actions = actions.clone()

        # Apply action randomizers
        for randomizer in self.randomizers:
            randomizer.actions(dt=self.scene.physics_dt, actions=actions)
            
        # Clip the actions between [-1, 1] to ensure they are within the expected range.
        actions = torch.clamp(actions, -1.0, 1.0)

        self._previous_actions = self._actions.clone()
        self._actions = actions

        # Thrust: first 3 (body-frame) or num_thrusters (direct); arms and rw are the same after that
        # thrust_dim = self._robot_cfg.num_thrusters if self._robot_cfg.direct_thruster_control else 3
        # self._thrust_action.fill_(0.0) # Reset thrust action

        # if self._robot_cfg.direct_thruster_control:
        #     # Direct thruster mode: one command per thruster, map [-1, 1] to [0, max_thrust]
        #     thrust_mag = (actions[:, :thrust_dim] * 0.5 + 0.5) * self._robot_cfg.max_thrust
        #     self._thrust_action[:, :, 2] = thrust_mag # .clamp(0.0, self._robot_cfg.max_thrust)
        # else:
        #     # High-level commands to thruster mapping (position-heading / body-frame)
        #     wp_actions = wp.from_torch(actions[:, :3].contiguous(), dtype=wp.vec3f)
        #     wp_thrust_action = wp.from_torch(self._thrust_action, dtype=wp.vec3f)
        #     wp.launch(
        #         kernel=compute_thruster_mapping,
        #         dim=self._num_envs,
        #         inputs=[
        #             wp_actions,
        #             wp_thrust_action,
        #             float(self._robot_cfg.max_thrust),
        #         ],
        #         device=self._device,
        #     )
        #     self._thrust_action = wp.to_torch(wp_thrust_action)
            
        # Arms control: absolute position, actions in [-1, 1] mapped to [lower_limit, upper_limit]
        # target = lower + (action * 0.5 + 0.5) * (upper - lower)
        self._previous_arm_position_targets = self.arm_position_targets.clone()
        # alpha = actions[:, thrust_dim:thrust_dim + 4] * 0.5 + 0.5  # remap [-1,1] -> [0,1]
        # self.arm_position_targets[:, 0] = self._shoulder_lower_limit + alpha[:, 0] * (self._shoulder_upper_limit - self._shoulder_lower_limit)  # Left shoulder
        # self.arm_position_targets[:, 1] = self._left_elbow_lower_limit + alpha[:, 1] * (self._left_elbow_upper_limit - self._left_elbow_lower_limit)  # Left elbow
        # self.arm_position_targets[:, 2] = self._shoulder_lower_limit + alpha[:, 2] * (self._shoulder_upper_limit - self._shoulder_lower_limit)  # Right shoulder
        # self.arm_position_targets[:, 3] = self._right_elbow_lower_limit + alpha[:, 3] * (self._right_elbow_upper_limit - self._right_elbow_lower_limit)  # Right elbow
        
        self.arm_position_targets[:] = 0.0
        

        if self._robot_cfg.has_reaction_wheel:
            dt = self.scene.physics_dt * 6.0
            commanded_torque = (actions[:, -1] * self._robot_cfg.reaction_wheel_scale).unsqueeze(-1)
            omega_prev = self.omega_reaction_wheel.clone()
            self.omega_reaction_wheel = commanded_torque / self.b + (omega_prev - commanded_torque / self.b) * math.exp(-self.b * dt / self.J_rw)
            reaction_torque = -self.b * (omega_prev - commanded_torque / self.b) * math.exp(-self.b * dt / self.J_rw)
            # self._reaction_wheel_action[:, :, 2] = reaction_torque
            # self._reaction_wheel_to_body_torque_action[:, :, 2] = -reaction_torque
            self._reaction_wheel_action = reaction_torque

        self.scalar_logger.log(
            "robot_state", "AVG/thruster_norm", torch.linalg.norm(self._thrust_action[:, :, 2], dim=-1)
        )
        if self._robot_cfg.has_reaction_wheel:
            self.scalar_logger.log("robot_state", "AVG/rw_body_torque", self._reaction_wheel_to_body_torque_action[:, 0, 2])
            self.scalar_logger.log(
                "robot_state", "AVG/rw_omega",
                self.omega_reaction_wheel.squeeze(-1),
            )
            # self.scalar_logger.log(
            #     "robot_state", "AVG/reaction_wheel_velocity",
            #     self._robot.data.joint_vel[:, self._reaction_wheel_dof_idx].squeeze(-1),
            # )
        
    def compute_physics(self):
        pass

    def apply_actions(self):
        # Compute the physics
        super().apply_actions()
        for randomizer in self.randomizers:
            randomizer.update(dt=self.scene.physics_dt, actions=self._actions)

        # Thrusters
        self._robot.set_external_force_and_torque(
            self._thrust_action, torch.zeros_like(self._thrust_action), body_ids=self._thrusters_dof_idx
        )

        # Arms position control        
        self._robot.set_joint_position_target(self.arm_position_targets, joint_ids=self._arms_ids)
        """
        # Effort control
        # Scale actions from [-1, 1] to effort in Nm (max effort is 1Nm from config)
        effort_scale = 1.0  # Nm per unit action
        left_arm_effort = self._actions[:, 3:5] * effort_scale # left shoulder, left elbow
        right_arm_effort = self._actions[:, 5:7] * effort_scale # right shoulder, right elbow
        # print(self._left_levionarm_dof_idx) # 3 is left shoulder, 6 is left elbow
        # print(self._right_levionarm_dof_idx) # 4 is right shoulder, 7 is right elbow
        self._robot.set_joint_effort_target(
            left_arm_effort, joint_ids=self._left_levionarm_dof_idx
        )
        self._robot.set_joint_effort_target(
            right_arm_effort, joint_ids=self._right_levionarm_dof_idx
        )
        """

        # Reaction wheel
        if self._robot_cfg.has_reaction_wheel:
            # Apply the reaction wheel torque as an external torque on the base link (opposite on the wheel joint itself)
            # self._robot.set_external_force_and_torque(
            #     torch.zeros_like(self._reaction_wheel_to_body_torque_action),
            #     self._reaction_wheel_to_body_torque_action,
            #     body_ids=self._root_idx,
            # )
            self._robot.set_joint_effort_target(self._reaction_wheel_action, joint_ids=self._reaction_wheel_dof_idx)

    @property
    def reaction_wheel_velocity(self) -> torch.Tensor:
        """Reaction wheel joint velocity in rad/s. Shape is (num_instances,).
        
        Returns the internally computed velocity from the reaction wheel dynamics model,
        not the Isaac joint velocity.
        """
        return self.omega_reaction_wheel.squeeze(-1) if self._robot_cfg.has_reaction_wheel else torch.zeros(self._num_envs, device=self._device)

    def set_velocity(
        self,
        velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        # Arms and reaction wheel
        arms_rw_vel = torch.zeros((env_ids.shape[0], 5), device=self._device)
        velocity = torch.cat([velocity[:, :2], velocity[:, -1].unsqueeze(-1), arms_rw_vel], dim=1)
        position = torch.zeros_like(velocity)
        self._robot.write_joint_state_to_sim(position, velocity, env_ids=env_ids)

    # def configure_gym_env_spaces(self):
    #     single_action_space = spaces.Box(
    #         low=-1.0, high=1.0, shape=(self._robot_cfg.action_space,), dtype=np.float32
    #     )
    #     action_space = vector.utils.batch_space(single_action_space, self._num_envs)
    #     return single_action_space, action_space

    def activateSensors(self, sensor_type: str, filter: list):
        if sensor_type == "contacts":
            self._robot_cfg.contact_sensor_active = True
            if len(filter) > 0:
                self._robot_cfg.body_contact_forces.filter_prim_paths_expr = filter

    def register_sensors(self) -> None:
        # Contact sensor
        if self._robot_cfg.contact_sensor_active and "robot_contacts" not in self.scene.sensors:
            self.scene.sensors["robot_contacts"] = ContactSensor(self._robot_cfg.body_contact_forces)
            self.contacts: ContactSensor = self.scene["robot_contacts"]

        # Arm contact sensor — always registered; used to penalize arm collisions.
        if "robot_arm_contacts" not in self.scene.sensors:
            self.scene.sensors["robot_arm_contacts"] = ContactSensor(self._robot_cfg.arm_contact_forces)
            self.arm_contacts: ContactSensor = self.scene["robot_arm_contacts"]

    def create_robot_visualization(self) -> None:
        """Creates arrow markers at each thruster position pointing in the thrust direction."""
        marker_cfg = ARROW_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Robot/Pingu/thrusters"
        marker_cfg.markers["arrow"].arrow_body_length = 0.25
        marker_cfg.markers["arrow"].arrow_body_radius = 0.03
        marker_cfg.markers["arrow"].arrow_head_radius = 0.07
        marker_cfg.markers["arrow"].arrow_head_length = 0.12
        marker_cfg.markers["arrow"].visual_material.diffuse_color = (1.0, 0.5, 0.0)
        self._thruster_visualizer = VisualizationMarkers(marker_cfg)

    def update_robot_visualization(self) -> None:
        """Updates thruster arrows: world position, heading from thrust direction, scale from magnitude."""
        N = self._num_envs
        T = self._robot_cfg.num_thrusters  # 8

        # Thruster body poses in world frame — (N, T, 3) and (N, T, 4)
        world_positions = self._robot.data.body_link_pos_w[:, self._thrusters_dof_idx].reshape(-1, 3)  # (N*T, 3)
        thruster_quats = self._robot.data.body_link_quat_w[:, self._thrusters_dof_idx].reshape(-1, 4)  # (N*T, 4)

        # --- Orientations: thruster local +Z in world frame is the thrust direction ---
        z_local = torch.tensor([[0., 0., 1.]], device=self._device).expand(N * T, -1)  # (N*T, 3)
        dirs_w = math_utils.quat_apply(thruster_quats, z_local)  # (N*T, 3)

        # ARROW_CFG points along its local +X axis. Negate to get exhaust (opposite of thrust).
        # Build a Z-rotation quat from +X to -dirs_w: angle = atan2(-dy, -dx)
        angle = torch.atan2(-dirs_w[:, 1], -dirs_w[:, 0])  # (N*T,)
        half = angle * 0.5
        zeros = torch.zeros_like(half)
        # quat format: (w, x, y, z)
        world_quats = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)  # (N*T, 4)

        # --- Scale: proportional to normalized thrust; hidden (near-zero) when inactive ---
        thrust_mag = self._thrust_action[:, :, 2]  # (N, T)
        normalized = (thrust_mag / self._robot_cfg.max_thrust).clamp(0.0, 1.0).reshape(-1)  # (N*T,)
        scale_vals = normalized.clamp(min=0.01).unsqueeze(-1).expand(-1, 3).contiguous()  # (N*T, 3)

        self._thruster_visualizer.visualize(world_positions, world_quats, scales=scale_vals)

    ##
    # Derived base properties
    ##

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self._robot.data.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    ##
    # Derived root properties
    ##

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation root's center of mass frame.
        """
        return self._robot.data.body_state_w[:, self._root_idx]

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root.
        """
        return self._robot.data.body_pos_w[:, self._root_idx].squeeze()

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root.
        """
        return self._robot.data.body_quat_w[:, self._root_idx].squeeze()

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame.
        """
        return self._robot.data.body_vel_w[:, self._root_idx].squeeze()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.body_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.body_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with
        respect to the articulation root's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        articulation root's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_quat_w, self.root_ang_vel_w)

    ##
    # Derived Root Link Frame properties
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_link_pos_w[:, self._root_idx].squeeze() #

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._robot.data.body_link_quat_w[:, self._root_idx].squeeze() #

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._robot.data.body_link_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._robot.data.body_link_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_link_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    ##
    # Derived CoM frame properties
    ##

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_com_pos_w[:, self._root_idx].squeeze()

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_com_quat_w[:, self._root_idx].squeeze() #

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_vel_w[:, self._root_idx].squeeze() #

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_lin_vel_w[:, self._root_idx].squeeze() #

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_ang_vel_w[:, self._root_idx].squeeze() #

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_com_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_com_quat_w, self.root_com_ang_vel_w)
