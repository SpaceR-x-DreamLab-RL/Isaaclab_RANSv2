# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from gymnasium import spaces, vector

from isaaclab.assets import Articulation
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

        # Buffers
        self.initialize_buffers()

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
            self._reaction_wheel_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
            
        self.arm_position_targets = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

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

        self.scalar_logger.add_log("robot_state", "AVG/thrusters", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/reaction_wheel", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/joint_acceleration", "mean")

    def get_observations(self) -> torch.Tensor:
        return self._unaltered_actions

    def compute_rewards(self):
        # TODO: DT should be factored in?

        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self.scalar_logger.log("robot_state", "AVG/joint_acceleration", joint_accelerations)
        self.scalar_logger.log("robot_reward", "AVG/joint_acceleration", joint_accelerations)

        reward = joint_accelerations * self._robot_cfg.rew_joint_accel_scale
        if self._robot_cfg.direct_thruster_control:
            action_rate = torch.sum(torch.abs(self._unaltered_actions - self._previous_unaltered_actions), dim=1)
            self.scalar_logger.log("robot_state", "AVG/action_rate", action_rate)
            self.scalar_logger.log("robot_reward", "AVG/action_rate", action_rate)
            reward = reward + action_rate * self._robot_cfg.rew_action_rate_scale
        return torch.zeros_like(reward) #reward

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
        
        # Reset swing state
        self._swing_state[env_ids] = 0
        self._swing_timer[env_ids] = torch.rand(len(env_ids), device=self._device) * 2.0 # Random wait up to 2s
        self._swing_start_angle[env_ids] = -0.8
        self._swing_target_angle[env_ids] = -0.8
        self._swing_duration[env_ids] = 1.0

    def set_initial_conditions(self, env_ids: torch.Tensor):
        # thrust_reset = torch.zeros_like(self._thrust_action)
        # self._robot.set_external_force_and_torque(
        #    thrust_reset, thrust_reset, body_ids=self._thrusters_dof_idx, env_ids=env_ids
        # )
        locking_joints = torch.zeros((len(env_ids), 3), device=self._device) # [['/World/envs/env_0/Robot/joints/x_lock_joint', '/World/envs/env_0/Robot/joints/y_lock_joint', '/World/envs/env_0/Robot/joints/base_joint', '/World/envs/env_0/Robot/joints/left_shoulder_joint', '/World/envs/env_0/Robot/joints/right_shoulder_joint', '/World/envs/env_0/Robot/joints/rw_revolute_joint', '/World/envs/env_0/Robot/joints/left_elbow_joint', '/World/envs/env_0/Robot/joints/right_elbow_joint']]
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

        self._previous_actions = self._actions.clone()
        self._actions = actions

        # Arms control (shared: indices thrust_dim .. thrust_dim+4)
        """ Incremental position control (actions are changes to joint position targets)
        """
        self._robot_cfg.arms_action_scalar = 1.0
        
        # # Left shoulder
        # left_shoulder_displacement = self._robot.data.joint_pos[self._env_ids, self._left_levionarm_dof_idx[0]] + actions[:, 0] * self._robot_cfg.arms_action_scalar
        # self.arm_position_targets[:, 0] = torch.clamp(
        #     left_shoulder_displacement, self._shoulder_lower_limit, self._shoulder_upper_limit
        # )
        # # Left elbow
        # left_elbow_displacement = self._robot.data.joint_pos[self._env_ids, self._left_levionarm_dof_idx[1]] + actions[:, 1] * self._robot_cfg.arms_action_scalar
        # self.arm_position_targets[:, 1] = torch.clamp(
        #     left_elbow_displacement, self._left_elbow_lower_limit, self._left_elbow_upper_limit
        # )
        # # Right shoulder
        # right_shoulder_displacement = self._robot.data.joint_pos[self._env_ids, self._right_levionarm_dof_idx[0]] + actions[:, 2] * self._robot_cfg.arms_action_scalar
        # self.arm_position_targets[:, 2] = torch.clamp(
        #     right_shoulder_displacement, self._shoulder_lower_limit, self._shoulder_upper_limit
        # )
        # # Right elbow
        # right_elbow_displacement = self._robot.data.joint_pos[self._env_ids, self._right_levionarm_dof_idx[1]] + actions[:, 3] * self._robot_cfg.arms_action_scalar
        # self.arm_position_targets[:, 3] = torch.clamp(
        #     right_elbow_displacement, self._right_elbow_lower_limit, self._right_elbow_upper_limit
        # )
        
        if self._robot_cfg.has_reaction_wheel:
            # Reaction wheel: last action (index thrust_dim+4 or -1)
            self._reaction_wheel_action = (
                actions[:, -1] * self._robot_cfg.reaction_wheel_scale
            ).unsqueeze(-1)

        # Log data for monitoring
        if self._robot_cfg.has_reaction_wheel:
            self.scalar_logger.log("robot_state", "AVG/reaction_wheel", self._reaction_wheel_action[:, 0])

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
            self._robot.set_joint_effort_target(self._reaction_wheel_action, joint_ids=self._reaction_wheel_dof_idx)

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
        if self._robot_cfg.contact_sensor_active:
            self.scene.sensors["robot_contacts"] = ContactSensor(self._robot_cfg.body_contact_forces)
            self.contacts: ContactSensor = self.scene["robot_contacts"]

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
