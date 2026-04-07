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

from ..robots_cfg import CuboRobotCfg

from .robot_core import RobotCore

import numpy as np
import warp as wp
from ..utils import compute_thruster_mapping


class CuboRobot(RobotCore):

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        robot_cfg: CuboRobotCfg = CuboRobotCfg(),
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
            self.omega_reation_wheel = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._thrusters_dof_idx, _ = self._robot.find_bodies(self._robot_cfg.thrusters_dof_name)
        self._root_idx, _ = self._robot.find_bodies([self._robot_cfg.root_id_name])

        if self._robot_cfg.has_reaction_wheel:
            self._reaction_wheel_dof_idx, _ = self._robot.find_joints(self._robot_cfg.reaction_wheel_dof_name)

    def create_logs(self):
        super().create_logs()

        self.scalar_logger.add_log("robot_state", "AVG/thruster_norm", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_body_torque", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_omega", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_torque_sq", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/rw_overspin", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/rw_torque_sq", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/rw_overspin", "mean")

    @property
    def eval_data_keys(self) -> list[str]:
        keys = [
            "position",
            "heading",
            "linear_velocity",
            "angular_velocity",
            "thrust_action",
            "actions",
            "unaltered_actions",
        ]
        if self._robot_cfg.has_reaction_wheel:
            keys += ["reaction_wheel_action", "omega_reaction_wheel"]
        return keys

    @property
    def eval_data_specs(self) -> dict[str, list[str]]:
        num_thrusters = self._robot_cfg.num_thrusters
        specs = {
            "position": [".robot_pos.x.m", ".robot_pos.y.m", ".robot_pos.z.m"],
            "heading": [".robot_heading.rad"],
            "linear_velocity": [".robot_lin_vel.x.m/s", ".robot_lin_vel.y.m/s", ".robot_lin_vel.z.m/s"],
            "angular_velocity": [".robot_ang_vel.x.rad/s", ".robot_ang_vel.y.rad/s", ".robot_ang_vel.z.rad/s"],
            "thrust_action": [f".thruster{i}.force.N" for i in range(num_thrusters)],
            "actions": [f".robot_actions{i}.u" for i in range(self._robot_cfg.action_space)],
            "unaltered_actions": [f".robot_unaltered_actions{i}.u" for i in range(self._robot_cfg.action_space)],
        }
        if self._robot_cfg.has_reaction_wheel:
            specs["reaction_wheel_action"] = [".reaction_wheel_action.u"]
            specs["omega_reaction_wheel"] = [".omega_reaction_wheel.rad/s"]
        return specs

    @property
    def eval_data(self) -> dict:
        data = {
            "position": self.root_pos_w,
            "heading": self.heading_w,
            "linear_velocity": self.root_lin_vel_b,
            "angular_velocity": self.root_ang_vel_b,
            "thrust_action": self._thrust_action[..., -1],
            "actions": self._actions,
            "unaltered_actions": self._unaltered_actions,
        }
        if self._robot_cfg.has_reaction_wheel:
            data["reaction_wheel_action"] = self._reaction_wheel_action
            data["omega_reaction_wheel"] = self.omega_reation_wheel
        return data

    def get_observations(self) -> torch.Tensor:
        return self._previous_actions

    def compute_rewards(self):
        """
        Reward term reference for Cubo (reaction-wheel only)
        -----------------------------------------------------
        Single penalty applied on top of the task-level reward:

          rw_torque_sq  (rew_rw_torque_scale, negative)
              Penalizes action^2 (the normalized RW command, squared).
              Quadratic growth makes high-torque commands disproportionately
              expensive while near-zero nudges are nearly free.  Because the
              cost accumulates every step, sustained torque is also penalized
              — the policy learns to apply brief, gentle corrections.

          rw_overspin  (rew_rw_overspin_scale, negative)
              Penalizes (omega_rw / max_rw_speed)^2.  Keeps the wheel speed
              within the physical operating range (max ~20 rad/s in the lab).
              Quadratic so low speeds are nearly free but approaching the
              limit becomes very costly.
        """

        reward = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)

        if self._robot_cfg.has_reaction_wheel:
            # # Penalize high torque commands
            rw_torque_sq = torch.square(self._actions[:, -1])
            self.scalar_logger.log("robot_state", "AVG/rw_torque_sq", rw_torque_sq)
            self.scalar_logger.log("robot_reward", "AVG/rw_torque_sq", rw_torque_sq * self._robot_cfg.rew_rw_torque_scale)
            reward = reward + rw_torque_sq * self._robot_cfg.rew_rw_torque_scale

            # Penalize high wheel speed (normalized by physical max)
            rw_overspin = torch.square(self.omega_reation_wheel.squeeze(-1) / self._robot_cfg.max_rw_speed)
            self.scalar_logger.log("robot_state", "AVG/rw_overspin", rw_overspin)
            self.scalar_logger.log("robot_reward", "AVG/rw_overspin", rw_overspin * self._robot_cfg.rew_rw_overspin_scale)
            reward = reward + rw_overspin * self._robot_cfg.rew_rw_overspin_scale

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

    def set_initial_conditions(self, env_ids: torch.Tensor):
        # thrust_reset = torch.zeros_like(self._thrust_action)
        # self._robot.set_external_force_and_torque(
        #    thrust_reset, thrust_reset, body_ids=self._thrusters_dof_idx, env_ids=env_ids
        # )
        locking_joints = torch.zeros((len(env_ids), 4), device=self._device) # [['/World/envs/env_0/Robot/joints/x_lock_joint', '/World/envs/env_0/Robot/joints/y_lock_joint', '/World/envs/env_0/Robot/joints/base_joint', '/World/envs/env_0/Robot/joints/reaction_wheel_joint', ]]
        self._robot.set_joint_velocity_target(locking_joints, env_ids=env_ids)
        self._robot.set_joint_position_target(locking_joints, env_ids=env_ids)

        if self._robot_cfg.has_reaction_wheel:
            rw_reset = torch.zeros((len(env_ids), 1), device=self._device, dtype=torch.float32)
            self._robot.set_joint_velocity_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)
            self._robot.set_joint_effort_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)
            self.omega_reation_wheel[env_ids] = 0
            self._reaction_wheel_to_body_torque_action[env_ids] = 0

    def process_actions(self, actions: torch.Tensor):
        """
        Process the actions for the robot. Expects continuous actions in the range [-1, 1]. This operates when the flag `direct_thruster_control` is False. Thrust (body-frame or direct) uses leading dimensions; auxiliary actuators use trailing
        indices (reaction wheel at -1, others at -2, -3, ... if added).

        - actions[:, -1] = reaction wheel (if has_reaction_wheel)
        """
        # Common: slice to robot action dim, store for obs/rewards, apply randomizers, update _actions (both modes)
        actions = actions[:, : self._dim_robot_act].float()
        self._previous_unaltered_actions = self._unaltered_actions.clone()

        self._unaltered_actions = actions.clone()
        for randomizer in self.randomizers:
            randomizer.actions(dt=self.scene.physics_dt, actions=actions)
            
        # Clip the actions between [-1, 1] to ensure they are within the expected range.
        actions = torch.clamp(actions, -1.0, 1.0)
            
        self._previous_actions = self._actions.clone()
        self._actions = actions
        
        # # Thrust: first 3 (body-frame) or num_thrusters (direct); arms and rw are the same after that
        # thrust_dim = self._robot_cfg.num_thrusters if self._robot_cfg.direct_thruster_control else 3

        # self._thrust_action.fill_(0.0)

        # if self._robot_cfg.direct_thruster_control:
        #     # Direct thruster mode: one command per thruster
        #     thrust_mag = (actions[:, :thrust_dim] * 0.5 + 0.5) * self._robot_cfg.max_thrust
        #     self._thrust_action[:, :, 2] = thrust_mag # .clamp(0.0, self._robot_cfg.max_thrust)

        # else:
        #     # Body-frame control: leading 3 = (forward, left/right, yaw); pass signed to kernel
        #     wp_actions = wp.from_torch(actions[:, :3].contiguous(), 
        #     dtype=wp.vec3f)
        #     body_act = actions[:, :3].float().contiguous()
        #     wp_actions = wp.from_torch(body_act, dtype=wp.vec3f)
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
        

        if self._robot_cfg.has_reaction_wheel:
            dt = self.scene.physics_dt * 6.0
            commanded_torque = (actions[:, -1] * self._robot_cfg.reaction_wheel_scale).unsqueeze(-1)
            omega_prev = self.omega_reation_wheel.clone()
            self.omega_reation_wheel = commanded_torque / self.b + (omega_prev - commanded_torque / self.b) * math.exp(-self.b * dt / self.J_rw)
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
                self.omega_reation_wheel.squeeze(-1),
            )
            # self.scalar_logger.log(
            #     "robot_state", "AVG/reaction_wheel_velocity",
            #     self._robot.data.joint_vel[:, self._reaction_wheel_dof_idx].squeeze(-1),
            # )

    def compute_physics(self):
        pass  # Model motor + ackermann steering here

    def apply_actions(self):
        # Compute the physics
        super().apply_actions()
        for randomizer in self.randomizers:
            randomizer.update(dt=self.scene.physics_dt, actions=self._actions)

        self._robot.set_external_force_and_torque(
            self._thrust_action, torch.zeros_like(self._thrust_action), body_ids=self._thrusters_dof_idx
        )
        
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
        return  self.omega_reation_wheel.squeeze(-1) if self._robot_cfg.has_reaction_wheel else torch.zeros(self._num_envs, device=self._device) # TODO: This values doesn't seem acurate: self._robot.data.joint_vel[:, self._reaction_wheel_dof_idx].squeeze(-1)

    def set_velocity(
        self,
        velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        # Arms and reaction wheel
        rw_vel = torch.zeros((env_ids.shape[0], 1), device=self._device)
        velocity = torch.cat([velocity[:, :2], velocity[:, -1].unsqueeze(-1), rw_vel], dim=1)
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
        return self._robot.data.body_link_pos_w[:, self._root_idx].squeeze()

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._robot.data.body_link_quat_w[:, self._root_idx].squeeze()

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
        return self._robot.data.body_com_quat_w[:, self._root_idx].squeeze()

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_ang_vel_w[:, self._root_idx].squeeze()

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