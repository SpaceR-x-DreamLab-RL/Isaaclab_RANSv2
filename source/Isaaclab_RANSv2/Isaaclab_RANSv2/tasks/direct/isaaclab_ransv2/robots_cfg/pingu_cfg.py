# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.pingu import PINGU_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ..domain_randomization import (
    ActionsRescalerCfg,
    CoMRandomizationCfg,
    MassRandomizationCfg,
    NoisyActionsCfg,
    WrenchRandomizationCfg,
)

from .robot_core_cfg import RobotCoreCfg


@configclass
class PinguRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_name: str = "Pingu"

    robot_cfg: ArticulationCfg = PINGU_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker_height = 0.9
    has_reaction_wheel = True
    num_thrusters = 8
    direct_thruster_control = True

    thrusters_dof_name = [f"thruster_{i}_link" for i in range(1, num_thrusters + 1)]
    locking_joint_dof_name = ["x_lock_joint", "y_lock_joint", "base_joint"]
    left_levionarm_dof_name = ["left_shoulder_joint", "left_elbow_joint"]
    right_levionarm_dof_name = ["right_shoulder_joint", "right_elbow_joint"]
    reaction_wheel_dof_name = ["rw_revolute_joint"]
    reaction_wheel_scale = 8.0  # [Nm]: scale from [-1, 1] action to effort (max is 10.0 Nm from URDF)

    root_id_name = "base_link"
    rew_action_rate_scale = -0.12 / 8
    rew_joint_accel_scale = -2.5e-6

    max_thrust = 1.0
    """Maximum thrust of the thrusters in Newtons"""
    split_thrust = True
    """Split the thrust between the thrusters"""
    arms_action_scalar=1.0
    """Scalar to adjust the magnitude of the arm actions(shoulder + elbow)"""

    # Randomization
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name=root_id_name, max_delta=0.25
    )
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name=root_id_name, max_delta=0.05
    )
    wrench_rand_cfg = WrenchRandomizationCfg(
        enable=False,
        randomization_modes=["constant_uniform"],
        body_name=root_id_name,
        uniform_force=(0, 0.25),
        uniform_torque=(0, 0.05),
        normal_force=(0, 0.25),
        normal_torque=(0, 0.025),
    )
    noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 8)],
        max_delta=[0.1],
        std=[0.025],
        clip_actions=[(-1, 1)],
    )
    action_rescaler_cfg: ActionsRescalerCfg = ActionsRescalerCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 8)],
        rescaling_ranges=[(0.8, 1.0)],
        clip_actions=[(-1, 1)],
    )

    # Sensors
    body_contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
    )

    # Spaces (depend on direct_thruster_control)
    # [thrust_dims..., auxiliary_dims] with reaction wheel at -1, others at -2, -3, ...
    # When direct_thruster_control=False: 3 movement, 4 motors for 2 arms (shoulder + elbow) + 1 reaction wheel
    # When direct_thruster_control=True: num_thrusters thrust, 4 motors for 2 arms (shoulder + elbow) + 1 reaction wheel
    state_space: int = 0
    gen_space: int = 0  # TODO: Add the generative space from the randomization

    @property
    def action_space(self) -> int:
        if self.direct_thruster_control:
            return 1 #4 # Only arms actions
        return 1 #4 # Only arms actions

    @property
    def observation_space(self) -> int:
        """Robot observation dim (same as action dim: last action as obs)."""
        return self.action_space
