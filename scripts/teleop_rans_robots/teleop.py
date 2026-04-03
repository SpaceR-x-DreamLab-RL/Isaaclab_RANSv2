# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers)."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--rw_test",
    action="store_true",
    default=False,
    help="Run reaction wheel test mode: apply constant torque and plot velocity response.",
)
parser.add_argument("--rw_torque", type=float, default=1.0, help="Reaction wheel torque command in [-1, 1] for rw_test mode.")
parser.add_argument("--rw_duration", type=float, default=10.0, help="Duration in seconds for rw_test mode.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging

import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: 
import Isaaclab_RANSv2  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    
# import logger
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        # External cameras are not supported with XR teleop
        # Check for any camera configs and disable them
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            omni.log.warn(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    def map_action_to_robot(action, robot_name: str) -> torch.Tensor:
        """
        Map the teleoperation action to the specific robot's action space.

        Args:
            robot_name: The name of the robot in the environment.

        Returns:
            A torch.Tensor representing the mapped action for the robot.
        """
        if robot_name == "jetbot":
            if action[0] > 0:  # forward
                action = torch.tensor([[1, 1]], device=action.device)
            elif action[0] < 0:  # backward
                action = torch.tensor([[-1, -1]], device=action.device)
            elif action[1] > 0:  # left
                action = torch.tensor([[-1, 1]], device=action.device)
            elif action[1] < 0:  # right
                action = torch.tensor([[1, -1]], device=action.device)
            else:
                action = torch.tensor([[0, 0]], device=action.device)

            return action
        elif robot_name == "Cubo":
            # Direct thruster control
            new_action = -1 * torch.ones((1,9), dtype=torch.float32, device=action.device)
            new_action[:, 8:] = 0.0  # set arms and reaction wheel to 0 when no action is given
            if action[0] > 0:  # forward
                print("Mapping forward action")
                new_action[:, 3] = 1.0
                new_action[:, 6] = 1.0
            elif action[0] < 0:  # backward
                print("Mapping backward action")
                new_action[:, 2] = 1.0
                new_action[:, 7] = 1.0
            elif action[1] > 0:  # left
                print("Mapping left action")
                new_action[:, 1] = 1.0
                new_action[:, 4] = 1.0
            elif action[1] < 0:  # right
                print("Mapping right action")
                new_action[:, 0] = 1.0
                new_action[:, 5] = 1.0
            elif action[2] > 0:  # rotate ccw
                print("Mapping rotate ccw action")
                new_action[:, 0] = 1.0
                new_action[:, 2] = 1.0
                new_action[:, 4] = 1.0
                new_action[:, 6] = 1.0
            elif action[2] < 0:  # rotate cw
                print("Mapping rotate cw action")
                new_action[:, 1] = 1.0
                new_action[:, 3] = 1.0
                new_action[:, 5] = 1.0
                new_action[:, 7] = 1.0
            elif action[3] > 0:  # left arm elbow
                print("Mapping left arm elbow action +v")
                new_action[:, 8] = 1.0
            elif action[3] < 0:  # left arm elbow
                print("Mapping left arm elbow action -v")
                new_action[:, 8] = -1.0
            elif action[5] > 0:  # reaction wheel positive
                print("Mapping reaction wheel action +v")
                new_action[:, 8] = 1.0
            elif action[5] < 0:  # reaction wheel negative
                print("Mapping reaction wheel action -v")
                new_action[:, 8] = -1.0
            else:
                new_action = -1 * torch.ones((1,9), dtype=torch.float32, device=action.device)
                new_action[:, 8:] = 0.0  # set arms and reaction wheel to 0 when no action is given
            return new_action
        
            # new_action = torch.tensor([[0,0,0,0,0,0,0,0]], device=action.device)
            # if action[0] > 0:  # forward
            #     print("Mapping forward action")
            #     # new_action = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0]], device=action.device)
            #     new_action = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 0]], device=action.device)
            # elif action[0] < 0:  # backward
            #     print("Mapping backward action")
            #     # new_action = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]], device=action.device)
            #     new_action = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 1]], device=action.device)
            # elif action[1] > 0:  # left
            #     print("Mapping left action")
            #     new_action = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0]], device=action.device)
            # elif action[1] < 0:  # right
            #     print("Mapping right action")
            #     new_action = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0]], device=action.device)
            # elif action[2] > 0:  # rotate cw
            #     print("Mapping rotate cw action")
            #     new_action = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]], device=action.device)
            # elif action[2] < 0:  # rotate ccw
            #     print("Mapping rotate ccw action")
            #     new_action = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], device=action.device)
            # else:
            #     new_action = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], device=action.device)

            # return new_action
            new_action = torch.zeros((1,3), dtype=torch.float32, device=action.device)
            if action[0] > 0:  # forward
                print("Mapping forward action")
                new_action = torch.tensor([[1, 0, 0]], dtype=torch.float32, device=action.device)
            elif action[0] < 0:  # backward
                print("Mapping backward action")
                new_action = torch.tensor([[-1, 0, 0]], dtype=torch.float32, device=action.device)
            elif action[1] > 0:  # left
                print("Mapping left action")
                new_action = torch.tensor([[0, 1, 0]], dtype=torch.float32, device=action.device)
            elif action[1] < 0:  # right
                print("Mapping right action")
                new_action = torch.tensor([[0, -1, 0]], dtype=torch.float32, device=action.device)
            elif action[5] > 0:  # rotate cw (C key -> rz in keyboard output)
                print("Mapping rotate cw action")
                new_action = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=action.device)
            elif action[5] < 0:  # rotate ccw (V key -> rz in keyboard output)
                print("Mapping rotate ccw action")
                new_action = torch.tensor([[0, 0, -1]], dtype=torch.float32, device=action.device)
            else:
                new_action = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=action.device)

            return new_action

        elif robot_name == "FloatingPlatform":
            new_action = torch.zeros((1,8), dtype=torch.float32, device=action.device)
            if action[0] > 0:  # forward
                print("Mapping forward action")
                new_action = torch.tensor([[1, 0, 0]], dtype=torch.float32, device=action.device)
            elif action[0] < 0:  # backward
                print("Mapping backward action")
                new_action = torch.tensor([[-1, 0, 0]], dtype=torch.float32, device=action.device)
            elif action[1] > 0:  # left
                print("Mapping left action")
                new_action = torch.tensor([[0, 1, 0]], dtype=torch.float32, device=action.device)
            elif action[1] < 0:  # right
                print("Mapping right action")
                new_action = torch.tensor([[0, -1, 0]], dtype=torch.float32, device=action.device)
            elif action[2] > 0:  # rotate ccw
                print("Mapping rotate ccw action")
                new_action = torch.tensor([[0, 0, 1]], dtype=torch.float32, device=action.device)
            elif action[2] < 0:  # rotate cw
                print("Mapping rotate cw action")
                new_action = torch.tensor([[0, 0, -1]], dtype=torch.float32, device=action.device)
            else:
                new_action = torch.zeros((1,8), dtype=torch.float32, device=action.device)
                
            return new_action

        elif robot_name == "Pingu":
            # Direct thruster control
            new_action = -1 * torch.ones((1,13), dtype=torch.float32, device=action.device)
            new_action[:, 8:] = 0.0  # set arms and reaction wheel to 0 when no action is given
            if action[0] > 0:  # forward
                print("Mapping forward action")
                new_action[:, 3] = 1.0
                new_action[:, 6] = 1.0
            elif action[0] < 0:  # backward
                print("Mapping backward action")
                new_action[:, 2] = 1.0
                new_action[:, 7] = 1.0
            elif action[1] > 0:  # left
                print("Mapping left action")
                new_action[:, 1] = 1.0
                new_action[:, 4] = 1.0
            elif action[1] < 0:  # right
                print("Mapping right action")
                new_action[:, 0] = 1.0
                new_action[:, 5] = 1.0
            elif action[2] > 0:  # rotate ccw
                print("Mapping rotate ccw action")
                new_action[:, 0] = 1.0
                new_action[:, 2] = 1.0
                new_action[:, 4] = 1.0
                new_action[:, 6] = 1.0
            elif action[2] < 0:  # rotate cw
                print("Mapping rotate cw action")
                new_action[:, 1] = 1.0
                new_action[:, 3] = 1.0
                new_action[:, 5] = 1.0
                new_action[:, 7] = 1.0
            elif action[3] > 0:  # left arm elbow
                print("Mapping left arm elbow action +v")
                new_action[:, 10] = 1.0
            elif action[3] < 0:  # left arm elbow
                print("Mapping left arm elbow action -v")
                new_action[:, 10] = -1.0
            elif action[4] > 0:  # right arm elbow
                print("Mapping right arm elbow action +v")
                new_action[:, 10] = 1.0
            elif action[4] < 0:  # right arm elbow
                print("Mapping right arm elbow action -v")
                new_action[:, 10] = -1.0
            elif action[5] > 0:  # reaction wheel positive
                print("Mapping reaction wheel action +v")
                new_action[:, -1] = 1.0
            elif action[5] < 0:  # reaction wheel negative
                print("Mapping reaction wheel action -v")
                new_action[:, -1] = -1.0
            else:
                new_action = -1 * torch.ones((1,13), dtype=torch.float32, device=action.device)
                new_action[:, 8:] = 0.0  # set arms and reaction wheel to 0 when no action is given
            return new_action
            
            # Mapping thrusters
            # new_action = torch.tensor([[0,0,0,0,0,0,0,0]], device=action.device)
            # if action[0] > 0:  # forward
            #     print("Mapping forward action")
            #     new_action = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], device=action.device)
            # elif action[0] < 0:  # backward
            #     print("Mapping backward action")
            #     new_action = torch.tensor([[-1, 0, 0, 0, 0, 0, 0, 0]], device=action.device)
            # elif action[1] > 0:  # left
            #     print("Mapping left action")
            #     new_action = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], device=action.device)
            # elif action[1] < 0:  # right
            #     print("Mapping right action")
            #     new_action = torch.tensor([[0, -1, 0, 0, 0, 0, 0, 0]], device=action.device)
            # elif action[2] > 0:  # rotate cw
            #     print("Mapping rotate cw action")
            #     new_action = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0]], device=action.device)
            # elif action[2] < 0:  # rotate ccw
            #     print("Mapping rotate ccw action")
            #     new_action = torch.tensor([[0, 0, -1, 0, 0, 0, 0, 0]], device=action.device)
            # elif action[3] > 0:  # left arm elbow
            #     print("Mapping left arm elbow action +v")
            #     new_action = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0]], device=action.device)
            # elif action[3] < 0:  # left arm elbow
            #     print("Mapping left arm elbow action -v")
            #     new_action = torch.tensor([[0, 0, 0, 0, -1, 0, 0, 0]], device=action.device)
            # elif action[4] > 0:  # right arm elbow
            #     print("Mapping right arm elbow action +v")
            #     new_action = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0]], device=action.device)
            # elif action[4] < 0:  # right arm elbow
            #     print("Mapping right arm elbow action -v")
            #     new_action = torch.tensor([[0, 0, 0, 0, 0, 0, -1, 0]], device=action.device)
            # elif action[5] > 0:  # reaction wheel positive
            #     print("Mapping reaction wheel action +v")
            #     new_action = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]], device=action.device)
            # elif action[5] < 0:  # reaction wheel negative
            #     print("Mapping reaction wheel action -v")
            #     new_action = torch.tensor([[0, 0, 0, 0, 0, 0, 0, -1]], device=action.device)
            # else:
            #     new_action = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], device=action.device)

            # return new_action
        else:
            omni.log.warn(f"Unknown robot '{robot_name}'. Using default action mapping.")
            return action

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                omni.log.error("Supported devices: keyboard, spacemouse, gamepad, handtracking")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    omni.log.warn(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        omni.log.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    if args_cli.rw_test:
        # --- Reaction wheel test mode ---
        print(f"Reaction wheel test mode: torque={args_cli.rw_torque}, duration={args_cli.rw_duration}s")
        robot_name = env.robot_cfg.robot_name
        sim_dt = env.step_dt  # step dt (physics_dt * decimation)
        total_steps = int(args_cli.rw_duration / sim_dt)

        # Build a constant action that only drives the reaction wheel
        act_dim = env.action_space.shape[-1]
        base_action = torch.zeros((1, act_dim), device=env.device)
        if robot_name in ("Cubo", "Pingu"):
            # Reaction wheel is always the last action dimension
            base_action[:, -1] = args_cli.rw_torque
            # base_action[:, 0] = args_cli.rw_torque
            # base_action[:, 2] = args_cli.rw_torque
            # base_action[:, 4] = args_cli.rw_torque
            # base_action[:, 6] = args_cli.rw_torque
            # Arms 
            # base_action[:, 8:10] = -1.0
            # base_action[:, 10:12] = 1.0
        else:
            omni.log.error(f"rw_test not supported for robot '{robot_name}'")
            env.close()
            simulation_app.close()
            return

        # For Cubo with direct_thruster_control, thrusters expect [-1,1] mapped to [0,max].
        # Set thrusters to -1 so they produce zero thrust.
        if env.robot_cfg.direct_thruster_control:
            base_action[:, :env.robot_cfg.num_thrusters] = -1.0

        time_log = []
        rw_vel_log = []
        torque_log = []
        robot_ang_vel_log = []

        print(f"Running {total_steps} steps...")
        for step_i in range(total_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():  # set a breakpoint here to inspect the environment state during the test
                actions = base_action.repeat(env.num_envs, 1)
                env.step(actions)
                
                # if step_i > 300:
                #     base_action[:, :env.robot_cfg.num_thrusters] = -1.0

                # Read reaction wheel velocity from robot api (internally computed)
                rw_vel = env.robot_api.reaction_wheel_velocity.mean().item()
                robot_ang_vel = env.robot_api.root_com_ang_vel_w[:, -1].mean().item()
                t = (step_i + 1) * sim_dt
                
                # Get the actual applied torque from the robot's internal state
                applied_torque = env.robot_api._reaction_wheel_action.mean().item() #env.robot_api._reaction_wheel_action[:, :, 2].mean().item()

                time_log.append(t)
                rw_vel_log.append(rw_vel)
                torque_log.append(applied_torque)
                robot_ang_vel_log.append(robot_ang_vel)

        # --- Plot results ---
        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # # Velocity vs Time
        # axes[0].plot(time_log, rw_vel_log, linewidth=1.5)
        # axes[0].set_xlabel("Time (s)")
        # axes[0].set_ylabel("Reaction Wheel Velocity (rad/s)")
        # axes[0].set_title(f"{robot_name} - RW Velocity vs Time (torque cmd={args_cli.rw_torque})")
        # axes[0].grid(True)

        # # Torque vs Velocity (parametric: each point is one timestep)
        # axes[1].plot(torque_log, rw_vel_log, "o", markersize=2)
        # axes[1].set_xlabel("Applied Torque (Nm)")
        # axes[1].set_ylabel("Reaction Wheel Velocity (rad/s)")
        # axes[1].set_title(f"{robot_name} - Torque vs RW Velocity")
        # axes[1].grid(True)

        # plt.tight_layout()
        # plot_path = f"source/rw_test_{robot_name}.png"
        # plt.savefig(plot_path, dpi=150)
        # print(f"Plots saved to {plot_path}")
        # plt.show()
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker  # Import the ticker module

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Define the formatter for 4 decimal places
        formatter = ticker.FormatStrFormatter('%.2f')

        # Velocity vs Time
        axes[0].plot(time_log, rw_vel_log, linewidth=1.5)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Reaction Wheel Velocity (rad/s)")
        axes[0].set_title(f"{robot_name} - RW Velocity vs Time (torque cmd={args_cli.rw_torque})")
        axes[0].grid(True)

        # Apply formatting to both axes of the first subplot
        # axes[0].xaxis.set_major_formatter(formatter)
        # axes[0].yaxis.set_major_formatter(formatter)

        # Torque vs Time
        axes[1].plot(time_log, torque_log, linewidth=1.5, color='orange')
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Applied Torque (Nm)")
        axes[1].set_title(f"{robot_name} - Applied Torque vs Time")
        axes[1].grid(True)

        # Apply formatting to both axes of the second subplot
        # axes[1].xaxis.set_major_formatter(formatter)
        axes[1].yaxis.set_major_formatter(formatter)

        # Torque vs Velocity (parametric)
        axes[2].plot(torque_log, rw_vel_log, "o", markersize=2)
        axes[2].set_xlabel("Applied Torque (Nm)")
        axes[2].set_ylabel("Reaction Wheel Velocity (rad/s)")
        axes[2].set_title(f"{robot_name} - Torque vs RW Velocity")
        axes[2].grid(True)

        # Apply formatting to both axes of the third subplot
        axes[2].xaxis.set_major_formatter(formatter)
        # axes[2].yaxis.set_major_formatter(formatter)

        # Robot angular velocity vs Time
        axes[3].plot(time_log, robot_ang_vel_log, linewidth=1.5, color="green")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Robot Angular Velocity (rad/s)")
        axes[3].set_title(f"{robot_name} - Robot Angular Velocity vs Time")
        axes[3].grid(True)

        # Apply formatting to both axes of the fourth subplot
        # axes[3].xaxis.set_major_formatter(formatter)
        axes[3].yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plot_path = f"source/rw_test_{robot_name}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plots saved to {plot_path}")
        # plt.show()
        
        # Print final values
        print(f"Final RW Velocity: {rw_vel_log[-1]:.4f} rad/s")
        print(f"Final Applied Torque: {torque_log[-1]:.4f} Nm")
        print(f"Final Robot Angular Velocity: {robot_ang_vel_log[-1]:.4f} rad/s")

    else:
        # --- Normal teleoperation mode ---
        print("Teleoperation started. Press 'R' to reset the environment.")

        # simulate environment
        while simulation_app.is_running():
            # try:
                # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance() # [x, y, z, rx, ry, rz, G] G = gripper (+1.0 for open, -1.0 for close)
                # action[-1] = 0.0

                action = map_action_to_robot(action, env.robot_cfg.robot_name)
                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)
                    # apply actions
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False

            # except Exception as e:
            #     omni.log.error(f"Error during simulation step: {e}")
            #     breakpoint()
            #     break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
