# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-MPPI player for floating platform tasks.

This script uses MPPI to compute thruster commands from the current state and task goal.
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run task with Warp-MPPI controller.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps. 0 means run until app closes.")
parser.add_argument("--ref_lookahead", type=float, default=10.0, help="Straight-line lookahead distance [m].")
parser.add_argument(
    "--use_obs_goal",
    action="store_true",
    default=False,
    help="Derive goal position/heading from task observations.",
)
parser.add_argument(
    "--obs_task_type",
    type=str,
    default="auto",
    choices=["auto", "go_to_pose", "go_to_position", "go_through_positions", "go_through_poses"],
    help="Task observation layout used to infer goal from observations.",
)

parser.add_argument("--mppi_samples", type=int, default=100, help="Number of sampled trajectories.")
parser.add_argument("--mppi_horizon", type=int, default=100, help="Planning horizon length.")
parser.add_argument("--temperature", type=float, default=0.02, help="MPPI temperature parameter.")
parser.add_argument("--noise_std_x", type=float, default=0.015, help="MPPI noise std dev for body-x input.")
parser.add_argument("--noise_std_y", type=float, default=0.015, help="MPPI noise std dev for body-y input.")
parser.add_argument("--noise_std_yaw", type=float, default=0.01*0.0, help="MPPI noise std dev for yaw input.")
parser.add_argument("--noise_std_rw", type=float, default=0.0, help="MPPI noise std dev for reaction wheel input.")

parser.add_argument("--cost_pos", type=float, default=100.0, help="Position tracking cost weight.")
parser.add_argument("--cost_heading", type=float, default=8.0, help="Heading tracking cost weight.")
parser.add_argument("--cost_vel", type=float, default=0.0, help="Linear velocity cost weight.")
parser.add_argument("--cost_omega", type=float, default=0.0, help="Angular velocity cost weight.")
parser.add_argument("--cost_u", type=float, default=0.0, help="Control effort cost weight.")

parser.add_argument("--mass", type=float, default=34.0, help="Platform mass [kg].")
parser.add_argument("--j_zz", type=float, default=0.4343, help="Yaw inertia [kg*m^2].")
parser.add_argument("--motor_thrust", type=float, default=None, help="Max thrust per motor [N].")
parser.add_argument("--motor_arm", type=float, default=0.25, help="Motor arm length [m].")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
import Isaaclab_RANSv2  # noqa: F401
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.agents.mppi_control import MPPIController, MPPIWeights
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.agents.straight_line_guidance import StraightLineGuidance
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.agents.task_obs_to_lqr import TaskObsToLqrStateConverter
from isaaclab_tasks.utils.hydra import hydra_task_config


def _obs_to_tensor(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        return next(iter(obs.values()))
    return obs


def _extract_goal(task_eval: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if "target_position" in task_eval:
        goal_pos = task_eval["target_position"][:, :2]
    elif "target_positions" in task_eval and "target_index" in task_eval:
        goal_positions = task_eval["target_positions"]
        idx = task_eval["target_index"].long()
        idx = torch.clamp(idx, min=0, max=goal_positions.shape[1] - 1)
        env_ids = torch.arange(goal_positions.shape[0], device=goal_positions.device)
        goal_pos = goal_positions[env_ids, idx, :2]
    else:
        raise KeyError(
            "Task eval_data must provide either 'target_position' or ('target_positions' and 'target_index')."
        )

    goal_heading = None
    if "target_heading" in task_eval:
        goal_heading = task_eval["target_heading"]
    elif "target_headings" in task_eval and "target_index" in task_eval:
        goal_headings = task_eval["target_headings"]
        idx = task_eval["target_index"].long()
        idx = torch.clamp(idx, min=0, max=goal_headings.shape[1] - 1)
        env_ids = torch.arange(goal_headings.shape[0], device=goal_headings.device)
        goal_heading = goal_headings[env_ids, idx]

    return goal_pos, goal_heading


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.unsqueeze(0)
    return t


def _ensure_1d(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        return t.unsqueeze(0)
    if t.ndim == 2 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    if not hasattr(env.unwrapped, "robot_api") or not hasattr(env.unwrapped, "task_api"):
        raise RuntimeError("This MPPI player expects AutoEnvGen-style env with robot_api/task_api exposed.")

    obs, _ = env.reset()
    step = 0
    num_envs = env_cfg.scene.num_envs

    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("This script supports only Box action spaces.")
    if env.action_space.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected action dimension 3 or 4, got {env.action_space.shape[-1]}"
        )

    robot_api = env.unwrapped.robot_api
    task_api = env.unwrapped.task_api

    motor_thrust = args_cli.motor_thrust
    if motor_thrust is None:
        motor_thrust = float(getattr(robot_api._robot_cfg, "max_thrust", 1.0))

    weights = MPPIWeights(
        pos=args_cli.cost_pos,
        heading=args_cli.cost_heading,
        vel=args_cli.cost_vel,
        omega=args_cli.cost_omega,
        u=args_cli.cost_u,
    )

    mppi_specs = {
        "number_of_sampled_trajectories": args_cli.mppi_samples,
        "number_of_iterations_per_sample": args_cli.mppi_horizon,
        "noise_std_dev_pwm_x": args_cli.noise_std_x,
        "noise_std_dev_pwm_y": args_cli.noise_std_y,
        "noise_std_dev_pwm_theta": args_cli.noise_std_yaw,
        "noise_std_dev_pwm_rw": args_cli.noise_std_rw,
        "temperature": args_cli.temperature,
    }

    controller = MPPIController(
        mppi_specs,
        device=str(env_cfg.sim.device),
        num_envs=num_envs,
        action_dim=env.action_space.shape[-1],
        dt=dt,
        mass=args_cli.mass,
        j_zz=args_cli.j_zz,
        motor_thrust=motor_thrust,
        motor_arm=args_cli.motor_arm,
        weights=weights,
    )

    guidance = None
    if args_cli.ref_lookahead > 0.0:
        guidance = StraightLineGuidance(
            num_envs=num_envs,
            device=str(env_cfg.sim.device),
            lookahead_distance=args_cli.ref_lookahead,
        )

    obs_to_lqr = None
    if args_cli.use_obs_goal:
        converter_task_name = args_cli.task
        if hasattr(env_cfg, "task_name"):
            converter_task_name = getattr(env_cfg, "task_name")
        elif hasattr(env_cfg, "env") and hasattr(env_cfg.env, "task_name"):
            converter_task_name = getattr(env_cfg.env, "task_name")

        obs_to_lqr = TaskObsToLqrStateConverter(
            task_name=converter_task_name,
            task_type=args_cli.obs_task_type,
            device=str(env_cfg.sim.device),
        )

    print("MPPI tracking enabled.")

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            obs_tensor = _obs_to_tensor(obs)

            pos_w = _ensure_2d(robot_api.root_link_pos_w)
            vel_b = _ensure_2d(robot_api.root_link_lin_vel_b)
            yaw_w = _ensure_1d(robot_api.heading_w)
            omega_b = _ensure_2d(robot_api.root_link_ang_vel_b)[:, 2]

            if obs_to_lqr is not None:
                obs_tensor = _ensure_2d(obs_tensor)
                lqr_state = obs_to_lqr(obs_tensor)
                pos_err_b = lqr_state[:, 4:6]
                c = torch.cos(yaw_w)
                s = torch.sin(yaw_w)
                pos_err_w = torch.stack(
                    (c * pos_err_b[:, 0] - s * pos_err_b[:, 1], s * pos_err_b[:, 0] + c * pos_err_b[:, 1]),
                    dim=-1,
                )
                goal_pos = pos_w[:, :2] - pos_err_w
                goal_heading = yaw_w - lqr_state[:, 3]
            else:
                task_eval = task_api.eval_data
                goal_pos, goal_heading = _extract_goal(task_eval)
                goal_pos = _ensure_2d(goal_pos)
                if goal_heading is not None:
                    goal_heading = _ensure_1d(goal_heading)

            if guidance is not None and args_cli.use_obs_goal:
                _, debug = guidance.compute_position_error_body(
                    pos_w=pos_w[:, :2],
                    heading_w=yaw_w,
                    goal_pos_w=goal_pos,
                )
                goal_pos = torch.stack((debug["ref_pos_x"], debug["ref_pos_y"]), dim=-1)

            actions = controller.compute_control(
                pos_w=pos_w,
                lin_vel_b=vel_b,
                yaw_w=yaw_w,
                ang_vel_b=omega_b,
                goal_pos_w=goal_pos,
                goal_heading_w=goal_heading,
                dt=dt,
            )

            low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=actions.device)
            high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=actions.device)
            actions = torch.max(torch.min(actions, high), low)

            obs, _, terminated, truncated, _ = env.step(actions)

        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break

        if torch.any(terminated) or torch.any(truncated):
            obs, _ = env.reset()
            controller.reset()

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
