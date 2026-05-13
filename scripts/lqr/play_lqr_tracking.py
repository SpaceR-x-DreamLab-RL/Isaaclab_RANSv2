# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-LQR player with moving straight-line references.

This script performs trajectory tracking by feeding a moving reference each timestep.
"""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run task with Warp-LQR and straight-line tracking references.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many steps. 0 means run until app closes.")
parser.add_argument(
    "--k_matrix_csv",
    type=str,
    default="source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/agents/K_LQR_no_wheel.csv",
    help="Path to LQR gain matrix CSV (expected shape: 4x9).",
)
parser.add_argument("--ref_lookahead", type=float, default=0.075, help="Straight-line lookahead distance [m].")
parser.add_argument(
    "--obs_task_type",
    type=str,
    default="auto",
    choices=["auto", "go_to_pose", "go_to_position", "go_through_positions", "go_through_poses"],
    help="Task observation layout used to build base LQR state. 'auto' infers from task.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
import Isaaclab_RANSv2  # noqa: F401
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.agents.lqr_control import LQRController
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
    """Extract current goal position and optional heading from task eval data."""
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
        raise RuntimeError("This tracking player expects AutoEnvGen-style env with robot_api/task_api exposed.")

    obs, _ = env.reset()
    step = 0
    num_envs = env_cfg.scene.num_envs

    controller = LQRController(
        gain_matrix_path=args_cli.k_matrix_csv,
        device=str(env_cfg.sim.device),
        num_envs=num_envs,
    )

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

    guidance = StraightLineGuidance(
        num_envs=num_envs,
        device=str(env_cfg.sim.device),
        lookahead_distance=args_cli.ref_lookahead,
    )

    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("This script supports only Box action spaces.")
    if env.action_space.shape[-1] != 4:
        raise ValueError(f"Expected action dimension 4, got {env.action_space.shape[-1]}")

    robot_api = env.unwrapped.robot_api
    task_api = env.unwrapped.task_api

    # Initialize path anchors from current state.
    task_eval = task_api.eval_data
    goal_pos, _ = _extract_goal(task_eval)
    # guidance.reset_paths(robot_api.root_link_pos_w[:, :2], goal_pos)

    print(f"Tracking mode enabled. Base obs converter: {obs_to_lqr.task_type}")

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            obs_tensor = _obs_to_tensor(obs)
            lqr_state = obs_to_lqr(obs_tensor)

            task_eval = task_api.eval_data
            goal_pos, _ = _extract_goal(task_eval)

            pos_err_b, _dbg = guidance.compute_position_error_body(
                pos_w=robot_api.root_link_pos_w[:, :2],
                heading_w=robot_api.heading_w,
                goal_pos_w=goal_pos,
            )

            # Keep velocity/theta channels from obs-based LQR mapping,
            # but overwrite position channels with local trajectory-tracking errors.
            lqr_state[:, 4:6] = pos_err_b

            lqr_actions = controller.compute_control(lqr_state, dt)

            actions = torch.zeros((num_envs, env.action_space.shape[-1]), dtype=torch.float32, device=lqr_actions.device)
            actions[:, 0] = lqr_actions[:, 0]
            actions[:, 1] = lqr_actions[:, 1]
            actions[:, 2] = lqr_actions[:, 2]
            actions[:, -1] = lqr_actions[:, 3]

            low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=actions.device)
            high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=actions.device)
            actions = torch.max(torch.min(actions, high), low)

            obs, _, terminated, truncated, _ = env.step(actions)

        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break

        if torch.any(terminated) or torch.any(truncated):
            obs, _ = env.reset()
            controller.reset_integrals()
            task_eval = task_api.eval_data
            goal_pos, _ = _extract_goal(task_eval)
            guidance.reset_paths(robot_api.root_link_pos_w[:, :2], goal_pos)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
