# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal Warp-LQR player."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run task with minimal Warp-LQR control.")
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
    default=(
        "source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/agents/K_LQR.csv"
    ),
    help="Path to LQR gain matrix CSV (expected shape: 4x9).",
)
parser.add_argument(
    "--obs_task_type",
    type=str,
    default="auto",
    choices=["auto", "go_to_pose", "go_to_position", "go_through_positions", "go_through_poses"],
    help="Task observation layout used to build LQR state. 'auto' infers from --task.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)

import isaaclab_tasks  # noqa: F401
import Isaaclab_RANSv2  # noqa: F401
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.agents.lqr_control import LQRController
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.agents.task_obs_to_lqr import TaskObsToLqrStateConverter
from isaaclab_tasks.utils.hydra import hydra_task_config


def _obs_to_tensor(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        return next(iter(obs.values()))
    return obs


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: dict):
    """Run environment with Warp LQR controller."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.reset()
    step = 0
    num_envs = env_cfg.scene.num_envs

    # ====================
    # masses = env.unwrapped.robot.root_physx_view.get_masses()  # [num_envs, num_bodies]
    # print("Mass tensor shape:", tuple(masses.shape))
    # print("Total vehicle mass per env [kg]:", masses.sum(dim=1))
    # base_idx, _ = env.unwrapped.robot.find_bodies(["base_link"])
    # print("base_link mass per env [kg]:", masses[:, base_idx].squeeze(-1))
    # body_names = env.unwrapped.robot.body_names

    # masses = env.unwrapped.robot.root_physx_view.get_masses()[0] # first env
    # for i, (name, m) in enumerate(zip(body_names, masses.tolist())):
    #     print(i, name, m)
    # print("sum:", float(masses.sum()))
    # ====================

    controller = LQRController(
        gain_matrix_path=args_cli.k_matrix_csv,
        device=str(env_cfg.sim.device),
        num_envs=num_envs,
    )

    # Prefer resolved task name from env config (e.g. AutoEnvGen env.task_name override),
    # then fall back to the gym task id.
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
    print(f"Using observation converter layout: {obs_to_lqr.task_type}")

    if isinstance(env.action_space, gym.spaces.Box):
        if env.action_space.shape[-1] != 4:
            raise ValueError(f"Expected action dimension 4, got {env.action_space.shape[-1]}")
    else:
        raise ValueError("This minimal script currently supports only Box action spaces.")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            obs_tensor = _obs_to_tensor(obs)
            lqr_state = obs_to_lqr(obs_tensor)
            print(f"LQR state: {lqr_state}")
            lqr_actions = controller.compute_control(lqr_state, dt)

            # Map controller outputs to robot action semantics when direct_thruster_control=False:
            # actions[:, 0]  -> forward/backward thrust
            # actions[:, 1]  -> left/right thrust
            # actions[:, 2]  -> yaw thrust
            # actions[:, -1] -> reaction wheel
            actions = torch.zeros((num_envs, env.action_space.shape[-1]), dtype=torch.float32, device=lqr_actions.device)
            # make first action be 0.1
            actions[:, 0] = lqr_actions[:, 0]
            actions[:, 1] = lqr_actions[:, 1]
            actions[:, 2] = lqr_actions[:, 2]
            actions[:, -1] = lqr_actions[:, 3]

            if isinstance(env.action_space, gym.spaces.Box):
                low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=actions.device)
                high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=actions.device)
                actions = torch.max(torch.min(actions, high), low)

            print(f"Actions: {actions}")
            obs, _, terminated, truncated, _ = env.step(actions)

        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            break

        if torch.any(terminated) or torch.any(truncated):
            obs, _ = env.reset()
            controller.reset_integrals()

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
