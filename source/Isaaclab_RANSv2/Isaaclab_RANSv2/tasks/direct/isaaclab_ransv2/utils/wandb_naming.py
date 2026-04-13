# Copyright (c) 2022-2026, The Isaac Lab Project Developers. (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utils for Weights & Biases (wandb) run naming and project layout.

By default, runs are logged in a library-agnostic way:
- Project: (wandb project / experiment folder): environment name (e.g. "GoToPosition-Cubo").
- Run name: <date-time>_<algorithm>_<robot_name>_<task_name>_<learning_library>

Sub-projects can override these helpers or the call sites to change behavior.
"""

from __future__ import annotations

from typing import Any


def get_wandb_env_name(env_cfg: Any, task_id: str | None = None) -> str:
    """Return the environment name used as wandb project and experiment folder.

    Same name regardless of rsl_rl, skrl, rl_games, etc.

    Args:
        env_cfg: Environment config. If it has robot_name and task_name (e.g. AutoEnvGenCfg),
            returns "{task_name}-{robot_name}". Otherwise falls back to task_id or "unknown".
        task_id: Optional task identifier (e.g. args_cli.task) when env_cfg has no robot/task.

    Returns:
        A string, for example "GoToPosition-Cubo", or the task_id / "unknown".
    """
    robot_name = getattr(env_cfg, "robot_name", None)
    task_name = getattr(env_cfg, "task_name", None)
    if robot_name is not None and task_name is not None:
        return f"{task_name}-{robot_name}"
    if task_id:
        return task_id
    return "unknown"


def get_wandb_run_name(
    date_time_str: str,
    algorithm: str,
    robot_name: str,
    task_name: str,
    learning_library: str,
) -> str:
    """Build the default wandb run name and log-dir run segment.

    Format: <date-time>_<algorithm>_<robot_name>_<task_name>_<learning_library>

    Args:
        date_time_str: Timestamp string (e.g. from datetime.now().strftime).
        algorithm: Algorithm name (e.g. from --algorithm: "ppo", "ppo-discrete").
        robot_name: Robot name from env (e.g. env_cfg.robot_name).
        task_name: Task name from env (e.g. env_cfg.task_name).
        learning_library: Library identifier, e.g. "rsl_rl", "skrl-torch", "rl_games".

    Returns:
        Full run name string.
    """
    return f"{date_time_str}_{algorithm}_{robot_name}_{task_name}_{learning_library}"


def get_robot_and_task_from_env_cfg(env_cfg: Any) -> tuple[str, str]:
    """Get robot_name and task_name from env config, with fallbacks."""
    robot_name = getattr(env_cfg, "robot_name", None) or "unknown"
    task_name = getattr(env_cfg, "task_name", None) or "unknown"
    return str(robot_name), str(task_name)
