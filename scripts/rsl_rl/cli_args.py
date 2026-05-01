# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm name for run naming (e.g. ppo, ppo-discrete). Default derived from agent config.",
    )
    
def _auto_detect_entry_point(task_name: str, cfg_yaml: dict) -> str | None:
    """Pick the rsl_rl ``*_cfg_entry_point`` whose registry default matches the saved yaml.

    Compares the yaml's ``actor.class_name`` (and ``actor.distribution_cfg.class_name`` if
    present) against each registered entry point's registry default. This avoids having
    to pass ``--algorithm`` at eval time when the checkpoint already encodes the model
    architecture and distribution.
    """
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    actor_yaml = cfg_yaml.get("actor")
    if not isinstance(actor_yaml, dict):
        return None
    yaml_actor_class = actor_yaml.get("class_name")
    yaml_dist_class = None
    yaml_dist = actor_yaml.get("distribution_cfg")
    if isinstance(yaml_dist, dict):
        yaml_dist_class = yaml_dist.get("class_name")

    spec_kwargs = gym.spec(task_name.split(":")[-1]).kwargs
    candidates = [
        k for k in spec_kwargs if k.startswith("rsl_rl") and k.endswith("_cfg_entry_point")
    ]

    best = None
    for key in candidates:
        try:
            cfg = load_cfg_from_registry(task_name, key)
        except Exception:
            continue
        actor = getattr(cfg, "actor", None)
        actor_class = getattr(actor, "class_name", None) if actor is not None else None
        if yaml_actor_class is not None and actor_class != yaml_actor_class:
            continue
        # if the yaml has a distribution_cfg, prefer entry points whose default also matches
        actor_dist = getattr(actor, "distribution_cfg", None)
        actor_dist_class = getattr(actor_dist, "class_name", None) if actor_dist is not None else None
        if yaml_dist_class is not None and actor_dist_class is not None and actor_dist_class != yaml_dist_class:
            # remember as a weak fallback in case nothing matches both fields
            best = best or key
            continue
        return key
    return best


def load_rsl_rl_cfg(
    checkpoint_path: str, task_name: str, entry_point_key: str = "rsl_rl_cfg_entry_point"
) -> RslRlOnPolicyRunnerCfg:
    """Load configuration for RSL-RL agent from checkpoint.

    Loads the saved ``agent.yaml`` and recursively merges it onto a freshly-instantiated
    runner cfg from the gym registry, so nested fields stay as typed configclass instances
    instead of plain dicts. This is required so ``handle_deprecated_rsl_rl_cfg`` can use
    attribute access on ``agent_cfg.actor`` / ``agent_cfg.critic`` / ``agent_cfg.policy``.
    Using ``cfg_cls(**yaml)`` (as in ``load_cfg_from_checkpoint``) does not recurse and
    leaves nested fields as dicts, which trips the deprecation handler.

    The registry default must use the same model class as the checkpoint (e.g. RNNModel
    for an RNN run), otherwise ``from_dict`` raises ``KeyError`` on architecture-specific
    fields. ``entry_point_key`` is treated as a hint only — if it doesn't match the yaml's
    ``actor.class_name``, an auto-detected entry point is used instead.

    Args:
        checkpoint_path: The path to the checkpoint file.
        task_name: The gym task id used to look up the cfg entry point.
        entry_point_key: Preferred cfg entry point. Falls back to auto-detection from the
            yaml's ``actor.class_name`` / ``distribution_cfg.class_name`` when this hint
            doesn't match the checkpoint.

    Returns:
        The loaded configuration for RSL-RL agent.
    """
    import os

    import yaml
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    config_file_path = os.path.join(os.path.dirname(checkpoint_path), "params", "agent.yaml")
    if not os.path.exists(config_file_path):
        raise ValueError(f"Could not find configuration file '{config_file_path}' in the checkpoint directory.")
    print(f"[INFO]: Loading configuration from: {config_file_path}")
    with open(config_file_path, encoding="utf-8") as f:
        cfg_yaml = yaml.full_load(f)

    # verify the supplied hint matches the yaml; otherwise auto-detect from actor class
    detected = _auto_detect_entry_point(task_name, cfg_yaml)
    if detected is not None and detected != entry_point_key:
        print(
            f"[INFO]: Overriding entry point '{entry_point_key}' with auto-detected"
            f" '{detected}' based on actor class in the saved yaml."
        )
        entry_point_key = detected

    # start from a freshly-typed registry default, then recursively overlay the yaml
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, entry_point_key)

    # `from_dict` recurses into nested mappings and requires every yaml key to already
    # exist on the target. For free-form dict fields like `wandb_kwargs` the yaml may
    # carry extra keys (e.g. `name` injected by train.py's wandb naming) that the
    # registry default doesn't have. Replace those wholesale before `from_dict`.
    for free_form_field in ("wandb_kwargs",):
        if isinstance(cfg_yaml.get(free_form_field), dict):
            setattr(rslrl_cfg, free_form_field, cfg_yaml.pop(free_form_field))

    # `distribution_cfg` is polymorphic (Gaussian / Heteroscedastic / Beta) and the
    # registry default may use a different subclass than the saved checkpoint. Recursing
    # into a Gaussian instance with Beta-specific keys (or vice versa) raises KeyError.
    # Pop and assign as a plain dict — the runner consumes `to_dict()` anyway, and the
    # deprecation handler only checks `distribution_cfg is not None`.
    for model_name in ("actor", "critic"):
        model_yaml = cfg_yaml.get(model_name)
        if isinstance(model_yaml, dict) and "distribution_cfg" in model_yaml and hasattr(rslrl_cfg, model_name):
            setattr(getattr(rslrl_cfg, model_name), "distribution_cfg", model_yaml.pop("distribution_cfg"))

    rslrl_cfg.from_dict(cfg_yaml)
    return rslrl_cfg


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
