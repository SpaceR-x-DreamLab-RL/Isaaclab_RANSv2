"""Train DreamerV3 against an Isaac Lab env via the ZMQ bridge.

Run with the DreamerV3 venv (after sim_server.py is already running):
    ${DREAMER_VENV}/bin/python scripts/dreamer/train.py \\
        --logdir logs/dreamer/pingu_gotoPose \\
        --steps 2000000 \\
        --configs defaults size12m \\
        --socket /tmp/isaaclab_dreamer.sock \\
        --wandb
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import gymnasium as gym

# Load ZmqEnvClient directly by file path — avoids triggering the Isaac Lab
# package __init__.py chain (which imports isaaclab_tasks, not in dreamer venv).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CLIENT_FILE = os.path.join(
    _REPO_ROOT, "source", "Isaaclab_RANSv2", "Isaaclab_RANSv2",
    "tasks", "direct", "isaaclab_ransv2", "utils", "zmq_env_client.py",
)
_spec = importlib.util.spec_from_file_location("zmq_env_client", _CLIENT_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ZmqEnvClient = _mod.ZmqEnvClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket",  type=str,  default="/tmp/isaaclab_dreamer.sock")
    parser.add_argument("--logdir",  type=str,  default="logs/dreamer")
    parser.add_argument("--steps",   type=int,  default=int(2e6))
    # Available size presets: size1m, size12m, size50m, size200m, size600m
    parser.add_argument("--configs", nargs="+", default=["defaults", "size12m"])
    parser.add_argument("--wandb",   action="store_true", default=False,
                        help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    if args.wandb:
        # Derive a clean run name from the last path component of logdir
        # e.g. "logs/dreamer/pingu_gotoPose" → "dreamer_pingu_gotoPose"
        run_name = "dreamer_" + args.logdir.rstrip("/").split("/")[-1]
        os.environ.setdefault("WANDB_PROJECT", "AutoEnvGen_PPO")
        os.environ.setdefault("WANDB_ENTITY",  "spacer-rl")
        os.environ.setdefault("WANDB_NAME",    run_name)

    # Register our ZmqEnvClient as a standard gymnasium env so that
    # dreamerv3's own make_env() can find it via "gymnasium.IsaacLabBridge-v0".
    socket_path = args.socket
    gym.register(
        id="IsaacLabBridge-v0",
        entry_point=lambda **kwargs: ZmqEnvClient(socket_path=socket_path),
        disable_env_checker=True,
    )

    # Append {timestamp} placeholder so dreamerv3's main.py fills it in,
    # giving each run a unique directory (e.g. logs/dreamer/foo_20260422_120000).
    logdir = args.logdir.rstrip("/") + "_{timestamp}"

    # Build the argv list that dreamerv3/main.py expects, then hand off to it.
    # dreamerv3 uses elements.Flags for parsing — nested keys use dot notation.
    dreamer_argv = [
        "--task",   "gymnasium_IsaacLabBridge-v0",
        "--logdir", logdir,
        f"--run.steps={args.steps}",
    ]
    for name in args.configs:
        dreamer_argv += ["--configs", name]

    if args.wandb:
        # elements.Flags splits on comma when default is a tuple — no brackets,
        # no spaces, or they become part of the element strings.
        dreamer_argv += ["--logger.outputs", "jsonl,scope,wandb"]

    from dreamerv3 import main as dreamer_main
    dreamer_main.main(dreamer_argv)


if __name__ == "__main__":
    main()
