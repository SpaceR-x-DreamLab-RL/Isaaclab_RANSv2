"""Visualize a trained DreamerV3 policy by resuming from checkpoint with GUI.

NOTE: This fork (AndrejOrsula/dreamerv3 @ 4049794d) does not have an eval_only
mode — dreamerv3/main.py hardwires embodied.run.train.  The only way to watch
the trained policy is to resume training from the checkpoint with Isaac Sim
rendering enabled (no --headless on the sim server side).  Gradient updates
continue but the policy IS the trained one, so you see learned behaviour.

For pure metrics (no rendering) use TensorBoard or parse metrics.jsonl instead.

Usage
-----
Window 1 — sim server WITH GUI (drop --headless):
    ${ISAAC_SIM_PYTHON} scripts/dreamer/sim_server.py \\
        --robot Pingu --task-name GoToPose --num_envs 1 \\
        --socket /tmp/isaaclab_dreamer.sock

Window 2 — resume from checkpoint:
    ${DREAMER_VENV}/bin/python scripts/dreamer/play.py \\
        --logdir logs/dreamer/pingu_gotoPose_<timestamp> \\
        --steps 5000 \\
        --configs defaults size12m \\
        --socket /tmp/isaaclab_dreamer.sock

Pass the EXACT timestamped logdir from training so the checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import gymnasium as gym


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
    parser.add_argument("--socket",  type=str, default="/tmp/isaaclab_dreamer.sock")
    parser.add_argument("--logdir",  type=str, required=True,
                        help="EXACT timestamped logdir from training (checkpoint is loaded from here).")
    parser.add_argument("--steps",   type=int, default=5000,
                        help="Steps to run (watch this many steps then exit).")
    parser.add_argument("--configs", nargs="+", default=["defaults", "size12m"],
                        help="Must match the configs used during training.")
    args = parser.parse_args()

    socket_path = args.socket
    gym.register(
        id="IsaacLabBridge-v0",
        entry_point=lambda **kwargs: ZmqEnvClient(socket_path=socket_path),
        disable_env_checker=True,
    )

    dreamer_argv = [
        "--task",   "gymnasium_IsaacLabBridge-v0",
        "--logdir", args.logdir,          # resumes checkpoint from this exact dir
        f"--run.steps={args.steps}",
    ]
    for name in args.configs:
        dreamer_argv += ["--configs", name]

    from dreamerv3 import main as dreamer_main
    dreamer_main.main(dreamer_argv)


if __name__ == "__main__":
    main()
