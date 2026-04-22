"""Launch an Isaac Lab env as a ZMQ server for DreamerV3.

Run with Isaac Sim's Python (Terminal 1):
    ${ISAAC_SIM_PYTHON} scripts/dreamer/sim_server.py \\
        --robot Pingu --task-name GoToPose \\
        --num_envs 1 --socket /tmp/isaaclab_dreamer.sock --headless

Then start DreamerV3 training in a second terminal (Terminal 2):
    ${DREAMER_VENV}/bin/python scripts/dreamer/train.py \\
        --logdir logs/dreamer/pingu_gotoPose \\
        --socket /tmp/isaaclab_dreamer.sock
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Serve an Isaac Lab env over ZMQ for DreamerV3.")
parser.add_argument("--task", type=str, default="Isaaclab-RANSv2-AutoEnvGen-v0", help="Gym env id.")
parser.add_argument("--robot", type=str, default="Pingu", help="Robot name (e.g. Pingu, FloatingPlatform).")
parser.add_argument("--task-name", type=str, default="GoToPose", help="Task name (e.g. GoToPose, GoToPosition).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--socket", type=str, default="/tmp/isaaclab_dreamer.sock", help="IPC socket path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- all Isaac Sim imports must come AFTER AppLauncher ---
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import Isaaclab_RANSv2.tasks  # noqa: F401, E402
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.environments.auto_env_gen_cfg import AutoEnvGenCfg  # noqa: E402
from Isaaclab_RANSv2.tasks.direct.isaaclab_ransv2.utils.zmq_env_server import ZmqEnvServer  # noqa: E402


def main():
    env_cfg = AutoEnvGenCfg()
    env_cfg.robot_name = args_cli.robot
    env_cfg.task_name = args_cli.task_name
    env_cfg.scene.num_envs = args_cli.num_envs

    print(f"[sim_server] Robot={env_cfg.robot_name}  Task={env_cfg.task_name}  num_envs={args_cli.num_envs}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    server = ZmqEnvServer(env, socket_path=args_cli.socket)
    server.serve()  # blocks until client sends 'close'
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
