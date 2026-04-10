# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with ROS2 bridge."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import os
from tensordict import TensorDict

# add argparse arguments
parser = argparse.ArgumentParser(description="Deploy an RL agent with RSL-RL via ROS2 bridge.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--overload-experiment-cfg", 
    action="store_true", 
    default=True, help="Overload experiment config. If set to True, it will load the cfg of the model that was used for training."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--ros_bridge",
    action="store_true",
    default=True,
    help="Enable ROS2 bridge mode for deployment testing.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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
import os
import time
import torch

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
import threading
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import Isaaclab_RANSv2.tasks  # noqa: F401


class IsaacLabROSNode(Node):
    """ROS2 node for Isaac Lab to communicate with RANS_DeployToRobot.
    
    Handles padded format:
    - obs: Task-specific observation + embedded task encoder
    - taskID_obs: One-hot task ID encoder
    - Action publishing back to RANS_DeployToRobot_DeployToRobot
    """
    
    def __init__(self, env_cfg, agent_cfg, resume_path, device, task_name):
        super().__init__('isaac_lab_eval')
        
        # Subscribers
        self.observation_sub = self.create_subscription(
            Float64MultiArray, 
            'isaac_lab/formatted_observation',
            self.obs_callback,
            10
        )
        
        # Publishers (Isaac Lab -> RANS_DeployToRobot)
        self.action_pub = self.create_publisher(
            Float64MultiArray,  # Actions as array
            'isaac_lab/action',
            10
        )
        
        # State variables
        self.latest_obs = None
        self.policy = None
        self.env = None
        self.num_tasks = 3  #GoToPose, taskIDVelocity, Rendezvous
        self.device = device
        self.task_name = task_name
        
        # Initialize environment and policy
        self._initialize_environment_and_policy(env_cfg, agent_cfg, resume_path)
        
        self.get_logger().info('Isaac Lab ROS Node initialized for deployment inference')
        
    def _initialize_environment_and_policy(self, env_cfg, agent_cfg, resume_path):
        """Initialize environment and load policy for deployment inference."""
        try:
            # Create environment using the task name
            self.env = gym.make(self.task_name, cfg=env_cfg, render_mode=None)
            
            # Convert to single-agent instance if required
            if isinstance(self.env.unwrapped, DirectMARLEnv):
                self.env = multi_agent_to_single_agent(self.env)
            
            # Wrap for rsl-rl
            self.env = RslRlVecEnvWrapper(self.env)
            
            # Load policy
            ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            ppo_runner.load(resume_path)
            self.policy = ppo_runner.get_inference_policy(device=self.device)
            
            # Reset environment to initialize internal state
            # Imp. for proper environment initialization
            obs = self.env.get_observations()
            self.get_logger().info(f'Environment reset successful. Observation shape: {obs["obs"].shape if isinstance(obs, dict) else "unknown"}')
            
            self.get_logger().info(f'Environment and policy loaded successfully. Device: {self.device}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize environment and policy: {str(e)}')
            raise
        
    def obs_callback(self, msg):
        """Receive obs from RANS_DeployToRobot bridge."""
        try:
            # Convert ROS message to tensor
            obs_np = np.array(msg.data, dtype=np.float32)
            obs = torch.from_numpy(obs_np).to(self.device)
            
            # Reshape if needed (assuming batch size of 1)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            self.latest_obs = obs
            self.get_logger().info(f'Received obs from RANS: shape={obs.shape}')
            
            # Run inference if we have all observations
            if self.policy is not None:
                self._run_inference()
                
        except Exception as e:
            self.get_logger().error(f'Error in obs_callback: {str(e)}')
            
    def _run_inference(self):
        """Run policy inference with the latest observation and taskID_obs."""
        try:
            with torch.inference_mode():
                # Create observation dictionary as expected by RSL_RL v3.3.0
                # get_actor_obs indexes obs by obs_groups["policy"] keys (e.g. "policy")
                obs = {"policy": self.latest_obs}

                # Debug: Log observation shapes
                self.get_logger().info(f'Policy inference: obs.shape={self.latest_obs.shape}')
                # Policy should be initialized by this point
                if self.policy is not None:
                    action = self.policy(obs)
                    self.env.unwrapped._pre_physics_step(action)
                    actions = self.env.unwrapped.robot_api._actions

                    self.get_logger().info(f'Policy output: action.shape={actions.shape}, values={actions.cpu().numpy()}')
                    self.publish_action(actions.squeeze(0))
                else:
                    self.get_logger().error('Policy is None during inference')
                
        except Exception as e:
            import traceback
            self.get_logger().error(f'Error in policy inference: {str(e)}\n{traceback.format_exc()}')
            # Publish zero action as fallback
            if self.latest_obs is not None:
                # Use the correct action space from the environment config
                action_dim = self.env.unwrapped.cfg.action_space if hasattr(self.env.unwrapped, 'cfg') else 9  # thrusters and reaction wheel
                self.get_logger().info(f'Fallback action: action_dim={action_dim}, env_cfg.action_space={getattr(self.env.unwrapped.cfg, "action_space", "N/A") if hasattr(self.env.unwrapped, "cfg") else "N/A"}')
                fallback_action = torch.zeros((self.latest_obs.shape[0], action_dim), device=self.latest_obs.device)
                self.publish_action(fallback_action.squeeze(0))
            
    def publish_action(self, action: torch.Tensor):
        """Publish action to RANS_DeployToRobot framework.
        
        Args:
            action: Action tensor from policy
        """
        try:
            msg = Float64MultiArray()
            
            # Convert tensor to flat array
            action_np = action.cpu().numpy().flatten()
            msg.data = action_np.tolist()
            
            # Add metadata about action structure
            msg.layout.dim.append(  # type: ignore
                MultiArrayDimension(
                    label="action",
                    size=len(action_np),
                    stride=len(action_np)
                )
            )
            
            self.action_pub.publish(msg)
            self.get_logger().info(f'Published action to RANS: shape={action.shape}, values={action.cpu().numpy()}')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing action: {str(e)}')


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    
    # Load environment config from checkpoint if available, otherwise parse from registry
    if args_cli.checkpoint:
        try:
            # Try to load environment config from checkpoint first
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_checkpoint
            checkpoint_env_cfg = load_cfg_from_checkpoint(args_cli.checkpoint, args_cli.task, "env_cfg_entry_point")
            print(f"[INFO]: Loaded environment configuration from checkpoint")
            
            # Override the Hydra-loaded config with checkpoint values
            if hasattr(checkpoint_env_cfg, 'observation_space'):
                env_cfg.observation_space = checkpoint_env_cfg.observation_space
                print(f"[INFO]: Set observation space from checkpoint: {env_cfg.observation_space}")
            if hasattr(checkpoint_env_cfg, 'action_space'):
                env_cfg.action_space = checkpoint_env_cfg.action_space
            if hasattr(checkpoint_env_cfg, 'state_space'):
                env_cfg.state_space = checkpoint_env_cfg.state_space
            if hasattr(checkpoint_env_cfg, 'gen_space'):
                env_cfg.gen_space = checkpoint_env_cfg.gen_space
            if hasattr(checkpoint_env_cfg, 'robot_name'):
                env_cfg.robot_name = checkpoint_env_cfg.robot_name
            if hasattr(checkpoint_env_cfg, 'tasks_names'):
                env_cfg.tasks_names = checkpoint_env_cfg.tasks_names
            if hasattr(checkpoint_env_cfg, 'type_of_training'):
                env_cfg.type_of_training = checkpoint_env_cfg.type_of_training
            
            # Override device and num_envs from CLI args
            if hasattr(env_cfg, 'sim'):
                if args_cli.device is not None:
                    env_cfg.sim.device = args_cli.device
                env_cfg.sim.use_fabric = not args_cli.disable_fabric
            if hasattr(env_cfg, 'scene'):
                if args_cli.num_envs is not None:
                    env_cfg.scene.num_envs = args_cli.num_envs
            
        except Exception as e:
            print(f"[WARNING]: Could not load environment config from checkpoint: {e}")
            print(f"[INFO]: Using default environment config from registry")
    
    if args_cli.overload_experiment_cfg:
        if args_cli.checkpoint:
            agent_cfg = cli_args.load_rsl_rl_cfg(args_cli.checkpoint, args_cli.task)
        else:
            raise ValueError("Missing checkpoint path for loading the experiment config.")
    else:
        # parse agent configuration
        agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Check if ROS2 bridge mode is enabled
    if args_cli.ros_bridge:
        print("[INFO] ROS2 bridge mode enabled for deployment inference")
        
        # Initialize ROS2
        rclpy.init()
        ros_node = IsaacLabROSNode(env_cfg, agent_cfg, resume_path, env_cfg.sim.device, args_cli.task)
        
        # Start ROS2 in a separate thread
        ros_executor = rclpy.executors.MultiThreadedExecutor()  # type: ignore
        ros_executor.add_node(ros_node)
        ros_thread = threading.Thread(target=ros_executor.spin, daemon=True)
        ros_thread.start()
        
        print("[INFO] Isaac Lab deployment inference with ROS2 bridge started")
        print("[INFO] ROS2 Topics:")
        print("  Subscribers (RANS_DeployToRobot Bridge -> Isaac Lab):")
        print("    - isaac_lab/formatted_observation (obs)")
        print("  Publishers (Isaac Lab -> RANS_DeployToRobot Bridge):")
        print("    - isaac_lab/action (policy actions)")
        print("[INFO] Waiting for observations from RANS_DeployToRobot bridge...")
        
        # Keep the script running to handle ROS2 communication
        try:
            while rclpy.ok():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[INFO] Shutting down...")
        finally:
            ros_executor.shutdown()
            ros_node.destroy_node()
            rclpy.shutdown()
            simulation_app.close()
        return
    
    # If not in ROS bridge mode, this script is for deployment only
    print("[ERROR] This script is designed for deployment inference with ROS2 bridge only.")
    print("[ERROR] Please use --ros_bridge flag to enable deployment mode.")
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore
    # close sim app
    simulation_app.close()