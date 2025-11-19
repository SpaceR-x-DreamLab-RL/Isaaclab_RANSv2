from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class AutoEnvGenCfg(DirectRLEnvCfg):
    # env
    decimation = 6
    episode_length_s = 20.0

    robot_name = "FloatingPlatform"
    task_name = "GoToPose"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=7.5, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)
    debug_vis: bool = True

    action_space = 0
    observation_space = 0
    state_space = 0
    gen_space = 0
