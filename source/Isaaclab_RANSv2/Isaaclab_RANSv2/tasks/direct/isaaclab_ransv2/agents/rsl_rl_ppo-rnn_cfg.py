# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 1000
    experiment_name = "AutoEnvGen_PPO_Pingu_GoToPose_All_Actuators_RNN_DomRand_v2"
    logger = "wandb"
    wandb_project = "AutoEnvGen_PPO_Pingu_GoToPose_All_Actuators_RNN_DomRand_v2"
    wandb_kwargs = {
        "project": "AutoEnvGen_PPO_Pingu_GoToPose_All_Actuators_RNN_DomRand_v2",
        "entity": "spacer-rl",
        "group": "zeroG",
    }
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="tanh",
        # GRU-specific fields:
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )