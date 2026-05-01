# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPHypernetModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1000
    save_interval = 500
    experiment_name = "AutoEnvGen_PPO_Hypernet"
    logger = "wandb"
    wandb_project = "AutoEnvGen_PPO_Hypernet"
    wandb_kwargs = {
        "project": "AutoEnvGen_PPO_Hypernet",
        "entity": "spacer-rl",
        "group": "zeroG",
    }
    # NOTE: the hypernet model expects a *_context observation set in addition to the regular one.
    # Replace "semantic" below with the actual obs group your environment emits for the hypernet context stream.
    obs_groups = {
        "actor": ["policy"],
        "actor_context": ["semantic"],
        "critic": ["policy"],
        "critic_context": ["semantic"],
    }
    actor = RslRlMLPHypernetModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        network_type="hybrid",
        use_embeddings=True,
        embeddings_size=64,
        generator_hidden_dims=[256, 256, 256],
    )
    critic = RslRlMLPHypernetModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        network_type="hybrid",
        use_embeddings=True,
        embeddings_size=64,
        generator_hidden_dims=[256, 256, 256],
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
