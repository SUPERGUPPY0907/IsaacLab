# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlBelmGenpoActorCriticCfg,
    RslRlGenpoActorCriticCfg,
    RslRlGenpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class H1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "h1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class H1FlatPPORunnerCfg(H1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "h1_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class H1RoughGenPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyFlowRunner"
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "h1_rough_genpo"
    wandb_project = "h1_rough_genpo"
    policy = RslRlGenpoActorCriticCfg(
        std=1.0,
        flow_num_steps=5,
        mix_para=0.95,
        time_dim=32,
        time_hidden_dims=[64, 64],
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlGenpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        compress_coef=0.01,
        use_compress=False,
        use_entropy=False,
    )


@configclass
class H1RoughBELMGenPORunnerCfg(H1RoughGenPORunnerCfg):
    class_name = "OnPolicyFlowRunner"
    experiment_name = "h1_rough_belmgenpo"
    wandb_project = "h1_rough_belmgenpo"
    policy = RslRlBelmGenpoActorCriticCfg(
        std=1.0,
        flow_num_steps=5,
        mix_para=0.95,
        lag_coeff=0.97,
        time_dim=32,
        time_hidden_dims=[64, 64],
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = H1RoughGenPORunnerCfg().algorithm.replace(class_name="BELMGenPO")

