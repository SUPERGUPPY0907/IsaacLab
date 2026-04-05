# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlBelmGenpoActorCriticCfg,
    RslRlGenpoActorCriticCfg,
    RslRlGenpoAlgorithmCfg,
    RslRlGenpoPlusPlusAlgorithmCfg,
    RslRlGenpoPushforwardClipAlgorithmCfg,
    RslRlGenpoU0ClipAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AntPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "ant_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
    )

@configclass
class AntGenPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyFlowRunner"
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 1000
    experiment_name = "ant_genpo"
    # empirical_normalization = False
    policy = RslRlGenpoActorCriticCfg(
        std=1.0,
        flow_num_steps=5,
        mix_para=0.95,
        time_dim = 32,
        time_hidden_dims = [64, 64],
        init_noise_std=1.0,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="mish",

    )
    algorithm = RslRlGenpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
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
class AntBELMGenPORunnerCfg(AntGenPORunnerCfg):
    experiment_name = "ant_belmgenpo"
    policy = RslRlBelmGenpoActorCriticCfg(
        std=1.0,
        flow_num_steps=5,
        mix_para=0.95,
        lag_coeff=0.95,
        time_dim=32,
        time_hidden_dims=[64, 64],
        init_noise_std=1.0,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="mish",
    )
    algorithm = AntGenPORunnerCfg().algorithm.replace(class_name="BELMGenPO")


@configclass
class AntSPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_spo"
    algorithm = AntPPORunnerCfg().algorithm.replace(class_name="SPO")

@configclass
class AntSGenPORunnerCfg(AntGenPORunnerCfg):
    experiment_name = "ant_sgenpo"
    algorithm = AntGenPORunnerCfg().algorithm.replace(class_name="SGenPO", learning_rate=5e-4)


@configclass
class AntGenPOPlusPlusRunnerCfg(AntGenPORunnerCfg):
    experiment_name = "ant_genpo_plus_plus"
    algorithm = RslRlGenpoPlusPlusAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        compress_coef=0.01,
        use_compress=False,
        use_entropy=False,
        lambda_dir=0.0,
        directional_num_samples=4,
        directional_advantage_quantile=0.75,
        directional_max_groups=32,
        directional_eps=1.0e-8,
        lambda_mirror = 0.01,
    )


@configclass
class AntGenPOPFClipRunnerCfg(AntGenPORunnerCfg):
    experiment_name = "ant_genpo_pfclip"
    algorithm = RslRlGenpoPushforwardClipAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        compress_coef=0.01,
        use_compress=False,
        use_entropy=False,
        log_clip_delta=None,
        pf_num_samples=2,
    )


@configclass
class AntGenPOU0ClipRunnerCfg(AntGenPORunnerCfg):
    experiment_name = "ant_genpo_u0clip"
    algorithm = RslRlGenpoU0ClipAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        compress_coef=0.01,
        use_compress=False,
        use_entropy=False,
        log_clip_delta=10000,
    )
