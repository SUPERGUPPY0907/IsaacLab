# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlBelmGenpoActorCriticCfg,
    RslRlFpoActorCriticCfg,
    RslRlFpoAlgorithmCfg,
    RslRlGenpoActorCriticCfg,
    RslRlGenpoAlgorithmCfg,
    RslRlOnPolicyFlowRunnerCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPolicyFlowActorCriticCfg,
    RslRlPolicyFlowAlgorithmCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    wandb_project = "franka_lift_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LiftCubeFPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyFlowRunner"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift_fpo"
    wandb_project = "franka_lift_fpo"
    policy = RslRlFpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        timestep_embed_dim=8,
        sampling_steps=64,
        cfm_loss_reduction="sqrt",
        action_perturb_std=0.02,
    )
    algorithm = RslRlFpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=16,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        weight_decay=1.0e-4,
        schedule="fixed",
        gamma=0.98,
        lam=0.95,
        desired_kl=1.0e-4,
        max_grad_norm=1.0,
        trust_region_mode="aspo",
        ema_decay=0.95,
        ema_warmup_steps=500,
    )


@configclass
class LiftCubePolicyFlowRunnerCfg(RslRlOnPolicyFlowRunnerCfg):
    class_name = "OnPolicyFlowRunner"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift_policyflow"
    wandb_project = "franka_lift_policyflow"
    policy = RslRlPolicyFlowActorCriticCfg(
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        flow_embedding_dim=64,
        flow_sample_steps=10,
        flow_sample_step_schedule="uniform_continuous",
        flow_interpolation_type="rectified_flow",
        flow_timestep_embedding_type="fourier",
        flow_conditioning="linear",
        flow_use_ema=False,
        flow_ema_rate=0.995,
        variance_std_init=1.0,
        variance_log_std_min=-20.0,
        variance_log_std_max=4.0,
    )
    algorithm = RslRlPolicyFlowAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        gaussian_entropy_loss_scale=0.0025,
        brownian_reg_loss_scale=0.0025,
        optimizer="adamw",
        optimizer_kwargs={"weight_decay": 1.0e-5},
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
    )


@configclass
class LiftCubeGenPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyFlowRunner"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift_genpo"
    wandb_project = "franka_lift_genpo"
    policy = RslRlGenpoActorCriticCfg(
        std=1.0,
        flow_num_steps=5,
        mix_para=0.95,
        time_dim=32,
        time_hidden_dims=[64, 64],
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlGenpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        compress_coef=0.01,
        use_compress=False,
        use_entropy=False,
    )


@configclass
class LiftCubeBELMGenPORunnerCfg(LiftCubeGenPORunnerCfg):
    experiment_name = "franka_lift_belmgenpo"
    wandb_project = "franka_lift_belmgenpo"
    policy = RslRlBelmGenpoActorCriticCfg(
        std=1.0,
        flow_num_steps=5,
        mix_para=0.95,
        lag_coeff=0.97,
        a_coeff=0.03,
        b_coeff=0.97,
        eps_coeff=1.0,
        time_dim=32,
        time_hidden_dims=[64, 64],
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = LiftCubeGenPORunnerCfg().algorithm.replace(class_name="BELMGenPO")
