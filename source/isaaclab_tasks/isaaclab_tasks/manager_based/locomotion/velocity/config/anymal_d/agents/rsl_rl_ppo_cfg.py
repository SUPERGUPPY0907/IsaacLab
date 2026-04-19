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
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import anymal


@configclass
class AnymalDRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough"
    wandb_project = "anymal_d_rough_ppo"
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


@configclass
class AnymalDRoughFPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyFlowRunner"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough_fpo"
    wandb_project = "anymal_d_rough_fpo"
    policy = RslRlFpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        timestep_embed_dim=8,
        sampling_steps=64,
        cfm_loss_reduction="sqrt",
        action_perturb_std=0.02,
    )
    algorithm = RslRlFpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=32,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=1.0e-4,
        max_grad_norm=1.0,
        trust_region_mode="aspo",
        ema_decay=0.95,
        ema_warmup_steps=500,
    )


@configclass
class AnymalDRoughGenPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyFlowRunner"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough_genpo"
    wandb_project = "anymal_d_rough_genpo"
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
class AnymalDRoughBELMGenPORunnerCfg(AnymalDRoughGenPORunnerCfg):
    experiment_name = "anymal_d_rough_belmgenpo"
    wandb_project = "anymal_d_rough_belmgenpo"
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
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AnymalDRoughGenPORunnerCfg().algorithm.replace(class_name="BELMGenPO")


@configclass
class AnymalDFlatPPORunnerCfg(AnymalDRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat"
        self.wandb_project = "anymal_d_flat_ppo"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class AnymalDFlatPPORunnerWithSymmetryCfg(AnymalDFlatPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=anymal.compute_symmetric_states
        ),
    )


@configclass
class AnymalDRoughPPORunnerWithSymmetryCfg(AnymalDRoughPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=anymal.compute_symmetric_states
        ),
    )
