# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlGenpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the GenPO actor-critic networks."""

    class_name: str = "ActorCriticGenPO"
    """The policy class name. Default is ActorCriticGenPO."""

    flow_num_steps: int = MISSING
    """The number of steps taken by flow."""

    mix_para: float = MISSING

    time_dim: int = MISSING

    time_hidden_dims: list[int] = MISSING

    std: float = MISSING


@configclass
class RslRlBelmGenpoActorCriticCfg(RslRlGenpoActorCriticCfg):
    """Configuration for the BELM-GenPO actor-critic networks."""

    class_name: str = "ActorCriticBELMGenPO"
    """The policy class name. Default is ActorCriticBELMGenPO."""

    lag_coeff: float | None = 0.95
    """Legacy lag coefficient alias used by the BELM flow policy."""

    a_coeff: float | None = 0.05
    """Optional BELM coefficient applied to the current state term."""

    b_coeff: float | None = 0.95
    """Optional BELM coefficient applied to the lagged state term."""

    eps_coeff: float | None = 1.0
    """Optional BELM coefficient applied to the score-network term."""


@configclass
class RslRlLeapfrogActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the leapfrog-flow actor-critic networks."""

    class_name: str = "ActorCriticLeapfrog"
    """The policy class name. Default is ActorCriticLeapfrog."""

    flow_num_steps: int = 5
    """The number of leapfrog integration steps."""

    std: float = 1.0
    """The standard deviation used by leapfrog flow."""

    time_dim: int = 32
    """The time embedding dimension."""

    flow_interations: int = 5
    """The number of distillation iterations used by leapfrog flow."""

    flow_distill_batch_size: int = 256
    """The distillation batch size used by leapfrog flow."""

    time_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the time embedding MLP."""


@configclass
class RslRlMoserActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the Moser-flow actor-critic networks."""

    class_name: str = "ActorCriticMoser"
    """The policy class name. Default is ActorCriticMoser."""

    envelope_scale: float = 1.0
    """The envelope scale used by the Moser flow policy."""

    ode: str = "rk4"
    """The ODE solver used by the Moser flow policy."""

    flow_num_steps: int = 10
    """The number of ODE integration steps."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

@configclass
class RslRlGenpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the GenPO algorithm."""

    class_name: str = "GenPO"
    """The algorithm class name. Default is GenPO."""

    compress_coef: float | None = None

    use_compress: bool | None = None

    use_entropy: bool | None = None


@configclass
class RslRlGenpoPushforwardClipAlgorithmCfg(RslRlGenpoAlgorithmCfg):
    """Configuration for the GenPO pushforward-clip algorithm."""

    class_name: str = "GenPOPFClip"
    """The algorithm class name. Default is GenPOPFClip."""

    log_clip_delta: float | None = None
    """The log-ratio clipping delta used in pushforward clipping."""

    pf_num_samples: int = 2
    """The number of pushforward proposal samples."""


@configclass
class RslRlGenpoU0ClipAlgorithmCfg(RslRlGenpoAlgorithmCfg):
    """Configuration for the GenPO U0-clip algorithm."""

    class_name: str = "GenPOU0Clip"
    """The algorithm class name. Default is GenPOU0Clip."""

    log_clip_delta: float | None = None
    """The log-ratio clipping delta used in section clipping."""


@configclass
class RslRlGenpoPlusPlusAlgorithmCfg(RslRlGenpoAlgorithmCfg):
    """Configuration for the GenPO++ algorithm."""

    class_name: str = "GenPO++"
    """The algorithm class name. Default is GenPO++."""

    lambda_dir: float = 0.0
    """The directional diversity regularization weight."""

    directional_num_samples: int = 4
    """The number of directional samples generated per anchor action."""

    directional_advantage_quantile: float = 0.75
    """The quantile threshold used to select directional anchors."""

    directional_max_groups: int = 32
    """The maximum number of anchor groups used for directional regularization."""

    directional_eps: float = 1.0e-8
    """Numerical epsilon used when normalizing directional latent vectors."""

    lambda_mirror: float = 0.0
    """The mirror consistency regularization weight."""


@configclass
class RslRlLeapfrogAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the leapfrog-flow PPO algorithm."""

    class_name: str = "LeapfrogPPO"
    """The algorithm class name. Default is LeapfrogPPO."""

    compress_coef: float = 0.0
    """The coefficient for leapfrog compression loss."""

    use_compress: bool = True
    """Whether to use leapfrog compression loss."""

    use_entropy: bool = True
    """Whether to use entropy loss."""


@configclass
class RslRlMoserAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Moser-flow PPO algorithm."""

    class_name: str = "MoserPPO"
    """The algorithm class name. Default is MoserPPO."""

    lambda_minus: float = 1.0
    """Regularization coefficient for the positivity term."""

    sigma: float = 1e-3
    """Numerical stabilization value for probability clamping."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool | None = None
    """This parameter is deprecated and will be removed in the future.

    Use `actor_obs_normalization` and `critic_obs_normalization` instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "policy": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the policy will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab2.3"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab2.3"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Default is False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
