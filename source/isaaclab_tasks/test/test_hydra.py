# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import functools
from collections.abc import Callable

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from isaaclab.utils import replace_strings_with_slices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import register_task_to_hydra


def hydra_task_config_test(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Copied from hydra.py hydra_task_config, since hydra.main requires a single point of entry,
    which will not work with multiple tests. Here, we replace hydra.main with hydra initialize
    and compose."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)

            # replace hydra.main with initialize and compose
            with initialize(config_path=None, version_base="1.3"):
                hydra_env_cfg = compose(config_name=task_name, overrides=sys.argv[1:])
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                if isinstance(agent_cfg, dict):
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator


def test_hydra():
    """Test the hydra configuration system."""

    # set hardcoded command line arguments
    sys.argv = [
        sys.argv[0],
        "env.decimation=42",  # test simple env modification
        "env.events.physics_material.params.asset_cfg.joint_ids='slice(0 ,1, 2)'",  # test slice setting
        "env.scene.robot.init_state.joint_vel={.*: 4.0}",  # test regex setting
        "env.rewards.feet_air_time=null",  # test setting to none
        "agent.max_iterations=3",  # test simple agent modification
    ]

    @hydra_task_config_test("Isaac-Velocity-Flat-H1-v0", "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        # env
        assert env_cfg.decimation == 42
        assert env_cfg.events.physics_material.params["asset_cfg"].joint_ids == slice(0, 1, 2)
        assert env_cfg.scene.robot.init_state.joint_vel == {".*": 4.0}
        assert env_cfg.rewards.feet_air_time is None
        # agent
        assert agent_cfg.max_iterations == 3

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_nested_iterable_dict():
    """Test the hydra configuration system when dict is nested in an Iterable."""

    @hydra_task_config_test("Isaac-Lift-Cube-Franka-v0", "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        # env
        assert env_cfg.scene.ee_frame.target_frames[0].name == "end_effector"
        assert env_cfg.scene.ee_frame.target_frames[0].offset.pos[2] == 0.1034

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_belm_hydra_policy_coeff_overrides():
    """Test that BELM policy coefficient overrides propagate through Hydra."""

    sys.argv = [
        sys.argv[0],
        "agent.policy.a_coeff=0.25",
        "agent.policy.b_coeff=0.75",
        "agent.policy.eps_coeff=2.0",
        "agent.policy.lag_coeff=null",
    ]

    @hydra_task_config_test("Isaac-Humanoid-v0", "belm_genpo")
    def main(env_cfg, agent_cfg):
        del env_cfg
        assert agent_cfg.policy.class_name == "ActorCriticBELMGenPO"
        assert agent_cfg.policy.a_coeff == 0.25
        assert agent_cfg.policy.b_coeff == 0.75
        assert agent_cfg.policy.eps_coeff == 2.0
        assert agent_cfg.policy.lag_coeff is None

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_g1_fpo_hydra_config_and_overrides():
    """Test that the G1 Rough FPO agent entry point resolves and accepts overrides."""

    sys.argv = [
        sys.argv[0],
        "agent.algorithm.clip_param=0.07",
        "agent.algorithm.knn_entropy_k=3",
        "agent.policy.sampling_steps=32",
    ]

    @hydra_task_config_test("Isaac-Velocity-Rough-G1-v0", "fpo")
    def main(env_cfg, agent_cfg):
        del env_cfg
        assert agent_cfg.class_name == "OnPolicyFlowRunner"
        assert agent_cfg.policy.class_name == "ActorCriticFPO"
        assert agent_cfg.algorithm.class_name == "FPO"
        assert agent_cfg.algorithm.clip_param == 0.07
        assert agent_cfg.algorithm.knn_entropy_k == 3
        assert agent_cfg.policy.sampling_steps == 32

    main()
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_h1_fpo_hydra_config_and_overrides():
    """Test that the H1 Rough FPO agent entry point resolves and accepts overrides."""

    sys.argv = [
        sys.argv[0],
        "agent.algorithm.clip_param=0.15",
        "agent.algorithm.ema_decay=0.9",
        "agent.policy.sampling_steps=32",
    ]

    @hydra_task_config_test("Isaac-Velocity-Rough-H1-v0", "fpo")
    def main(env_cfg, agent_cfg):
        del env_cfg
        assert agent_cfg.class_name == "OnPolicyFlowRunner"
        assert agent_cfg.policy.class_name == "ActorCriticFPO"
        assert agent_cfg.algorithm.class_name == "FPO"
        assert agent_cfg.algorithm.clip_param == 0.15
        assert agent_cfg.algorithm.ema_decay == 0.9
        assert agent_cfg.policy.sampling_steps == 32

    main()
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_humanoid_policyflow_hydra_config_and_overrides():
    """Test that the Humanoid PolicyFlow agent entry point resolves and accepts overrides."""

    sys.argv = [
        sys.argv[0],
        "agent.policy.flow_sample_steps=32",
        "agent.policy.flow_conditioning=mlp",
        "agent.algorithm.brownian_reg_loss_scale=0.01",
    ]

    @hydra_task_config_test("Isaac-Humanoid-v0", "policyflow")
    def main(env_cfg, agent_cfg):
        del env_cfg
        assert agent_cfg.class_name == "OnPolicyFlowRunner"
        assert agent_cfg.policy.class_name == "ActorCriticPolicyFlow"
        assert agent_cfg.algorithm.class_name == "PolicyFlow"
        assert agent_cfg.policy.flow_sample_steps == 32
        assert agent_cfg.policy.flow_conditioning == "mlp"
        assert agent_cfg.algorithm.brownian_reg_loss_scale == 0.01

    main()
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
