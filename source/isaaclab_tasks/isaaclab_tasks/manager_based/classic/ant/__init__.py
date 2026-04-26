# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "fpo": f"{agents.__name__}.rsl_rl_ppo_cfg:AntFPORunnerCfg",
        "policyflow": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPolicyFlowRunnerCfg",
        "genpo": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGenPORunnerCfg",
        "belmgenpo": f"{agents.__name__}.rsl_rl_ppo_cfg:AntBELMGenPORunnerCfg",
        "belm_genpo": f"{agents.__name__}.rsl_rl_ppo_cfg:AntBELMGenPORunnerCfg",
        "genpo_pp": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGenPOPlusPlusRunnerCfg",
        "genpo++": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGenPOPlusPlusRunnerCfg",
        "genpo_pfclip": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGenPOPFClipRunnerCfg",
        "genpo_u0clip": f"{agents.__name__}.rsl_rl_ppo_cfg:AntGenPOU0ClipRunnerCfg",
        "spo": f"{agents.__name__}.rsl_rl_ppo_cfg:AntSPORunnerCfg",
        "sgenpo": f"{agents.__name__}.rsl_rl_ppo_cfg:AntSGenPORunnerCfg",
    },
)
