# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect

from rsl_rl import runners as rsl_runners


def _discover_runner_names() -> list[str]:
    """Return available runner class names from rsl_rl.runners."""
    if hasattr(rsl_runners, "__all__"):
        names = [name for name in rsl_runners.__all__ if name.endswith("Runner")]
    else:
        names = [
            name for name, value in vars(rsl_runners).items() if inspect.isclass(value) and name.endswith("Runner")
        ]
    return sorted(set(names))


def build_runner(env, agent_cfg, log_dir: str | None):
    """Create an rsl_rl runner from the class name in agent_cfg."""
    runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    runner_cls = getattr(rsl_runners, runner_class_name, None)
    if runner_cls is None:
        available = ", ".join(_discover_runner_names())
        raise ValueError(
            f"Unsupported runner class: {runner_class_name}. "
            f"Available runner classes in rsl_rl.runners: {available}"
        )
    return runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
