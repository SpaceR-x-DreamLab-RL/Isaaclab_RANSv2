# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.scene import InteractiveScene

from ..tasks_cfg import WaypointDockingCfg

from .go_through_poses import GoThroughPosesTask


class WaypointDockingTask(GoThroughPosesTask):
    """GoThroughPoses variant for slow, precise navigation through closely-spaced waypoints.

    This task is the training entry-point for sensor-based docking on Pingu. It inherits
    the full GoThroughPoses observation/reward/goal structure and tunes the configuration
    for docking-regime training:
      - Waypoints are generated 0.3–1.5 m apart (vs. the default 1–5 m).
      - Linear and angular velocity are penalised heavily to enforce slow motion.
      - The robot always spawns at rest, within 1 m of the first waypoint.
      - No per-step time penalty keeps the early curriculum easy to solve.

    Arm actions are not used by this task. Pair with PinguNoArmsRobot so the RL policy
    does not waste action-space capacity on the arms.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: WaypointDockingCfg = WaypointDockingCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            scene=scene,
            task_cfg=task_cfg,
            task_uid=task_uid,
            num_envs=num_envs,
            device=device,
            env_ids=env_ids,
        )
