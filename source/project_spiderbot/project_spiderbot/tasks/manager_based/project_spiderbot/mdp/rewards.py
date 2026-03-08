# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def upright_posture(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Uprightness reward: dot(body_z_world, world_up). Range [-1, 1]."""
    quat_w = env.scene["robot"].data.root_quat_w  # (N,4) or (4,)

    # Ensure batched shape (N, 4)
    if quat_w.ndim == 1:
        quat_w = quat_w.unsqueeze(0)

    z_axis_world = math_utils.quat_apply(
        quat_w, torch.tensor([0.0, 0.0, 1.0], device=quat_w.device)
    )

    # Ensure output shape (N,)
    if z_axis_world.ndim == 1:
        # (3,) -> (1,3)
        z_axis_world = z_axis_world.unsqueeze(0)

    return z_axis_world[:, 2].clamp(-1.0, 1.0)

def ang_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize yaw rate (world frame). Returns w_z^2 per env."""
    ang_vel_w = env.scene["robot"].data.root_ang_vel_w  # (N, 3) or (3,)
    if ang_vel_w.ndim == 1:
        ang_vel_w = ang_vel_w.unsqueeze(0)
    wz = ang_vel_w[:, 2]
    return wz * wz