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
    if quat_w.ndim == 1:
        quat_w = quat_w.unsqueeze(0)

    # Batched body-up axis (N, 3) to match quat_apply batching
    up_b = torch.tensor(
        [0.0, 0.0, 1.0],
        device=quat_w.device,
        dtype=quat_w.dtype,
    ).expand(quat_w.shape[0], 3)

    # Rotate body-up into world frame: (N, 3)
    z_axis_world = math_utils.quat_apply(quat_w, up_b)

    # dot(z_axis_world, world_up) == z component
    return z_axis_world[:, 2].clamp(-1.0, 1.0)


def ang_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize yaw rate (world frame). Returns w_z^2 per env."""
    ang_vel_w = env.scene["robot"].data.root_ang_vel_w  # (N, 3) or (3,)
    if ang_vel_w.ndim == 1:
        ang_vel_w = ang_vel_w.unsqueeze(0)
    wz = ang_vel_w[:, 2]
    return wz * wz


def base_height_range_l2(env: ManagerBasedRLEnv, z_min: float, z_max: float) -> torch.Tensor:
    """
    Penalize base height if it goes outside [z_min, z_max].
    Returns per-env penalty >= 0 (0 when inside range).
    """
    pos_w = env.scene["robot"].data.root_pos_w  # (N,3) or (3,)
    if pos_w.ndim == 1:
        pos_w = pos_w.unsqueeze(0)

    z = pos_w[:, 2]

    below = (z_min - z).clamp(min=0.0)
    above = (z - z_max).clamp(min=0.0)

    # quadratic penalty outside the band
    return below * below + above * above


def log_curriculum_stage(env: ManagerBasedRLEnv, env_ids=None) -> torch.Tensor:
    """Log curriculum stage every step into extras['log'] (weight=0 reward term)."""
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    log = env.extras.setdefault("log", {})

    # stage is persisted by spiderbot_curriculums.switch_cmd_range_after_steps
    stage = float(getattr(env, "_spiderbot_curriculum_stage", 0.0))

    # Keep this for TensorBoard-style metrics (may or may not print in console)
    log["Metrics/base_velocity/curriculum_stage"] = stage

    # RETURN stage so it ALWAYS prints every iteration as Episode_Reward/curriculum_stage
    return torch.full((env.scene.num_envs,), stage, device=env.device)


def debug_common_step_counter(env: ManagerBasedRLEnv, env_ids=None) -> torch.Tensor:
    """Expose env.common_step_counter as an Episode_Reward line (weight=0)."""
    step = float(getattr(env, "common_step_counter", -1.0))
    return torch.full((env.scene.num_envs,), step, device=env.device)


def debug_curriculum_streak(env: ManagerBasedRLEnv, env_ids=None) -> torch.Tensor:
    """Expose env._spiderbot_curr_streak as an Episode_Reward line (weight=0)."""
    streak = float(getattr(env, "_spiderbot_curr_streak", -1.0))
    return torch.full((env.scene.num_envs,), streak, device=env.device)