from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils


@dataclass
class StageSpec:
    """Defines one curriculum stage and its promotion criteria."""
    lin_vel_x_range: tuple[float, float]
    max_vel_xy_err: float = 0.20
    max_bad_orient_frac: float = 0.20
    streak_required: int = 20


# ---------------------------------------------------------------------
# GLOBAL STAGES
# Edit these here to tune curriculum.
# ---------------------------------------------------------------------
STAGES: list[StageSpec] = [
    # Stage 0: Stand
    StageSpec(
        lin_vel_x_range=(0.0, 0.0),
        max_vel_xy_err=0.70,
        max_bad_orient_frac=0.40,
        streak_required=200,
    ),
    # Stage 1: small forward vx
    StageSpec(
        lin_vel_x_range=(0.0, 0.20),
        max_vel_xy_err=0.80,
        max_bad_orient_frac=0.40,
        streak_required=30,
    ),
]


def _ensure_log(env: ManagerBasedRLEnv) -> dict:
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    return env.extras.setdefault("log", {})


def _ensure_state(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "_spiderbot_curr_stage"):
        env._spiderbot_curr_stage = 0
    if not hasattr(env, "_spiderbot_curr_streak"):
        env._spiderbot_curr_streak = 0
    if not hasattr(env, "_spiderbot_curr_checks"):
        env._spiderbot_curr_checks = 0
    if not hasattr(env, "_spiderbot_curriculum_stage"):
        env._spiderbot_curriculum_stage = 0.0


def _get_cmd_vel_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    cmd_term = env.command_manager.get_term("base_velocity")
    cmd = cmd_term.command  # (N,3): [vx, vy, wz]
    return cmd[:, 0:2]


def _get_base_vel_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Match IsaacLab tracking: base-frame velocity
    v_b = env.scene["robot"].data.root_lin_vel_b  # (N,3)
    return v_b[:, 0:2]


def _bad_orient_frac(env: ManagerBasedRLEnv, limit_angle: float) -> float:
    quat_w = env.scene["robot"].data.root_quat_w  # (N,4)
    up_b = torch.tensor(
        [0.0, 0.0, 1.0],
        device=quat_w.device,
        dtype=quat_w.dtype,
    ).expand(quat_w.shape[0], 3)

    z_axis_world = math_utils.quat_apply(quat_w, up_b)
    cos_theta = z_axis_world[:, 2].clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta).abs()
    return (theta > limit_angle).float().mean().item()


# ---------------------------------------------------------------------
# EVENT-STYLE UPDATE (called every step via EventManager interval)
# IMPORTANT: must accept (env, env_ids, ...) for EventManager.
# ---------------------------------------------------------------------
def update_streak_every_step(
    env: ManagerBasedRLEnv,
    env_ids,
    tilt_limit_angle: float = 1.80,
    tag: str = "spiderbot_curriculum",
):
    _ensure_state(env)
    log = _ensure_log(env)

    env._spiderbot_curr_checks += 1

    stage_idx = int(env._spiderbot_curr_stage)
    stage_idx = max(0, min(stage_idx, len(STAGES) - 1))
    spec = STAGES[stage_idx]

    cmd_xy = _get_cmd_vel_xy(env)
    vel_xy = _get_base_vel_xy(env)
    vel_xy_err = torch.linalg.norm(vel_xy - cmd_xy, dim=1)
    err_mean = vel_xy_err.mean().item()

    bad_frac = _bad_orient_frac(env, tilt_limit_angle)

    # continuous logs (these should show up via your interval logging)
    log["Metrics/base_velocity/vel_xy_err_mean"] = float(err_mean)
    log["Metrics/base_velocity/bad_orient_frac"] = float(bad_frac)
    log["Metrics/curriculum/stage"] = float(stage_idx)
    log["Metrics/curriculum/streak"] = float(env._spiderbot_curr_streak)
    log["Metrics/curriculum/checks"] = float(env._spiderbot_curr_checks)

    good = (err_mean <= spec.max_vel_xy_err) and (bad_frac <= spec.max_bad_orient_frac)
    if good:
        env._spiderbot_curr_streak += 1
    else:
        env._spiderbot_curr_streak = 0

    if env._spiderbot_curr_streak >= spec.streak_required and stage_idx < (len(STAGES) - 1):
        env._spiderbot_curr_stage = stage_idx + 1
        env._spiderbot_curr_streak = 0
        print(f"[{tag}] PROMOTE -> stage {env._spiderbot_curr_stage}")

    env._spiderbot_curriculum_stage = float(env._spiderbot_curr_stage)
    return None


# ---------------------------------------------------------------------
# CURRICULUM APPLY (called by CurriculumManager schedule)
# ---------------------------------------------------------------------
def lin_vel_x_from_stage(env: ManagerBasedRLEnv, env_ids, old_value, **kwargs):
    _ensure_state(env)
    stage_idx = int(env._spiderbot_curr_stage)
    stage_idx = max(0, min(stage_idx, len(STAGES) - 1))
    env._spiderbot_curriculum_stage = float(stage_idx)
    return STAGES[stage_idx].lin_vel_x_range