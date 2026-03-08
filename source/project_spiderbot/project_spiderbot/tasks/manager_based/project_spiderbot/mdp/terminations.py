from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation_quat(env: ManagerBasedRLEnv, limit_angle: float) -> torch.Tensor:
    """Terminate when root body's up-axis deviates from world up by more than limit_angle."""
    quat_w = env.scene["robot"].data.root_quat_w  # (N,4) or (4,)
    if quat_w.ndim == 1:
        quat_w = quat_w.unsqueeze(0)

    z_axis_world = math_utils.quat_apply(
        quat_w, torch.tensor([0.0, 0.0, 1.0], device=quat_w.device)
    )
    if z_axis_world.ndim == 1:
        z_axis_world = z_axis_world.unsqueeze(0)

    # cos(theta) = dot(z_axis_world, world_up) = z component
    cos_theta = z_axis_world[:, 2].clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta).abs()
    return theta > limit_angle