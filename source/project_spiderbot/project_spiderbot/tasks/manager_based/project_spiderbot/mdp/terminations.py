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

    # Batched body-up axis (N, 3) to match quat_apply batching
    up_b = torch.tensor(
        [0.0, 0.0, 1.0],
        device=quat_w.device,
        dtype=quat_w.dtype,
    ).expand(quat_w.shape[0], 3)

    # Rotate body-up into world frame: (N, 3)
    z_axis_world = math_utils.quat_apply(quat_w, up_b)

    # cos(theta) = dot(z_axis_world, world_up) = z component
    cos_theta = z_axis_world[:, 2].clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta).abs()
    return theta > limit_angle