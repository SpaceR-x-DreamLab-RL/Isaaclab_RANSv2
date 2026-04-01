"""Task-aware observation to LQR state conversion.

Converts task observations to the 6D LQR state expected by the controller:
    [vel_err_x, vel_err_y, ang_vel_err_z, theta_err, pos_err_x, pos_err_y]

Rules:
- Extra observation entries are ignored.
- Missing required entries are replaced with 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _TaskObsLayout:
    d_idx: int | None
    cos_bearing_idx: int | None
    sin_bearing_idx: int | None
    vx_idx: int | None
    vy_idx: int | None
    wz_idx: int | None
    theta_cos_idx: int | None = None
    theta_sin_idx: int | None = None


class TaskObsToLqrStateConverter:
    """Converts task observations into LQR state depending on selected task."""

    _LAYOUTS: dict[str, _TaskObsLayout] = {
        # [d, cos(bearing), sin(bearing), cos(theta_err), sin(theta_err), vx, vy, wz, ...]
        "go_to_pose": _TaskObsLayout(
            d_idx=0,
            cos_bearing_idx=1,
            sin_bearing_idx=2,
            vx_idx=5,
            vy_idx=6,
            wz_idx=7,
            theta_cos_idx=3,
            theta_sin_idx=4,
        ),
        # [d, cos(bearing), sin(bearing), vx, vy, wz, ...]
        "go_to_position": _TaskObsLayout(
            d_idx=0,
            cos_bearing_idx=1,
            sin_bearing_idx=2,
            vx_idx=3,
            vy_idx=4,
            wz_idx=5,
        ),
        # [vx, vy, wz, d, cos(bearing), sin(bearing), ...]
        "go_through_positions": _TaskObsLayout(
            d_idx=3,
            cos_bearing_idx=4,
            sin_bearing_idx=5,
            vx_idx=0,
            vy_idx=1,
            wz_idx=2,
        ),
        # [vx, vy, wz, d, cos(bearing), sin(bearing), cos(theta_err), sin(theta_err), ...]
        "go_through_poses": _TaskObsLayout(
            d_idx=3,
            cos_bearing_idx=4,
            sin_bearing_idx=5,
            vx_idx=0,
            vy_idx=1,
            wz_idx=2,
            theta_cos_idx=6,
            theta_sin_idx=7,
        ),
    }

    _ALIASES: dict[str, str] = {
        "go_to_pose": "go_to_pose",
        "gotopose": "go_to_pose",
        "go_to_position": "go_to_position",
        "gotoposition": "go_to_position",
        "go_through_positions": "go_through_positions",
        "gothroughpositions": "go_through_positions",
        "go_through_poses": "go_through_poses",
        "gothroughposes": "go_through_poses",
    }

    def __init__(self, task_name: str | None, task_type: str = "auto", device: str = "cuda") -> None:
        self._device = torch.device(device)
        self.task_type = self._resolve_task_type(task_name, task_type)
        self._layout = self._LAYOUTS[self.task_type]

    def _resolve_task_type(self, task_name: str | None, task_type: str) -> str:
        task_type_norm = task_type.strip().lower().replace("-", "_")
        if task_type_norm != "auto":
            if task_type_norm not in self._ALIASES:
                supported = ", ".join(sorted(self._LAYOUTS.keys()))
                raise ValueError(f"Unsupported --obs_task_type '{task_type}'. Supported: {supported}, auto")
            return self._ALIASES[task_type_norm]

        if task_name is None:
            return "go_to_pose"

        task_name_norm = task_name.lower().replace("-", "_")
        if "go_through_poses" in task_name_norm or "gothroughposes" in task_name_norm:
            return "go_through_poses"
        if "go_through_positions" in task_name_norm or "gothroughpositions" in task_name_norm:
            return "go_through_positions"
        if "go_to_position" in task_name_norm or "gotoposition" in task_name_norm:
            return "go_to_position"
        if "go_to_pose" in task_name_norm or "gotopose" in task_name_norm:
            return "go_to_pose"
        return "go_to_pose"

    @staticmethod
    def _index_is_valid(obs: torch.Tensor, idx: int | None) -> bool:
        return idx is not None and 0 <= idx < obs.shape[1]

    @staticmethod
    def _col_or_zero(obs: torch.Tensor, idx: int | None) -> torch.Tensor:
        if idx is None or idx < 0 or idx >= obs.shape[1]:
            return torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        return obs[:, idx]

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        obs = obs_tensor.to(device=self._device, dtype=torch.float32)

        d = self._col_or_zero(obs, self._layout.d_idx)
        cos_bearing = self._col_or_zero(obs, self._layout.cos_bearing_idx)
        sin_bearing = self._col_or_zero(obs, self._layout.sin_bearing_idx)

        # Convention: pos_err = pos - pos_ref in body frame.
        pos_err_x = -d * cos_bearing
        pos_err_y = -d * sin_bearing

        has_theta = self._index_is_valid(obs, self._layout.theta_cos_idx) and self._index_is_valid(
            obs, self._layout.theta_sin_idx
        )
        if has_theta:
            # theta_task is typically (theta_ref - theta_vehicle).
            theta_err = -torch.atan2(obs[:, self._layout.theta_sin_idx], obs[:, self._layout.theta_cos_idx])
        else:
            theta_err = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)

        vel_err_x = self._col_or_zero(obs, self._layout.vx_idx)
        vel_err_y = self._col_or_zero(obs, self._layout.vy_idx)
        ang_vel_err = self._col_or_zero(obs, self._layout.wz_idx)

        return torch.stack((vel_err_x, vel_err_y, ang_vel_err, theta_err, pos_err_x, pos_err_y), dim=-1)
