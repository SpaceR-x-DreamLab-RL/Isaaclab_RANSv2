from __future__ import annotations

import torch


class StraightLineGuidance:
    """
    Goal-direction lookahead guidance (no path memory, no heading).

    Instead of following a fixed line, the reference is:
        current_position + lookahead_distance * direction_to_goal

    Output:
        body-frame position error to that moving reference
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        lookahead_distance: float = 0.05,
    ) -> None:
        self._num_envs = num_envs
        self._device = torch.device(device)
        self.lookahead_distance = float(lookahead_distance)

    @staticmethod
    def _world_to_body(vec_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
        c = torch.cos(heading_w)
        s = torch.sin(heading_w)
        x_b = c * vec_w[:, 0] + s * vec_w[:, 1]
        y_b = -s * vec_w[:, 0] + c * vec_w[:, 1]
        return torch.stack((x_b, y_b), dim=-1)

    def compute_position_error_body(
        self,
        pos_w: torch.Tensor,
        heading_w: torch.Tensor,
        goal_pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Body-frame position error to a moving reference defined
        by goal-direction lookahead.
        """

        # --- Direction to goal ---
        dir_w = goal_pos_w - pos_w
        dist = torch.norm(dir_w, dim=-1, keepdim=True).clamp(min=1e-6)
        dir_w = dir_w / dist  # unit vector

        # --- Lookahead reference (relative to current position) ---
        step = torch.minimum(
            dist,
            torch.tensor(self.lookahead_distance, device=pos_w.device)
        )
        ref_pos_w = pos_w + step * dir_w

        # --- Position error ---
        pos_err_w = pos_w - ref_pos_w
        pos_err_b = self._world_to_body(pos_err_w, heading_w)

        debug = {
            "distance_to_goal": dist.squeeze(-1),
            "ref_pos_x": ref_pos_w[:, 0],
            "ref_pos_y": ref_pos_w[:, 1],
        }

        return pos_err_b, debug