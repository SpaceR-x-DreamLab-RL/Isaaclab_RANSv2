"""Minimal Warp-based LQR controller.

State layout expected by this controller:
    [vel_err_x, vel_err_y, ang_vel_err_z, theta_err, pos_err_x, pos_err_y]

The controller appends integral states [int_theta, int_pos_x, int_pos_y], so K must be shape (4, 9).
"""

# pyright: reportInvalidTypeForm=false

import numpy as np
import torch
import warp as wp


@wp.kernel
def lqr_control_kernel(
    states: wp.array2d(dtype=wp.float32),
    commands: wp.array2d(dtype=wp.float32),
    integral_states: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    K: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()

    vel_err_x = states[tid, 0]
    vel_err_y = states[tid, 1]
    ang_vel_err = states[tid, 2]
    theta_err = states[tid, 3]
    pos_err_x = states[tid, 4]
    pos_err_y = states[tid, 5]

    integral_states[tid, 0] -= theta_err * dt #* 0.0
    integral_states[tid, 1] -= pos_err_x * dt #* 0.0
    integral_states[tid, 2] -= pos_err_y * dt #* 0.0

    x0 = vel_err_x
    x1 = vel_err_y
    x2 = ang_vel_err
    x3 = theta_err
    x4 = pos_err_x
    x5 = pos_err_y
    x6 = integral_states[tid, 0]
    x7 = integral_states[tid, 1]
    x8 = integral_states[tid, 2]

    for i in range(3):
        u = -(
            K[i, 0] * x0
            + K[i, 1] * x1
            + K[i, 2] * x2
            + K[i, 3] * x3
            + K[i, 4] * x4
            + K[i, 5] * x5
            + K[i, 6] * x6
            + K[i, 7] * x7
            + K[i, 8] * x8
        )
        commands[tid, i] = u

    # Negate y and z thrusts to match the expected action convention of the environment
    commands[tid, 1] = -commands[tid, 1]
    commands[tid, 2] = -commands[tid, 2]
    
    # Frame rotation is not needed since velocity errors are already in the body frame

    # commands[tid, 0] = 0.0
    # commands[tid, 1] = 0.0
    # commands[tid, 2] = 0.0
    commands[tid, 3] = 0.0


class LQRController:
    """Warp LQR helper that accepts torch states and returns torch actions."""

    def __init__(self, gain_matrix_path: str, device: str = "cuda", num_envs: int = 1) -> None:
        wp.init()

        self.device = device
        self.torch_device = torch.device(device)
        self.num_envs = num_envs

        k_np = np.loadtxt(gain_matrix_path, delimiter=",", dtype=np.float32)
        if k_np.shape != (3, 9):
            raise ValueError(f"Expected K matrix shape (3, 9), got {k_np.shape}")

        self.K = wp.array2d(k_np, dtype=wp.float32, device=self.device)
        self.integral_states = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)

        self._commands_torch = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.torch_device)
        self._commands_wp = wp.from_torch(self._commands_torch, dtype=wp.float32)

    def compute_control(self, states: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute actions from torch states.

        Args:
            states: Tensor with shape (num_envs, 6)
            dt: physics timestep
        Returns:
            Tensor with shape (num_envs, 4)
        """
        if states.ndim != 2 or states.shape[0] != self.num_envs or states.shape[1] != 6:
            raise ValueError(f"Expected states shape ({self.num_envs}, 6), got {tuple(states.shape)}")

        states_local = states.to(device=self.torch_device, dtype=torch.float32).contiguous()
        states_wp = wp.from_torch(states_local, dtype=wp.float32)

        wp.launch(
            kernel=lqr_control_kernel,
            dim=self.num_envs,
            inputs=[states_wp, self._commands_wp, self.integral_states, wp.float32(dt), self.K],
            device=self.device,
        )
        return self._commands_torch

    def reset_integrals(self) -> None:
        self.integral_states.zero_()
