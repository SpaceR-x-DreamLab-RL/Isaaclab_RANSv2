# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LQR controller for the Cubo floating platform — GoToPose task.

Dynamics (planar, body frame, linearised around target pose):

    d/dt [ex_b, ey_b, eθ, vx_b, vy_b, ω]  =  A · e  +  B · u

    A = [[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]

    B = [[0,       0,       0,         0      ],
         [0,       0,       0,         0      ],
         [0,       0,       0,         0      ],
         [1/m,     0,       0,         0      ],
         [0,       1/m,     0,         0      ],
         [0,       0,       1/Iz,      1/Iz   ]]

    u = [Fx_body (N), Fy_body (N), Mz_thrusters (N·m), τ_rw (N·m)]

GoToPose observation layout (indices 0-7):
    0  - distance to target (m)
    1  - cos(angle from heading to target pos) — body frame
    2  - sin(angle from heading to target pos) — body frame
    3  - cos(heading error to target heading)
    4  - sin(heading error to target heading)
    5  - vx  (body frame, m/s)
    6  - vy  (body frame, m/s)
    7  - ω   (rad/s)

Action layout (body-frame + reaction wheel, 4 dims):
    0  - forward/backward thrust  ∈ [-1, 1]
    1  - left/right thrust        ∈ [-1, 1]
    2  - yaw thrust (thrusters)   ∈ [-1, 1]
    3  - reaction wheel torque    ∈ [-1, 1]

Usage:
    python scripts/lqr/cubo_lqr.py --num_envs 64 [--headless]

    Optional tuning flags (all physical units):
        --mass          robot mass in kg          (default 23.0)
        --inertia_z     yaw inertia in kg·m²      (default 0.5)
        --f_max         max body-frame force  N   (default 1.0)
        --m_max         max thruster yaw torque Nm (default 0.5)
        --rw_max        max reaction wheel torque Nm (default 0.1)
        --qx            position error weight     (default 10.0)
        --qtheta        heading error weight       (default 5.0)
        --qv            velocity error weight      (default 1.0)
        --r_force       force effort weight        (default 0.1)
        --r_torque      torque effort weight       (default 0.1)
        --r_rw          reaction wheel cost weight (default 1.0)
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LQR controller for Cubo — GoToPose task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False)

# Physical parameters
parser.add_argument("--mass",       type=float, default=23.0,  help="Robot mass (kg).")
parser.add_argument("--inertia_z",  type=float, default=0.5,   help="Yaw inertia (kg·m²).")
parser.add_argument("--f_max",      type=float, default=1.0,   help="Max body-frame force per axis (N).")
parser.add_argument("--m_max",      type=float, default=0.5,   help="Max thruster yaw torque (N·m).")
parser.add_argument("--rw_max",     type=float, default=0.1,   help="Max reaction wheel torque (N·m).")

# LQR cost weights
parser.add_argument("--qx",        type=float, default=10.0,  help="Position error weight (Q diagonal).")
parser.add_argument("--qtheta",    type=float, default=5.0,   help="Heading error weight (Q diagonal).")
parser.add_argument("--qv",        type=float, default=1.0,   help="Velocity error weight (Q diagonal).")
parser.add_argument("--r_force",   type=float, default=0.1,   help="Force effort weight (R diagonal).")
parser.add_argument("--r_torque",  type=float, default=0.1,   help="Thruster torque effort weight (R diagonal).")
parser.add_argument("--r_rw",      type=float, default=1.0,   help="Reaction wheel effort weight (R diagonal).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Isaaclab_RANSv2.tasks  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# LQR gain computation (runs once on CPU at startup)
# ──────────────────────────────────────────────────────────────────────────────

def compute_lqr_gain(
    mass: float,
    inertia_z: float,
    f_max: float,
    m_max: float,
    rw_max: float,
    qx: float,
    qtheta: float,
    qv: float,
    r_force: float,
    r_torque: float,
    r_rw: float,
) -> np.ndarray:
    """Solve the continuous-time LQR problem and return the gain matrix K (4×6).

    Linearised body-frame model (state = [ex, ey, eθ, vx, vy, ω]):

        ė = A·e + B·u,  u = [Fx(N), Fy(N), Mz_thrust(N·m), τ_rw(N·m)]

    Returns:
        K  (4, 6) ndarray — optimal gain, u* = -K @ e
    """
    # State matrix (double integrator, 3 DOF)
    A = np.zeros((6, 6))
    A[0, 3] = 1.0   # ėx  = vx
    A[1, 4] = 1.0   # ėy  = vy
    A[2, 5] = 1.0   # ėθ  = ω

    # Input matrix (physical units: N and N·m)
    B = np.zeros((6, 4))
    B[3, 0] = 1.0 / mass        # v̇x  = Fx / m
    B[4, 1] = 1.0 / mass        # v̇y  = Fy / m
    B[5, 2] = 1.0 / inertia_z   # ω̇   = Mz / Iz
    B[5, 3] = 1.0 / inertia_z   # ω̇  += τ_rw / Iz

    # Cost matrices
    Q = np.diag([qx, qx, qtheta, qv, qv, qv])
    R = np.diag([r_force, r_force, r_torque, r_rw])

    # Solve CARE: A'P + PA - PBR⁻¹B'P + Q = 0
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)   # K = R⁻¹ B' P  →  (4, 6)

    print("[LQR] Gain matrix K:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print(f"[LQR] Closed-loop eigenvalues: {np.linalg.eigvals(A - B @ K)}")
    return K


# ──────────────────────────────────────────────────────────────────────────────
# State extraction from GoToPose observations
# ──────────────────────────────────────────────────────────────────────────────

def obs_to_state_error(obs: torch.Tensor) -> torch.Tensor:
    """Extract the 6D body-frame error state from GoToPose observations.

    GoToPose obs layout:
        0  distance to target (m)
        1  cos(angle robot→target, body frame)
        2  sin(angle robot→target, body frame)
        3  cos(heading error)
        4  sin(heading error)
        5  vx body frame (m/s)
        6  vy body frame (m/s)
        7  ω  (rad/s)

    Args:
        obs: (num_envs, obs_dim) tensor — full observation from the environment.

    Returns:
        e:  (num_envs, 6) tensor — [ex_b, ey_b, eθ, vx_b, vy_b, ω]
    """
    dist    = obs[:, 0]          # distance to target (metres)
    cos_a   = obs[:, 1]          # cos of angle robot→target in body frame
    sin_a   = obs[:, 2]          # sin of angle robot→target in body frame
    cos_eth = obs[:, 3]          # cos of heading error
    sin_eth = obs[:, 4]          # sin of heading error
    vx      = obs[:, 5]
    vy      = obs[:, 6]
    omega   = obs[:, 7]

    ex_b = dist * cos_a                         # forward position error (body)
    ey_b = dist * sin_a                         # lateral position error (body)
    e_th = torch.atan2(sin_eth, cos_eth)        # heading error  ∈ (-π, π]

    return torch.stack([ex_b, ey_b, e_th, vx, vy, omega], dim=1)  # (N, 6)


# ──────────────────────────────────────────────────────────────────────────────
# Batched LQR policy
# ──────────────────────────────────────────────────────────────────────────────

def lqr_policy(
    obs: torch.Tensor,
    K: torch.Tensor,
    f_max: float,
    m_max: float,
    rw_max: float,
) -> torch.Tensor:
    """Apply the LQR control law to a batch of observations.

    u_physical = -K @ e  (physical units: N, N·m)
    Then normalise to action space [-1, 1].

    Args:
        obs:    (num_envs, obs_dim)
        K:      (4, 6) LQR gain tensor on the correct device
        f_max:  max force (N) corresponding to action=1
        m_max:  max thruster yaw torque (N·m) corresponding to action=1
        rw_max: max reaction wheel torque (N·m) corresponding to action=1

    Returns:
        actions: (num_envs, 4) tensor, clipped to [-1, 1]
    """
    e = obs_to_state_error(obs)             # (N, 6)
    u = -(e @ K.T)                          # (N, 4)  u = -K e

    # Scale physical outputs → action range [-1, 1]
    scale = torch.tensor([f_max, f_max, m_max, rw_max], device=obs.device)
    actions = (u / scale).clamp(-1.0, 1.0)
    return actions


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Compute LQR gains (CPU, runs once) ──────────────────────────────
    K_np = compute_lqr_gain(
        mass=args_cli.mass,
        inertia_z=args_cli.inertia_z,
        f_max=args_cli.f_max,
        m_max=args_cli.m_max,
        rw_max=args_cli.rw_max,
        qx=args_cli.qx,
        qtheta=args_cli.qtheta,
        qv=args_cli.qv,
        r_force=args_cli.r_force,
        r_torque=args_cli.r_torque,
        r_rw=args_cli.r_rw,
    )

    # ── 2. Create environment ──────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        "Isaaclab-RANSv2-AutoEnvGen-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.robot_name = "Cubo"
    env_cfg.task_name  = "GoToPose"

    env = gym.make("Isaaclab-RANSv2-AutoEnvGen-v0", cfg=env_cfg)

    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space:      {env.action_space}")

    device = env.unwrapped.device
    K = torch.tensor(K_np, dtype=torch.float32, device=device)

    # ── 3. Control loop ────────────────────────────────────────────────────
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]    # (num_envs, obs_dim)

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = lqr_policy(
                obs,
                K,
                f_max=args_cli.f_max,
                m_max=args_cli.m_max,
                rw_max=args_cli.rw_max,
            )
            obs_dict, rewards, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"]

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
