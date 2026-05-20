"""Warp-based MPPI controller for the 2D floating platform."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import warp as wp


@wp.func
def _wrap_angle(angle: wp.float32) -> wp.float32:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def _body_to_world(v_b: wp.vec3f, yaw: wp.float32) -> wp.vec3f:
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    return wp.vec3f(c * v_b[0] - s * v_b[1], s * v_b[0] + c * v_b[1], 0.0)


@wp.func
def ctrl3_to_8motors(u: wp.vec3f) -> None:
    """Map body-frame control to 8 motor commands in [0, 1]."""
    F_motor_0 = 0.0
    F_motor_1 = 0.0
    F_motor_2 = 0.0
    F_motor_3 = 0.0
    F_motor_4 = 0.0
    F_motor_5 = 0.0
    F_motor_6 = 0.0
    F_motor_7 = 0.0

    if u[0] != 0.0:
        mag = wp.abs(u[0])
        if u[0] > 0.0:
            F_motor_1 += mag
            F_motor_6 += mag
        else:
            F_motor_2 += mag
            F_motor_5 += mag

    if u[1] != 0.0:
        mag = wp.abs(u[1])
        if u[1] > 0.0:
            F_motor_0 += mag
            F_motor_3 += mag
        else:
            F_motor_4 += mag
            F_motor_7 += mag

    if u[2] != 0.0:
        mag = wp.abs(u[2])
        if u[2] < 0.0:
            F_motor_0 += mag
            F_motor_2 += mag
            F_motor_4 += mag
            F_motor_6 += mag
        else:
            F_motor_1 += mag
            F_motor_3 += mag
            F_motor_5 += mag
            F_motor_7 += mag

    F_motor_0 = wp.clamp(F_motor_0, 0.0, 1.0)
    F_motor_1 = wp.clamp(F_motor_1, 0.0, 1.0)
    F_motor_2 = wp.clamp(F_motor_2, 0.0, 1.0)
    F_motor_3 = wp.clamp(F_motor_3, 0.0, 1.0)
    F_motor_4 = wp.clamp(F_motor_4, 0.0, 1.0)
    F_motor_5 = wp.clamp(F_motor_5, 0.0, 1.0)
    F_motor_6 = wp.clamp(F_motor_6, 0.0, 1.0)
    F_motor_7 = wp.clamp(F_motor_7, 0.0, 1.0)

    return F_motor_0, F_motor_1, F_motor_2, F_motor_3, F_motor_4, F_motor_5, F_motor_6, F_motor_7


@wp.func
def compute_thrust_forces_and_moments(
    F_motor_0: wp.float32,
    F_motor_1: wp.float32,
    F_motor_2: wp.float32,
    F_motor_3: wp.float32,
    F_motor_4: wp.float32,
    F_motor_5: wp.float32,
    F_motor_6: wp.float32,
    F_motor_7: wp.float32,
    motor_thrust: wp.float32,
    motor_arm: wp.float32,
    tau_rw: wp.float32,
) -> None:
    F_0 = motor_thrust * F_motor_0 * wp.vec2f(0.0, 1.0)
    F_1 = motor_thrust * F_motor_1 * wp.vec2f(1.0, 0.0)
    F_2 = motor_thrust * F_motor_2 * wp.vec2f(-1.0, 0.0)
    F_3 = motor_thrust * F_motor_3 * wp.vec2f(0.0, 1.0)
    F_4 = motor_thrust * F_motor_4 * wp.vec2f(0.0, -1.0)
    F_5 = motor_thrust * F_motor_5 * wp.vec2f(-1.0, 0.0)
    F_6 = motor_thrust * F_motor_6 * wp.vec2f(1.0, 0.0)
    F_7 = motor_thrust * F_motor_7 * wp.vec2f(0.0, -1.0)

    M_0 = -motor_arm * motor_thrust * F_motor_0
    M_1 = motor_arm * motor_thrust * F_motor_1
    M_2 = -motor_arm * motor_thrust * F_motor_2
    M_3 = motor_arm * motor_thrust * F_motor_3
    M_4 = -motor_arm * motor_thrust * F_motor_4
    M_5 = motor_arm * motor_thrust * F_motor_5
    M_6 = -motor_arm * motor_thrust * F_motor_6
    M_7 = motor_arm * motor_thrust * F_motor_7

    F_T_b = F_0 + F_1 + F_2 + F_3 + F_4 + F_5 + F_6 + F_7
    F_T = wp.vec3f(F_T_b[0], F_T_b[1], 0.0)

    M_T_z = M_0 + M_1 + M_2 + M_3 + M_4 + M_5 + M_6 + M_7
    M_T_z -= tau_rw
    M_T = wp.vec3f(0.0, 0.0, M_T_z)

    return F_T, M_T


@wp.func
def compute_derivatives(
    vel_b: wp.vec3f,
    yaw: wp.float32,
    omega: wp.float32,
    Fb: wp.vec3f,
    Mb: wp.vec3f,
    m: wp.float32,
    J_zz: wp.float32,
) -> None:
    dot_pos = _body_to_world(vel_b, yaw)
    dot_vb = wp.vec3f(Fb[0] / m + vel_b[1] * omega, Fb[1] / m - vel_b[0] * omega, 0.0)
    dot_omega = Mb[2] / J_zz
    dot_yaw = omega
    return dot_pos, dot_vb, dot_omega, dot_yaw


@wp.func
def euler_update(
    position: wp.vec3f,
    linear_vel: wp.vec3f,
    yaw: wp.float32,
    omega: wp.float32,
    dt: wp.float32,
    dot_pos: wp.vec3f,
    dot_vb: wp.vec3f,
    dot_omega: wp.float32,
    dot_yaw: wp.float32,
) -> None:
    position += dt * dot_pos
    linear_vel += dt * dot_vb
    omega += dt * dot_omega
    yaw = _wrap_angle(yaw + dt * dot_yaw)
    return position, linear_vel, yaw, omega


@wp.kernel
def _sample_inputs_sequences(
    num_envs: wp.int32,
    num_samples: wp.int32,
    horizon: wp.int32,
    seed: wp.int32,
    previous_action: wp.array(dtype=wp.float32),
    noise_std_dev: wp.float32,
    noisy_action: wp.array(dtype=wp.float32),
    bias_cost_term: wp.array(dtype=wp.float32),
    temperature: wp.float32,
):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)

    env_id = tid // (num_samples * horizon)
    local = tid - env_id * num_samples * horizon
    traj_id = local // horizon
    t = local - traj_id * horizon

    previous = previous_action[env_id * horizon + t]
    sampled = previous + noise_std_dev * wp.randn(state)
    noisy = wp.clamp(sampled, -1.0, 1.0)
    noisy_action[tid] = noisy

    bias_cost = 0.0
    if noise_std_dev > 1.0e-12:
        delta_u = noisy - previous
        bias_cost = temperature * (previous * delta_u / (noise_std_dev ** 2.0))

    wp.atomic_add(bias_cost_term, env_id * num_samples + traj_id, bias_cost)


@wp.kernel
def _convert_inputs_sequences_into_trajectories(
    num_envs: wp.int32,
    num_samples: wp.int32,
    horizon: wp.int32,
    pos0: wp.array(dtype=wp.vec3f),
    vel0: wp.array(dtype=wp.vec3f),
    yaw0: wp.array(dtype=wp.float32),
    omega0: wp.array(dtype=wp.float32),
    position_memory: wp.array(dtype=wp.vec3f),
    linear_vel_memory: wp.array(dtype=wp.vec3f),
    yaw_memory: wp.array(dtype=wp.float32),
    omega_memory: wp.array(dtype=wp.float32),
    pwm_x: wp.array(dtype=wp.float32),
    pwm_y: wp.array(dtype=wp.float32),
    pwm_theta: wp.array(dtype=wp.float32),
    pwm_rw: wp.array(dtype=wp.float32),
    dt: wp.float32,
    m: wp.float32,
    J_zz: wp.float32,
    motor_thrust: wp.float32,
    motor_arm: wp.float32,
):
    tid = wp.tid()
    env_id = tid // num_samples
    traj_id = tid - env_id * num_samples

    current_position = pos0[env_id]
    current_linear_vel = vel0[env_id]
    current_yaw = yaw0[env_id]
    current_omega = omega0[env_id]

    for step in range(horizon):
        idx = (env_id * num_samples + traj_id) * horizon + step

        pwm_cmd = wp.vec3f(pwm_x[idx], pwm_y[idx], pwm_theta[idx])
        pwm_rw_cmd = pwm_rw[idx]

        F0, F1, F2, F3, F4, F5, F6, F7 = ctrl3_to_8motors(pwm_cmd)
        Fb, Mb = compute_thrust_forces_and_moments(
            F0,
            F1,
            F2,
            F3,
            F4,
            F5,
            F6,
            F7,
            motor_thrust,
            motor_arm,
            pwm_rw_cmd,
        )

        dot_pos, dot_vb, dot_omega, dot_yaw = compute_derivatives(
            current_linear_vel,
            current_yaw,
            current_omega,
            Fb,
            Mb,
            m,
            J_zz,
        )

        current_position, current_linear_vel, current_yaw, current_omega = euler_update(
            current_position,
            current_linear_vel,
            current_yaw,
            current_omega,
            dt,
            dot_pos,
            dot_vb,
            dot_omega,
            dot_yaw,
        )

        position_memory[idx] = current_position
        linear_vel_memory[idx] = current_linear_vel
        yaw_memory[idx] = current_yaw
        omega_memory[idx] = current_omega


@wp.kernel
def _evaluate_trajectories(
    num_envs: wp.int32,
    num_samples: wp.int32,
    horizon: wp.int32,
    position_memory: wp.array(dtype=wp.vec3f),
    linear_vel_memory: wp.array(dtype=wp.vec3f),
    yaw_memory: wp.array(dtype=wp.float32),
    omega_memory: wp.array(dtype=wp.float32),
    pwm_x: wp.array(dtype=wp.float32),
    pwm_y: wp.array(dtype=wp.float32),
    pwm_theta: wp.array(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
    goal_pos: wp.array(dtype=wp.vec2f),
    goal_heading: wp.array(dtype=wp.float32),
    has_heading: wp.int32,
    w_pos: wp.float32,
    w_heading: wp.float32,
    w_vel: wp.float32,
    w_omega: wp.float32,
    w_u: wp.float32,
):
    tid = wp.tid()

    env_id = tid // (num_samples * horizon)
    local = tid - env_id * num_samples * horizon
    traj_id = local // horizon
    idx = tid

    pos = position_memory[idx]
    vel = linear_vel_memory[idx]
    yaw = yaw_memory[idx]
    omega = omega_memory[idx]

    goal = goal_pos[env_id]
    dx = pos[0] - goal[0]
    dy = pos[1] - goal[1]
    pos_cost = dx * dx + dy * dy

    heading_cost = 0.0
    if has_heading != 0:
        heading_err = _wrap_angle(yaw - goal_heading[env_id])
        heading_cost = heading_err * heading_err

    vel_cost = vel[0] * vel[0] + vel[1] * vel[1]
    omega_cost = omega * omega

    u_cost = pwm_x[idx] * pwm_x[idx] + pwm_y[idx] * pwm_y[idx] + pwm_theta[idx] * pwm_theta[idx]

    total_cost = (
        w_pos * pos_cost
        + w_heading * heading_cost
        + w_vel * vel_cost
        + w_omega * omega_cost
        + w_u * u_cost
    )

    wp.atomic_add(costs, env_id * num_samples + traj_id, total_cost)


@wp.kernel
def _compute_min_cost(
    num_samples: wp.int32,
    costs: wp.array(dtype=wp.float32),
    min_cost: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_id = tid // num_samples
    wp.atomic_min(min_cost, env_id, costs[tid])


@wp.kernel
def _compute_eta(
    num_samples: wp.int32,
    costs: wp.array(dtype=wp.float32),
    min_cost: wp.array(dtype=wp.float32),
    bias_cost_term: wp.array(dtype=wp.float32),
    temperature: wp.float32,
    eta: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_id = tid // num_samples
    normalized_cost = costs[tid] + bias_cost_term[tid] - min_cost[env_id]
    exponent = wp.clamp(-normalized_cost / temperature, -60.0, 60.0)
    wp.atomic_add(eta, env_id, wp.exp(exponent))


@wp.kernel
def _compute_final_weights(
    num_samples: wp.int32,
    costs: wp.array(dtype=wp.float32),
    min_cost: wp.array(dtype=wp.float32),
    bias_cost_term: wp.array(dtype=wp.float32),
    temperature: wp.float32,
    eta: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_id = tid // num_samples
    normalized_cost = costs[tid] + bias_cost_term[tid] - min_cost[env_id]
    exponent = wp.clamp(-normalized_cost / temperature, -60.0, 60.0)
    if eta[env_id] > 1e-10:
        weights[tid] = wp.exp(exponent) / eta[env_id]
    else:
        weights[tid] = 1.0 / wp.float32(num_samples)


@wp.kernel
def _compute_weighted_average(
    num_envs: wp.int32,
    num_samples: wp.int32,
    horizon: wp.int32,
    weights: wp.array(dtype=wp.float32),
    sampled_actions: wp.array(dtype=wp.float32),
    optimal_action: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_id = tid // (num_samples * horizon)
    local = tid - env_id * num_samples * horizon
    traj_id = local // horizon
    t = local - traj_id * horizon

    weight = weights[env_id * num_samples + traj_id]
    wp.atomic_add(optimal_action, env_id * horizon + t, sampled_actions[tid] * weight)


@wp.kernel
def _shift_sequence(
    num_envs: wp.int32,
    horizon: wp.int32,
    u: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_id = tid // (horizon - 1)
    t = tid - env_id * (horizon - 1)
    u[env_id * horizon + t] = u[env_id * horizon + t + 1]


@wp.kernel
def apply_mppi_control_kernel(
    commands: wp.array2d(dtype=wp.float32),
    opt_x: wp.array(dtype=wp.float32),
    opt_y: wp.array(dtype=wp.float32),
    opt_theta: wp.array(dtype=wp.float32),
    opt_rw: wp.array(dtype=wp.float32),
    horizon: wp.int32,
    action_dim: wp.int32,
):
    tid = wp.tid()
    idx = tid * horizon

    u_x = opt_x[idx]
    u_y = opt_y[idx]
    u_theta = opt_theta[idx]
    u_rw = opt_rw[idx]

    commands[tid, 0] = u_x
    commands[tid, 1] = -u_y
    commands[tid, 2] = u_theta

    for j in range(3, action_dim):
        commands[tid, j] = 0.0


@dataclass(frozen=True)
class MPPIWeights:
    pos: float = 25.0
    heading: float = 8.0
    vel: float = 0.1
    omega: float = 0.1
    u: float = 0.01


class MPPIController:
    """Warp MPPI helper that accepts torch states and returns torch actions."""

    def __init__(
        self,
        specs: dict,
        device: str = "cuda",
        num_envs: int = 1,
        action_dim: int = 8,
        dt: float = 0.01,
        mass: float = 34.0,
        j_zz: float = 0.4343,
        motor_thrust: float = 1.0,
        motor_arm: float = 0.25,
        weights: MPPIWeights | None = None,
    ) -> None:
        wp.init()

        self.device = device
        self.torch_device = torch.device(device)
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.dt = float(dt)
        self.mass = float(mass)
        self.j_zz = float(j_zz)
        self.motor_thrust = float(motor_thrust)
        self.motor_arm = float(motor_arm)
        self.weights = weights or MPPIWeights()

        self.number_of_sampled_trajectories = int(specs.get("number_of_sampled_trajectories", 256))
        self.number_of_iterations_per_sample = int(specs.get("number_of_iterations_per_sample", 32))
        self.noise_std_dev = {
            "pwm_x": float(specs.get("noise_std_dev_pwm_x", 0.3)),
            "pwm_y": float(specs.get("noise_std_dev_pwm_y", 0.3)),
            "pwm_theta": float(specs.get("noise_std_dev_pwm_theta", 0.2)),
            "pwm_rw": float(specs.get("noise_std_dev_pwm_rw", 0.0)),
        }
        self.temperature = max(float(specs.get("temperature", 0.05)), 1.0e-6)

        self._traj_count = self.num_envs * self.number_of_sampled_trajectories
        self._seq_len = self._traj_count * self.number_of_iterations_per_sample
        self._env_seq_len = self.num_envs * self.number_of_iterations_per_sample

        self._build_buffers()

    def _build_buffers(self) -> None:
        self.optimal_sequence = {
            "pwm_x": wp.zeros((self._env_seq_len,), device=self.device, dtype=wp.float32),
            "pwm_y": wp.zeros((self._env_seq_len,), device=self.device, dtype=wp.float32),
            "pwm_theta": wp.zeros((self._env_seq_len,), device=self.device, dtype=wp.float32),
            "pwm_rw": wp.zeros((self._env_seq_len,), device=self.device, dtype=wp.float32),
        }
        self.sampled_sequences = {
            "pwm_x": wp.zeros((self._seq_len,), device=self.device, dtype=wp.float32),
            "pwm_y": wp.zeros((self._seq_len,), device=self.device, dtype=wp.float32),
            "pwm_theta": wp.zeros((self._seq_len,), device=self.device, dtype=wp.float32),
            "pwm_rw": wp.zeros((self._seq_len,), device=self.device, dtype=wp.float32),
        }

        self.position_memory = wp.zeros((self._seq_len,), device=self.device, dtype=wp.vec3f)
        self.linear_vel_memory = wp.zeros((self._seq_len,), device=self.device, dtype=wp.vec3f)
        self.yaw_memory = wp.zeros((self._seq_len,), device=self.device, dtype=wp.float32)
        self.omega_memory = wp.zeros((self._seq_len,), device=self.device, dtype=wp.float32)

        self.costs = wp.zeros((self._traj_count,), device=self.device, dtype=wp.float32)
        self.bias_cost_term = wp.zeros((self._traj_count,), device=self.device, dtype=wp.float32)
        self.min_cost = wp.zeros((self.num_envs,), device=self.device, dtype=wp.float32)
        self.weights_wp = wp.zeros((self._traj_count,), device=self.device, dtype=wp.float32)
        self.eta = wp.zeros((self.num_envs,), device=self.device, dtype=wp.float32)

        self._actions_torch = torch.zeros((self.num_envs, self.action_dim), device=self.torch_device, dtype=torch.float32)
        self._actions_wp = wp.from_torch(self._actions_torch, dtype=wp.float32)

    def reset(self) -> None:
        for seq in self.optimal_sequence.values():
            seq.zero_()

    def _reset_material_buffers(self) -> None:
        self.costs.zero_()
        self.bias_cost_term.zero_()
        self.min_cost.fill_(float("inf"))
        self.weights_wp.zero_()
        self.eta.zero_()

    def _shift_sequence(self) -> None:
        if self.number_of_iterations_per_sample <= 1:
            return
        dim = self.num_envs * (self.number_of_iterations_per_sample - 1)
        for key in self.optimal_sequence:
            wp.launch(
                kernel=_shift_sequence,
                dim=dim,
                inputs=[
                    self.num_envs,
                    self.number_of_iterations_per_sample,
                    self.optimal_sequence[key],
                ],
                device=self.device,
            )

    def _sample_inputs_sequences(self) -> None:
        for (key_prev, previous_action), (key_noise, noise_std_dev), (key_sampled, noisy_action) in zip(
            self.optimal_sequence.items(),
            self.noise_std_dev.items(),
            self.sampled_sequences.items(),
        ):
            wp.launch(
                kernel=_sample_inputs_sequences,
                dim=self._seq_len,
                inputs=[
                    self.num_envs,
                    self.number_of_sampled_trajectories,
                    self.number_of_iterations_per_sample,
                    int(np.random.randint(0, 100000)),
                    previous_action,
                    noise_std_dev,
                    noisy_action,
                    self.bias_cost_term,
                    self.temperature,
                ],
                device=self.device,
            )

    def _convert_inputs_sequences_into_trajectories(
        self,
        pos_w: torch.Tensor,
        lin_vel_b: torch.Tensor,
        yaw_w: torch.Tensor,
        ang_vel_b: torch.Tensor,
    ) -> None:
        pos_wp = wp.from_torch(pos_w.contiguous(), dtype=wp.vec3f)
        vel_wp = wp.from_torch(lin_vel_b.contiguous(), dtype=wp.vec3f)
        yaw_wp = wp.from_torch(yaw_w.contiguous(), dtype=wp.float32)
        omega_wp = wp.from_torch(ang_vel_b.contiguous(), dtype=wp.float32)

        wp.launch(
            kernel=_convert_inputs_sequences_into_trajectories,
            dim=self._traj_count,
            inputs=[
                self.num_envs,
                self.number_of_sampled_trajectories,
                self.number_of_iterations_per_sample,
                pos_wp,
                vel_wp,
                yaw_wp,
                omega_wp,
                self.position_memory,
                self.linear_vel_memory,
                self.yaw_memory,
                self.omega_memory,
                self.sampled_sequences["pwm_x"],
                self.sampled_sequences["pwm_y"],
                self.sampled_sequences["pwm_theta"],
                self.sampled_sequences["pwm_rw"],
                self.dt,
                self.mass,
                self.j_zz,
                self.motor_thrust,
                self.motor_arm,
            ],
            device=self.device,
        )

    def _evaluate_trajectories(
        self,
        goal_pos_w: torch.Tensor,
        goal_heading_w: torch.Tensor | None,
    ) -> None:
        goal_pos_wp = wp.from_torch(goal_pos_w.contiguous(), dtype=wp.vec2f)
        if goal_heading_w is None:
            goal_heading_w = torch.zeros((self.num_envs,), device=goal_pos_w.device, dtype=goal_pos_w.dtype)
            has_heading = 0
        else:
            has_heading = 1

        goal_heading_wp = wp.from_torch(goal_heading_w.contiguous(), dtype=wp.float32)

        wp.launch(
            kernel=_evaluate_trajectories,
            dim=self._seq_len,
            inputs=[
                self.num_envs,
                self.number_of_sampled_trajectories,
                self.number_of_iterations_per_sample,
                self.position_memory,
                self.linear_vel_memory,
                self.yaw_memory,
                self.omega_memory,
                self.sampled_sequences["pwm_x"],
                self.sampled_sequences["pwm_y"],
                self.sampled_sequences["pwm_theta"],
                self.costs,
                goal_pos_wp,
                goal_heading_wp,
                has_heading,
                float(self.weights.pos),
                float(self.weights.heading),
                float(self.weights.vel),
                float(self.weights.omega),
                float(self.weights.u),
            ],
            device=self.device,
        )

    def _compute_weights(self) -> None:
        wp.launch(
            kernel=_compute_min_cost,
            dim=self._traj_count,
            inputs=[
                self.number_of_sampled_trajectories,
                self.costs,
                self.min_cost,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_compute_eta,
            dim=self._traj_count,
            inputs=[
                self.number_of_sampled_trajectories,
                self.costs,
                self.min_cost,
                self.bias_cost_term,
                self.temperature,
                self.eta,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_compute_final_weights,
            dim=self._traj_count,
            inputs=[
                self.number_of_sampled_trajectories,
                self.costs,
                self.min_cost,
                self.bias_cost_term,
                self.temperature,
                self.eta,
                self.weights_wp,
            ],
            device=self.device,
        )

    def _compute_weighted_average(self) -> None:
        for key in self.optimal_sequence:
            self.optimal_sequence[key].zero_()

        for (key_sampled, sampled_actions), (key_opt, optimal_action) in zip(
            self.sampled_sequences.items(),
            self.optimal_sequence.items(),
        ):
            wp.launch(
                kernel=_compute_weighted_average,
                dim=self._seq_len,
                inputs=[
                    self.num_envs,
                    self.number_of_sampled_trajectories,
                    self.number_of_iterations_per_sample,
                    self.weights_wp,
                    sampled_actions,
                    optimal_action,
                ],
                device=self.device,
            )

    def _apply_mppi_control(self) -> torch.Tensor:
        wp.launch(
            kernel=apply_mppi_control_kernel,
            dim=self.num_envs,
            inputs=[
                self._actions_wp,
                self.optimal_sequence["pwm_x"],
                self.optimal_sequence["pwm_y"],
                self.optimal_sequence["pwm_theta"],
                self.optimal_sequence["pwm_rw"],
                self.number_of_iterations_per_sample,
                self.action_dim,
            ],
            device=self.device,
        )
        return self._actions_torch

    def compute_control(
        self,
        pos_w: torch.Tensor,
        lin_vel_b: torch.Tensor,
        yaw_w: torch.Tensor,
        ang_vel_b: torch.Tensor,
        goal_pos_w: torch.Tensor,
        goal_heading_w: torch.Tensor | None,
        dt: float,
    ) -> torch.Tensor:
        """Compute actions from the current state and goal."""
        self.dt = float(dt)

        self._reset_material_buffers()
        self._shift_sequence()
        self._sample_inputs_sequences()
        self._convert_inputs_sequences_into_trajectories(pos_w, lin_vel_b, yaw_w, ang_vel_b)
        self._evaluate_trajectories(goal_pos_w, goal_heading_w)
        self._compute_weights()
        self._compute_weighted_average()
        return self._apply_mppi_control()
