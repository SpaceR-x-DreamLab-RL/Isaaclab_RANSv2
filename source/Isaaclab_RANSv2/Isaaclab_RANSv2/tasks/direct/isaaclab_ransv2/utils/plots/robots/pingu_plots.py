from . import BaseRobotPlots, Registerable
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PinguPlots(BaseRobotPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info: dict, folder_path: list, plot_cfg: dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.robot_name = "Pingu"

        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if key.startswith("todo")
                )

        self.labels_to_plot = list(keys_set)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _has_trajectories(self) -> bool:
        return self.trajectories_to_plot is not None

    def _first_trajectory(self) -> pd.DataFrame | None:
        if not self._has_trajectories():
            return None
        first_id = self.trajectories_to_plot["trajectory_id"].iloc[0]
        return self.trajectories_to_plot[self.trajectories_to_plot["trajectory_id"] == first_id].copy()

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = os.path.join(self._save_plots_folder_path, f"{self.robot_name}_{name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    # ── Plot: 8 thruster commands (mean ± std over all trajectories) ─────────

    def plot_thruster_commands(self):
        if not self._has_trajectories():
            return

        thruster_cols = sorted(
            [c for c in self.trajectories_to_plot.columns if c.startswith("thrust_action_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        if not thruster_cols:
            print("[WARN] No 'thrust_action_*' columns found in trajectory data.")
            return

        colors = [
            "#FF3D50", "#FFA034", "#2FA1FF", "#A734FF",
            "#4DFF3D", "#FF3DBB", "#FFFF3D", "#623652",
        ]

        grouped = self.trajectories_to_plot.groupby("step")

        fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
        fig.suptitle("Thruster commands (mean ± std over all trajectories)", fontsize=13)

        for i, col in enumerate(thruster_cols):
            ax = axes[i // 2, i % 2]
            stats = grouped[col].agg(mean="mean", std="std").reset_index()
            steps = stats["step"].values
            mean = stats["mean"].values
            std = stats["std"].fillna(0).values
            color = colors[i % len(colors)]

            ax.plot(steps, mean, color=color, linewidth=1.4)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color=color)
            ax.set_ylabel("Force (N)")
            ax.set_title(f"Thruster {i}")
            ax.grid(True, alpha=0.3)

        for ax in axes[-1]:
            ax.set_xlabel("Step")

        plt.tight_layout()
        self._save(fig, "thruster_commands")

    # ── Plot: reaction wheel command ─────────────────────────────────────────

    def plot_reaction_wheel_command(self):
        if not self._has_trajectories():
            return

        col = "reaction_wheel_action"
        if col not in self.trajectories_to_plot.columns:
            print(f"[WARN] Column '{col}' not found in trajectory data.")
            return

        grouped = self.trajectories_to_plot.groupby("step")
        stats = grouped[col].agg(mean="mean", std="std").reset_index()
        steps = stats["step"].values
        mean = stats["mean"].values
        std = stats["std"].fillna(0).values

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, mean, color="#FF3D50", linewidth=1.6, label="mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="#FF3D50", label="± std")
        ax.set_xlabel("Step")
        ax.set_ylabel("Torque (u)")
        ax.set_title("Reaction wheel command (mean ± std over all trajectories)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "reaction_wheel_command")

    # ── Plot: robot linear & angular velocity magnitudes (track_velocities) ──

    def _is_track_velocities(self) -> bool:
        """Heuristic: track_velocities task exposes these target columns."""
        if not self._has_trajectories():
            return False
        cols = self.trajectories_to_plot.columns
        return any(c in cols for c in ("linear_velocity_target",
                                       "lateral_velocity_target",
                                       "angular_velocity_target"))

    def plot_velocity_magnitudes_over_time(self):
        """Plot robot linear-velocity magnitude (body frame, matches task linear
        target) and angular velocity (world-frame yaw, matches task angular
        target) over time (mean ± std across trajectories)."""
        if not self._has_trajectories():
            return

        df = self.trajectories_to_plot
        # Body-frame linear velocity is what the task targets (linear_velocity_target
        # is compared against root_com_lin_vel_b[:, 0]). Prefer world-frame yaw for
        # angular velocity since the task compares angular_velocity_target against
        # root_com_ang_vel_w[:, 2]; fall back to body-frame z if the world-frame
        # column isn't present (older eval runs).
        required = ["linear_velocity_x", "linear_velocity_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] Missing velocity columns for plot: {missing}")
            return

        ang_col = "angular_velocity_w_z" if "angular_velocity_w_z" in df.columns else "angular_velocity_z"
        if ang_col not in df.columns:
            print(f"[WARN] Missing angular velocity column for plot: {ang_col}")
            return

        df = df.copy()
        df["linear_velocity_mag"] = np.sqrt(
            df["linear_velocity_x"].values ** 2 + df["linear_velocity_y"].values ** 2
        )

        grouped = df.groupby("step")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Robot velocities over time (mean ± std over trajectories)", fontsize=13)

        # Linear velocity magnitude
        stats = grouped["linear_velocity_mag"].agg(mean="mean", std="std").reset_index()
        steps = stats["step"].values
        mean = stats["mean"].values
        std = stats["std"].fillna(0).values
        axes[0].plot(steps, mean, color="#2FA1FF", linewidth=1.6, label="robot")
        axes[0].fill_between(steps, mean - std, mean + std, alpha=0.25, color="#2FA1FF")
        if "linear_velocity_target" in df.columns:
            tgt = grouped["linear_velocity_target"].agg(mean="mean", std="std").reset_index()
            axes[0].plot(tgt["step"].values, tgt["mean"].values,
                         color="#FF3D50", linewidth=1.2, linestyle="--", label="target")
        axes[0].set_ylabel("Linear velocity magnitude (m/s)")
        axes[0].set_title("Linear velocity magnitude (body frame)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Angular velocity (world-frame z / yaw rate, matches task target)
        stats = grouped[ang_col].agg(mean="mean", std="std").reset_index()
        steps = stats["step"].values
        mean = stats["mean"].values
        std = stats["std"].fillna(0).values
        axes[1].plot(steps, mean, color="#A734FF", linewidth=1.6, label="robot")
        axes[1].fill_between(steps, mean - std, mean + std, alpha=0.25, color="#A734FF")
        if "angular_velocity_target" in df.columns:
            tgt = grouped["angular_velocity_target"].agg(mean="mean", std="std").reset_index()
            axes[1].plot(tgt["step"].values, tgt["mean"].values,
                         color="#FF3D50", linewidth=1.2, linestyle="--", label="target")
        frame = "world" if ang_col.startswith("angular_velocity_w") else "body"
        axes[1].set_ylabel(f"Angular velocity z ({frame} frame) (rad/s)")
        axes[1].set_title(f"Angular velocity (yaw, {frame} frame)")
        axes[1].set_xlabel("Step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "velocity_magnitudes_over_time")

    def plot_linear_velocity_components_over_time(self):
        """Plot robot linear velocity broken down into x and y body-frame
        components over time (mean ± std across trajectories)."""
        if not self._has_trajectories():
            return

        df = self.trajectories_to_plot
        required = ["linear_velocity_x", "linear_velocity_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] Missing velocity columns for plot: {missing}")
            return

        grouped = df.groupby("step")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Robot linear velocity components over time (mean ± std)", fontsize=13)

        components = [
            ("linear_velocity_x", "Linear velocity x (body frame)", "#FF3D50", "linear_velocity_target"),
            ("linear_velocity_y", "Linear velocity y (body frame)", "#FFA034", "lateral_velocity_target"),
        ]

        for ax, (col, title, color, target_col) in zip(axes, components):
            stats = grouped[col].agg(mean="mean", std="std").reset_index()
            steps = stats["step"].values
            mean = stats["mean"].values
            std = stats["std"].fillna(0).values
            ax.plot(steps, mean, color=color, linewidth=1.6, label="robot")
            ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color=color)
            if target_col in df.columns:
                tgt = grouped[target_col].agg(mean="mean", std="std").reset_index()
                ax.plot(tgt["step"].values, tgt["mean"].values,
                        color="#2FA1FF", linewidth=1.2, linestyle="--", label="target")
            ax.set_ylabel("Velocity (m/s)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        self._save(fig, "linear_velocity_components_over_time")

    # ── Plot: all trajectories overlaid (track_velocities) ──────────────────

    def plot_all_trajectories_velocities(self):
        """Overlay every trajectory's linear-magnitude, linear-x, linear-y and
        angular (yaw) velocities through time. Each trajectory is a thin line;
        target / effective target is drawn on top per-step (mean across traj)."""
        if not self._has_trajectories():
            return

        df = self.trajectories_to_plot
        required = ["linear_velocity_x", "linear_velocity_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] Missing velocity columns for plot: {missing}")
            return

        ang_col = "angular_velocity_w_z" if "angular_velocity_w_z" in df.columns else "angular_velocity_z"
        if ang_col not in df.columns:
            print(f"[WARN] Missing angular velocity column for plot: {ang_col}")
            return

        df = df.copy()
        df["linear_velocity_mag"] = np.sqrt(
            df["linear_velocity_x"].values ** 2 + df["linear_velocity_y"].values ** 2
        )

        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(
            "Robot velocities over time — all trajectories overlaid",
            fontsize=13,
        )

        panels = [
            ("linear_velocity_mag", "Linear velocity magnitude (body, m/s)",
             "#2FA1FF", "linear_velocity_target"),
            ("linear_velocity_x",   "Linear velocity x (body, m/s)",
             "#FF3D50", "linear_velocity_target"),
            ("linear_velocity_y",   "Linear velocity y (body, m/s)",
             "#FFA034", "lateral_velocity_target"),
            (ang_col, f"Angular velocity z ({'world' if ang_col.endswith('w_z') else 'body'}, rad/s)",
             "#A734FF", "effective_angular_velocity_target"),
        ]

        for ax, (col, title, color, target_col) in zip(axes, panels):
            # Plot every trajectory as a thin translucent line
            for traj_id, traj_df in df.groupby("trajectory_id"):
                ax.plot(
                    traj_df["step"].values,
                    traj_df[col].values,
                    color=color,
                    linewidth=0.7,
                    alpha=0.25,
                )
            # Overlay per-step mean across trajectories
            stats = df.groupby("step")[col].mean().reset_index()
            ax.plot(stats["step"].values, stats[col].values,
                    color=color, linewidth=2.0, label="mean across traj")
            # Target line (per-step mean of target)
            if target_col in df.columns:
                tgt = df.groupby("step")[target_col].mean().reset_index()
                ax.plot(tgt["step"].values, tgt[target_col].values,
                        color="black", linewidth=1.2, linestyle="--", label=target_col)
            elif target_col == "effective_angular_velocity_target" and "angular_velocity_target" in df.columns:
                tgt = df.groupby("step")["angular_velocity_target"].mean().reset_index()
                ax.plot(tgt["step"].values, tgt["angular_velocity_target"].values,
                        color="black", linewidth=1.2, linestyle="--", label="angular_velocity_target")
            ax.set_title(title)
            ax.set_ylabel(title.split(" (")[-1].rstrip(")"))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        self._save(fig, "all_trajectories_velocities")

    # ── Plot: single trajectory detail (track_velocities) ───────────────────

    def plot_single_trajectory_velocities(self, trajectory_id: int | None = None):
        """Plot linear (magnitude + x + y) and angular velocity through time for
        a single trajectory, with targets overlaid. If trajectory_id is None,
        the first one is plotted."""
        if not self._has_trajectories():
            return

        df_all = self.trajectories_to_plot
        required = ["linear_velocity_x", "linear_velocity_y"]
        missing = [c for c in required if c not in df_all.columns]
        if missing:
            print(f"[WARN] Missing velocity columns for plot: {missing}")
            return

        ang_col = "angular_velocity_w_z" if "angular_velocity_w_z" in df_all.columns else "angular_velocity_z"
        if ang_col not in df_all.columns:
            print(f"[WARN] Missing angular velocity column for plot: {ang_col}")
            return

        if trajectory_id is None:
            trajectory_id = int(df_all["trajectory_id"].iloc[0])

        df = df_all[df_all["trajectory_id"] == trajectory_id].copy()
        if df.empty:
            print(f"[WARN] No data for trajectory_id={trajectory_id}")
            return

        df["linear_velocity_mag"] = np.sqrt(
            df["linear_velocity_x"].values ** 2 + df["linear_velocity_y"].values ** 2
        )

        steps = df["step"].values

        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(
            f"Single trajectory velocities through time (trajectory_id={trajectory_id})",
            fontsize=13,
        )

        # Panel 1: linear velocity magnitude
        axes[0].plot(steps, df["linear_velocity_mag"].values,
                     color="#2FA1FF", linewidth=1.6, label="robot |v|")
        if "linear_velocity_target" in df.columns:
            axes[0].plot(steps, np.abs(df["linear_velocity_target"].values),
                         color="black", linewidth=1.2, linestyle="--",
                         label="|linear_velocity_target|")
        axes[0].set_ylabel("Linear speed (m/s)")
        axes[0].set_title("Linear velocity magnitude (body frame)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Panel 2: linear velocity x (body)
        axes[1].plot(steps, df["linear_velocity_x"].values,
                     color="#FF3D50", linewidth=1.6, label="robot x")
        if "linear_velocity_target" in df.columns:
            axes[1].plot(steps, df["linear_velocity_target"].values,
                         color="black", linewidth=1.2, linestyle="--",
                         label="linear_velocity_target")
        axes[1].set_ylabel("Velocity x (m/s)")
        axes[1].set_title("Linear velocity x (body frame)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: linear velocity y (body)
        axes[2].plot(steps, df["linear_velocity_y"].values,
                     color="#FFA034", linewidth=1.6, label="robot y")
        if "lateral_velocity_target" in df.columns:
            axes[2].plot(steps, df["lateral_velocity_target"].values,
                         color="black", linewidth=1.2, linestyle="--",
                         label="lateral_velocity_target")
        axes[2].set_ylabel("Velocity y (m/s)")
        axes[2].set_title("Linear velocity y (body frame)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Panel 4: angular velocity (yaw). Also overlay body-frame z if we have
        # both, so the user can see any body↔world divergence (non-zero roll/pitch).
        axes[3].plot(steps, df[ang_col].values,
                     color="#A734FF", linewidth=1.6,
                     label=f"robot ({'world' if ang_col.endswith('w_z') else 'body'}) z")
        other_ang = "angular_velocity_z" if ang_col == "angular_velocity_w_z" else "angular_velocity_w_z"
        if other_ang in df.columns:
            axes[3].plot(steps, df[other_ang].values,
                         color="#A734FF", linewidth=1.0, alpha=0.5, linestyle=":",
                         label=f"robot ({'body' if other_ang == 'angular_velocity_z' else 'world'}) z")
        if "effective_angular_velocity_target" in df.columns:
            axes[3].plot(steps, df["effective_angular_velocity_target"].values,
                         color="black", linewidth=1.2, linestyle="--",
                         label="effective target (ramped)")
        if "angular_velocity_target" in df.columns:
            axes[3].plot(steps, df["angular_velocity_target"].values,
                         color="gray", linewidth=1.0, linestyle=":",
                         label="angular_velocity_target (goal)")
        axes[3].set_ylabel("Angular velocity z (rad/s)")
        axes[3].set_xlabel("Step")
        axes[3].set_title("Angular velocity (yaw)")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, f"single_trajectory_{trajectory_id}_velocities")

    # ── Plot: arm joint commands ─────────────────────────────────────────────

    def plot_arm_commands(self):
        if not self._has_trajectories():
            return

        arm_cols = {
            "L shoulder": ("left_arm_position_x", "#FF3D50"),
            "L elbow":    ("left_arm_position_y", "#FFA034"),
            "R shoulder": ("right_arm_position_x", "#2FA1FF"),
            "R elbow":    ("right_arm_position_y", "#A734FF"),
        }

        available = {label: (col, color) for label, (col, color) in arm_cols.items()
                     if col in self.trajectories_to_plot.columns}
        if not available:
            print("[WARN] No arm position columns found in trajectory data.")
            return

        grouped = self.trajectories_to_plot.groupby("step")

        fig, axes = plt.subplots(len(available), 1, figsize=(12, 4 * len(available)), sharex=True)
        if len(available) == 1:
            axes = [axes]
        fig.suptitle("Arm joint positions (mean ± std over all trajectories)", fontsize=13)

        for ax, (label, (col, color)) in zip(axes, available.items()):
            stats = grouped[col].agg(mean="mean", std="std").reset_index()
            steps = stats["step"].values
            mean = stats["mean"].values
            std = stats["std"].fillna(0).values

            ax.plot(steps, mean, color=color, linewidth=1.4, label=label)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color=color)
            ax.set_ylabel("Joint position (rad)")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        self._save(fig, "arm_commands")

    # ── Entry point ───────────────────────────────────────────────────────────

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)

        if self._has_trajectories():
            self.plot_thruster_commands()
            self.plot_reaction_wheel_command()
            self.plot_arm_commands()
            if self._is_track_velocities():
                self.plot_velocity_magnitudes_over_time()
                self.plot_linear_velocity_components_over_time()
                self.plot_all_trajectories_velocities()
                self.plot_single_trajectory_velocities()
