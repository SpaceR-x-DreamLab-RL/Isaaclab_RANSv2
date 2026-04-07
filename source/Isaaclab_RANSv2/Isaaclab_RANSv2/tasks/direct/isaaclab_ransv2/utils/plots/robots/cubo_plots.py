from . import BaseRobotPlots, Registerable
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CuboPlots(BaseRobotPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info: dict, folder_path: list, plot_cfg: dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.robot_name = "Cubo"

        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if key.startswith("todo")
                )

        self.labels_to_plot = list(keys_set)

    # -- Helpers ---------------------------------------------------------------

    def _has_trajectories(self) -> bool:
        return self.trajectories_to_plot is not None

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = os.path.join(self._save_plots_folder_path, f"{self.robot_name}_{name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    # -- Plot: angular velocity over time (z-axis) ----------------------------

    def plot_angular_velocity(self):
        """Angular velocity (z-axis) over time with target overlay."""
        if not self._has_trajectories():
            return

        # 3D vector → _x, _y, _z suffixes; 1D scalar → bare name
        ang_vel_col = "angular_velocity_z"
        target_col = "angular_velocity_target"
        if ang_vel_col not in self.trajectories_to_plot.columns:
            print(f"[WARN] Column '{ang_vel_col}' not found in trajectory data.")
            return

        grouped = self.trajectories_to_plot.groupby("step")

        fig, ax = plt.subplots(figsize=(12, 5))

        # Actual angular velocity
        stats = grouped[ang_vel_col].agg(mean="mean", std="std").reset_index()
        steps = stats["step"].values
        mean = stats["mean"].values
        std = stats["std"].fillna(0).values
        ax.plot(steps, mean, color="#2FA1FF", linewidth=1.6, label="ang vel (z)")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#2FA1FF")

        # Target angular velocity
        if target_col in self.trajectories_to_plot.columns:
            t_stats = grouped[target_col].agg(mean="mean").reset_index()
            ax.plot(t_stats["step"].values, t_stats["mean"].values,
                    color="#FF3D50", linewidth=1.4, linestyle="--", label="target")

        ax.set_xlabel("Step")
        ax.set_ylabel("Angular velocity (rad/s)")
        ax.set_title("Angular velocity tracking (mean +/- std)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "angular_velocity")

    # -- Plot: reaction wheel command ------------------------------------------

    def plot_reaction_wheel_command(self):
        """Reaction wheel torque command over time."""
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
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="#FF3D50", label="+/- std")
        ax.set_xlabel("Step")
        ax.set_ylabel("Torque (u)")
        ax.set_title("Reaction wheel command (mean +/- std)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "reaction_wheel_command")

    # -- Plot: reaction wheel internal speed -----------------------------------

    def plot_reaction_wheel_omega(self):
        """Internal reaction wheel angular speed over time."""
        if not self._has_trajectories():
            return

        col = "omega_reaction_wheel"
        if col not in self.trajectories_to_plot.columns:
            print(f"[WARN] Column '{col}' not found in trajectory data.")
            return

        grouped = self.trajectories_to_plot.groupby("step")
        stats = grouped[col].agg(mean="mean", std="std").reset_index()
        steps = stats["step"].values
        mean = stats["mean"].values
        std = stats["std"].fillna(0).values

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, mean, color="#A734FF", linewidth=1.6, label="mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="#A734FF", label="+/- std")
        ax.set_xlabel("Step")
        ax.set_ylabel("Wheel speed (rad/s)")
        ax.set_title("Reaction wheel internal speed (mean +/- std)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "reaction_wheel_omega")

    # -- Plot: 8 thruster commands ---------------------------------------------

    def plot_thruster_commands(self):
        """Individual thruster force over time (mean +/- std)."""
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
        fig.suptitle("Thruster commands (mean +/- std over all trajectories)", fontsize=13)

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

    # -- Plot: combined overview (ang vel + RW command + RW omega) -------------

    def plot_overview(self):
        """Three-panel overview: angular velocity tracking, RW command, RW speed."""
        if not self._has_trajectories():
            return

        ang_vel_col = "angular_velocity_z"
        target_col = "angular_velocity_target"
        rw_cmd_col = "reaction_wheel_action"
        rw_omega_col = "omega_reaction_wheel"

        needed = [ang_vel_col, rw_cmd_col, rw_omega_col]
        missing = [c for c in needed if c not in self.trajectories_to_plot.columns]
        if missing:
            print(f"[WARN] Missing columns for overview plot: {missing}")
            return

        grouped = self.trajectories_to_plot.groupby("step")

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle("Cubo reaction wheel control overview", fontsize=14)

        # Panel 1: angular velocity
        ax = axes[0]
        stats = grouped[ang_vel_col].agg(mean="mean", std="std").reset_index()
        ax.plot(stats["step"], stats["mean"], color="#2FA1FF", linewidth=1.6, label="ang vel (z)")
        ax.fill_between(stats["step"], stats["mean"] - stats["std"].fillna(0),
                        stats["mean"] + stats["std"].fillna(0), alpha=0.2, color="#2FA1FF")
        if target_col in self.trajectories_to_plot.columns:
            t_stats = grouped[target_col].agg(mean="mean").reset_index()
            ax.plot(t_stats["step"], t_stats["mean"], color="#FF3D50", linestyle="--", linewidth=1.4, label="target")
        ax.set_ylabel("Angular vel (rad/s)")
        ax.set_title("Angular velocity tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: RW command
        ax = axes[1]
        stats = grouped[rw_cmd_col].agg(mean="mean", std="std").reset_index()
        ax.plot(stats["step"], stats["mean"], color="#FF3D50", linewidth=1.6)
        ax.fill_between(stats["step"], stats["mean"] - stats["std"].fillna(0),
                        stats["mean"] + stats["std"].fillna(0), alpha=0.25, color="#FF3D50")
        ax.set_ylabel("Torque (u)")
        ax.set_title("Reaction wheel command")
        ax.grid(True, alpha=0.3)

        # Panel 3: RW internal speed
        ax = axes[2]
        stats = grouped[rw_omega_col].agg(mean="mean", std="std").reset_index()
        ax.plot(stats["step"], stats["mean"], color="#A734FF", linewidth=1.6)
        ax.fill_between(stats["step"], stats["mean"] - stats["std"].fillna(0),
                        stats["mean"] + stats["std"].fillna(0), alpha=0.25, color="#A734FF")
        ax.set_ylabel("Wheel speed (rad/s)")
        ax.set_title("Reaction wheel internal speed")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "overview")

    # -- Plot: phase portrait (angular velocity vs RW command) -----------------

    def plot_phase_portrait(self):
        """Phase portrait: body angular velocity vs reaction wheel command.

        Each trajectory is drawn as a thin line; the mean trajectory is drawn bold.
        Shows how the policy maps angular velocity error to RW torque."""
        if not self._has_trajectories():
            return

        ang_vel_col = "angular_velocity_z"
        rw_cmd_col = "reaction_wheel_action"
        if ang_vel_col not in self.trajectories_to_plot.columns or rw_cmd_col not in self.trajectories_to_plot.columns:
            print("[WARN] Missing columns for phase portrait.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        # Individual trajectories
        for traj_id, traj_df in self.trajectories_to_plot.groupby("trajectory_id"):
            ax.plot(traj_df[ang_vel_col].values, traj_df[rw_cmd_col].values,
                    color="#2FA1FF", alpha=0.15, linewidth=0.8)

        # Mean trajectory
        grouped = self.trajectories_to_plot.groupby("step")
        mean_ang = grouped[ang_vel_col].mean().values
        mean_rw = grouped[rw_cmd_col].mean().values
        ax.plot(mean_ang, mean_rw, color="#FF3D50", linewidth=2.0, label="mean trajectory")

        # Mark start and end of mean trajectory
        ax.scatter(mean_ang[0], mean_rw[0], color="#4DFF3D", s=80, zorder=5, label="start")
        ax.scatter(mean_ang[-1], mean_rw[-1], color="#FF3DBB", s=80, zorder=5, marker="X", label="end")

        ax.set_xlabel("Body angular velocity (rad/s)")
        ax.set_ylabel("RW command (u)")
        ax.set_title("Phase portrait: angular velocity vs RW command")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("auto")
        self._save(fig, "phase_portrait")

    # -- Entry point -----------------------------------------------------------

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)

        if self._has_trajectories():
            self.plot_angular_velocity()
            self.plot_reaction_wheel_command()
            self.plot_reaction_wheel_omega()
            self.plot_thruster_commands()
            self.plot_overview()
            self.plot_phase_portrait()
