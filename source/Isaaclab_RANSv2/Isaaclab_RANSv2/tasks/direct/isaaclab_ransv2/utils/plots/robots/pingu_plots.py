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
