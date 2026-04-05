from . import BaseTaskPlots, Registerable
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GoToPosePlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "go_to_pose"

        # Box plots
        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if key.startswith("final_position_distance")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("final_orientation_error")
                )
        self.labels_box_plot = list(keys_set)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _has_trajectories(self) -> bool:
        return hasattr(self, "trajectories_to_plot") and self.trajectories_to_plot is not None

    def _first_trajectory(self) -> pd.DataFrame | None:
        """Return the rows belonging to the first available trajectory_id."""
        if not self._has_trajectories():
            return None
        first_id = self.trajectories_to_plot["trajectory_id"].iloc[0]
        return self.trajectories_to_plot[self.trajectories_to_plot["trajectory_id"] == first_id].copy()

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    # ── Plot: position distance mean ± std over steps ─────────────────────────

    def plot_position_distance_over_time(self):
        if not self._has_trajectories():
            return
        col = "position_distance"
        if col not in self.trajectories_to_plot.columns:
            print(f"[WARN] Column '{col}' not found in trajectory data.")
            return

        stats = (
            self.trajectories_to_plot
            .groupby("step")[col]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        steps = stats["step"].values
        mean  = stats["mean"].values
        std   = stats["std"].fillna(0).values

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, mean, color="#2FA1FF", linewidth=1.8, label="mean")
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.25, color="#2FA1FF", label="± std")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position distance (m)")
        ax.set_title("Position distance to target over time (all trajectories)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "position_distance_over_time")

    # ── Plot: 8 thruster activations for one trajectory ───────────────────────

    def plot_thruster_activations(self):
        if not self._has_trajectories():
            return
        traj = self._first_trajectory()
        if traj is None:
            return

        thruster_cols = sorted(
            [c for c in traj.columns if c.startswith("thrust_action_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        if not thruster_cols:
            print("[WARN] No 'thrust_action_*' columns found in trajectory data.")
            return

        colors = [
            "#FF3D50", "#FFA034", "#2FA1FF", "#A734FF",
            "#4DFF3D", "#FF3DBB", "#FFFF3D", "#623652",
        ]
        steps = traj["step"].values

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, col in enumerate(thruster_cols):
            ax.plot(steps, traj[col].values,
                    color=colors[i % len(colors)],
                    linewidth=1.4,
                    label=f"Thruster {i}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Force (N)")
        ax.set_title("Thruster forces over time (trajectory 0)")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3)
        self._save(fig, "thruster_activations")

    # ── Plot: reaction wheel + angular velocity + arm joints ──────────────────

    def plot_reaction_wheel_and_arms(self):
        if not self._has_trajectories():
            return
        traj = self._first_trajectory()
        if traj is None:
            return

        steps = traj["step"].values

        # Three sub-panels
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("Reaction wheel, angular velocity & arm joints (trajectory 0)", fontsize=12)

        # ── Panel 1: reaction wheel torque ────────────────────────────────────
        ax = axes[0]
        rw_col = "reaction_wheel_action"
        if rw_col in traj.columns:
            ax.plot(steps, traj[rw_col].values, color="#FF3D50", linewidth=1.6)
        else:
            print(f"[WARN] Column '{rw_col}' not found.")
        ax.set_ylabel("Torque (u)")
        ax.set_title("Reaction wheel torque")
        ax.grid(True, alpha=0.3)

        # ── Panel 2: body angular velocity (x, y, z) ─────────────────────────
        ax = axes[1]
        av_cols = {"x": "angular_velocity_x", "y": "angular_velocity_y", "z": "angular_velocity_z"}
        av_colors = {"x": "#2FA1FF", "y": "#FFA034", "z": "#4DFF3D"}
        for axis_name, col in av_cols.items():
            if col in traj.columns:
                ax.plot(steps, traj[col].values,
                        color=av_colors[axis_name],
                        linewidth=1.4,
                        label=f"ω_{axis_name}")
        ax.set_ylabel("Angular velocity (rad/s)")
        ax.set_title("Body angular velocity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Panel 3: arm joint positions ──────────────────────────────────────
        ax = axes[2]
        arm_cols = {
            "L shoulder": ("left_arm_position_x",  "#FF3D50"),
            "L elbow":    ("left_arm_position_y",  "#FFA034"),
            "R shoulder": ("right_arm_position_x", "#2FA1FF"),
            "R elbow":    ("right_arm_position_y", "#A734FF"),
        }
        for label, (col, color) in arm_cols.items():
            if col in traj.columns:
                ax.plot(steps, traj[col].values, color=color, linewidth=1.4, label=label)
            else:
                print(f"[WARN] Column '{col}' not found.")
        ax.set_xlabel("Step")
        ax.set_ylabel("Joint position (rad)")
        ax.set_title("Arm joint positions")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, "reaction_wheel_and_arms")

    # ── Entry point ───────────────────────────────────────────────────────────

    def plot(self):
        for label_to_plot in self.labels_box_plot:
            self.boxplot(label_to_plot)

        if self._has_trajectories():
            self.plot_position_distance_over_time()
            self.plot_thruster_activations()
            self.plot_reaction_wheel_and_arms()
