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

    # ── Entry point ───────────────────────────────────────────────────────────

    def plot(self):
        for label_to_plot in self.labels_box_plot:
            self.boxplot(label_to_plot)

        if self._has_trajectories():
            self.plot_position_distance_over_time()
