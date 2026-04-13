#!/usr/bin/env python3
"""
Aggregate evaluation metrics across seeds, generate comparative plots and a
summary text file (mean ± std across all seeds).

Usage (from the project root):
    python scripts/rsl_rl/plot_metrics.py <run_dir_1> [run_dir_2 ...]
"""

import argparse
import glob
import os
import sys
import traceback
import yaml
import pandas as pd
import numpy as np

# ── Import the existing plot factories ────────────────────────────────────────
_PLOTS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../source/Isaaclab_RANSv2/Isaaclab_RANSv2"
        "/tasks/direct/isaaclab_ransv2/utils/plots",
    )
)
sys.path.insert(0, _PLOTS_DIR)

from tasks import TaskPlotsFactory  # noqa: E402
from robots import RobotPlotsFactory  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_run(run_dir: str, task_name: str):
    """Return (metrics_df, trajectories_df, env_info) for one run directory."""
    metrics_dir = os.path.join(run_dir, "metrics")

    pattern = os.path.join(metrics_dir, "*_metrics.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No *_metrics.csv found in {metrics_dir}")
    df = pd.read_csv(matches[0])

    traj_files = glob.glob(os.path.join(metrics_dir, f"detailed_trajectories_{task_name}.csv"))
    traj_df = pd.read_csv(traj_files[0]) if traj_files else pd.DataFrame()

    env_info_path = os.path.join(metrics_dir, "env_info.yaml")
    env_info: dict = {}
    if os.path.exists(env_info_path):
        with open(env_info_path) as fh:
            env_info = yaml.safe_load(fh) or {}

    return df, traj_df, env_info


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot and summarize multi-seed evaluation metrics."
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories, one per seed.")
    parser.add_argument("--task", default="GoToPose", help="Task name (default: GoToPose)")
    parser.add_argument("--robot", default="Pingu", help="Robot name (default: Pingu)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder. Defaults to <parent of first run dir>/multiseed_results/",
    )
    args = parser.parse_args()

    task_name = args.task
    robot_name = args.robot

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.run_dirs[0])), "multiseed_results"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {out_dir}")

    # ── Load all seeds ────────────────────────────────────────────────────────
    all_dfs = []
    dfs: dict = {}
    trajectories_dfs: dict = {}
    labels: dict = {}
    env_info: dict = {}

    for seed_idx, run_dir in enumerate(args.run_dirs):
        seed_label = f"seed_{seed_idx + 1}"
        group_key = f"{task_name}_group-{seed_idx}_{seed_label}"
        try:
            df, traj_df, env_info = load_run(run_dir, task_name)
        except Exception as exc:
            print(f"[WARN] Could not load {run_dir}: {exc}")
            continue

        dfs[group_key] = [df]
        # Only include trajectory df if it actually has a 'trajectory' column;
        # passing an empty DataFrame causes KeyError inside BaseTaskPlots.__init__
        trajectories_dfs[group_key] = [traj_df] if "trajectory_id" in traj_df.columns else []
        labels[group_key] = [seed_label]
        all_dfs.append(df)
        print(f"[INFO] Loaded seed {seed_idx + 1}: {run_dir}")

    if not all_dfs:
        print("[ERROR] No valid runs loaded, aborting.")
        sys.exit(1)

    # ── Build a second dfs dict with all seeds pooled into one group ─────────
    combined_group_key = f"{task_name}_group-0_all_seeds"
    dfs_combined = {combined_group_key: all_dfs}
    traj_all = [t for group in trajectories_dfs.values() for t in group]
    trajectories_dfs_combined = {combined_group_key: traj_all} if traj_all else {}
    labels_combined = {combined_group_key: [f"seed_{i+1}" for i in range(len(all_dfs))]}

    # ── Plot helpers ──────────────────────────────────────────────────────────
    def run_task_plots(dfs_arg, traj_arg, labels_arg, out_folder, title_suffix):
        has_traj = any(v for v in traj_arg.values()) if traj_arg else False
        traj_final = traj_arg if has_traj else {}
        cfg = {
            "title": f"{robot_name} — {task_name} {title_suffix}",
            "box_colors": ["#FF3D50", "#FFA034", "#2FA1FF", "#A734FF", "#4DFF3D"],
            "runs_names": list(dfs_arg.keys()),
            "zoom_in": False,
        }
        os.makedirs(out_folder, exist_ok=True)
        try:
            task_plots = TaskPlotsFactory.create(
                task_name,
                dfs=dfs_arg,
                trajectories_dfs=traj_final,
                labels=labels_arg,
                env_info=env_info,
                folder_path=out_folder,
                plot_cfg=cfg,
            )
            task_plots.plot()
            print(f"[INFO] Task plots saved → {out_folder}")
        except Exception as exc:
            print(f"[WARN] Task plots failed ({title_suffix}): {exc}")
            traceback.print_exc()

        try:
            robot_plots = RobotPlotsFactory.create(
                robot_name,
                dfs=dfs_arg,
                trajectories_dfs=traj_final,
                labels=labels_arg,
                env_info=env_info,
                folder_path=out_folder,
                plot_cfg=cfg,
            )
            robot_plots.plot()
            print(f"[INFO] Robot plots saved → {out_folder}")
        except Exception as exc:
            print(f"[WARN] Robot plots skipped ('{robot_name}' not in registry): {exc}")
            traceback.print_exc()

    # ── Per-seed comparison plots ─────────────────────────────────────────────
    per_seed_labels = {k: [k.split("_")[-1]] for k in dfs}
    run_task_plots(dfs, trajectories_dfs, per_seed_labels,
                   os.path.join(out_dir, "per_seed"), "(per seed)")

    # ── Combined (all seeds pooled) plot ──────────────────────────────────────
    run_task_plots(dfs_combined, trajectories_dfs_combined, labels_combined,
                   os.path.join(out_dir, "combined"), "(all seeds combined)")

    # ── Text summary (mean ± std, 4 decimal places) ───────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    numeric_cols = combined.select_dtypes(include="number").columns

    summary_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(summary_path, "w") as fh:
        checkpoint_lines = "".join(
            f"  [{i+1}] {rd}\n" for i, rd in enumerate(args.run_dirs)
        )
        header = (
            f"Multi-seed evaluation summary\n"
            f"Robot : {robot_name}\n"
            f"Task  : {task_name}\n"
            f"Seeds : {len(all_dfs)}\n\n"
            f"Checkpoints evaluated:\n{checkpoint_lines}"
            f"{'=' * 60}\n\n"
        )
        fh.write(header)
        print(header, end="")

        for col in numeric_cols:
            vals = combined[col].dropna()
            if vals.empty:
                continue
            mean = vals.mean()
            std = vals.std()
            line = f"{col}\n  mean = {mean:.4f}  std = {std:.4f}\n\n"
            fh.write(line)
            print(line, end="")

    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
