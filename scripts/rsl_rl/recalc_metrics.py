#!/usr/bin/env python3
"""
Recalculate evaluation metrics from already-saved detailed trajectory CSV files,
without re-running the simulation.

The trajectory CSV already excludes the done step (the step where IsaacLab resets
the env), so the last row per trajectory_id is the last valid pre-reset observation.

Usage (from the project root):
    python scripts/rsl_rl/recalc_metrics.py <run_dir_1> [run_dir_2 ...]

Example:
    python scripts/rsl_rl/recalc_metrics.py \\
        logs/rsl_rl/AutoEnvGen_PPO.../2026-04-04_15-57-17_ppo_Pingu_GoToPose_rsl_rl_seed_1 \\
        logs/rsl_rl/AutoEnvGen_PPO.../2026-04-04_19-04-36_ppo_Pingu_GoToPose_rsl_rl_seed_2
"""

import argparse
import glob
import os
import sys
import traceback
import numpy as np
import pandas as pd


# ── Per-task metric definitions ───────────────────────────────────────────────

def _metrics_go_to_pose(last: pd.Series) -> dict:
    metrics = {}

    if "position_distance" in last.index:
        metrics["final_position_distance.m"] = last["position_distance"]

    if "target_heading" in last.index and "heading" in last.index:
        err = last["target_heading"] - last["heading"]
        metrics["final_orientation_error.rad"] = float(
            abs(np.arctan2(np.sin(err), np.cos(err)))
        )

    return metrics


TASK_METRIC_FNS = {
    "GoToPose": _metrics_go_to_pose,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_trajectories(run_dir: str, task_name: str) -> pd.DataFrame | None:
    pattern = os.path.join(run_dir, "metrics", f"detailed_trajectories_{task_name}.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[WARN] No trajectory file found at: {pattern}")
        return None
    df = pd.read_csv(files[0])
    print(f"[INFO] Loaded {df['trajectory_id'].nunique()} trajectories "
          f"({len(df)} rows) from {files[0]}")
    return df


def compute_metrics(traj_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """
    For each trajectory_id take its last step (last valid pre-reset observation)
    and compute per-trajectory metrics.
    """
    metric_fn = TASK_METRIC_FNS.get(task_name)
    if metric_fn is None:
        print(f"[WARN] No metric function registered for task '{task_name}'. "
              f"Available: {list(TASK_METRIC_FNS.keys())}")
        return pd.DataFrame()

    last_steps = (
        traj_df
        .loc[traj_df.groupby("trajectory_id")["step"].idxmax()]
        .reset_index(drop=True)
    )

    rows = [metric_fn(row) for _, row in last_steps.iterrows()]
    return pd.DataFrame(rows)


def metrics_save_path(run_dir: str, task_name: str) -> str:
    """Reproduce the filename convention used by eval_metrics.py."""
    base = os.path.basename(run_dir)
    parts = base.split("_")
    if len(parts) > 4:
        parts[4] = task_name
    name = "_".join(parts)
    return os.path.join(run_dir, "metrics", f"{name}_metrics.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Recalculate metrics from saved trajectory CSVs."
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories, one per seed.")
    parser.add_argument("--task", default="GoToPose", help="Task name (default: GoToPose)")
    args = parser.parse_args()

    all_summaries = []

    for run_dir in args.run_dirs:
        print(f"\n{'='*60}")
        print(f"[INFO] {run_dir}")

        try:
            traj_df = load_trajectories(run_dir, args.task)
            if traj_df is None:
                continue

            metrics_df = compute_metrics(traj_df, args.task)
            if metrics_df.empty:
                continue

            save_path = metrics_save_path(run_dir, args.task)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            metrics_df.to_csv(save_path, index=False)
            print(f"[INFO] Saved {len(metrics_df)} rows → {save_path}")

            # Print per-metric summary for this run
            for col in metrics_df.columns:
                vals = metrics_df[col].dropna()
                print(f"  {col:45s}  mean={vals.mean():.4f}  std={vals.std():.4f}")

            all_summaries.append(metrics_df)

        except Exception as exc:
            print(f"[ERROR] Failed for {run_dir}: {exc}")
            traceback.print_exc()

    # Cross-seed summary
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print(f"Cross-seed summary ({len(all_summaries)} runs)")
        print(f"{'='*60}")
        combined = pd.concat(all_summaries, ignore_index=True)
        for col in combined.columns:
            vals = combined[col].dropna()
            print(f"  {col:45s}  mean={vals.mean():.4f}  std={vals.std():.4f}")


if __name__ == "__main__":
    main()
