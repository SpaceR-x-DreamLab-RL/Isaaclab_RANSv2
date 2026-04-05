#!/bin/bash
# Re-run IsaacLab evaluation for one or more existing run directories.
#
# The script finds the latest model checkpoint inside each run directory,
# extracts the robot/task names from the folder name, and calls eval.py.
# After all evaluations it re-runs plot_metrics.py to refresh the plots.
#
# Usage (from the project root):
#   bash scripts/rsl_rl/reeval.sh <run_dir_1> [run_dir_2 ...]
#
# Example:
#   bash scripts/rsl_rl/reeval.sh \
#     logs/rsl_rl/AutoEnvGen_PPO.../2026-04-04_15-57-17_ppo_Pingu_GoToPose_rsl_rl_seed_1 \
#     logs/rsl_rl/AutoEnvGen_PPO.../2026-04-04_19-04-36_ppo_Pingu_GoToPose_rsl_rl_seed_2

# ── Configuration ─────────────────────────────────────────────────────────────
TASK_ENV="Isaaclab-RANSv2-AutoEnvGen-v0"
NUM_ENVS=128
RUNS_PER_ENV=4
CUSTOM_PLOTS_LOG_NAME="reeval_plots"

PYTHON_EXE="${ISAACSIM_ROOT_PATH}/python.sh"

# ── Parse arguments ───────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    echo "[ERROR] No run directories provided."
    echo "Usage: bash scripts/rsl_rl/reeval.sh <run_dir_1> [run_dir_2 ...]"
    exit 1
fi

RUN_DIRS=("$@")

# ── Helpers ───────────────────────────────────────────────────────────────────

# Extract field N (0-indexed) from an underscore-split folder name.
# Folder format: {date}_{time}_{algo}_{Robot}_{Task}_rsl_rl_seed_{N}
#   split[0]=date  [1]=time  [2]=algo  [3]=Robot  [4]=Task
_field() {
    basename "$1" | cut -d'_' -f$(( $2 + 1 ))
}

# Find the checkpoint with the highest iteration number inside a run dir.
_latest_checkpoint() {
    local run_dir="$1"
    ls "$run_dir"/model_*.pt 2>/dev/null \
        | sort -t_ -k2 -V \
        | tail -1
}

# ── Evaluation loop ───────────────────────────────────────────────────────────
EVAL_RUN_DIRS=()
ROBOT_NAME=""
TASK_NAME=""

for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo "========================================"
    echo "Re-evaluating: ${RUN_DIR}"
    echo "========================================"

    if [ ! -d "$RUN_DIR" ]; then
        echo "✗ Directory not found, skipping: ${RUN_DIR}"
        continue
    fi

    # Extract robot and task from folder name
    _ROBOT=$(_field "$RUN_DIR" 3)
    _TASK=$(_field  "$RUN_DIR" 4)

    # Sanity-check that all runs share the same robot/task
    if [ -n "$ROBOT_NAME" ] && [ "$_ROBOT" != "$ROBOT_NAME" ]; then
        echo "[WARN] Robot mismatch: expected ${ROBOT_NAME}, got ${_ROBOT}. Skipping."
        continue
    fi
    if [ -n "$TASK_NAME" ] && [ "$_TASK" != "$TASK_NAME" ]; then
        echo "[WARN] Task mismatch: expected ${TASK_NAME}, got ${_TASK}. Skipping."
        continue
    fi
    ROBOT_NAME="$_ROBOT"
    TASK_NAME="$_TASK"

    CHECKPOINT=$(_latest_checkpoint "$RUN_DIR")
    if [ -z "$CHECKPOINT" ]; then
        echo "✗ No model_*.pt checkpoint found in ${RUN_DIR}"
        continue
    fi
    echo "[INFO] Using checkpoint: ${CHECKPOINT}"

    $PYTHON_EXE scripts/rsl_rl/eval.py \
        --task="${TASK_ENV}" \
        --headless \
        --num_envs=${NUM_ENVS} \
        --runs-per-env=${RUNS_PER_ENV} \
        env.robot_name="${ROBOT_NAME}" \
        env.task_name="${TASK_NAME}" \
        --checkpoint="${CHECKPOINT}"

    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed: ${RUN_DIR}"
        EVAL_RUN_DIRS+=("$RUN_DIR")
    else
        echo "✗ Evaluation failed:    ${RUN_DIR}"
    fi
    echo "----------------------------------------"
done

# ── Plots & summary ───────────────────────────────────────────────────────────
if [ ${#EVAL_RUN_DIRS[@]} -gt 0 ]; then
    # Derive EXPERIMENT_NAME from the parent folder of the first run dir
    EXPERIMENT_NAME=$(basename "$(dirname "${EVAL_RUN_DIRS[0]}")")

    echo ""
    echo "========================================"
    echo "Generating plots and metrics summary..."
    echo "========================================"

    $PYTHON_EXE scripts/rsl_rl/plot_metrics.py \
        --task="${TASK_NAME}" \
        --robot="${ROBOT_NAME}" \
        --out="logs/plots/${EXPERIMENT_NAME}/${CUSTOM_PLOTS_LOG_NAME}" \
        "${EVAL_RUN_DIRS[@]}"

    if [ $? -eq 0 ]; then
        echo "✓ Plots and summary saved to: logs/plots/${EXPERIMENT_NAME}/${CUSTOM_PLOTS_LOG_NAME}"
    else
        echo "✗ Plotting step failed."
    fi
else
    echo "[WARN] No successful evaluations — skipping plots."
fi

echo "All done!"
