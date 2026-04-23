#!/bin/bash

# ── Configuration ─────────────────────────────────────────────────────────────
TASK_ENV="Isaaclab-RANSv2-AutoEnvGen-v0"
ROBOT_NAME="Pingu"
TASK_NAME="TrackVelocities"
CUSTOM_PLOTS_LOG_NAME="PPO_${ROBOT_NAME}_${TASK_NAME}_Plots"
NUM_SEEDS=4
NUM_EVAL_ENVS=128
RUNS_PER_ENV=4 #Skip first reset on by default, technically n-1 runs per env
AGENT= "rsl_rl_cfg_entry_point" #"rsl_rl_rnn_cfg_entry_point" #rsl_rl_cfg_entry_point


# Path to the PPO config (relative to project root)
_PPO_CFG="source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/agents/rsl_rl_ppo_cfg.py"

# Read experiment_name and max_iterations directly from the Python config
EXPERIMENT_NAME=$(grep -m1 'experiment_name\s*=' "$_PPO_CFG" | sed "s/.*=\s*['\"]//;s/['\"].*//")
MAX_ITERATIONS=$(grep -m1 'max_iterations\s*=' "$_PPO_CFG" | sed 's/[^0-9]//g')

if [ -z "$EXPERIMENT_NAME" ] || [ -z "$MAX_ITERATIONS" ]; then
    echo "[ERROR] Could not read EXPERIMENT_NAME or MAX_ITERATIONS from $_PPO_CFG"
    exit 1
fi

echo "[INFO] experiment_name : ${EXPERIMENT_NAME}"
echo "[INFO] max_iterations  : ${MAX_ITERATIONS}"

CHECKPOINT_ITER=$((MAX_ITERATIONS - 1))

PYTHON_EXE="${ISAACSIM_ROOT_PATH}/python.sh"
LOG_BASE_DIR="logs/rsl_rl/${EXPERIMENT_NAME}"

TRAIN_ARGS="--task=${TASK_ENV} env.robot_name=${ROBOT_NAME} env.task_name=${TASK_NAME} --headless --agent=${AGENT}"
EVAL_ARGS="--task=${TASK_ENV} --headless --num_envs=${NUM_EVAL_ENVS} --runs-per-env=${RUNS_PER_ENV} env.robot_name=${ROBOT_NAME} env.task_name=${TASK_NAME}"

# ── Training ──────────────────────────────────────────────────────────────────
CHECKPOINTS=()
RUN_DIRS=()

for SEED in $(seq 1 ${NUM_SEEDS}); do
    echo "========================================"
    echo "Training seed ${SEED}/${NUM_SEEDS}"
    echo "========================================"

    $PYTHON_EXE scripts/rsl_rl/train.py ${TRAIN_ARGS} --seed=${SEED}

    if [ $? -ne 0 ]; then
        echo "✗ Training failed for seed ${SEED}, skipping evaluation for this seed."
        continue
    fi

    # Pick the newest run directory for this seed (created just now)
    RUN_DIR=$(ls -td "${LOG_BASE_DIR}"/*_seed_${SEED} 2>/dev/null | head -1)

    if [ -z "$RUN_DIR" ]; then
        echo "✗ No run directory found for seed ${SEED} under ${LOG_BASE_DIR}"
        continue
    fi

    CHECKPOINT="${RUN_DIR}/model_${CHECKPOINT_ITER}.pt"
    if [ -f "$CHECKPOINT" ]; then
        CHECKPOINTS+=("$CHECKPOINT")
        RUN_DIRS+=("$RUN_DIR")
        echo "✓ Checkpoint found: ${CHECKPOINT}"
    else
        echo "✗ Expected checkpoint not found: ${CHECKPOINT}"
    fi
done

# ── Evaluation ────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Starting evaluation for ${#CHECKPOINTS[@]} checkpoint(s)..."
echo "========================================"

EVAL_RUN_DIRS=()

for i in "${!CHECKPOINTS[@]}"; do
    CHECKPOINT="${CHECKPOINTS[$i]}"
    RUN_DIR="${RUN_DIRS[$i]}"

    echo "Running evaluation for: ${CHECKPOINT}"
    $PYTHON_EXE scripts/rsl_rl/eval.py ${EVAL_ARGS} --checkpoint=${CHECKPOINT}

    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed: ${CHECKPOINT}"
        EVAL_RUN_DIRS+=("$RUN_DIR")
    else
        echo "✗ Evaluation failed:    ${CHECKPOINT}"
    fi
    echo "----------------------------------------"
done

# ── Plots & summary ───────────────────────────────────────────────────────────
if [ ${#EVAL_RUN_DIRS[@]} -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "Generating plots and metrics summary..."
    echo "========================================"

    $PYTHON_EXE scripts/rsl_rl/plot_metrics.py \
        --task="${TASK_NAME}" \
        --robot="${ROBOT_NAME}" \
        --out="logs/rsl_rl/${EXPERIMENT_NAME}/${CUSTOM_PLOTS_LOG_NAME}" \
        "${EVAL_RUN_DIRS[@]}"

    if [ $? -eq 0 ]; then
        echo "✓ Plots and summary saved."
    else
        echo "✗ Plotting step failed."
    fi
else
    echo "[WARN] No successful evaluations — skipping plots."
fi

echo "All done!"