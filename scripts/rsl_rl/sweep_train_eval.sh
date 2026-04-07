#!/bin/bash
# ── Reward sweep: train + eval for different reward configurations ────────────
#
# Sweeps over combinations of:
#   - rew_rw_torque_scale   (robot cfg)  : penalty on action^2
#   - rew_rw_overspin_scale (robot cfg)  : penalty on (omega/max)^2
#   - ang_vel_exponential_reward_coeff (task cfg) : sharpness of the task reward
#
# Each combo gets a unique experiment name so logs don't collide.
# Original config files are restored at the end (or on Ctrl-C).

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
ROBOT_CFG="source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/robots_cfg/cubo_cfg.py"
TASK_CFG="source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/tasks_cfg/track_velocities_cfg.py"
PPO_CFG="source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/agents/rsl_rl_ppo_cfg.py"

# ── Backup originals & restore on exit ────────────────────────────────────────
cp "$ROBOT_CFG" "${ROBOT_CFG}.bak"
cp "$TASK_CFG"  "${TASK_CFG}.bak"
cp "$PPO_CFG"   "${PPO_CFG}.bak"

cleanup() {
    echo ""
    echo "[INFO] Restoring original config files..."
    mv "${ROBOT_CFG}.bak" "$ROBOT_CFG"
    mv "${TASK_CFG}.bak"  "$TASK_CFG"
    mv "${PPO_CFG}.bak"   "$PPO_CFG"
    echo "[INFO] Done."
}
trap cleanup EXIT

# ── Sweep values ──────────────────────────────────────────────────────────────
# Torque penalty:   how much to penalize action magnitude
TORQUE_SCALES=( -0.3 -0.7 -1.5 )
# Overspin penalty: how much to penalize high wheel speed
OVERSPIN_SCALES=( -0.01 -0.1 -0.5 )
# Task reward sharpness: smaller = sharper (more reward near zero error)
ANG_VEL_COEFFS=( 0.25 0.5 1.0 )

# ── Common settings ───────────────────────────────────────────────────────────
TASK_ENV="Isaaclab-RANSv2-AutoEnvGen-v0"
ROBOT_NAME="Cubo"
TASK_NAME="TrackVelocities"
NUM_SEEDS=1
NUM_EVAL_ENVS=128
RUNS_PER_ENV=2
PYTHON_EXE="${ISAACSIM_ROOT_PATH}/python.sh"

TRAIN_ARGS="--task=${TASK_ENV} env.robot_name=${ROBOT_NAME} env.task_name=${TASK_NAME} --headless"
EVAL_ARGS="--task=${TASK_ENV} --headless --num_envs=${NUM_EVAL_ENVS} --runs-per-env=${RUNS_PER_ENV} env.robot_name=${ROBOT_NAME} env.task_name=${TASK_NAME}"

# ── Helper: patch a Python config value ───────────────────────────────────────
# Usage: patch_cfg <file> <param_name> <new_value>
# Matches lines like:  rew_rw_torque_scale = -0.7
patch_cfg() {
    local file="$1" param="$2" value="$3"
    sed -i "s/\(${param}\s*[=:]\s*\)[^ #]*/\1${value}/" "$file"
}

# ── Sweep loop ────────────────────────────────────────────────────────────────
RUN_IDX=0
TOTAL=$(( ${#TORQUE_SCALES[@]} * ${#OVERSPIN_SCALES[@]} * ${#ANG_VEL_COEFFS[@]} ))

for TORQUE in "${TORQUE_SCALES[@]}"; do
  for OVERSPIN in "${OVERSPIN_SCALES[@]}"; do
    for ANG_COEFF in "${ANG_VEL_COEFFS[@]}"; do
      RUN_IDX=$((RUN_IDX + 1))

      # Readable tag for this combo (replace minus signs for folder names)
      TAG="torq${TORQUE}_ospin${OVERSPIN}_angc${ANG_COEFF}"
      EXPERIMENT_NAME="sweep_cubo_rw_${TAG}"

      echo ""
      echo "╔══════════════════════════════════════════════════════════════╗"
      echo "║  Sweep run ${RUN_IDX}/${TOTAL}: ${TAG}"
      echo "║  rew_rw_torque_scale=${TORQUE}"
      echo "║  rew_rw_overspin_scale=${OVERSPIN}"
      echo "║  ang_vel_exponential_reward_coeff=${ANG_COEFF}"
      echo "╚══════════════════════════════════════════════════════════════╝"

      # ── Patch configs ──────────────────────────────────────────────────
      patch_cfg "$ROBOT_CFG" "rew_rw_torque_scale"   "$TORQUE"
      patch_cfg "$ROBOT_CFG" "rew_rw_overspin_scale" "$OVERSPIN"
      patch_cfg "$TASK_CFG"  "ang_vel_exponential_reward_coeff" "$ANG_COEFF"
      patch_cfg "$PPO_CFG"   "experiment_name" "\"${EXPERIMENT_NAME}\""

      LOG_BASE_DIR="logs/rsl_rl/${EXPERIMENT_NAME}"
      MAX_ITERATIONS=$(grep -m1 'max_iterations\s*=' "$PPO_CFG" | sed 's/[^0-9]//g')
      CHECKPOINT_ITER=$((MAX_ITERATIONS - 1))

      # ── Train ──────────────────────────────────────────────────────────
      CHECKPOINTS=()
      EVAL_RUN_DIRS=()

      for SEED in $(seq 1 ${NUM_SEEDS}); do
          echo "[TRAIN] seed ${SEED}/${NUM_SEEDS}"
          $PYTHON_EXE scripts/rsl_rl/train.py ${TRAIN_ARGS} --seed=${SEED} || {
              echo "✗ Training failed for seed ${SEED}, skipping."
              continue
          }

          RUN_DIR=$(ls -td "${LOG_BASE_DIR}"/*_seed_${SEED} 2>/dev/null | head -1)
          CHECKPOINT="${RUN_DIR}/model_${CHECKPOINT_ITER}.pt"

          if [ -f "$CHECKPOINT" ]; then
              CHECKPOINTS+=("$CHECKPOINT")

              # ── Eval ───────────────────────────────────────────────────
              echo "[EVAL] ${CHECKPOINT}"
              $PYTHON_EXE scripts/rsl_rl/eval.py ${EVAL_ARGS} --checkpoint="${CHECKPOINT}" && {
                  EVAL_RUN_DIRS+=("$RUN_DIR")
                  echo "✓ Eval done."
              } || echo "✗ Eval failed."
          else
              echo "✗ Checkpoint not found: ${CHECKPOINT}"
          fi
      done

      # ── Plots ──────────────────────────────────────────────────────────
      if [ ${#EVAL_RUN_DIRS[@]} -gt 0 ]; then
          PLOTS_DIR="${LOG_BASE_DIR}/${EXPERIMENT_NAME}_Plots"
          $PYTHON_EXE scripts/rsl_rl/plot_metrics.py \
              --task="${TASK_NAME}" \
              --robot="${ROBOT_NAME}" \
              --out="${PLOTS_DIR}" \
              "${EVAL_RUN_DIRS[@]}" \
          && echo "✓ Plots saved to ${PLOTS_DIR}" \
          || echo "✗ Plotting failed."
      fi

    done
  done
done

echo ""
echo "========================================"
echo "Sweep complete: ${TOTAL} configurations."
echo "========================================"
