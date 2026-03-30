
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="scripts/rsl_rl/eval.py"
TASK="Isaaclab-RANSv2-AutoEnvGen-v0"
NUM_ENVS=128
runs_per_env=4
BASE_LOG_DIR="logs/rsl_rl/AutoEnvGen_PPO"
robot="Pingu"
task_name="GoToPose"
PYTHON_EXE="${ISAACSIM_ROOT_PATH}/python.sh"

# Common arguments that apply to all evaluations
COMMON_ARGS="--task=${TASK} --headless --num_envs=${NUM_ENVS} --runs-per-env=${runs_per_env} env.robot_name=${robot} env.task_name=${task_name}"

# Array of checkpoint paths (relative to BASE_LOG_DIR)

#PPO
# CHECKPOINTS=(
#     2026-03-19_15-50-06_ppo_Pingu_GoToPose_rsl_rl_seed_1/model_499.pt
#     2026-03-19_16-00-50_ppo_Pingu_GoToPose_rsl_rl_seed_3/model_499.pt
#     2026-03-19_15-55-30_ppo_Pingu_GoToPose_rsl_rl_seed_2/model_499.pt
#     2026-03-19_16-06-11_ppo_Pingu_GoToPose_rsl_rl_seed_4/model_499.pt
#     2026-03-19_16-11-27_ppo_Pingu_GoToPose_rsl_rl_seed_5/model_499.pt
# )

#PPO + DR
CHECKPOINTS=(
    2026-03-30_08-35-14_ppo_Pingu_GoToPose_rsl_rl_seed_1/model_499.pt
    2026-03-30_08-46-00_ppo_Pingu_GoToPose_rsl_rl_seed_2/model_499.pt
    2026-03-30_08-57-02_ppo_Pingu_GoToPose_rsl_rl_seed_3/model_499.pt
    2026-03-30_09-07-50_ppo_Pingu_GoToPose_rsl_rl_seed_4/model_499.pt
    2026-03-30_09-18-44_ppo_Pingu_GoToPose_rsl_rl_seed_5/model_499.pt
)


# Function to run evaluation for a single checkpoint
run_evaluation() {
    local checkpoint_path="$1"
    local full_checkpoint_path="${BASE_LOG_DIR}/${checkpoint_path}"
    
    echo "Running evaluation for checkpoint: ${checkpoint_path}"
    $PYTHON_EXE ${SCRIPT_PATH} ${COMMON_ARGS} --checkpoint=${full_checkpoint_path}
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for: ${checkpoint_path}"
    else
        echo "✗ Evaluation failed for: ${checkpoint_path}"
    fi
    echo "----------------------------------------"
}

# Main execution: iterate through all checkpoints
echo "Starting control evaluation for ${#CHECKPOINTS[@]} checkpoints..."
echo "Task: ${TASK}"
echo "Number of environments: ${NUM_ENVS}"
echo "========================================"

for checkpoint in "${CHECKPOINTS[@]}"; do
    run_evaluation "${checkpoint}"
done

echo "All evaluations completed!"