#!/bin/bash
TASK="GoToPose"
ROBOT_NAME="Pingu"

PYTHON_EXE="${ISAACSIM_ROOT_PATH}/python.sh"

for SEED in {1..5}
do
    echo "Training seed $SEED"
    $PYTHON_EXE scripts/rsl_rl/train.py --task=Isaaclab-RANSv2-AutoEnvGen-v0 env.robot_name=$ROBOT_NAME env.task_name=$TASK --headless --seed=$SEED
done