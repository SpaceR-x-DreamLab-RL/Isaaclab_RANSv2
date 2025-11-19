import torch
from tasks import TaskPlotsFactory
from robots import RobotPlotsFactory

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml

def main():
    
    list_of_folders = [
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_12-34-20_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        ],
        [
            "/workspace/isaaclab/logs/rsl_rl/multitask_racing_baseline_noTrackInfo/2025-07-13_12-23-18_rsl-rl_ppo-memory_RaceGates_Leatherback_r-0_seed-42",
        ],
    ]

    plot_cfg = {
        "title": "",
        "box_colors": [
            "#FF3D50", 
            "#FFA034",
            "#2FA1FF", 
            "#A734FF",
            "#FFFF3D", 
            "#4DFF3D",
            "#FF3DBB", 
            "#623652", 
        ],
        "runs_names": [
            "Expert 37",
            "Expert 42",
        ],
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/plots/orbital_mass" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    trajectories_dfs = {}
    labels = {}
    list_of_tasks = []
    
    for group_idx, group in enumerate(list_of_folders):
        task_name = group[0].split("/")[-1].split("_")[4]
        robot_name = group[0].split("/")[-1].split("_")[5]
        # Extract seed from folder path (e.g., "seed-42" from the folder name)
        seed_info = group[0].split("/")[-1].split("_")[-1]  # Gets "seed-42"
        # Use the run name from plot_cfg if available, otherwise use group index
        run_name = plot_cfg["runs_names"][group_idx] if group_idx < len(plot_cfg["runs_names"]) else f"group-{group_idx}"
        group_key = f"{task_name}_group-{group_idx}_{run_name}"  # Creates "RaceGates_group-0_seed-42"
        dfs[group_key] = []
        trajectories_dfs[group_key] = []
        labels[group_key] = []
        list_of_tasks.append(task_name)
        
        for folder_path in group:
            try:
                # Load metrics CSV
                experiment_name = glob.glob(os.path.join(folder_path, "metrics", "*_metrics.csv"))[0]
                metrics_file_path = os.path.join(folder_path, "metrics", experiment_name)
                df = pd.read_csv(metrics_file_path)
                dfs[group_key].append(df)
                labels[group_key].append(experiment_name.split("/")[-1])

                # Load trajectories CSV
                trajectories_pattern = os.path.join(folder_path, "metrics", f"extracted_trajectories_{task_name}.csv")
                trajectories_files = glob.glob(trajectories_pattern)
                if trajectories_files:
                    trajectories_file_path = trajectories_files[0]
                    trajectories_df = pd.read_csv(trajectories_file_path)
                    trajectories_dfs[group_key].append(trajectories_df)
                else:
                    print(f"Warning: No trajectories file found for {folder_path}")
                    trajectories_dfs[group_key].append(pd.DataFrame())  # Empty DataFrame as fallback

                # Load env info
                env_info_file_path = os.path.join(folder_path, "metrics", "env_info.yaml")
                with open(env_info_file_path, 'r') as f:
                    env_info = yaml.safe_load(f)
                    
            except Exception as e:
                print(f"Error reading file in {folder_path}: {e}")
                exit(0)

    # Group by task
    task_dfs = {}
    task_trajectories_dfs = {}
    
    for task in list_of_tasks:
        task_dfs[task] = {}
        task_trajectories_dfs[task] = {}
        for key in dfs:
            if task in key:
                task_dfs[task][key] = dfs[key]
                task_trajectories_dfs[task][key] = trajectories_dfs[key]

    # Create plots for each task
    for task_name in task_dfs:
        task_plots_factory = TaskPlotsFactory.create(
            task_name, 
            dfs=task_dfs[task_name],
            trajectories_dfs=task_trajectories_dfs[task_name],
            labels=labels,
            env_info=env_info,
            folder_path=save_plots_folder_path,
            plot_cfg=plot_cfg,
        )
        task_plots_factory.plot()
        
        robot_plots_factory = RobotPlotsFactory.create(
            robot_name, 
            dfs=dfs,
            trajectories_dfs=trajectories_dfs,
            labels=labels,
            env_info=env_info,
            folder_path=save_plots_folder_path,
            plot_cfg=plot_cfg,
        )
        robot_plots_factory.plot()

if __name__ == "__main__":
    main()
    