from . import BaseTaskPlots, Registerable
import torch

class GoThroughPosesPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "go_through_poses"

        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                # keys_set.update(
                #     key for key in df.columns if "num_goals_reached_in_" in key
                # )
                # for i in range(env_info["num_goals"]):
                #     label = f"time_to_reach_goal_num_{i}"
                #     keys_set.update(
                #         key for key in df.columns if label in key
                #     )

                keys_set.update(
                    key for key in df.columns if "spl." in key
                )
                keys_set.update(
                    key for key in df.columns if "orientation_error_following_path" in key
                )
                

        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)