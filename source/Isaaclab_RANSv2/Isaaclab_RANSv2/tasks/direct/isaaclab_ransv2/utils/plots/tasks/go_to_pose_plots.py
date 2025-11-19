from . import BaseTaskPlots, Registerable
import torch

class GoToPosePlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "go_to_pose"

        # Box plots
        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                # keys_set.update(
                #     key for key in df.columns if "_steps_jerk" in key
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("time_to_reach_position_threshold")
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("num_steps_position_overshoot")
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("overshoot_distance_error")
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("trajectory_efficiency")
                # )
                keys_set.update(
                    key for key in df.columns if key.startswith("final_position_distance")
                )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("final_position_heading_error")
                # )
                keys_set.update(
                    key for key in df.columns if key.startswith("final_orientation_error")
                )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("success_rate")
                # )
        self.labels_box_plot = list(keys_set)

        # # Dot plots
        # keys_set = set()
        # for group_dfs in dfs.values():
        #     for df in group_dfs:
        #         keys_set.update(
        #             key for key in df.columns if key.startswith("success_rate")
        #         )
        # self.labels_dots = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_box_plot:
            self.boxplot(label_to_plot)

        # for label_to_plot in self.labels_dots:
        #     self.dotplot(label_to_plot)

        # if len(self._trajectories_dfs) > 0:
        #     self.plot_xy_trajectories_0_centered()
        #     self.plot_position_distance_over_time()
        #     self.plot_linear_velocity_over_time()
        #     self.plot_angular_velocity_over_time()
        #     self.plot_actions_over_time()