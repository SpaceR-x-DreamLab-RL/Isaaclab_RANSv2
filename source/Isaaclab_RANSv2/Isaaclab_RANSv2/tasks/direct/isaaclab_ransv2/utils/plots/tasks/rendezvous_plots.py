from . import BaseTaskPlots, Registerable
import torch

class RendezvousPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "rendezvous"

        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if "mean_orientation_error" in key
                )
                keys_set.update(
                    key for key in df.columns if "mean_orientation_error_path" in key
                )
            
        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)

        self.rendezvous_sr()


    def rendezvous_sr(self):
        for key, value in self._env_info.items():
            failed_trajectories_list = [item['failed_trajectories'] for item in value]
            sr_num_gates_list = [item['sr_num_gates'] for item in value]
            
            # Calculate the mean for failed_trajectories
            mean_failed_trajectories = sum(failed_trajectories_list) / len(failed_trajectories_list)
            
            # Calculate the mean for sr_num_gates
            mean_sr_num_gates = sum(sr_num_gates_list) / len(sr_num_gates_list)
            
            # Print the results for the current key, formatted to two decimal places
            print(f"Key: {key}")
            # print(f"  Mean of failed_trajectories: {mean_failed_trajectories:.2f}")
            print(f"  Mean of SR: {mean_sr_num_gates:.2f}")
            print("-" * 20)