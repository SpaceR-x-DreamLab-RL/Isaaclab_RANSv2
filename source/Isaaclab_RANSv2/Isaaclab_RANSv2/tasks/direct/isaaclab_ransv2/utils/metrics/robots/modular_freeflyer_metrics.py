from . import BaseRobotMetrics, Registerable
import torch

class ModularFreeflyerMetrics(BaseRobotMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, robot_name: str) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, robot_name=robot_name)
    
    # @BaseRobotMetrics.register
    # def mass(self):
    #     print("[INFO][METRICS][ROBOT] Mass metrics")

    #     masked_masses = self.trajectories['masses'] * self.trajectories_masks
    #     avg_mass_per_trajectory = torch.stack([torch.mean(row[:end_idx]) for row, end_idx in zip(masked_masses, self.last_true_index)])
    #     self.metrics["avg_mass_per_trajectory.kg"] = avg_mass_per_trajectory