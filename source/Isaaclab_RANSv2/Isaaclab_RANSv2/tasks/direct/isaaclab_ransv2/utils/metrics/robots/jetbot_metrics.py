from . import BaseRobotMetrics, Registerable
import torch
class JetbotMetrics(BaseRobotMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, robot_name: str) -> None:
        super().__init__(env=env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, robot_name=robot_name)

    @BaseRobotMetrics.register
    def mean_wheel_action(self):
        print("[INFO][METRICS][ROBOT] Mean wheel action")
        masked_left_wheel_action = self.trajectories['left_wheel_action'] * self.trajectories_masks
        masked_right_wheel_action = self.trajectories['right_wheel_action'] * self.trajectories_masks

        mean_left_wheel_action = torch.mean(masked_left_wheel_action, dim=1)
        mean_right_wheel_action = torch.mean(masked_right_wheel_action, dim=1)

        self.metrics["mean_left_wheel_action.rad/s"] = mean_left_wheel_action
        self.metrics["mean_right_wheel_action.rad/s"] = mean_right_wheel_action