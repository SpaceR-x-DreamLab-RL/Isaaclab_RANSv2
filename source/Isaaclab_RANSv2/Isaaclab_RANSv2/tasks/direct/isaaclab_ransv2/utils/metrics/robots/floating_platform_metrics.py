from . import BaseRobotMetrics, Registerable

class FloatingPlatformMetrics(BaseRobotMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, robot_name: str) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, robot_name=robot_name)
    
    @BaseRobotMetrics.register
    def some_metrics_here(self):
        print("[INFO][METRICS][ROBOT] Test metrics")


    #TODO tumbling