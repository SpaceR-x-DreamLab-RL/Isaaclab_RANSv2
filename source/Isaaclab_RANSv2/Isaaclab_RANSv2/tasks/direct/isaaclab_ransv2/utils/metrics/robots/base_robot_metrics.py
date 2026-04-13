import torch


def _expand_runs(trajectories: dict, cutoff_indices: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Reshape (num_envs, total_steps) → (num_envs * num_runs, max_run_len).

    Each run becomes an independent row so per-run metrics are computed correctly
    when num_runs_per_env > 1.

    cutoff_indices[env, run] is the exclusive end step (the done step) for that run.
    """
    if cutoff_indices.ndim == 1:
        cutoff_indices = cutoff_indices.unsqueeze(1)

    num_envs, num_runs = cutoff_indices.shape
    device = cutoff_indices.device

    starts = torch.cat([
        torch.zeros(num_envs, 1, dtype=torch.long, device=device),
        cutoff_indices[:, :-1],
    ], dim=1)

    run_lens = (cutoff_indices - starts).clamp(min=0)
    max_run_len = int(run_lens.max().item())
    total_rows = num_envs * num_runs

    if max_run_len == 0:
        expanded = {k: torch.zeros(total_rows, 1, *v.shape[2:], dtype=v.dtype, device=device)
                    for k, v in trajectories.items()}
        return expanded, torch.zeros(total_rows, 1, dtype=torch.bool, device=device)

    expanded: dict = {}
    for k, v in trajectories.items():
        expanded[k] = torch.zeros(total_rows, max_run_len, *v.shape[2:], dtype=v.dtype, device=device)
    expanded_masks = torch.zeros(total_rows, max_run_len, dtype=torch.bool, device=device)

    for env_idx in range(num_envs):
        for run_idx in range(num_runs):
            row = env_idx * num_runs + run_idx
            start = int(starts[env_idx, run_idx].item())
            end = int(cutoff_indices[env_idx, run_idx].item())
            rlen = end - start
            if rlen <= 0:
                continue
            for k, v in trajectories.items():
                expanded[k][row, :rlen] = v[env_idx, start:end]
            expanded_masks[row, :rlen] = True

    return expanded, expanded_masks


class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)

        # Unique for each subclass + inherit from parent class
        cls._registry = getattr(super(cls, cls), '_registry', {}).copy()

        for name, value in cls.__dict__.items():
            # If an attribute is a function and has our marker, register it.
            if callable(value) and getattr(value, '_auto_register', False):
                cls._registry[name] = value

    @staticmethod
    def register(func: callable) -> callable:
        """Decorator that simply marks a function so that __init_subclass__
        knows it should be placed in the registry.
        """
        func._auto_register = True
        return func

    @classmethod
    def get_registered_methods(cls) -> dict[str, callable]:
        """Retrieve registered methods."""
        return cls._registry


class BaseRobotMetrics(AutoRegister):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, robot_name: str) -> None:
        self.env = env
        self.folder_path = folder_path
        self.physics_dt = physics_dt
        self.step_dt = step_dt

        self.robot_name = robot_name

        self.metrics = {}
        self.env_info = {}

    def populate_env_info(self)-> None:
        """Populate environment information. Subclass should implement this method."""
        pass

    def generate_metrics(
            self,
            trajectories: dict,
            cutoff_indices: torch.Tensor,
            trajectories_masks: torch.Tensor,
        ) -> None:

        expanded_traj, expanded_masks = _expand_runs(trajectories, cutoff_indices)
        self.trajectories = expanded_traj
        self.trajectories_masks = expanded_masks
        self.cutoff_indices = expanded_masks.sum(dim=1) - 1

        for metric_fnc in self.get_registered_methods().values():
            metric_fnc(self)

        for key, value in self.metrics.items():
            self.metrics

    @property
    def last_true_index(self) -> torch.Tensor:
        """Returns the last true index of a masked tensor along the first dimension.
            Args:
                masked_tensor (torch.Tensor): The tensor to find the last true index for.
            Returns:
                torch.Tensor: A tensor containing the last true indices for each row.
        """
        # traj_len = self.trajectories_masks.shape[1]
        # last_true_idx = torch.argmax(~self.trajectories_masks.int(), dim=1)
        # all_true = torch.all(self.trajectories_masks, dim=1)
        # last_true_idx[all_true] = traj_len
        # return last_true_idx
        return self.trajectories_masks.sum(dim=1) - 1
    
    @AutoRegister.register
    def action_rate(self):
        # print("[INFO][METRICS][ROBOT] Action rate")
        # masked_unaltered_actions = self.trajectories['actions'] * self.trajectories_masks.unsqueeze(-1)
        # action_rate = torch.mean(
        #     torch.sum(
        #         torch.square(masked_unaltered_actions[:, 1:] - masked_unaltered_actions[:, :-1]), dim=-1
        #     )[:, 1:], dim=1)
        # self.metrics["mean_trajectory_action_rate.u"] = action_rate
        pass

    @AutoRegister.register
    def energy(self):
        # print("[INFO][METRICS][ROBOT] Energy")
        # masked_actions = self.trajectories['actions'] * self.trajectories_masks.unsqueeze(-1)

        # energy = torch.stack([torch.mean(row[:end_idx]) for row, end_idx in zip(masked_actions, self.last_true_index)])
        # energy = torch.mean(torch.sum(masked_actions ** 2, dim=-1), dim=1)
        # self.metrics["mean_trajectory_energy.u"] = energy
        pass