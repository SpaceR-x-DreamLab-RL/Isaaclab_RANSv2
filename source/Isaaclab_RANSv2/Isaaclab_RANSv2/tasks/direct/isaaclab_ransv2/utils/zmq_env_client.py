"""ZMQ client: gymnasium.Env interface backed by ZmqEnvServer.

Runs inside the DreamerV3 venv (/opt/dreamer_venv). Has no Isaac Sim imports.

Usage:
    env = ZmqEnvClient(socket_path="/tmp/isaaclab_dreamer.sock")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""
from __future__ import annotations

import pickle

import gymnasium as gym
import numpy as np
import zmq


class ZmqEnvClient(gym.Env):
    """Gymnasium env that proxies all calls to a remote ZmqEnvServer over IPC."""

    _PICKLE_PROTO = 5

    def __init__(self, socket_path: str = "/tmp/isaaclab_dreamer.sock"):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(f"ipc://{socket_path}")

        meta = self._call("spaces")
        raw_obs_space = meta["observation_space"]
        self.action_space: gym.Space = meta["action_space"]
        self.num_envs: int = meta.get("num_envs", 1)

        # Isaac Lab wraps obs as Dict({"policy": Box(...)}).  DreamerV3's
        # FromGymnasium renames the space key to "image" but leaves the
        # actual step() dict key as "policy" — causing a KeyError at runtime.
        # Flatten here so both the space and the data are a plain array.
        if isinstance(raw_obs_space, gym.spaces.Dict) and list(raw_obs_space.spaces.keys()) == ["policy"]:
            policy_space: gym.Space = raw_obs_space["policy"]
            self._obs_key: str | None = "policy"
        else:
            policy_space = raw_obs_space
            self._obs_key = None

        # Probe reset to discover which log metrics Isaac Lab exposes in
        # info["log"].  _reset_idx() always populates extras["log"] on reset,
        # so the keys are present even if all values are zero at this point.
        probe_reply = self._call("reset", kwargs={"seed": None, "options": None})
        probe_info = probe_reply.get("info", {})
        self._log_keys: list[str] = sorted(probe_info.get("log", {}).keys())

        if self._log_keys:
            # Expose policy obs under "image" plus every log metric as
            # "log/<key>".  FromGymnasium sees a Dict space, flattens it, and
            # DreamerV3 automatically averages log/ keys over episodes and
            # includes them in the training metrics.
            self.observation_space: gym.Space = gym.spaces.Dict({
                "image": policy_space,
                **{
                    f"log/{k}": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(), dtype=np.float32
                    )
                    for k in self._log_keys
                },
            })
        else:
            self.observation_space = policy_space

    # ------------------------------------------------------------------
    def _call(self, cmd: str, **kwargs) -> dict:
        self._sock.send(pickle.dumps({"cmd": cmd, **kwargs}, protocol=self._PICKLE_PROTO))
        reply = pickle.loads(self._sock.recv())
        if "error" in reply:
            raise RuntimeError(f"[ZmqEnvClient] Server error: {reply['error']}")
        return reply

    def _flatten_obs(self, obs) -> np.ndarray:
        """Unwrap Isaac Lab's {'policy': array} dict to a plain array."""
        if self._obs_key is not None and isinstance(obs, dict):
            return obs[self._obs_key]
        if isinstance(obs, dict) and "policy" in obs:
            return obs["policy"]
        return obs

    def _make_obs(self, raw_obs, info: dict):
        """Build the observation: plain array when no log keys, Dict otherwise."""
        flat = self._flatten_obs(raw_obs)
        if not self._log_keys:
            return flat
        log = info.get("log", {})
        return {
            "image": flat,
            **{f"log/{k}": np.float32(log.get(k, 0.0)) for k in self._log_keys},
        }

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        reply = self._call("reset", kwargs={"seed": seed, "options": options})
        info = reply.get("info", {})
        return self._make_obs(reply["obs"], info), info

    def step(self, action):
        reply = self._call("step", action=np.asarray(action))
        info = reply.get("info", {})
        return (
            self._make_obs(reply["obs"], info),
            reply["reward"],
            reply["terminated"],
            reply["truncated"],
            info,
        )

    def close(self):
        """Disconnect this client. Server stays alive for the next connection."""
        try:
            self._call("close")
        finally:
            self._sock.close()
            self._ctx.term()

    def shutdown(self):
        """Tell the server to close the Isaac Lab env and exit entirely."""
        try:
            self._call("shutdown")
        finally:
            self._sock.close()
            self._ctx.term()
