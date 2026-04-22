"""ZMQ server: wraps an Isaac Lab gymnasium env and serves it over a Unix socket.

Run inside Isaac Sim's Python. DreamerV3 (in its own venv) connects as a client.

Protocol: REQ/REP over IPC, messages serialized with pickle.
"""
from __future__ import annotations

import pickle

import numpy as np
import zmq


def _to_numpy(val):
    """Recursively convert torch tensors / nested structures to numpy."""
    try:
        import torch
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, dict):
        return {k: _to_numpy(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_to_numpy(v) for v in val)
    return val


def _action_to_tensor(action_np: np.ndarray, device: str) -> "torch.Tensor":
    """Convert numpy action from client into a CUDA tensor for Isaac Lab."""
    import torch
    return torch.from_numpy(action_np).to(device)


class ZmqEnvServer:
    """Serve a gymnasium env over a ZMQ IPC socket (REP side)."""

    _PICKLE_PROTO = 5

    def __init__(self, env, socket_path: str = "/tmp/isaaclab_dreamer.sock"):
        self._env = env
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(f"ipc://{socket_path}")
        print(f"[ZmqEnvServer] Listening on ipc://{socket_path}")

    # ------------------------------------------------------------------
    def serve(self) -> None:
        """Block and handle requests until a 'close' command arrives."""
        while True:
            msg = pickle.loads(self._sock.recv())
            cmd = msg["cmd"]

            if cmd == "spaces":
                reply = {
                    "observation_space": self._env.observation_space,
                    "action_space": self._env.action_space,
                    "num_envs": getattr(self._env.unwrapped, "num_envs", 1),
                }
            elif cmd == "reset":
                obs, info = self._env.reset(**msg.get("kwargs", {}))
                reply = {"obs": _to_numpy(obs), "info": _to_numpy(info)}
            elif cmd == "step":
                device = getattr(self._env.unwrapped, "device", "cpu")
                action = _action_to_tensor(msg["action"], device)
                obs, reward, terminated, truncated, info = self._env.step(action)
                reply = {
                    "obs": _to_numpy(obs),
                    "reward": float(_to_numpy(reward).flat[0]),
                    "terminated": bool(_to_numpy(terminated).flat[0]),
                    "truncated": bool(_to_numpy(truncated).flat[0]),
                    "info": _to_numpy(info),
                }
            elif cmd == "close":
                # Client disconnecting (e.g. dreamerv3 make_agent() probes spaces then
                # closes). Keep the server alive so the next client can reconnect.
                self._sock.send(pickle.dumps({"ok": True}, protocol=self._PICKLE_PROTO))
                continue
            elif cmd == "shutdown":
                # Full termination — close the Isaac Lab env and exit.
                self._env.close()
                self._sock.send(pickle.dumps({"ok": True}, protocol=self._PICKLE_PROTO))
                break
            else:
                reply = {"error": f"Unknown command: {cmd}"}

            self._sock.send(pickle.dumps(reply, protocol=self._PICKLE_PROTO))

        self._sock.close()
        self._ctx.term()
