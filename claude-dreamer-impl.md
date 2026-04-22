# DreamerV3 Integration — Implementation Record

## Context

This document records every decision, file change, and fix made to integrate DreamerV3 (MBRL) into the Isaaclab_RANSv2 project. It is written so that anyone picking this up cold can understand what was done, why each choice was made, and how to operate the result.

---

## 1. Starting State

Before this work, the repo had:
- **Isaac Lab** (SpaceR-x-DreamLab-RL fork, v2.2.0) as the simulation framework
- **RSL_RL** (leggedrobotics, v3.3.0) as the only RL training backend
- **No DreamerV3, no MBRL, no ZMQ** anywhere in the codebase or Docker setup

The name "DreamLab" in the GitHub org (`SpaceR-x-DreamLab-RL`) is unrelated to the DreamerV3 algorithm — it refers to the lab, not the algorithm.

---

## 2. Why DreamerV3 (MBRL)?

DreamerV3 is a **model-based RL** algorithm. It learns a world model (a latent-space dynamics model) and plans/trains inside that model, requiring far fewer real environment interactions than model-free methods like PPO.

For space robotics tasks (GoToPose, Rendezvous, etc.) where:
- Reset is expensive (full physics re-initialization)
- Episode length is long (40 s at 60 Hz = 2400 steps)
- Sample efficiency matters

DreamerV3 is a strong candidate. RSL_RL PPO typically needs 4096 parallel envs running for thousands of iterations. DreamerV3 can learn from a single env running sequentially.

**Step count comparison:**

| | RSL_RL PPO | DreamerV3 |
|---|---|---|
| Env steps | 65.5M (4096 envs × 16 × 1000 iters) | 2M (1 env) |
| Why it works | Brute-force parallelism | World model imagines ~15× more steps internally |

**Fork used:** `AndrejOrsula/dreamerv3` — a fork specifically maintained for compatibility with Isaac Lab / robotics simulation workflows.

**Pinned commit:** `4049794d4135e41c691f18da38a9af7541b01553` (dated 2025-07-16)

---

## 3. Core Architectural Decision: Why a Separate Venv + ZMQ Bridge

### The Problem

DreamerV3 runs on **JAX**. Isaac Sim (and RSL_RL) run on **PyTorch**. Both require CUDA, but their CUDA library stacks are not the same version or build. Installing JAX into Isaac Sim's bundled Python (`${ISAACSIM_ROOT_PATH}/python.sh`) risks:

- CUDA version conflicts (Isaac Sim pins specific CUDA versions)
- JAX overwriting or shadowing PyTorch CUDA libraries
- Silent failures where JAX appears to import but runs on CPU

### The Solution: Separate Venv + IPC Bridge

```
Isaac Sim Python (/isaac-sim/python.sh)      dreamer_venv (/opt/dreamer_venv)
───────────────────────────────────────      ────────────────────────────────
PyTorch, Isaac Lab, RSL_RL                   JAX, DreamerV3, embodied
ZmqEnvServer (REP socket)         ◄──────►  ZmqEnvClient (REQ socket)
                                   IPC Unix socket
                                   /tmp/isaaclab_dreamer.sock
```

**Why IPC Unix socket over TCP?**
- ~10× lower latency than TCP between containers
- No port management
- Both processes share the same filesystem inside the container

**Why REQ/REP pattern?**
- Synchronous RL training naturally maps to request/response
- One step = one REQ → one REP, no buffering complexity

**Why not docker-compose two-service split?**
- The repo already has Singularity support baked into the Dockerfile (lines 117–134)
- Singularity (used on HPC/SLURM clusters) does not support multi-container orchestration
- Single container with two processes and a Unix socket is simpler and portable

---

## 4. Files Changed

### 4.1 `docker/Dockerfile.base`

#### Added `python3` and `python3-venv` to apt block

```dockerfile
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    libglib2.0-0 \
    ncurses-term \
    python3 \        # ← added
    python3-venv \   # ← added
    wget && \
```

**Why:** The Isaac Sim base Ubuntu image has no system `python3` — only `${ISAACSIM_ROOT_PATH}/python.sh`. Without `python3`, `python3 -m venv` fails. These are installed unconditionally (lightweight, ~5 MB).

#### DreamerV3 install block (full final form)

```dockerfile
## DreamerV3 — isolated venv to keep JAX separate from Isaac Sim's PyTorch
ARG DREAMER_DEV=false
ARG DREAMER_PATH="/root/dreamerv3"
ARG DREAMER_REMOTE="https://github.com/AndrejOrsula/dreamerv3.git"
ARG DREAMER_BRANCH="main"
ARG DREAMER_COMMIT_SHA="4049794d4135e41c691f18da38a9af7541b01553"
ARG DREAMER_VENV="/opt/dreamer_venv"
RUN "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir pyzmq
RUN if [[ "${DEV,,}" = true && "${DREAMER_DEV,,}" = true ]]; then \
    git clone "${DREAMER_REMOTE}" "${DREAMER_PATH}" --branch "${DREAMER_BRANCH}" && \
    git -C "${DREAMER_PATH}" reset --hard "${DREAMER_COMMIT_SHA}" && \
    sed -i '1s/^/import elements\n/' "${DREAMER_PATH}/embodied/envs/from_gymnasium.py" && \
    python3 -m venv "${DREAMER_VENV}" && \
    "${DREAMER_VENV}/bin/pip" install --no-cache-dir --upgrade pip && \
    "${DREAMER_VENV}/bin/pip" install --no-cache-dir pyzmq gymnasium numpy && \
    "${DREAMER_VENV}/bin/pip" install --no-cache-dir --editable "${DREAMER_PATH}" ; \
    fi
ENV DREAMER_VENV=${DREAMER_VENV}
ENV DREAMER_PATH=${DREAMER_PATH}
```

Key points:
- `DREAMER_DEV=false` — safe default; enabled only when user answers `y` to the build prompt
- DreamerV3 goes into `/opt/dreamer_venv`, never into Isaac Sim's Python
- `pyzmq` installed into Isaac Sim's Python (server side) and into the venv (client side)
- `gymnasium` and `numpy` explicitly installed in venv (dreamerv3's own `setup.py` doesn't always pull them)
- `sed` patch baked in at clone time (fixes missing `import elements` bug — see Bug 4 below)

### 4.2 `docker/docker-compose.yaml`

Added one line to the `isaac-lab-base` build args:

```yaml
- DREAMER_DEV=${DREAMER_DEV:-false}
```

**Why:** Docker Compose forwards this env var to the Dockerfile ARG. The interactive prompt in `container.py` sets it before the build starts.

### 4.3 `docker/container.py`

Added two lines before `ci.start()`:

```python
dreamer_reply = input("[INFO] Include DreamerV3 (MBRL) in this build? [y/N]: ").strip().lower()
ci.environ["DREAMER_DEV"] = "true" if dreamer_reply in ("y", "yes") else "false"
```

**Flow:**
```
python3 docker/container.py start
  → prompt: "Include DreamerV3? [y/N]"
       y → ci.environ["DREAMER_DEV"] = "true"  → venv + JAX installed
       N → ci.environ["DREAMER_DEV"] = "false" → skipped entirely
  → ci.start() → docker compose build → Dockerfile ARG picks it up
```

---

## 5. Files Created

### 5.1 `source/.../utils/zmq_env_server.py`

**Purpose:** Wraps any gymnasium env and serves it over a ZMQ IPC socket. Runs inside Isaac Sim's Python.

**Key design points:**
- `_to_numpy()` helper recursively converts PyTorch tensors to numpy before sending. Isaac Lab envs return GPU tensors; pickle can't serialize CUDA tensors without this.
- `cmd = "close"` — client disconnecting. Server **keeps running** to accept the next client connection. This is critical because DreamerV3's `make_agent()` opens a temporary connection to read spaces then immediately closes it before training starts.
- `cmd = "shutdown"` — true termination. Closes the Isaac Lab env and exits the serve loop. Send this only when you're done training entirely.
- Pickle protocol 5 used throughout.

**Protocol (all messages are `pickle.dumps(dict)`):**

| Command | Request fields | Reply fields | Server after |
|---------|---------------|--------------|--------------|
| `spaces` | — | `observation_space`, `action_space`, `num_envs` | continues |
| `reset` | `kwargs: {seed, options}` | `obs`, `info` | continues |
| `step` | `action: np.ndarray` | `obs`, `reward`, `terminated`, `truncated`, `info` | continues |
| `close` | — | `ok: True` | **continues** (waits for next client) |
| `shutdown` | — | `ok: True` | **exits** |

**Why `close` ≠ `shutdown`:** DreamerV3 internally calls `make_agent()` which creates a throw-away env, reads `obs_space`/`act_space`, then calls `env.close()`. If `close` terminated the server, the sim would die before training ever started. The server must survive this probe and wait for the real training connection.

### 5.2 `source/.../utils/zmq_env_client.py`

**Purpose:** Implements `gymnasium.Env` interface, proxying all calls to the ZMQ server. Runs inside the dreamer venv.

**Key design points:**
- **Zero Isaac Lab imports.** Only `gymnasium`, `numpy`, `zmq`.
- `close()` — sends `close` command, disconnects socket. Server stays alive.
- `shutdown()` — sends `shutdown` command. Use this only to stop the sim server entirely after training.

**Import rule:** Never import via `from Isaaclab_RANSv2.tasks...`. That triggers the package `__init__.py` chain which imports `isaaclab_tasks` (not in dreamer venv). Always load directly:
```python
import importlib.util
spec = importlib.util.spec_from_file_location("zmq_env_client", "/path/to/zmq_env_client.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ZmqEnvClient = mod.ZmqEnvClient
```

### 5.3 `scripts/dreamer/sim_server.py`

**Purpose:** Launches Isaac Sim + ZMQ server. Run with Isaac Sim's Python.

**CLI arguments:**

| Arg | Default | Meaning |
|-----|---------|---------|
| `--robot` | `Pingu` | Robot name → `AutoEnvGenCfg.robot_name` |
| `--task-name` | `GoToPose` | Task name → `AutoEnvGenCfg.task_name` |
| `--num_envs` | `1` | Keep at 1–4 for DreamerV3 |
| `--socket` | `/tmp/isaaclab_dreamer.sock` | IPC socket path |
| `+ AppLauncher args` | — | `--headless`, `--device`, etc. |

All Isaac Sim imports come **after** `AppLauncher(args_cli)`. `sys.argv` is cleared after parsing to prevent Isaac Sim's internal arg parser from choking on unknown args.

### 5.4 `scripts/dreamer/train.py`

**Purpose:** DreamerV3 training entry point. Runs entirely inside the dreamer venv.

**How it works:** Registers `ZmqEnvClient` as a standard gymnasium env, then hands off to `dreamerv3.main.main()`. This delegates all config loading, env wrapping (`embodied.envs.FromGymnasium`), replay, driver, and training loop to dreamerv3's own machinery.

```python
gym.register(
    id="IsaacLabBridge-v0",
    entry_point=lambda **kwargs: ZmqEnvClient(socket_path=socket_path),
    disable_env_checker=True,
)
from dreamerv3 import main as dreamer_main
dreamer_main.main([
    "--task", "gymnasium_IsaacLabBridge-v0",   # note: _ not .
    "--logdir", args.logdir,
    f"--run.steps={args.steps}",
    "--configs", name, ...
])
```

**CLI arguments:**

| Arg | Default | Meaning |
|-----|---------|---------|
| `--socket` | `/tmp/isaaclab_dreamer.sock` | Must match `sim_server.py --socket` |
| `--logdir` | `logs/dreamer` | Checkpoints, replays, config saved here |
| `--steps` | `2000000` | Total environment steps |
| `--configs` | `defaults size12m` | DreamerV3 config presets to stack |

**Available size presets** (this fork uses size names, not `small`/`medium`/`large`):

| Config | Params | Use when |
|--------|--------|----------|
| `size1m` | ~1M | Fast debugging |
| `size12m` | ~12M | Simple tasks (GoToPose, GoToPosition) |
| `size50m` | ~50M | Complex tasks (Rendezvous, RaceWaypoints) |
| `size200m` | ~200M | Full capability |
| `size600m` | ~600M | Research only |

---

## 6. Bugs Encountered and Fixed

### Bug 1: `python3: command not found` (Docker build step 7/14)

```
/bin/bash: line 1: python3: command not found  (exit code 127)
```

**Cause:** The Isaac Sim base image has no system Python. Only `${ISAACSIM_ROOT_PATH}/python.sh` exists.

**Fix:** Added `python3` and `python3-venv` to the apt-get install block unconditionally.

---

### Bug 2: `ModuleNotFoundError: No module named 'isaaclab_tasks'`

```
File ".../Isaaclab_RANSv2/__init__.py", line 11: from .tasks import *
File ".../tasks/__init__.py", line 12: from isaaclab_tasks.utils import import_packages
ModuleNotFoundError: No module named 'isaaclab_tasks'
```

**Cause:** Importing `ZmqEnvClient` via `from Isaaclab_RANSv2.tasks...` inside the dreamer venv triggers every `__init__.py` up the chain. `tasks/__init__.py` imports `isaaclab_tasks` which isn't in the dreamer venv. `zmq_env_client.py` itself has zero Isaac Lab deps.

**Fix:** Load the file directly with `importlib.util.spec_from_file_location`, bypassing all `__init__.py` files. Applied in `train.py` and in the verification command.

---

### Bug 3: `AttributeError: module 'embodied' has no attribute 'Config'`

```
AttributeError: module 'embodied' has no attribute 'Config'
```

**Cause:** The original `train.py` was written against the standard danijar DreamerV3 API. The AndrejOrsula fork separates config into the `elements` library. `embodied` handles runtime (Driver, Replay, Env); `elements` handles config loading, CLI parsing, and spaces.

**Fix:** Scrapped the manual training loop. Instead register `ZmqEnvClient` as a gymnasium env and call `dreamerv3.main.main(argv)` directly, delegating all config/training infrastructure to dreamerv3's own correctly-implemented code.

---

### Bug 4: `NameError: name 'elements' is not defined` in `from_gymnasium.py`

```
File "/root/dreamerv3/embodied/envs/from_gymnasium.py", line 136:
    return elements.Space(space.dtype, space.shape, space.low, space.high)
NameError: name 'elements' is not defined
```

**Cause:** Missing `import elements` at the top of `from_gymnasium.py` in the pinned commit (`4049794d`) of the fork. The file uses `elements.Space` but never imports it.

**Fix:** Patch with `sed` immediately after the clone in the Dockerfile:
```dockerfile
sed -i '1s/^/import elements\n/' "${DREAMER_PATH}/embodied/envs/from_gymnasium.py"
```
For the already-running container: run the same `sed` directly on `/root/dreamerv3/embodied/envs/from_gymnasium.py`.

---

### Bug 5: `ModuleNotFoundError: No module named 'gymnasium'` in dreamer venv

```
File ".../zmq_env_client.py", line 15: import gymnasium as gym
ModuleNotFoundError: No module named 'gymnasium'
```

**Cause:** `gymnasium` and `numpy` are deps of `zmq_env_client.py` but were not explicitly installed in the dreamer venv. DreamerV3's own `setup.py` does not always pull them transitively.

**Fix:** Added both to the venv pip install line in the Dockerfile:
```dockerfile
"${DREAMER_VENV}/bin/pip" install --no-cache-dir pyzmq gymnasium numpy && \
```
For the already-running container: `${DREAMER_VENV}/bin/pip install gymnasium numpy`.

---

### Bug 6: `KeyError: 'medium'` in dreamerv3 configs

```
File "/root/dreamerv3/dreamerv3/main.py", line 28: config = config.update(configs[name])
KeyError: 'medium'
```

**Cause:** The AndrejOrsula fork does not use `small`/`medium`/`large` preset names. It uses size-based names: `size1m`, `size12m`, `size50m`, `size200m`, `size600m`.

**Fix:** Changed default `--configs` in `train.py` from `["defaults"]` to `["defaults", "size12m"]`. Use `size12m` for GoToPose/GoToPosition, `size50m` for more complex tasks.

---

### Bug 7: `ValueError: not enough values to unpack` for task name

```
File "/root/dreamerv3/dreamerv3/main.py", line 213:
    suite, task = config.task.split('_', 1)
ValueError: not enough values to unpack (expected 2, got 1)
```

**Cause:** The task format in this fork uses `_` as the suite separator, not `.`. We were passing `gymnasium.IsaacLabBridge-v0` but it expects `gymnasium_IsaacLabBridge-v0`.

**Fix:** Changed `--task` value in `train.py`:
```python
"--task", "gymnasium_IsaacLabBridge-v0",   # underscore, not dot
```

---

### Bug 8: Sim server exits after DreamerV3 `make_agent()` probe

**Symptom:** Sim server terminal appeared to exit cleanly right after DreamerV3 printed "Compiling train and report...". Training then hung or crashed on first `reset()`.

**Cause:** DreamerV3's internal `make_agent()` function creates a temporary env just to read `obs_space` and `act_space`, then immediately calls `env.close()`. Our original `ZmqEnvServer` treated `close` as a full shutdown — it closed the Isaac Lab env and broke the serve loop. By the time DreamerV3's actual training loop tried to connect, the socket was dead.

JAX compilation (which happens between the probe and actual training) takes 2–5 minutes, making it look like the server "crashed during compilation" when actually it was killed by the probe's `close`.

**Fix:** Split `close` and `shutdown` into two distinct commands:
- `close` → client disconnects, server **continues** waiting for the next connection (Isaac Lab env stays open)
- `shutdown` → server closes Isaac Lab env and exits entirely

Updated in both `zmq_env_server.py` and `zmq_env_client.py`. The `ZmqEnvClient.close()` method sends `close`; a new `ZmqEnvClient.shutdown()` method sends `shutdown` for when you truly want to stop the sim.

---

### Bug 9: `IndexError: too many indices for tensor of dimension 1` on first `reset()`

```
File ".../tasks/go_to_pose.py", line 301:
    self._target_positions[env_ids, :2] - self._robot.root_link_pos_w[self._env_ids, :2][env_ids]
IndexError: too many indices for tensor of dimension 1
```

**Cause:** With `num_envs=1`, `root_link_pos_w` returns a 1D tensor of shape `(3,)` instead of `(1, 3)` — a tensor squeeze edge case in Isaac Lab's articulation property. The existing task code then tries to index that 1D tensor with `[self._env_ids, :2]` (two indices), which fails. This code was always run with `num_envs=4096` in RSL_RL and was never exercised at `num_envs=1`.

**Original fix (superseded):** Temporarily changed `--num_envs` default from 1 to 2 as a workaround. This was superseded by the proper fix below.

---

### Bug 10: `AssertionError: shapes (42,) != (21,)` — checkpoint shape mismatch + obs space problem

```
AssertionError: [Chex] Assertion assert_trees_all_equal_shapes failed:
Trees 0 and 1 differ in leaves 'dec/vec/image/pred/bias': shapes: (42,) != (21,)
```

**Two interacting issues:**

**Issue A — Stale checkpoint:** The `num_envs=2` workaround for Bug 9 caused `ZmqEnvClient.observation_space = Box(shape=(2, 21))`. DreamerV3's `FromGymnasium` flattens this to a 42-dim obs and builds a decoder with bias shape `(42,)`. That checkpoint is now incompatible with any run using `num_envs=1` (which produces a correct 21-dim obs).

**Issue B — Root cause of Bug 9:** The `squeeze()` edge case was not in `go_to_pose.py` but in `pingu.py`. Every robot property (`root_link_pos_w`, `root_pos_w`, `root_quat_w`, etc.) called `.squeeze()` with no dimension argument on a `(num_envs, feat_dim)` tensor. With `num_envs=1`, bare `.squeeze()` removes *all* size-1 dimensions including the batch axis, turning `(1, 3)` → `(3,)`. With `num_envs > 1` the batch axis is never size-1 so this was never triggered.

**Fix A — pingu.py:** Changed all 15 robot property methods from `.squeeze()` to `.squeeze(1)`. This squeezes only the single-body selection dimension (the `[:, self._root_idx]` axis), never the batch axis. Shape is always `(num_envs, feat_dim)` regardless of `num_envs`.

**Fix B — sim_server.py:** Reverted `--num_envs` default back to `1`. DreamerV3 now receives the correct `Box(shape=(21,))` observation space.

**Stale checkpoint:** Delete `logs/dreamer/pingu_gotoPose/` before the next run — the checkpoint built with the wrong 42-dim obs is incompatible:
```bash
rm -rf logs/dreamer/pingu_gotoPose/
```

**Files changed:**
- `source/Isaaclab_RANSv2/.../robots/pingu.py` — 15 properties: `.squeeze()` → `.squeeze(1)`
- `scripts/dreamer/sim_server.py` — `--num_envs` default: `2` → `1`

---

### Bug 11: `KeyError: 'image'` during first training step

```
File ".../embodied/core/wrappers.py", line 225, in step
    obs[key] = np.asarray(obs[key], dtype)
KeyError: 'image'
```

**Cause:** `auto_env_gen.py:_get_observations()` returns `{"policy": tensor}` — a dict. Isaac Lab's gymnasium wrapper also sets `observation_space = Dict({"policy": Box(1, 21)})`. The DreamerV3 fork's `from_gymnasium.py` renames the obs *space* key to `"image"` when it builds the `elements.Space` (so `DtypeWrapper` is initialized expecting `"image"`), but `FromGymnasium._obs()` passes dict observations through unchanged (so the actual step data still has key `"policy"`). The mismatch causes `DtypeWrapper` to fail looking for `"image"`.

**Fix:** `ZmqEnvClient.__init__()` now detects `Dict({"policy": Box})` observation spaces and:
1. Flattens `observation_space` to just the inner `Box` (so DreamerV3 builds its space from a plain array space → uses `"image"` key consistently)
2. Adds `_flatten_obs()` helper that unwraps `{"policy": array}` → `array` in both `reset()` and `step()`

**Files changed:**
- `source/Isaaclab_RANSv2/.../utils/zmq_env_client.py` — `_obs_key` detection + `_flatten_obs()` in reset/step

---

## 7. Environment Details: Pingu + GoToPose

Spaces as computed by `AutoEnvGen.edit_cfg()` merging robot + task configs:

| Source | Space | Size |
|--------|-------|------|
| `PinguRobotCfg` | Observation (last action) | 13 |
| `GoToPoseCfg` | Observation (task state) | 8 |
| **Combined** | **Observation space** | **21** |
| `PinguRobotCfg` | Action: 8 thrusters + 4 arm joints + 1 reaction wheel | 13 |
| `GoToPoseCfg` | Action | 0 |
| **Combined** | **Action space** | **13** |

**Episode:** 40 s at 60 Hz with decimation 6 → ~400 policy steps per episode.

DreamerV3 confirms these at startup:
```
Observations
  image   Space(float32, shape=(1, 21), ...)
Actions
  action  Space(float32, shape=(1, 13), low=-1.0, high=1.0)
```
(`image` is dreamerv3's default key name for the primary observation vector from `FromGymnasium`.)

---

## 8. Verification Checklist (inside container)

Run after `python3 docker/container.py enter base`:

```bash
# 1. Venv Python exists
${DREAMER_VENV}/bin/python --version

# 2. DreamerV3 and embodied importable
${DREAMER_VENV}/bin/python -c "import dreamerv3; import embodied; print('dreamerv3 OK')"

# 3. JAX sees GPU (critical)
${DREAMER_VENV}/bin/python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
# If CpuDevice: reinstall JAX — see Section 9

# 4. pyzmq in Isaac Sim's Python
${ISAAC_SIM_PYTHON} -c "import zmq; print('zmq OK, version:', zmq.__version__)"

# 5. ZmqEnvClient importable without Isaac Lab
${DREAMER_VENV}/bin/python -c "
import importlib.util
spec = importlib.util.spec_from_file_location(
    'zmq_env_client',
    '${ISAAC_LAB_RANS_V2_PATH}/source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/utils/zmq_env_client.py'
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('ZmqEnvClient OK')
"
```

---

## 9. JAX GPU Fix (if step 3 shows CPU)

```bash
${DREAMER_VENV}/bin/pip install --upgrade "jax[cuda12_local]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## 10. End-to-End Usage

### Build

```bash
python3 docker/container.py start
# → [INFO] Include DreamerV3 (MBRL) in this build? [y/N]: y
```

### Run training — two tmux windows, both entered into the same container

**Window 1 — sim server:**
```bash
python3 docker/container.py enter base
${ISAAC_SIM_PYTHON} scripts/dreamer/sim_server.py \
    --robot Pingu \
    --task-name GoToPose \
    --num_envs 1 \
    --socket /tmp/isaaclab_dreamer.sock \
    --headless
# Wait for: [ZmqEnvServer] Listening on ipc:///tmp/isaaclab_dreamer.sock
```

**Window 2 — DreamerV3:**
```bash
python3 docker/container.py enter base
${DREAMER_VENV}/bin/python scripts/dreamer/train.py \
    --logdir logs/dreamer/pingu_gotoPose \
    --steps 2000000 \
    --configs defaults size12m \
    --socket /tmp/isaaclab_dreamer.sock
```

Both windows land in the same running container (`container.py enter` = `docker exec -it isaac-lab-base bash`). They share `/tmp/isaaclab_dreamer.sock`.

**Expected startup sequence:**
1. DreamerV3 connects → prints Observations/Actions spaces
2. DreamerV3 probes spaces via `make_agent()` → sends `close` → server stays alive
3. JAX compiles (~2–5 min, normal) → "Done compiling!"
4. DreamerV3 reconnects for actual training → `reset()` → `step()` loop begins
5. Sim server processes steps; DreamerV3 trains world model

### Switch robot/task

Change `--robot` and `--task-name` on the sim server side only. Train command is unchanged.

- **Robots:** `Pingu`, `FloatingPlatform`, `Cubo`, `Turtlebot2`, `Jetbot`, `IntBall2`, `Kingfisher`, `Leatherback`, `ModularFreeflyer`
- **Tasks:** `GoToPose`, `GoToPosition`, `Rendezvous`, `TrackVelocities`, `RaceWaypoints`

### Stopping cleanly

Ctrl+C on the train window stops DreamerV3. The sim server will keep running (waiting for a new connection). To stop the sim server too, either Ctrl+C it directly or from a third shell:
```bash
${DREAMER_VENV}/bin/python -c "
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('zmq_env_client',
    os.path.expandvars('\${ISAAC_LAB_RANS_V2_PATH}/source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/utils/zmq_env_client.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.ZmqEnvClient().shutdown()
"
```

---

### Bug 12: `KeyError: 'image'` — same symptom as Bug 11, different root cause

```
KeyError: 'image'
```

**Why Bug 11's fix was incomplete:** Bug 11's analysis assumed Isaac Lab's gymnasium wrapper exposes `observation_space = Dict({"policy": Box})`. It does not. `DirectRLEnv._configure_gym_env_spaces()` sets `self.observation_space = gym.vector.utils.batch_space(single_obs_space["policy"], num_envs)` — a **plain `Box(shape=(num_envs, obs_dim))`**, not a Dict. So the `isinstance(raw_obs_space, gym.spaces.Dict)` check in `ZmqEnvClient.__init__` always fails, leaving `_obs_key = None`.

**Cause:** With `_obs_key = None`, `_flatten_obs` returned the obs unchanged. Isaac Lab's `step()` returns `self.obs_buf = {"policy": tensor}` (a dict, confirmed at `direct_rl_env.py:389`). After `_to_numpy` on the server this becomes `{"policy": numpy_array}`. With `_obs_key = None`, `_flatten_obs` passes this dict straight through. `FromGymnasium` then wraps it as `{"image": {"policy": array}}`, and its `_flatten()` routine turns nested dicts into slash-joined keys, producing `{"image/policy": array}`. `UnifyDtypes` then looks for `"image"` → KeyError.

**Fix:** Added a fallback in `_flatten_obs` that unconditionally unwraps `{"policy": array}` when `_obs_key is None` but the obs is still a "policy"-keyed dict:

```python
def _flatten_obs(self, obs):
    if self._obs_key is not None and isinstance(obs, dict):
        return obs[self._obs_key]
    # Isaac Lab sets observation_space=Box but step()/reset() still return {"policy": array}
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return obs
```

**Files changed:**
- `source/Isaaclab_RANSv2/.../utils/zmq_env_client.py` — `_flatten_obs()` fallback for Box obs_space + dict step return

---

### Bug 13: `AttributeError: 'numpy.ndarray' object has no attribute 'to'` in sim server step

```
File ".../direct_rl_env.py", line 337, in step
    action = action.to(self.device)
AttributeError: 'numpy.ndarray' object has no attribute 'to'
```

**Cause:** `ZmqEnvServer.serve()` passed `msg["action"]` (a numpy array deserialized from pickle) directly to `self._env.step()`. Isaac Lab's `DirectRLEnv.step()` calls `action.to(self.device)` expecting a PyTorch tensor.

**Fix:** Added `_action_to_tensor(action_np, device)` helper in `zmq_env_server.py` that converts the incoming numpy array to a CUDA tensor, and call it before every `self._env.step()`:

```python
device = getattr(self._env.unwrapped, "device", "cpu")
action = _action_to_tensor(msg["action"], device)
obs, reward, terminated, truncated, info = self._env.step(action)
```

**Files changed:**
- `source/Isaaclab_RANSv2/.../utils/zmq_env_server.py` — `_action_to_tensor()` helper + convert action in step handler

---

### Bug 14: `ValueError: reward shape (1,) not in Space(float32, shape=())`

```
ValueError: Value for 'reward' with dtype float32, shape (1,), ... is not in Space(float32, shape=(), ...)
```

**Cause:** Isaac Lab's `step()` returns `reward`, `terminated`, `truncated` as tensors of shape `(num_envs,)` = `(1,)`. `_to_numpy` preserves that shape. `FromGymnasium._obs()` calls `np.float32(reward)` which in the installed NumPy version does NOT squeeze a `(1,)` array to a scalar — it returns a `(1,)` float32 array. `CheckSpaces` then rejects it because `obs_space["reward"]` has `shape=()`.

**Fix:** Squeeze all three scalars to Python primitives on the server side before pickling:

```python
"reward":     float(_to_numpy(reward).flat[0]),
"terminated": bool(_to_numpy(terminated).flat[0]),
"truncated":  bool(_to_numpy(truncated).flat[0]),
```

`.flat[0]` extracts the single element regardless of shape, and the `float()`/`bool()` cast gives a plain Python scalar that `np.float32()` / boolean logic in `from_gymnasium.py` can handle correctly.

**Files changed:**
- `source/Isaaclab_RANSv2/.../utils/zmq_env_server.py` — step reply: reward/terminated/truncated squeezed to Python scalars

---

## 11. File Map

```
Isaaclab_RANSv2/
├── docker/
│   ├── Dockerfile.base        ← python3/venv in apt; sed patch; dreamer venv block; DREAMER_DEV=false
│   ├── docker-compose.yaml    ← DREAMER_DEV build arg forwarded to Dockerfile
│   └── container.py           ← interactive DreamerV3 prompt before ci.start()
├── scripts/
│   └── dreamer/               ← NEW
│       ├── sim_server.py      ← Isaac Sim launch + ZMQ server (run with ${ISAAC_SIM_PYTHON})
│       └── train.py           ← DreamerV3 training entry point (run with ${DREAMER_VENV}/bin/python)
└── source/Isaaclab_RANSv2/Isaaclab_RANSv2/tasks/direct/isaaclab_ransv2/
    ├── robots/
    │   └── pingu.py           ← MODIFIED: 15× .squeeze() → .squeeze(1) (fixes num_envs=1 shape bug)
    └── utils/
        ├── zmq_env_server.py  ← NEW: gymnasium env → ZMQ REP server (close≠shutdown)
        └── zmq_env_client.py  ← NEW: ZMQ REQ client → gymnasium.Env (close/shutdown split)
```
