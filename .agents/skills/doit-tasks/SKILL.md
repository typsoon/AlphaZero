---
name: doit-tasks
description: Reference for the doit (pydoit) task automation defined in dodo.py, dodo_profile.py, and doit_systemd.py at the repo root — what each `doit <task>` command does, its dependencies, and its parameters. Use this whenever the user asks to build, test, lint, format, profile, run, or manage the AlphaZero project via doit, or asks "what does task X do" / "how do I run X".
---

# doit tasks

This repo drives builds, tests, linting, profiling, and service management through
[doit](https://pydoit.org/), configured in `dodo.py` (root), which imports
`dodo_profile.py` (all `profile_*` tasks) and `doit_systemd.py` (systemd service
tasks). Shared helpers (`PROJ_ROOT`, `BUILD_DIR`, `SUPPORTED_GAMES`, `NETWORK_PARAM`,
`resolve_network_path`, `run_protected`) live in `python/utils/__init__.py` so both
files can import them without a circular dependency on `dodo.py` itself. Run tasks
with `doit <task_name>` or `doit <task>:<subtask>` from the repo root. List all tasks
with `doit list`.

## Build & setup

- **`setup_vcpkg`** — Clones and bootstraps vcpkg locally, then installs the C++
  dependencies declared in the vcpkg manifest.
- **`patch_torchtrt`** — Patches `libtorchtrt.so`'s RUNPATH (via `patchelf`) so the
  C++ binaries can find TensorRT libraries without needing `LD_LIBRARY_PATH` set.
  Works around torch-tensorrt's Bazel-built, broken `$ORIGIN`-relative `DT_RUNPATH`.
  Soft-fails (returns success) if torch_tensorrt isn't installed.
- **`build`** — Configures (if needed) and builds the C++ engine/inference server
  with CMake, using ccache when available. Depends on `setup_vcpkg` and
  `patch_torchtrt`. Param `-t/--type` selects `debug` (default) or `release`; reuses
  the previous CMake configuration's build type when re-running.

## Formatting & linting

- **`check_format`** (subtasks: `python`, `cpp`, `cmake`, `connect4_puzzles`, `ts`) —
  Checks formatting without modifying files: `ruff format --check` (Python),
  `clang-format --dry-run -Werror` (C++), `cmake-format --check` (CMake files), a
  puzzle-JSON formatting checker, and `npm run format:check` in both
  `gameplay_client` and `gameplay_server`.
- **`format`** (subtasks: `python`, `cpp`, `cmake`, `connect4_puzzles`, `ts`) — Same
  scope as `check_format` but applies the fixes in place.
- **`lint`** (subtasks: `python`, `cpp`, `cmake`, `ts`) — Runs `ruff check` (Python),
  cached `clang-tidy` (C++, depends on `build`), `cmake-lint`, and `npm run lint` in
  the client/server packages.
- **`fix_python_lint`** — Runs `ruff check --fix` to auto-fix Python lint errors.

## Testing & validation

- **`test_python`** — Runs the Python integration test suite (`pytest python/test`).
- **`test_train_loss`** — Runs the AlphaZero training loss convergence test
  (`python/test/test_train_loss.py`).
- **`test_cpp`** — Runs the compiled C++ test binaries (inference server tests,
  `test_chess`, `test_mcts`). Depends on `build`.
- **`test_ts`** — Runs the gameplay server's TypeScript test suite (`npm run test`).
- **`validate_connect4_puzzles`** — Validates the Connect4 puzzle JSON files.
- **`validate_chess_puzzles`** — Validates the Chess puzzle JSON files.
- **`check_all`** — Umbrella CI task with no actions of its own; depends on
  `check_format`, `lint:python`, `lint:cmake`, `lint:ts`, `lint:cpp`,
  `validate_connect4_puzzles`, `validate_chess_puzzles`, `test_cpp`, `test_python`,
  `test_train_loss`, and `test_ts`.

## Running components

- **`run_client`** — Runs the gameplay client in dev mode (`npm run dev`).
- **`run_game_server`** — Runs the gameplay server in dev mode (`npm run start`).
- **`run_inference_server`** — Runs the built inference server binary directly.
  Params: `--game` (`connect4`/`chess`) and `--network_path` (defaults to the
  resolved checkpoint for the chosen game). Depends on `build`.
- **`clear_db`** — Deletes the gameplay server's SQLite database files
  (`games.db`, `-wal`, `-shm`).

## Performance evaluation & benchmarking

- **`test_performance`** — Runs the performance evaluator against a network and
  generates the HTML report. Params: `--network_path`, `--mcts_search_depth`,
  `--game`. Depends on `build`.
- **`benchmark_batch_size`** — Runs the inference server batch-size benchmark.
  Param: `--network_path`. Depends on `build`.
- **`generate_tensorrt_models`** — Iterates available checkpoints and generates
  scripted TensorRT models for them. Param: `--max_first_dim` (optional override).

## Profiling (`dodo_profile.py`)

Four profilers are used throughout the project — Valgrind cachegrind, Linux `perf`,
PyTorch Kineto, and Nsight Systems (`nsys`) — each wired up for two different
targets: the C++ self-play engine, and the Python training step in isolation.

### Profiling self-play

All four share the same parameter set (`--game`, `--network_path`, `--num_games`,
`--thread_count`, `--max_moves`, `--mcts_num_simulations`, `--mcts_batch_size`) and
first rebuild the `run_self_play` target, then depend on `setup_vcpkg`:

- **`profile_self_play_cachegrind`** — Profiles self-play with Valgrind's
  cachegrind; view results with `kcachegrind cachegrind.out`.
- **`profile_self_play_perf`** — Profiles self-play with Linux `perf record`
  (using `--call-graph dwarf` since libtorch's kernels often lack frame pointers,
  and detecting hybrid P-core/E-core CPUs to pick the right perf event); view with
  `perf report -g`.
- **`profile_self_play_kineto`** — Profiles self-play with PyTorch Kineto, writing
  `pytorch_profile.json` (view via `chrome://tracing` or ui.perfetto.dev).
- **`profile_self_play_nsys`** — Profiles self-play with Nsight Systems (`nsys
  profile`, tracking CUDA memory-API backtraces), writing `nsys_capture.nsys-rep`.

### Profiling training

These profile only `AlphaZeroTrainer.train()` (the optimizer step — forward,
backward, `optimizer.step()`), *not* self-play — use the `profile_self_play_*`
tasks above for that. They run `python -m python.tools.profile_train`, a standalone
script that seeds a real `ReplayBuffer` with synthetic random transitions (so no
self-play or checkpoint I/O is needed) and calls `trainer.train()` directly. All
four share the same parameter set (`--game`, `--replay_size`, `--training_iterations`,
`--batch_size`, `--minibatch_size`) and depend on `build` (for the `engine_bind`
pybind extension):

- **`profile_train_cachegrind`** — Profiles training with Valgrind's cachegrind;
  view results with `kcachegrind cachegrind_train.out`.
- **`profile_train_perf`** — Profiles training with Linux `perf record`
  (same `--call-graph dwarf` / hybrid-CPU handling as the self-play variant); view
  with `perf report -g --input perf_train.data`.
- **`profile_train_kineto`** — Profiles training with PyTorch Kineto (passes
  `--kineto-out` through to `profile_train.py`, which wraps the training steps in
  `torch.profiler.profile`), writing `pytorch_train_profile.json`.
- **`profile_train_nsys`** — Profiles training with Nsight Systems, writing
  `nsys_train_capture.nsys-rep`.

## Systemd service management (`doit_systemd.py`)

These manage the inference server as a per-game systemd **user** service
(`alphazero-inference@connect4.service`, `alphazero-inference@chess.service`):

- **`install_service_binary`** — Copies the built inference server binary to
  `~/.local/bin/alphazero-inference-server`. Depends on `build`.
- **`setup_service`** — Installs the systemd unit template to
  `~/.config/systemd/user/` and generates per-game env files under the
  `alphazero` platform config dir (from `inference.env.example`), then runs
  `systemctl --user daemon-reload`. Depends on `install_service_binary`.
- **`enable_service`** — Enables both game services to start on boot. Depends on
  `setup_service`.
- **`disable_service`** — Disables both game services.
- **`start_service`** — Starts both services and verifies they stay active,
  dumping recent journal logs on failure. Depends on `setup_service` and
  `install_service_binary`.
- **`restart_service`** — Same as `start_service` but restarts (to pick up config
  changes) instead of starting fresh.
- **`stop_service`** — Stops both game services.
