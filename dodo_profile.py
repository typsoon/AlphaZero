import sys
from pathlib import Path
from python.utils import (
    BUILD_DIR,
    PROJ_ROOT,
    SUPPORTED_GAMES,
    NETWORK_PARAM,
    resolve_network_path,
    run_protected,
)

OUT_DIR = PROJ_ROOT / "out"
_MKDIR_OUT_DIR = f"mkdir -p {OUT_DIR}"


def _get_profile_params(default_num_games: int, default_thread_count: int) -> list:
    return [
        {
            "name": "game",
            "short": "g",
            "long": "game",
            "type": str,
            "default": "connect4",
            "help": "The game to run inference for",
            "choices": SUPPORTED_GAMES,
        },
        NETWORK_PARAM,
        {
            "name": "num_games",
            "long": "num_games",
            "type": int,
            "default": default_num_games,
            "help": "Number of games to profile",
        },
        {
            "name": "thread_count",
            "long": "thread_count",
            "type": int,
            "default": default_thread_count,
            "help": "Number of threads to use",
        },
        {
            "name": "max_moves",
            "long": "max_moves",
            "type": int,
            "default": 512,
            "help": "Maximum number of moves per game before forcing a draw",
        },
        {
            "name": "mcts_num_simulations",
            "long": "mcts_num_simulations",
            "type": int,
            "default": 800,
            "help": "Number of MCTS simulations per move",
        },
        {
            "name": "mcts_batch_size",
            "long": "mcts_batch_size",
            "type": int,
            "default": 32,
            "help": "Batch size used within MCTS::search for leaf evaluation",
        },
    ]


def task_profile_self_play_cachegrind():
    """Run a profiler on the self_play execution using cachegrind."""
    profiling_bin = BUILD_DIR / "engine" / "profiling" / "run_self_play"

    def run_cachegrind(
        game,
        network_path,
        num_games,
        thread_count,
        max_moves,
        mcts_num_simulations,
        mcts_batch_size,
    ):
        network_path = resolve_network_path(network_path, game)
        # Empty "" is the kineto_out placeholder (skips kineto profiling) so the
        # positional mcts_num_simulations/mcts_batch_size args can still be reached.
        cmd = (
            f"valgrind --tool=cachegrind --cachegrind-out-file={OUT_DIR}/cachegrind.out "
            f'{profiling_bin} {game} {network_path} {num_games} {thread_count} {max_moves} "" '
            f"{mcts_num_simulations} {mcts_batch_size}"
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            f"cmake --build {BUILD_DIR} --target run_self_play -j$(nproc)",
            run_cachegrind,
            f"echo '\\033[32mRun `kcachegrind {OUT_DIR}/cachegrind.out` to view the profiling results.\\033[0m'",
        ],
        "params": _get_profile_params(default_num_games=2, default_thread_count=1),
        "task_dep": ["setup_vcpkg"],
    }


def _perf_cycles_event():
    """Pick the perf event(s) needed to see all CPU activity on this machine.

    On Intel hybrid (P-core/E-core) CPUs, plain "cycles" resolves to a single PMU
    and silently misses whatever ran on the other core type. Non-hybrid machines
    (AMD, older Intel, most CI/cloud boxes) don't have cpu_core/cpu_atom PMUs at
    all, and requesting them makes perf record fail outright. Detect which case
    we're in via the PMUs registered in sysfs rather than hardcoding either.
    """
    pmu_dir = Path("/sys/bus/event_source/devices")
    if (pmu_dir / "cpu_core").exists() and (pmu_dir / "cpu_atom").exists():
        return "cpu_core/cycles/,cpu_atom/cycles/"
    return "cycles"


def task_profile_self_play_perf():
    """Run a multithreaded profiler on the self_play execution using Linux perf."""
    profiling_bin = BUILD_DIR / "engine" / "profiling" / "run_self_play"

    def run_perf(
        game,
        network_path,
        num_games,
        thread_count,
        max_moves,
        mcts_num_simulations,
        mcts_batch_size,
    ):
        network_path = resolve_network_path(network_path, game)
        # --call-graph dwarf is used instead of -g (frame-pointer) because
        # libtorch's optimized kernels are often built without frame pointers,
        # which otherwise breaks call-graph attribution.
        # Empty "" is the kineto_out placeholder (skips kineto profiling) so the
        # positional mcts_num_simulations/mcts_batch_size args can still be reached.
        cmd = (
            f"perf record -e {_perf_cycles_event()} --call-graph dwarf "
            f"-o {OUT_DIR}/perf.data {profiling_bin} {game} {network_path} {num_games} "
            f'{thread_count} {max_moves} "" {mcts_num_simulations} {mcts_batch_size}'
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            f"cmake --build {BUILD_DIR} --target run_self_play -j$(nproc)",
            run_perf,
            f"echo '\\033[32mRun `perf report -g --input {OUT_DIR}/perf.data` to view the multithreaded profiling results.\\033[0m'",
        ],
        "params": _get_profile_params(default_num_games=10, default_thread_count=4),
        "task_dep": ["setup_vcpkg"],
    }


def task_profile_self_play_kineto():
    """Run a profiler on the self_play execution using PyTorch Kineto."""
    profiling_bin = BUILD_DIR / "engine" / "profiling" / "run_self_play"

    def run_kineto(
        game,
        network_path,
        num_games,
        thread_count,
        max_moves,
        mcts_num_simulations,
        mcts_batch_size,
    ):
        network_path = resolve_network_path(network_path, game)
        out_file = OUT_DIR / "pytorch_profile.json"
        cmd = (
            f"{profiling_bin} {game} {network_path} {num_games} {thread_count} {max_moves} "
            f"{out_file} {mcts_num_simulations} {mcts_batch_size}"
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            f"cmake --build {BUILD_DIR} --target run_self_play -j$(nproc)",
            run_kineto,
            f"echo '\\033[32mProfile saved to {OUT_DIR}/pytorch_profile.json. Open chrome://tracing or ui.perfetto.dev to view it.\\033[0m'",
        ],
        "params": _get_profile_params(default_num_games=2, default_thread_count=1),
        "task_dep": ["setup_vcpkg"],
    }


def task_profile_self_play_nsys():
    """Run a profiler on the self_play execution using Nsight Systems (nsys)."""
    profiling_bin = BUILD_DIR / "engine" / "profiling" / "run_self_play"

    def run_nsys(
        game,
        network_path,
        num_games,
        thread_count,
        max_moves,
        mcts_num_simulations,
        mcts_batch_size,
    ):
        network_path = resolve_network_path(network_path, game)
        # --cudabacktrace=memory:1000000 captures a CPU backtrace for any CUDA memory
        # API call (memcpy/memset) that takes over 1ms, so slow calls can be traced
        # back to the exact source line that issued them - the plain CUDA API name
        # alone (e.g. cudaMemcpyAsync) doesn't say which of many call sites it was.
        # Empty "" is the kineto_out placeholder (skips kineto profiling) so the
        # positional mcts_num_simulations/mcts_batch_size args can still be reached.
        cmd = (
            "nsys profile --cudabacktrace=memory:1000000 --force-overwrite=true "
            f"-o {OUT_DIR}/nsys_capture {profiling_bin} {game} {network_path} {num_games} "
            f'{thread_count} {max_moves} "" {mcts_num_simulations} {mcts_batch_size}'
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            f"cmake --build {BUILD_DIR} --target run_self_play -j$(nproc)",
            run_nsys,
            f"echo '\\033[32mProfile saved to {OUT_DIR}/nsys_capture.nsys-rep. Run `nsys stats {OUT_DIR}/nsys_capture.nsys-rep` or open it in the Nsight Systems UI to view it.\\033[0m'",
        ],
        "params": _get_profile_params(default_num_games=2, default_thread_count=1),
        "task_dep": ["setup_vcpkg"],
    }


def _get_train_profile_params(default_training_iterations: int) -> list:
    return [
        {
            "name": "game",
            "short": "g",
            "long": "game",
            "type": str,
            "default": "connect4",
            "help": "The game to profile training for",
            "choices": SUPPORTED_GAMES,
        },
        {
            "name": "replay_size",
            "long": "replay_size",
            "type": int,
            "default": 8192,
            "help": "Number of synthetic transitions to seed the replay buffer with",
        },
        {
            "name": "training_iterations",
            "long": "training_iterations",
            "type": int,
            "default": default_training_iterations,
            "help": "Number of optimizer steps to profile",
        },
        {
            "name": "batch_size",
            "long": "batch_size",
            "type": int,
            "default": 256,
            "help": "Training batch size",
        },
        {
            "name": "minibatch_size",
            "long": "minibatch_size",
            "type": int,
            "default": 4096,
            "help": "Training minibatch size",
        },
    ]


def _profile_train_cmd(
    game, replay_size, training_iterations, batch_size, minibatch_size
):
    return (
        f"{sys.executable} -m python.tools.profile_train --game {game} "
        f"--replay-size {replay_size} --training-iterations {training_iterations} "
        f"--batch-size {batch_size} --minibatch-size {minibatch_size}"
    )


def task_profile_train_cachegrind():
    """Run a profiler on AlphaZeroTrainer.train() (no self-play) using cachegrind."""

    def run_cachegrind(
        game, replay_size, training_iterations, batch_size, minibatch_size
    ):
        cmd = (
            f"valgrind --tool=cachegrind --cachegrind-out-file={OUT_DIR}/cachegrind_train.out "
            + _profile_train_cmd(
                game, replay_size, training_iterations, batch_size, minibatch_size
            )
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            run_cachegrind,
            f"echo '\\033[32mRun `kcachegrind {OUT_DIR}/cachegrind_train.out` to view the profiling results.\\033[0m'",
        ],
        "params": _get_train_profile_params(default_training_iterations=20),
        "task_dep": ["build"],
    }


def task_profile_train_perf():
    """Run a profiler on AlphaZeroTrainer.train() (no self-play) using Linux perf."""

    def run_perf(game, replay_size, training_iterations, batch_size, minibatch_size):
        cmd = (
            f"perf record -e {_perf_cycles_event()} --call-graph dwarf "
            f"-o {OUT_DIR}/perf_train.data "
            + _profile_train_cmd(
                game, replay_size, training_iterations, batch_size, minibatch_size
            )
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            run_perf,
            f"echo '\\033[32mRun `perf report -g --input {OUT_DIR}/perf_train.data` to view the profiling results.\\033[0m'",
        ],
        "params": _get_train_profile_params(default_training_iterations=100),
        "task_dep": ["build"],
    }


def task_profile_train_kineto():
    """Run a profiler on AlphaZeroTrainer.train() (no self-play) using PyTorch Kineto."""

    def run_kineto(game, replay_size, training_iterations, batch_size, minibatch_size):
        out_file = OUT_DIR / "pytorch_train_profile.json"
        cmd = (
            _profile_train_cmd(
                game, replay_size, training_iterations, batch_size, minibatch_size
            )
            + f" --kineto-out {out_file}"
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            run_kineto,
            f"echo '\\033[32mProfile saved to {OUT_DIR}/pytorch_train_profile.json. Open chrome://tracing or ui.perfetto.dev to view it.\\033[0m'",
        ],
        "params": _get_train_profile_params(default_training_iterations=20),
        "task_dep": ["build"],
    }


def task_profile_train_cprofile():
    """Run a Python-level profiler on AlphaZeroTrainer.train() (no self-play) using cProfile."""

    def run_cprofile(
        game, replay_size, training_iterations, batch_size, minibatch_size
    ):
        out_file = OUT_DIR / "train_cprofile.prof"
        cmd = (
            _profile_train_cmd(
                game, replay_size, training_iterations, batch_size, minibatch_size
            )
            + f" --cprofile-out {out_file}"
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            run_cprofile,
            f"echo '\\033[32mProfile saved to {OUT_DIR}/train_cprofile.prof. Run "
            f"`python -m pstats {OUT_DIR}/train_cprofile.prof` or `snakeviz {OUT_DIR}/train_cprofile.prof` to view it.\\033[0m'",
        ],
        "params": _get_train_profile_params(default_training_iterations=100),
        "task_dep": ["build"],
    }


def task_profile_train_nsys():
    """Run a profiler on AlphaZeroTrainer.train() (no self-play) using Nsight Systems (nsys)."""

    def run_nsys(game, replay_size, training_iterations, batch_size, minibatch_size):
        cmd = (
            "nsys profile --cudabacktrace=memory:1000000 --force-overwrite=true "
            f"-o {OUT_DIR}/nsys_train_capture "
            + _profile_train_cmd(
                game, replay_size, training_iterations, batch_size, minibatch_size
            )
        )
        return run_protected(cmd)

    return {
        "actions": [
            _MKDIR_OUT_DIR,
            run_nsys,
            f"echo '\\033[32mProfile saved to {OUT_DIR}/nsys_train_capture.nsys-rep. Run `nsys stats {OUT_DIR}/nsys_train_capture.nsys-rep` or open it in the Nsight Systems UI to view it.\\033[0m'",
        ],
        "params": _get_train_profile_params(default_training_iterations=100),
        "task_dep": ["build"],
    }
