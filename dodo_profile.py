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

# Maps JSON training-param field names that differ from doit param names.
_JSON_FIELD_MAP = {
    "mcts_simulations": "mcts_num_simulations",
    "fast_mcts_simulations": "fast_mcts_num_simulations",
    "initial_network": "network_path",
    "games_in_each_iteration": "num_games",
}


def _merge_params_file(params_file: str, params: dict, field_map: dict = None) -> dict:
    """Load a training-params JSON and merge it into *params*, returning a new dict.

    CLI flags always win over the file; the file only fills in values that were
    not explicitly overridden on the command line (i.e. still at their doit default).
    Unknown JSON keys are silently ignored so the full training JSON can be passed
    without needing to strip training-only fields first.

    *field_map* overrides the default _JSON_FIELD_MAP for callers whose doit param
    names diverge from it (e.g. run_inference_server's mcts_search_depth).
    """
    if not params_file:
        return params
    import json

    with open(params_file) as f:
        data = json.load(f)

    if field_map is None:
        field_map = _JSON_FIELD_MAP

    merged = dict(params)
    for key, value in data.items():
        param_name = field_map.get(key, key)
        if param_name in merged:
            merged[param_name] = value
    return merged


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
        {
            "name": "fast_mcts_num_simulations",
            "long": "fast_mcts_num_simulations",
            "type": int,
            "default": 100,
            "help": "Playout-cap-randomization 'fast' search simulation count "
            "(see full_search_probability)",
        },
        {
            "name": "full_search_probability",
            "long": "full_search_probability",
            "type": float,
            "default": 0.25,
            "help": "Probability a move uses the full mcts_num_simulations search "
            "instead of the cheaper fast_mcts_num_simulations one",
        },
        {
            "name": "use_gumbel_search",
            "long": "use_gumbel_search",
            "type": bool,
            "default": False,
            "help": "Use MCTS::search_gumbel() (Gumbel-Top-k + sequential halving) "
            "instead of the default Dirichlet-noised PUCT search",
        },
        {
            "name": "max_num_considered_actions",
            "long": "max_num_considered_actions",
            "type": int,
            "default": 16,
            "help": "Gumbel search only: full search's Gumbel-Top-k candidate-set size (m)",
        },
        {
            "name": "transposition_cache_entries",
            "long": "transposition_cache_entries",
            "type": int,
            "default": 1_000_000,
            "help": "Self-play NN-inference transposition cache slot count. 0 disables it",
        },
        {
            "name": "chess_encoder_history",
            "long": "chess_encoder_history",
            "type": int,
            "default": 0,
            "help": "Length of the state history for the chess encoder (e.g. 4 for historical nets).",
        },
        {
            "name": "params_file",
            "long": "params_file",
            "type": str,
            "default": "",
            "help": "Path to a training-params JSON (e.g. training_params/chess_params_gumbell.json). "
            "Values from the file are used as defaults; explicit CLI flags still override them.",
        },
        {
            "name": "value_network_path",
            "long": "value_network_path",
            "type": str,
            "default": "",
            "help": "Optional second network to evaluate values, separating policy and value inference",
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
        fast_mcts_num_simulations,
        full_search_probability,
        use_gumbel_search,
        max_num_considered_actions,
        transposition_cache_entries,
        chess_encoder_history,
        value_network_path,
        params_file,
    ):
        p = _merge_params_file(params_file, locals())
        game = p["game"]
        network_path = p["network_path"]
        num_games = p["num_games"]
        thread_count = p["thread_count"]
        max_moves = p["max_moves"]
        mcts_num_simulations = p["mcts_num_simulations"]
        mcts_batch_size = p["mcts_batch_size"]
        fast_mcts_num_simulations = p["fast_mcts_num_simulations"]
        full_search_probability = p["full_search_probability"]
        use_gumbel_search = p["use_gumbel_search"]
        max_num_considered_actions = p["max_num_considered_actions"]
        transposition_cache_entries = p["transposition_cache_entries"]
        value_network_path = p.get("value_network_path", "")
        if value_network_path:
            value_network_path = resolve_network_path(value_network_path, game)

        cmd = (
            f"valgrind --tool=cachegrind --cachegrind-out-file={OUT_DIR}/cachegrind.out "
            f'{profiling_bin} {game} {network_path} {num_games} {thread_count} {max_moves} "" '
            f"{mcts_num_simulations} {mcts_batch_size} {fast_mcts_num_simulations} "
            f"{full_search_probability} {int(use_gumbel_search)} {max_num_considered_actions} "
            f"{transposition_cache_entries} {chess_encoder_history} {value_network_path}"
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
        fast_mcts_num_simulations,
        full_search_probability,
        use_gumbel_search,
        max_num_considered_actions,
        transposition_cache_entries,
        chess_encoder_history,
        value_network_path,
        params_file,
    ):
        p = _merge_params_file(params_file, locals())
        game = p["game"]
        network_path = p["network_path"]
        num_games = p["num_games"]
        thread_count = p["thread_count"]
        max_moves = p["max_moves"]
        mcts_num_simulations = p["mcts_num_simulations"]
        mcts_batch_size = p["mcts_batch_size"]
        fast_mcts_num_simulations = p["fast_mcts_num_simulations"]
        full_search_probability = p["full_search_probability"]
        use_gumbel_search = p["use_gumbel_search"]
        max_num_considered_actions = p["max_num_considered_actions"]
        transposition_cache_entries = p["transposition_cache_entries"]
        chess_encoder_history = p["chess_encoder_history"]
        network_path = resolve_network_path(network_path, game)
        value_network_path = p.get("value_network_path", "")
        if value_network_path:
            value_network_path = resolve_network_path(value_network_path, game)

        # Profiling only works with TensorRT models, so convert PyTorch .pt paths to .pt_trt
        if network_path.endswith(".pt"):
            import os

            dirname, basename = os.path.split(network_path)
            network_path = os.path.join(dirname, "tensorrt", basename + "_trt")
        # --call-graph dwarf is used instead of -g (frame-pointer) because
        # libtorch's optimized kernels are often built without frame pointers,
        # which otherwise breaks call-graph attribution.
        # Empty "" is the kineto_out placeholder (skips kineto profiling) so the
        # positional mcts_num_simulations/mcts_batch_size args can still be reached.
        cmd = (
            f"perf record -e {_perf_cycles_event()} --call-graph dwarf "
            f"-o {OUT_DIR}/perf.data {profiling_bin} {game} {network_path} {num_games} "
            f'{thread_count} {max_moves} "" {mcts_num_simulations} {mcts_batch_size} '
            f"{fast_mcts_num_simulations} {full_search_probability} {int(use_gumbel_search)} "
            f"{max_num_considered_actions} {transposition_cache_entries} {chess_encoder_history} "
            f"{value_network_path}"
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
        fast_mcts_num_simulations,
        full_search_probability,
        use_gumbel_search,
        max_num_considered_actions,
        transposition_cache_entries,
        chess_encoder_history,
        value_network_path,
        params_file,
    ):
        p = _merge_params_file(params_file, locals())
        game = p["game"]
        network_path = p["network_path"]
        num_games = p["num_games"]
        thread_count = p["thread_count"]
        max_moves = p["max_moves"]
        mcts_num_simulations = p["mcts_num_simulations"]
        mcts_batch_size = p["mcts_batch_size"]
        fast_mcts_num_simulations = p["fast_mcts_num_simulations"]
        full_search_probability = p["full_search_probability"]
        use_gumbel_search = p["use_gumbel_search"]
        max_num_considered_actions = p["max_num_considered_actions"]
        transposition_cache_entries = p["transposition_cache_entries"]
        chess_encoder_history = p["chess_encoder_history"]
        network_path = resolve_network_path(network_path, game)
        value_network_path = p.get("value_network_path", "")
        if value_network_path:
            value_network_path = resolve_network_path(value_network_path, game)

        # Profiling only works with TensorRT models, so convert PyTorch .pt paths to .pt_trt
        if network_path.endswith(".pt"):
            import os

            dirname, basename = os.path.split(network_path)
            network_path = os.path.join(dirname, "tensorrt", basename + "_trt")
        out_file = OUT_DIR / "pytorch_profile.json"
        cmd = (
            f"{profiling_bin} {game} {network_path} {num_games} {thread_count} {max_moves} "
            f"{OUT_DIR}/pytorch_profile.json {mcts_num_simulations} {mcts_batch_size} "
            f"{fast_mcts_num_simulations} {full_search_probability} {int(use_gumbel_search)} "
            f"{max_num_considered_actions} {transposition_cache_entries} {chess_encoder_history} "
            f"{value_network_path}"
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
        fast_mcts_num_simulations,
        full_search_probability,
        use_gumbel_search,
        max_num_considered_actions,
        transposition_cache_entries,
        params_file,
    ):
        p = _merge_params_file(params_file, locals())
        game = p["game"]
        network_path = p["network_path"]
        num_games = p["num_games"]
        thread_count = p["thread_count"]
        max_moves = p["max_moves"]
        mcts_num_simulations = p["mcts_num_simulations"]
        mcts_batch_size = p["mcts_batch_size"]
        fast_mcts_num_simulations = p["fast_mcts_num_simulations"]
        full_search_probability = p["full_search_probability"]
        use_gumbel_search = p["use_gumbel_search"]
        max_num_considered_actions = p["max_num_considered_actions"]
        transposition_cache_entries = p["transposition_cache_entries"]
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
            f'{thread_count} {max_moves} "" {mcts_num_simulations} {mcts_batch_size} '
            f"{fast_mcts_num_simulations} {full_search_probability} "
            f"{int(use_gumbel_search)} {max_num_considered_actions} "
            f"{transposition_cache_entries}"
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
        {
            "name": "network_arch",
            "long": "network_arch",
            "type": str,
            "default": "legacy",
            "choices": (("legacy", ""), ("chess_v2", "")),
            "help": "Network architecture to profile. Use 'chess_v2' to profile "
            "the live history-encoder bootstrap net; 'legacy' (default) keeps "
            "the original behavior.",
        },
        {
            "name": "chess_encoder_history",
            "long": "chess_encoder_history",
            "type": str,
            "default": "",
            "help": "chess_v2 only: history length N in {1,4,8} for the "
            "63-plane ChessEncoderV2History input (empty = 19-plane default). "
            "Sizes both the net and the synthetic states. For the live run: 4.",
        },
    ]


def _profile_train_cmd(
    game,
    replay_size,
    training_iterations,
    batch_size,
    minibatch_size,
    network_arch="legacy",
    chess_encoder_history="",
):
    cmd = (
        f"{sys.executable} -m python.tools.profile_train --game {game} "
        f"--replay-size {replay_size} --training-iterations {training_iterations} "
        f"--batch-size {batch_size} --minibatch-size {minibatch_size} "
        f"--network-arch {network_arch}"
    )
    if chess_encoder_history != "":
        cmd += f" --chess-encoder-history {chess_encoder_history}"
    return cmd


def task_profile_train_cachegrind():
    """Run a profiler on AlphaZeroTrainer.train() (no self-play) using cachegrind."""

    def run_cachegrind(
        game,
        replay_size,
        training_iterations,
        batch_size,
        minibatch_size,
        network_arch,
        chess_encoder_history,
    ):
        cmd = (
            f"valgrind --tool=cachegrind --cachegrind-out-file={OUT_DIR}/cachegrind_train.out "
            + _profile_train_cmd(
                game,
                replay_size,
                training_iterations,
                batch_size,
                minibatch_size,
                network_arch,
                chess_encoder_history,
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

    def run_perf(
        game,
        replay_size,
        training_iterations,
        batch_size,
        minibatch_size,
        network_arch,
        chess_encoder_history,
    ):
        cmd = (
            f"perf record -e {_perf_cycles_event()} --call-graph dwarf "
            f"-o {OUT_DIR}/perf_train.data "
            + _profile_train_cmd(
                game,
                replay_size,
                training_iterations,
                batch_size,
                minibatch_size,
                network_arch,
                chess_encoder_history,
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

    def run_kineto(
        game,
        replay_size,
        training_iterations,
        batch_size,
        minibatch_size,
        network_arch,
        chess_encoder_history,
    ):
        out_file = OUT_DIR / "pytorch_train_profile.json"
        cmd = (
            _profile_train_cmd(
                game,
                replay_size,
                training_iterations,
                batch_size,
                minibatch_size,
                network_arch,
                chess_encoder_history,
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
        game,
        replay_size,
        training_iterations,
        batch_size,
        minibatch_size,
        network_arch,
        chess_encoder_history,
    ):
        out_file = OUT_DIR / "train_cprofile.prof"
        cmd = (
            _profile_train_cmd(
                game,
                replay_size,
                training_iterations,
                batch_size,
                minibatch_size,
                network_arch,
                chess_encoder_history,
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

    def run_nsys(
        game,
        replay_size,
        training_iterations,
        batch_size,
        minibatch_size,
        network_arch,
        chess_encoder_history,
    ):
        cmd = (
            "nsys profile --cudabacktrace=memory:1000000 --force-overwrite=true "
            f"-o {OUT_DIR}/nsys_train_capture "
            + _profile_train_cmd(
                game,
                replay_size,
                training_iterations,
                batch_size,
                minibatch_size,
                network_arch,
                chess_encoder_history,
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
