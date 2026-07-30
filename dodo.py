import os
import shutil
import sys
from python.utils import (
    PROJ_ROOT,
    BUILD_DIR,
    SUPPORTED_GAMES,
    NETWORK_PARAM,
    load_dotenv,
    resolve_network_path,
    run_protected,
)
from doit_systemd import *  # noqa: F403
from dodo_profile import *  # noqa: F403

load_dotenv()

# Machine-local paths, overridable via .env (see .env.example) or real env
# vars - defaults below match the original z1201659 training machine's setup,
# so behavior is unchanged there unless it also opts into a .env file.
CUDA_COMPILER = os.environ.get("ALPHAZERO_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc")
TENSORRT_INCLUDE_DIR = os.environ.get(
    "ALPHAZERO_TENSORRT_INCLUDE_DIR",
    "/mnt/storage/users/z1201659/tensorrt_sdk/TensorRT-11.1.0.106/include",
)

DOIT_CONFIG = {
    "verbosity": 2,
    # The venv now runs on conda's python3.12 (2026-07-26 OS upgrade removed the
    # system 3.12), which lacks the `_gdbm` extension doit's default `dbm` backend
    # needs. sqlite3 (stdlib) is always available. See venv-py314-breakage notes.
    "backend": "sqlite3",
}

GLOBAL_EXCLUDES = [
    "build",
    "pybind11",
    "node_modules",
    "_deps",
    "libtorch",
    ".git",
    "vcpkg",
    "vcpkg_installed",
]


def task_patch_torchtrt():
    """Patch libtorchtrt.so's RUNPATH so C++ binaries can load TensorRT without LD_LIBRARY_PATH.

    torch-tensorrt is built with Bazel and ships with broken $ORIGIN-relative RUNPATH entries
    that don't resolve after pip install. Since it uses DT_RUNPATH (not DT_RPATH), the parent
    executable's RPATH cannot compensate. This task appends the real library directories to
    libtorchtrt.so's own RUNPATH using patchelf, making it work from any context.
    """
    import subprocess as _sp
    import sys as _sys

    def do_patch():
        try:
            result = _sp.run(
                [
                    _sys.executable,
                    "-c",
                    "import torch_tensorrt, nvidia, os, sys; "
                    "trt = os.path.join(os.path.dirname(torch_tensorrt.__file__), 'lib', 'libtorchtrt.so'); "
                    "trt_libs = os.path.join(os.path.dirname(os.path.dirname(torch_tensorrt.__file__)), 'tensorrt_libs'); "
                    "cuda_lib = os.path.join(nvidia.__path__[0], 'cuda_runtime', 'lib'); "
                    "torch_lib = os.path.realpath(os.path.join(os.path.dirname(torch_tensorrt.__file__), '..', 'torch', 'lib')); "
                    "print(trt + ':' + trt_libs + ':' + cuda_lib + ':' + torch_lib)",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("torch_tensorrt not found, skipping patch")
                return True
            # Take the last colon-bearing line (ignore TRT logger noise on stdout)
            line = [
                line
                for line in result.stdout.strip().splitlines()
                if line.count(":") >= 3
            ][-1]
            libtorchtrt, trt_libs, cuda_lib, torch_lib = line.split(":", 3)
        except Exception as e:
            print(f"Could not locate torch_tensorrt libraries: {e}")
            return True  # soft failure — TRT simply not installed

        rpath_to_add = f"{trt_libs}:{cuda_lib}:{torch_lib}"

        # Idempotency: skip if already patched
        check = _sp.run(
            ["patchelf", "--print-rpath", libtorchtrt], capture_output=True, text=True
        )
        if trt_libs in check.stdout:
            print("libtorchtrt.so already patched, skipping")
            return True

        print(f"Patching {libtorchtrt}")
        print(f"  Adding RPATH: {rpath_to_add}")
        r = _sp.run(["patchelf", "--add-rpath", rpath_to_add, libtorchtrt])
        return r.returncode == 0

    return {
        "actions": [do_patch],
        "verbosity": 2,
        "uptodate": [False],  # always re-check; the action itself is idempotent
    }


def with_report(cmd):
    """Wraps a shell command with colored PASS/FAIL output."""
    return f"{cmd} && printf '\\033[32mPASS\\033[0m\\n' || {{ printf '\\033[31mFAIL\\033[0m\\n'; exit 1; }}"


def get_clang_format_cmd(args):
    """Returns the correct clang-format command using fd (if available) or falling back to find."""
    excludes = GLOBAL_EXCLUDES

    fd_bin = shutil.which("fd") or shutil.which("fdfind")
    if fd_bin:
        excludes_str = " ".join([f"-E {e}" for e in excludes])
        return f"{fd_bin} -e cpp -e h -e hpp {excludes_str} -X clang-format {args}"
    else:
        excludes_find = " -o ".join([f"-name {e}" for e in excludes])
        find_cmd = f"find . -type d \\( {excludes_find} \\) -prune -o -type f \\( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \\) -print"
        return f"{find_cmd} | xargs -r clang-format {args}"


def get_cmake_files_cmd(cmd, args):
    """Returns the correct command for cmake files using fd (if available) or falling back to find."""
    excludes = GLOBAL_EXCLUDES

    fd_bin = shutil.which("fd") or shutil.which("fdfind")
    if fd_bin:
        excludes_str = " ".join([f"-E {e}" for e in excludes])
        return (
            f"{fd_bin} 'CMakeLists\\.txt|\\.cmake$' -t f {excludes_str} -X {cmd} {args}"
        )
    else:
        excludes_find = " -o ".join([f"-name {e}" for e in excludes])
        find_cmd = f"find . -type d \\( {excludes_find} \\) -prune -o -type f \\( -name 'CMakeLists.txt' -o -name '*.cmake' \\) -print"
        return f"{find_cmd} | xargs -r {cmd} {args}"


def get_ruff_cmd(cmd_prefix):
    """Returns the correct ruff command with exclusions."""
    excludes = GLOBAL_EXCLUDES
    excludes_str = " ".join([f"--exclude {e}" for e in excludes])
    return f"{cmd_prefix} . {excludes_str}"


def get_clang_tidy_cmd(args):
    """Returns the correct clang-tidy command using fd (if available) or falling back to find."""
    excludes = GLOBAL_EXCLUDES

    import sys
    import os
    import glob

    env_root = os.path.dirname(os.path.dirname(sys.executable))
    omp_paths = glob.glob(f"{env_root}/lib/gcc/*/*/include")
    if omp_paths:
        args += f" --extra-arg=-isystem{omp_paths[0]}"

    fd_bin = shutil.which("fd") or shutil.which("fdfind")
    if fd_bin:
        excludes_str = " ".join([f"-E {e}" for e in excludes])
        return f"{fd_bin} -j 4 -e cpp -e h -e hpp {excludes_str} -x clang-tidy -p build {args}"
    else:
        excludes_find = " -o ".join([f"-name {e}" for e in excludes])
        find_cmd = f"find . -type d \\( {excludes_find} \\) -prune -o -type f \\( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \\) -print"
        return f"{find_cmd} | xargs -P $(nproc) -n 1 -r clang-tidy -p build {args}"


def task_check_format():
    """Check formatting across different languages."""
    yield {
        "name": "python",
        "actions": [with_report(get_ruff_cmd("ruff format --check"))],
    }
    yield {
        "name": "cpp",
        "actions": [with_report(get_clang_format_cmd("--dry-run -Werror"))],
    }
    yield {
        "name": "cmake",
        "actions": [with_report(get_cmake_files_cmd("cmake-format", "--check"))],
    }
    yield {
        "name": "connect4_puzzles",
        "actions": [
            with_report(
                f"{sys.executable} -m performance_evaluation.check_format_puzzles"
            )
        ],
    }
    client_dir = PROJ_ROOT / "gameplay_client"
    server_dir = PROJ_ROOT / "gameplay_server"
    yield {
        "name": "ts",
        "actions": [
            with_report(f"cd {client_dir} && npm install && npm run format:check"),
            with_report(f"cd {server_dir} && npm install && npm run format:check"),
        ],
    }


def task_lint():
    """Lint code across different languages."""
    yield {
        "name": "python",
        "actions": [with_report(get_ruff_cmd("ruff check"))],
    }
    yield {
        "name": "cpp",
        "actions": [
            with_report(
                f"{sys.executable} {PROJ_ROOT}/python/utils/clang_tidy_cached.py {PROJ_ROOT} {BUILD_DIR}"
            )
        ],
        "task_dep": ["build"],
    }
    yield {
        "name": "cmake",
        "actions": [with_report(get_cmake_files_cmd("cmake-lint", ""))],
    }
    client_dir = PROJ_ROOT / "gameplay_client"
    server_dir = PROJ_ROOT / "gameplay_server"
    yield {
        "name": "ts",
        "actions": [
            with_report(f"cd {client_dir} && npm install && npm run lint"),
            with_report(f"cd {server_dir} && npm install && npm run lint"),
        ],
    }


def task_format():
    """Format code across different languages."""
    yield {
        "name": "python",
        "actions": [get_ruff_cmd("ruff format")],
    }
    yield {
        "name": "cpp",
        "actions": [get_clang_format_cmd("-i")],
    }
    yield {
        "name": "cmake",
        "actions": [get_cmake_files_cmd("cmake-format", "-i")],
    }
    yield {
        "name": "connect4_puzzles",
        "actions": [f"{sys.executable} -m performance_evaluation.format_puzzles"],
    }
    client_dir = PROJ_ROOT / "gameplay_client"
    server_dir = PROJ_ROOT / "gameplay_server"
    yield {
        "name": "ts",
        "actions": [
            f"cd {client_dir} && npm install && npm run format",
            f"cd {server_dir} && npm install && npm run format",
        ],
    }


def task_fix_python_lint():
    """Fix Python lint errors using Ruff."""
    return {"actions": [get_ruff_cmd("ruff check --fix")]}


def task_setup_vcpkg():
    """Clone and bootstrap vcpkg locally, then install C++ dependencies."""
    vcpkg_dir = PROJ_ROOT / "vcpkg"
    vcpkg_bin = vcpkg_dir / "vcpkg"
    vcpkg_bootstrap = vcpkg_dir / "bootstrap-vcpkg.sh"
    return {
        "actions": [
            f"test -d {vcpkg_dir} || git clone https://github.com/microsoft/vcpkg.git {vcpkg_dir}",
            f"test -f {vcpkg_bin} || {vcpkg_bootstrap}",
        ]
    }


def task_build():
    """Build the C++ components locally, optionally using ccache if available."""
    cmake_cache = BUILD_DIR / "CMakeCache.txt"
    compile_commands = BUILD_DIR / "compile_commands.json"
    toolchain = PROJ_ROOT / "vcpkg" / "scripts" / "buildsystems" / "vcpkg.cmake"

    default_build_type = "debug"
    if cmake_cache.exists():
        with open(cmake_cache) as f:
            if "CMAKE_BUILD_TYPE:STRING=Release" in f.read():
                default_build_type = "release"

    def build_action(build_type):
        build_type = build_type.capitalize()

        run_cmake = True
        if cmake_cache.exists():
            with open(cmake_cache) as f:
                if f"CMAKE_BUILD_TYPE:STRING={build_type}" in f.read():
                    run_cmake = False

        import sys
        cmake_cmd = (
            f"cmake -S {PROJ_ROOT} -B {BUILD_DIR} "
            f"-DCMAKE_BUILD_TYPE={build_type} "
            f"-DCMAKE_PREFIX_PATH=$({sys.executable} -c 'import torch; print(torch.utils.cmake_prefix_path)') "
            f"-DCMAKE_CUDA_COMPILER={CUDA_COMPILER} "
            # Machine-local TensorRT SDK download (NvInfer.h - pip's tensorrt
            # package ships no C++ headers). Safe to pass unconditionally: the
            # CMake logic only enables native TRT-engine loading when this
            # header set's major version matches the active venv's pip
            # tensorrt runtime major version, so it silently no-ops when
            # building against a venv on a different TensorRT major (e.g.
            # .12_ml_venv's TRT 10.9 vs this SDK's TRT 11). Both paths are
            # overridable per-machine via .env - see .env.example.
            f"-DALPHAZERO_TENSORRT_INCLUDE_DIR={TENSORRT_INCLUDE_DIR} "
            f"-DPython3_EXECUTABLE={sys.executable} "
            f"-DPYTHON_EXECUTABLE={sys.executable} "
            "-DBUILD_TESTS=ON "
            "-DVCPKG_MANIFEST_FEATURES=test "
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
        )

        if shutil.which("ccache"):
            cmake_cmd += " -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache"

        if run_cmake:
            if not run_protected(cmake_cmd):
                return False

        if not run_protected(f"cmake --build {BUILD_DIR} -j$(nproc)"):
            return False

        run_protected(
            f"test -f {compile_commands} && ln -sf {compile_commands} {PROJ_ROOT} || true"
        )
        return True

    return {
        "actions": [build_action],
        "params": [
            {
                "name": "build_type",
                "short": "t",
                "long": "type",
                "type": str,
                "default": default_build_type,
                "choices": (("debug", "Debug mode"), ("release", "Release mode")),
                "help": "Build type (debug or release)",
            }
        ],
        "task_dep": ["setup_vcpkg", "patch_torchtrt"],
    }


def task_test_python():
    """Run Python integration tests."""
    return {"actions": [with_report(f"pytest {PROJ_ROOT / 'python' / 'test'}")]}


def task_test_train_loss():
    """Run the AlphaZero training loss convergence test."""
    return {
        "actions": [
            with_report(
                f"pytest -s {PROJ_ROOT / 'python' / 'test' / 'test_train_loss.py'}"
            )
        ]
    }


def task_test_cpp():
    """Run C++ tests."""
    test_bin = BUILD_DIR / "inference_server" / "tests" / "inference_server_tests"
    test_chess = BUILD_DIR / "engine" / "test_chess"
    test_mcts = BUILD_DIR / "engine" / "test_mcts"
    test_replay_buffer = BUILD_DIR / "engine" / "test_replay_buffer"
    test_inference_cache = BUILD_DIR / "engine" / "test_inference_cache"
    test_resignation = BUILD_DIR / "engine" / "test_resignation"
    test_inference_backend = BUILD_DIR / "engine" / "test_inference_backend"
    return {
        "actions": [
            with_report(f"{test_bin}"),
            with_report(f"{test_chess}"),
            with_report(f"{test_mcts}"),
            # CUDA_VISIBLE_DEVICES="" hides CUDA from this process before it starts -
            # see the comment above main() in test_replay_buffer.cpp for why that's
            # needed (a CUDA-driver-init/CppUTest leak-detector interaction crashes
            # the binary otherwise, unrelated to ReplayBuffer's own correctness).
            with_report(f'CUDA_VISIBLE_DEVICES="" {test_replay_buffer}'),
            with_report(f"{test_inference_cache}"),
            with_report(f"{test_resignation}"),
            # Needs real CUDA (unlike the above) to exercise TensorRT -
            # test_inference_backend.cpp's main() works around the same class
            # of CppUTest/CUDA leak-detector crash differently (thread-safe
            # tracking + per-test ignoreAllLeaksInTest, see its comments)
            # since disabling CUDA isn't an option for a TensorRT test.
            with_report(f"{test_inference_backend}"),
        ],
        "task_dep": ["build"],
    }


def task_test_ts():
    """Run TypeScript server tests."""
    ts_dir = PROJ_ROOT / "gameplay_server"
    return {"actions": [with_report(f"cd {ts_dir} && npm install && npm run test")]}


def task_check_all():
    """Run all CI checks."""
    return {
        "actions": [],
        "task_dep": [
            "check_format",
            "lint:python",
            "lint:cmake",
            "lint:ts",
            "lint:cpp",
            "validate_connect4_puzzles",
            "validate_chess_puzzles",
            "test_cpp",
            "test_python",
            "test_train_loss",
            "test_ts",
        ],
    }


def task_validate_connect4_puzzles():
    """Validate the Connect4 puzzle JSON files."""
    return {
        "actions": [
            with_report(f"{sys.executable} -m performance_evaluation.validate_puzzles")
        ]
    }


def task_validate_chess_puzzles():
    """Validate the Chess puzzle JSON files."""
    return {
        "actions": [
            with_report(
                f"{sys.executable} -m performance_evaluation.validate_chess_puzzles"
            )
        ]
    }


def task_test_performance():
    """Run performance evaluation and generate the HTML report."""
    inference_bin = BUILD_DIR / "inference_server" / "inference_server"

    def run_evaluator(network_path, mcts_search_depth, game, chess_encoder_history):
        network_path = resolve_network_path(network_path, game)
        cmd = (
            f"{sys.executable} -m performance_evaluation.evaluator "
            f"--network-path {network_path} --inference-binary {inference_bin} --game {game}"
        )
        if mcts_search_depth is not None:
            cmd += f" --mcts-search-depth {mcts_search_depth}"
        if chess_encoder_history:
            cmd += f" --chess-encoder-history {chess_encoder_history}"
        return run_protected(cmd)

    return {
        "actions": [
            run_evaluator,
            with_report(f"{sys.executable} -m performance_evaluation.generate_report"),
        ],
        "params": [
            NETWORK_PARAM,
            {
                "name": "mcts_search_depth",
                "long": "mcts_search_depth",
                "type": int,
                "default": None,
                "help": "MCTS search depth (optional)",
            },
            {
                "name": "game",
                "long": "game",
                "type": str,
                "default": "connect4",
                "help": "The game to run puzzle evaluation for",
                "choices": SUPPORTED_GAMES,
            },
            {
                "name": "chess_encoder_history",
                "long": "chess_encoder_history",
                "type": int,
                "default": 0,
                "help": "Chess only: input encoding the net expects. 0 = "
                "19-plane default; 1/4/8 = ChessEncoderV2History(N). Use 4 for "
                "the chess-v2 history bootstrap net.",
            },
        ],
        "task_dep": ["build"],
    }


def task_run_arena():
    """Play head-to-head arena matches between two checkpoints and report the score."""
    from dodo_profile import _merge_params_file

    arena_bin = BUILD_DIR / "engine" / "profiling" / "run_arena"

    def run_arena(
        game,
        network_path_a,
        network_path_b,
        num_games,
        thread_count,
        max_moves,
        mcts_num_simulations,
        mcts_batch_size,
        use_gumbel_search,
        max_num_considered_actions,
        transposition_cache_entries,
        params_file,
    ):
        p = _merge_params_file(params_file, locals())
        game = p["game"]
        network_path_a = p["network_path_a"]
        network_path_b = p["network_path_b"]
        num_games = p["num_games"]
        thread_count = p["thread_count"]
        max_moves = p["max_moves"]
        mcts_num_simulations = p["mcts_num_simulations"]
        mcts_batch_size = p["mcts_batch_size"]
        use_gumbel_search = p["use_gumbel_search"]
        max_num_considered_actions = p["max_num_considered_actions"]
        transposition_cache_entries = p["transposition_cache_entries"]

        # A defaults to the current (latest) checkpoint via the same resolution
        # test_performance uses; B defaults to the newest archive in
        # checkpoints/<game>/old_checkpoint (the 2-hourly milestone snapshots),
        # so a bare `doit run_arena --game chess` answers "did the last two
        # hours of training make the network stronger?".
        network_path_a = resolve_network_path(network_path_a, game)
        if not network_path_b:
            archive_dir = PROJ_ROOT / "checkpoints" / game / "old_checkpoint"
            candidates = sorted(archive_dir.glob("*.pt_trt")) or sorted(
                archive_dir.glob("*.pt_scripted")
            )
            if not candidates:
                raise ValueError(
                    f"network_path_b not given and no archived checkpoints found in "
                    f"{archive_dir} - pass --network_path_b or populate the archive"
                )
            network_path_b = candidates[-1]

        cmd = (
            f"{arena_bin} {game} {network_path_a} {network_path_b} {num_games} "
            f"{thread_count} {max_moves} {mcts_num_simulations} {mcts_batch_size} "
            f"{int(use_gumbel_search)} {max_num_considered_actions} "
            f"{transposition_cache_entries}"
        )
        return run_protected(cmd)

    return {
        "actions": [
            f"cmake --build {BUILD_DIR} --target run_arena -j$(nproc)",
            run_arena,
        ],
        "params": [
            {
                "name": "game",
                "long": "game",
                "type": str,
                "default": "chess",
                "help": "The game to play the arena match in",
                "choices": SUPPORTED_GAMES,
            },
            {
                "name": "network_path_a",
                "long": "network_path_a",
                "type": str,
                "default": "",
                "help": "Engine A's checkpoint (default: current latest)",
            },
            {
                "name": "network_path_b",
                "long": "network_path_b",
                "type": str,
                "default": "",
                "help": "Engine B's checkpoint (default: newest old_checkpoint archive)",
            },
            {
                "name": "num_games",
                "long": "num_games",
                "type": int,
                "default": 40,
                "help": "Number of games (colors alternate every game)",
            },
            {
                "name": "thread_count",
                "long": "thread_count",
                "type": int,
                "default": 8,
                "help": "Concurrent games (one MCTS pair + arenas per thread)",
            },
            {
                "name": "max_moves",
                "long": "max_moves",
                "type": int,
                "default": 512,
                "help": "Ply cutoff per game (cutoff games score as draws)",
            },
            {
                "name": "mcts_num_simulations",
                "long": "mcts_num_simulations",
                "type": int,
                "default": 800,
                "help": "Simulations per move for both engines",
            },
            {
                "name": "mcts_batch_size",
                "long": "mcts_batch_size",
                "type": int,
                "default": 64,
                "help": "MCTS inference batch size",
            },
            {
                "name": "use_gumbel_search",
                "long": "use_gumbel_search",
                "type": int,
                "default": 1,
                "help": "1 = Gumbel root selection (recommended for arenas), 0 = plain PUCT",
            },
            {
                "name": "max_num_considered_actions",
                "long": "max_num_considered_actions",
                "type": int,
                "default": 16,
                "help": "Gumbel-Top-k candidate set size",
            },
            {
                "name": "transposition_cache_entries",
                "long": "transposition_cache_entries",
                "type": int,
                "default": 1000000,
                "help": "Per-engine inference cache slots (0 disables)",
            },
            {
                "name": "params_file",
                "long": "params_file",
                "type": str,
                "default": "",
                "help": "JSON file with any of the above keys (CLI flags win)",
            },
        ],
        "task_dep": ["setup_vcpkg", "patch_torchtrt"],
    }


def task_run_client():
    """Run the gameplay client in dev mode."""
    client_dir = PROJ_ROOT / "gameplay_client"
    return {"actions": [f"cd {client_dir} && npm install && npm run dev"]}


def task_run_game_server():
    """Run the gameplay server in dev mode."""
    server_dir = PROJ_ROOT / "gameplay_server"
    return {"actions": [f"cd {server_dir} && npm install && npm run start"]}


def task_benchmark_batch_size():
    """Run the batch size benchmark."""
    inference_bin = BUILD_DIR / "inference_server" / "inference_server"

    def run_benchmark(network_path):
        network_path = resolve_network_path(network_path, "connect4")
        cmd = with_report(
            f"{sys.executable} -m performance_evaluation.benchmark_batch_size --network-path {network_path} --inference-binary {inference_bin}"
        )
        return run_protected(cmd)

    return {
        "actions": [run_benchmark],
        "params": [NETWORK_PARAM],
        "task_dep": ["build"],
    }


def task_run_inference_server():
    """Run the inference server directly from the terminal."""
    from dodo_profile import _merge_params_file

    inference_bin = BUILD_DIR / "inference_server" / "inference_server"
    # mcts_simulations is training_params/*.json's name for search depth;
    # everything else this task cares about (mcts_batch_size,
    # chess_encoder_history, use_gumbel_search, max_num_considered_actions,
    # full_search_probability, fast_mcts_simulations) already matches its
    # JSON key 1:1, so only this one entry needs remapping.
    _INFERENCE_JSON_FIELD_MAP = {"mcts_simulations": "mcts_search_depth"}

    def run_server(
        network_path,
        game,
        device,
        socket,
        mcts_search_depth,
        mcts_batch_size,
        chess_encoder_history,
        use_gumbel_search,
        max_num_considered_actions,
        full_search_probability,
        fast_mcts_simulations,
        params_file,
    ):
        p = _merge_params_file(params_file, locals(), _INFERENCE_JSON_FIELD_MAP)
        network_path = p["network_path"]
        game = p["game"]
        device = p["device"]
        socket = p["socket"]
        mcts_search_depth = p["mcts_search_depth"]
        mcts_batch_size = p["mcts_batch_size"]
        chess_encoder_history = p["chess_encoder_history"]
        use_gumbel_search = p["use_gumbel_search"]
        max_num_considered_actions = p["max_num_considered_actions"]
        full_search_probability = p["full_search_probability"]
        fast_mcts_simulations = p["fast_mcts_simulations"]

        network_path = resolve_network_path(network_path, game)
        cmd = (
            f"{inference_bin} --network-path {network_path} --game {game}"
            f" --device {device} --mcts-search-depth {mcts_search_depth}"
            f" --mcts-batch-size {mcts_batch_size}"
        )
        if socket:
            cmd += f" --socket {socket}"
        # chess-only; only pass when a history encoder is requested.
        if game == "chess" and chess_encoder_history:
            cmd += f" --chess-encoder-history {chess_encoder_history}"
        # The remaining flags mirror training_params/*.json's self-play search
        # config (mcts_simulations/mcts_batch_size above already do) - only
        # passed when they'd change the binary's own defaults, so the command
        # line stays uncluttered for the common case.
        if use_gumbel_search:
            cmd += " --use-gumbel-search"
        if max_num_considered_actions != 16:
            cmd += f" --max-num-considered-actions {max_num_considered_actions}"
        if full_search_probability != 1.0:
            cmd += f" --full-search-probability {full_search_probability}"
        if fast_mcts_simulations:
            cmd += f" --fast-mcts-simulations {fast_mcts_simulations}"
        return run_protected(cmd)

    return {
        "actions": [run_server],
        "params": [
            {
                "name": "game",
                "long": "game",
                "type": str,
                "default": "connect4",
                "help": "The game to run inference for",
                "choices": SUPPORTED_GAMES,
            },
            NETWORK_PARAM,
            {
                "name": "device",
                "long": "device",
                "type": str,
                "default": "cuda",
                "help": "Inference device (cuda or cpu)",
            },
            {
                "name": "socket",
                "long": "socket",
                "type": str,
                "default": "",
                "help": "Unix socket path (empty = auto-generate under /tmp)",
            },
            {
                "name": "mcts_search_depth",
                "long": "mcts-search-depth",
                "type": int,
                "default": 800,
                "help": "MCTS simulations per move",
            },
            {
                "name": "mcts_batch_size",
                "long": "mcts-batch-size",
                "type": int,
                "default": 32,
                "help": "Inference batch size",
            },
            {
                "name": "chess_encoder_history",
                "long": "chess-encoder-history",
                "type": int,
                "default": 0,
                "help": "Chess only: 0 = 19-plane v1; 1/4/8 = history encoder",
            },
            {
                "name": "use_gumbel_search",
                "long": "use-gumbel-search",
                "type": bool,
                "default": False,
                "help": "Use Gumbel-Top-k root sampling instead of plain-PUCT "
                "search (matches training_params/*.json's use_gumbel_search)",
            },
            {
                "name": "max_num_considered_actions",
                "long": "max-num-considered-actions",
                "type": int,
                "default": 16,
                "help": "Gumbel-Top-k candidate set size, only used with "
                "--use-gumbel-search",
            },
            {
                "name": "full_search_probability",
                "long": "full-search-probability",
                "type": float,
                "default": 1.0,
                "help": "Fraction of requests using the full mcts-search-depth; "
                "the rest use --fast-mcts-simulations (default 1.0 = always full, "
                "the sensible choice for serving; self-play uses lower values to "
                "make training games cheaper)",
            },
            {
                "name": "fast_mcts_simulations",
                "long": "fast-mcts-simulations",
                "type": int,
                "default": 0,
                "help": "Simulation count for the \"fast\" branch above, only "
                "used when --full-search-probability < 1.0",
            },
            {
                "name": "params_file",
                "long": "params_file",
                "type": str,
                "default": "",
                "help": "JSON file with any of: mcts_simulations (-> "
                "mcts_search_depth), mcts_batch_size, chess_encoder_history, "
                "use_gumbel_search, max_num_considered_actions, "
                "full_search_probability, fast_mcts_simulations. A full "
                "training_params/*.json works directly (other keys ignored); "
                "CLI flags win over file values.",
            },
        ],
        "task_dep": ["build"],
    }


def task_clear_db():
    """Clear the games database by removing the SQLite files."""
    return {
        "actions": [
            "rm -f gameplay_server/games.db gameplay_server/games.db-wal gameplay_server/games.db-shm"
        ]
    }


def task_generate_tensorrt_models():
    """Iterate through available checkpoints and generate TensorRT scripted models."""

    def run_generate(max_first_dim, backend, checkpoint_dir):
        cmd = f"{sys.executable} -m python.tools.generate_tensorrt_models"
        # Only forward an explicit override - leaving it unset lets
        # AlphaZeroNetwork.tensorrt_and_save_network use its own default.
        if max_first_dim is not None:
            cmd += f" --max_first_dim {max_first_dim}"
        if backend is not None:
            cmd += f" --backend {backend}"
        if checkpoint_dir is not None:
            cmd += f" --checkpoint_dir {checkpoint_dir}"
        return run_protected(cmd)

    return {
        "actions": [run_generate],
        "params": [
            {
                "name": "max_first_dim",
                "long": "max_first_dim",
                "type": int,
                "default": None,
                "help": "Max first dim of input (max TensorRT batch size). Defaults to "
                "AlphaZeroNetwork.tensorrt_and_save_network's own default if not set.",
            },
            {
                "name": "backend",
                "long": "backend",
                "type": str,
                "default": None,
                "help": "Compilation backend ('torch_tensorrt' or 'onnx'). Defaults to "
                "AlphaZeroNetwork.tensorrt_and_save_network's own default if not set.",
            },
            {
                "name": "checkpoint_dir",
                "long": "checkpoint_dir",
                "type": str,
                "default": None,
                "help": "Root directory containing per-game checkpoint subdirectories. "
                "Defaults to 'checkpoints'.",
            },
        ],
        "verbosity": 2,
    }
