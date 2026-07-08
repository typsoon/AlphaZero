import shutil
import sys
from python.utils import (
    PROJ_ROOT,
    BUILD_DIR,
    SUPPORTED_GAMES,
    NETWORK_PARAM,
    resolve_network_path,
    run_protected,
)
from doit_systemd import *  # noqa: F403
from dodo_profile import *  # noqa: F403

DOIT_CONFIG = {
    "verbosity": 2,
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

        cmake_cmd = (
            f"cmake -S {PROJ_ROOT} -B {BUILD_DIR} "
            f"-DCMAKE_BUILD_TYPE={build_type} "
            "-DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') "
            "-DPython3_EXECUTABLE=$(which python) "
            "-DPYTHON_EXECUTABLE=$(which python) "
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
    return {
        "actions": [
            with_report(f"{test_bin}"),
            with_report(f"{test_chess}"),
            with_report(f"{test_mcts}"),
            with_report(f"{test_replay_buffer}"),
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

    def run_evaluator(network_path, mcts_search_depth, game):
        network_path = resolve_network_path(network_path, game)
        cmd = (
            f"{sys.executable} -m performance_evaluation.evaluator "
            f"--network-path {network_path} --inference-binary {inference_bin} --game {game}"
        )
        if mcts_search_depth is not None:
            cmd += f" --mcts-search-depth {mcts_search_depth}"
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
        ],
        "task_dep": ["build"],
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
    inference_bin = BUILD_DIR / "inference_server" / "inference_server"

    def run_server(network_path, game):
        network_path = resolve_network_path(network_path, game)
        cmd = f"{inference_bin} --network-path {network_path} --game {game}"
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

    def run_generate(max_first_dim):
        cmd = f"{sys.executable} -m python.tools.generate_tensorrt_models"
        # Only forward an explicit override - leaving it unset lets
        # AlphaZeroNetwork.tensorrt_and_save_network use its own default.
        if max_first_dim is not None:
            cmd += f" --max_first_dim {max_first_dim}"
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
            }
        ],
        "verbosity": 2,
    }
