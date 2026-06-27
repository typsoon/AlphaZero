import shutil
import sys
from python.utils import PROJ_ROOT, BUILD_DIR
from doit_systemd import *  # noqa: F403

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

NETWORK_PARAM = {
    "name": "network_path",
    "long": "network_path",
    "type": str,
    "default": str(PROJ_ROOT / "checkpoints" / "scripted" / "AZNetwork_0.pt_scripted"),
    "help": "Path to the scripted network file",
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
        return (
            f"{fd_bin} -e cpp -e h -e hpp {excludes_str} -x clang-tidy -p build {args}"
        )
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
        "actions": [with_report(get_clang_tidy_cmd(""))],
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

    cmake_cmd = (
        f"cmake -S {PROJ_ROOT} -B {BUILD_DIR} "
        "-DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') "
        "-DPython3_EXECUTABLE=$(which python) "
        "-DPYTHON_EXECUTABLE=$(which python) "
        "-DBUILD_TESTS=ON "
        "-DVCPKG_MANIFEST_FEATURES=test "
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    )

    # Automatically use ccache to speed up compilation if the user has it installed
    if shutil.which("ccache"):
        cmake_cmd += (
            " -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache"
        )

    return {
        "actions": [
            f"test -f {cmake_cache} || {cmake_cmd}",
            f"cmake --build {BUILD_DIR} -j$(nproc)",
            f"test -f {compile_commands} && ln -sf {compile_commands} {PROJ_ROOT} || true",
        ],
        "task_dep": ["setup_vcpkg"],
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
    return {
        "actions": [with_report(f"{test_bin}")],
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
            # "lint:cpp", # Removed because it takes too long to run on every check_all
            "validate_connect4_puzzles",
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


def task_test_performance():
    """Run performance evaluation and generate the HTML report."""
    inference_bin = BUILD_DIR / "inference_server" / "inference_server"

    def run_evaluator(network_path, mcts_search_depth):
        import subprocess

        cmd = f"{sys.executable} -m performance_evaluation.evaluator --network-path {network_path} --inference-binary {inference_bin}"
        if mcts_search_depth is not None:
            cmd += f" --mcts-search-depth {mcts_search_depth}"

        full_cmd = with_report(cmd)
        result = subprocess.run(full_cmd, shell=True)
        return result.returncode == 0

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
    return {
        "actions": [
            with_report(
                f"{sys.executable} -m performance_evaluation.benchmark_batch_size --network-path %(network_path)s --inference-binary {inference_bin}"
            )
        ],
        "params": [NETWORK_PARAM],
        "task_dep": ["build"],
    }


def task_clear_db():
    """Clear the games database by removing the SQLite files."""
    return {
        "actions": [
            "rm -f gameplay_server/games.db gameplay_server/games.db-wal gameplay_server/games.db-shm"
        ]
    }


def task_profile_self_play_cachegrind():
    """Run a profiler on the self_play execution using cachegrind."""
    profiling_bin = BUILD_DIR / "engine" / "profiling" / "run_self_play"
    return {
        "actions": [
            f"cmake --build {BUILD_DIR} --target run_self_play -j$(nproc)",
            f"valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out {profiling_bin} %(network_path)s %(num_games)s %(thread_count)s",
            "echo '\\033[32mRun `kcachegrind cachegrind.out` to view the profiling results.\\033[0m'",
        ],
        "params": [
            NETWORK_PARAM,
            {
                "name": "num_games",
                "long": "num_games",
                "type": int,
                "default": 2,
                "help": "Number of games to profile",
            },
            {
                "name": "thread_count",
                "long": "thread_count",
                "type": int,
                "default": 1,
                "help": "Number of threads to use",
            },
        ],
        "task_dep": ["setup_vcpkg"],
    }


def task_profile_self_play_perf():
    """Run a multithreaded profiler on the self_play execution using Linux perf."""
    profiling_bin = BUILD_DIR / "engine" / "profiling" / "run_self_play"
    return {
        "actions": [
            f"cmake --build {BUILD_DIR} --target run_self_play -j$(nproc)",
            f"perf record -g -o perf.data {profiling_bin} %(network_path)s %(num_games)s %(thread_count)s",
            "echo '\\033[32mRun `perf report -g` to view the multithreaded profiling results.\\033[0m'",
        ],
        "params": [
            NETWORK_PARAM,
            {
                "name": "num_games",
                "long": "num_games",
                "type": int,
                "default": 10,
                "help": "Number of games to profile",
            },
            {
                "name": "thread_count",
                "long": "thread_count",
                "type": int,
                "default": 4,
                "help": "Number of threads to use (defaults to 4 for multithreaded profiling)",
            },
        ],
        "task_dep": ["setup_vcpkg"],
    }
