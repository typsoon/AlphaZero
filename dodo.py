import shutil
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


def task_check_cmake_format():
    """Check CMake formatting using cmake-format."""
    return {"actions": [with_report(get_cmake_files_cmd("cmake-format", "--check"))]}


def task_check_cmake_lint():
    """Lint CMake code using cmake-lint."""
    return {"actions": [with_report(get_cmake_files_cmd("cmake-lint", ""))]}


def task_format_cmake():
    """Format CMake code using cmake-format."""
    return {"actions": [get_cmake_files_cmd("cmake-format", "-i")]}


def get_ruff_cmd(cmd_prefix):
    """Returns the correct ruff command with exclusions."""
    excludes = GLOBAL_EXCLUDES
    excludes_str = " ".join([f"--exclude {e}" for e in excludes])
    return f"{cmd_prefix} . {excludes_str}"


def task_check_python_format():
    """Check Python formatting using Ruff."""
    return {"actions": [with_report(get_ruff_cmd("ruff format --check"))]}


def task_check_python_lint():
    """Lint Python code using Ruff."""
    return {"actions": [with_report(get_ruff_cmd("ruff check"))]}


def task_check_cpp_format():
    """Check C++ formatting using clang-format."""
    return {"actions": [with_report(get_clang_format_cmd("--dry-run -Werror"))]}


def get_clang_tidy_cmd(args):
    """Returns the correct clang-tidy command using fd (if available) or falling back to find."""
    excludes = GLOBAL_EXCLUDES

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


def task_check_cpp_lint():
    """Lint C++ code using clang-tidy."""
    return {"actions": [with_report(get_clang_tidy_cmd(""))], "task_dep": ["build"]}


def task_format_python():
    """Format Python code using Ruff."""
    return {"actions": [get_ruff_cmd("ruff format")]}


def task_fix_python_lint():
    """Fix Python lint errors using Ruff."""
    return {"actions": [get_ruff_cmd("ruff check --fix")]}


def task_format_cpp():
    """Format C++ code using clang-format."""
    return {"actions": [get_clang_format_cmd("-i")]}


def task_setup_vcpkg():
    """Clone and bootstrap vcpkg locally, then install C++ dependencies."""
    vcpkg_dir = PROJ_ROOT / "vcpkg"
    vcpkg_bin = vcpkg_dir / "vcpkg"
    vcpkg_bootstrap = vcpkg_dir / "bootstrap-vcpkg.sh"
    return {
        "actions": [
            f"test -d {vcpkg_dir} || git clone https://github.com/microsoft/vcpkg.git {vcpkg_dir}",
            f"test -f {vcpkg_bin} || {vcpkg_bootstrap}",
            f"{vcpkg_bin} install",
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


def task_check_ts_format():
    """Check TypeScript formatting."""
    client_dir = PROJ_ROOT / "gameplay_client"
    server_dir = PROJ_ROOT / "gameplay_server"
    return {
        "actions": [
            with_report(f"cd {client_dir} && npm install && npm run format:check"),
            with_report(f"cd {server_dir} && npm install && npm run format:check"),
        ]
    }


def task_check_ts_lint():
    """Lint TypeScript code."""
    client_dir = PROJ_ROOT / "gameplay_client"
    server_dir = PROJ_ROOT / "gameplay_server"
    return {
        "actions": [
            with_report(f"cd {client_dir} && npm install && npm run lint"),
            with_report(f"cd {server_dir} && npm install && npm run lint"),
        ]
    }


def task_format_ts():
    """Format TypeScript code."""
    client_dir = PROJ_ROOT / "gameplay_client"
    server_dir = PROJ_ROOT / "gameplay_server"
    return {
        "actions": [
            f"cd {client_dir} && npm install && npm run format",
            f"cd {server_dir} && npm install && npm run format",
        ]
    }


def task_check_all():
    """Run all CI checks."""
    return {
        "actions": [],
        "task_dep": [
            "check_python_format",
            "check_python_lint",
            "check_cpp_format",
            # "check_cpp_lint", # Removed because it takes too long to run on every check_all
            "check_cmake_format",
            "check_cmake_lint",
            "check_format_connect4_puzzles",
            "check_ts_format",
            "check_ts_lint",
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
        "actions": [with_report("python -m performance_evaluation.validate_puzzles")]
    }


def task_format_connect4_puzzles():
    """Format the Connect4 puzzle JSON files."""
    return {"actions": ["python -m performance_evaluation.format_puzzles"]}


def task_check_format_connect4_puzzles():
    """Check if the Connect4 puzzle JSON files are correctly formatted."""
    return {
        "actions": [
            with_report("python -m performance_evaluation.check_format_puzzles")
        ]
    }


def task_test_performance():
    """Run performance evaluation and generate the HTML report."""
    default_network_path = (
        PROJ_ROOT / "checkpoints" / "scripted" / "AZNetwork_0.pt_scripted"
    )
    inference_bin = BUILD_DIR / "inference_server" / "inference_server"
    return {
        "actions": [
            with_report(
                f"python -m performance_evaluation.evaluator --network-path {default_network_path} --inference-binary {inference_bin}"
            ),
            with_report("python -m performance_evaluation.generate_report"),
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
    default_network_path = (
        PROJ_ROOT / "checkpoints" / "scripted" / "AZNetwork_0.pt_scripted"
    )
    inference_bin = BUILD_DIR / "inference_server" / "inference_server"
    return {
        "actions": [
            with_report(
                f"python -m performance_evaluation.benchmark_batch_size --network-path {default_network_path} --inference-binary {inference_bin}"
            )
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
