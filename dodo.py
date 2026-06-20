import shutil
from doit_systemd import *
DOIT_CONFIG = {
    "verbosity": 2,
}


def with_report(cmd):
    """Wraps a shell command with colored PASS/FAIL output."""
    return f"{cmd} && printf '\\033[32mPASS\\033[0m\\n' || {{ printf '\\033[31mFAIL\\033[0m\\n'; exit 1; }}"


def get_clang_format_cmd(args):
    """Returns the correct clang-format command using fd (if available) or falling back to find."""
    excludes = [
        "build",
        "pybind11",
        "node_modules",
        "_deps",
        "libtorch",
        ".git",
        "vcpkg",
    ]

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
    excludes = [
        "build",
        "pybind11",
        "node_modules",
        "_deps",
        "libtorch",
        ".git",
        "vcpkg",
    ]

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


def task_check_python_format():
    """Check Python formatting using Ruff."""
    return {"actions": [with_report("ruff format --check python gameplay")]}


def task_check_python_lint():
    """Lint Python code using Ruff."""
    return {"actions": [with_report("ruff check python gameplay")]}


def task_check_cpp_format():
    """Check C++ formatting using clang-format."""
    return {"actions": [with_report(get_clang_format_cmd("--dry-run -Werror"))]}


def task_format_python():
    """Format Python code using Ruff."""
    return {"actions": ["ruff format python gameplay"]}


def task_fix_python_lint():
    """Fix Python linting errors using Ruff."""
    return {"actions": ["ruff check --fix python gameplay"]}


def task_format_cpp():
    """Format C++ code using clang-format."""
    return {"actions": [get_clang_format_cmd("-i")]}

def task_setup_vcpkg():
    """Clone and bootstrap vcpkg locally, then install C++ dependencies."""
    return {
        "actions": [
            "test -d vcpkg || git clone https://github.com/microsoft/vcpkg.git",
            "test -f vcpkg/vcpkg || ./vcpkg/bootstrap-vcpkg.sh",
            "./vcpkg/vcpkg install spdlog nlohmann-json json-schema-validator asio crow cpputest"
        ]
    }


def task_build():
    """Build the C++ components locally, optionally using ccache if available."""
    cmake_cmd = (
        "cmake -S . -B build "
        "-DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') "
        "-DPython3_EXECUTABLE=$(which python) "
        "-DPYTHON_EXECUTABLE=$(which python) "
        "-DBUILD_TESTS=ON "
        "-DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake"
    )

    # Automatically use ccache to speed up compilation if the user has it installed
    if shutil.which("ccache"):
        cmake_cmd += (
            " -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache"
        )

    return {
        "actions": [
            f"test -f build/CMakeCache.txt || {cmake_cmd}",
            "cmake --build build -j$(nproc)",
            "test -f build/compile_commands.json && ln -sf build/compile_commands.json . || true"
        ],
        "task_dep": ["setup_vcpkg"]
    }


def task_test_python():
    """Run Python integration tests."""
    return {"actions": [with_report("pytest python/test_integration.py")]}


def task_test_cpp():
    """Run C++ tests."""
    return {
        "actions": [with_report("build/inference_server/tests/inference_server_tests")],
        "task_dep": ["build"],
    }


def task_test_ts():
    """Run TypeScript server tests."""
    return {
        "actions": [with_report("cd gameplay_server && npm install && npm run test")]
    }


def task_check_all():
    """Run all CI checks."""
    return {
        "actions": [],
        "task_dep": [
            "check_python_format",
            "check_python_lint",
            "check_cpp_format",
            "check_cmake_format",
            "check_cmake_lint",
            "test_cpp",
            "test_python",
            "test_ts",
        ],
    }
