import shutil

DOIT_CONFIG = {
    "verbosity": 2,
}

def with_report(cmd):
    """Wraps a shell command with colored PASS/FAIL output."""
    return f"{cmd} && printf '\\033[32mPASS\\033[0m\\n' || {{ printf '\\033[31mFAIL\\033[0m\\n'; exit 1; }}"

def get_clang_format_cmd(args):
    """Returns the correct clang-format command using fd (if available) or falling back to find."""
    excludes = ["build", "pybind11", "node_modules", "_deps", "libtorch", ".git"]
    
    fd_bin = shutil.which("fd") or shutil.which("fdfind")
    if fd_bin:
        excludes_str = " ".join([f"-E {e}" for e in excludes])
        return f"{fd_bin} -e cpp -e h -e hpp {excludes_str} -X clang-format {args}"
    else:
        excludes_find = " -o ".join([f"-name {e}" for e in excludes])
        find_cmd = f"find . -type d \\( {excludes_find} \\) -prune -o -type f \\( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \\) -print"
        return f"{find_cmd} | xargs -r clang-format {args}"

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

def task_test_cpp():
    """Run C++ tests."""
    return {"actions": [with_report("build/inference_server/tests/inference_server_tests")]}

def task_test_python():
    """Run Python integration tests."""
    return {"actions": [with_report("pytest python/test_integration.py")]}

def task_check_all():
    """Run all CI checks."""
    return {
        "actions": [],
        "task_dep": ["check_python_format", "check_python_lint", "check_cpp_format", "test_cpp", "test_python"],
    }

def task_setup_service():
    """Install the inference server as a systemd user service."""
    return {
        "actions": [
            "mkdir -p ~/.config/systemd/user ~/.config/alphazero",
            "cp inference_server/alphazero-inference.service ~/.config/systemd/user/",
            "test -f ~/.config/alphazero/inference.env || cp inference_server/inference.env.example ~/.config/alphazero/inference.env",
            "systemctl --user daemon-reload",
        ]
    }

def task_enable_service():
    """Enable the inference server systemd service to start on boot."""
    return {
        "actions": [
            "systemctl --user enable alphazero-inference.service",
        ]
    }

def task_disable_service():
    """Disable the inference server systemd service."""
    return {
        "actions": [
            "systemctl --user disable alphazero-inference.service",
        ]
    }

def task_start_service():
    """Start the inference server systemd service immediately."""
    return {
        "actions": [
            "systemctl --user start alphazero-inference.service",
        ]
    }
