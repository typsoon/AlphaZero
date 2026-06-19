DOIT_CONFIG = {
    "verbosity": 2,
}


def with_report(cmd):
    """Wraps a shell command with colored PASS/FAIL output."""
    return f"{cmd} && printf '\\033[32mPASS\\033[0m\\n' || {{ printf '\\033[31mFAIL\\033[0m\\n'; exit 1; }}"


CPP_FILES_CMD = "find . -type d \\( -name build -o -name pybind11 -o -name node_modules -o -name _deps -o -name .git \\) -prune -o -type f \\( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \\) -print"


def task_check_python_format():
    """Check Python formatting using Ruff."""
    return {"actions": [with_report("ruff format --check python gameplay")]}


def task_check_python_lint():
    """Lint Python code using Ruff."""
    return {"actions": [with_report("ruff check python gameplay")]}


def task_check_cpp_format():
    """Check C++ formatting using clang-format."""
    return {
        "actions": [
            with_report(f"{CPP_FILES_CMD} | xargs -r clang-format --dry-run -Werror")
        ]
    }


def task_format_python():
    """Format Python code using Ruff."""
    return {"actions": ["ruff format python gameplay"]}


def task_fix_python_lint():
    """Fix Python linting errors using Ruff."""
    return {"actions": ["ruff check --fix python gameplay"]}


def task_format_cpp():
    """Format C++ code using clang-format."""
    return {"actions": [f"{CPP_FILES_CMD} | xargs -r clang-format -i"]}


def task_check_all():
    """Run all CI checks."""
    return {
        "actions": [],
        "task_dep": ["check_python_format", "check_python_lint", "check_cpp_format"],
    }
