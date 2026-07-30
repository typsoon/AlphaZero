---
name: no-hardcoded-paths
description: Prevents hardcoding machine-specific absolute paths (SDK dirs, venv locations, compiler binaries, usernames) into version-controlled scripts; use .env instead.
---

# No Hardcoded Paths

This repo is checked out and run on multiple machines (this Arch dev machine, a
remote training machine, CI, etc.) with different filesystem layouts. A path
that's correct on one machine (`/opt/cuda/bin/nvcc`, `/mnt/storage/users/<user>/...`)
is silently wrong on another - see the TensorRT SDK path and CUDA compiler
path in `dodo.py`, and the venv paths in `hist_run/`, all of which broke this
way and were fixed by moving to `.env`.

## Guidelines

**DO NOT** write a literal absolute path into a committed file (`dodo.py`,
`*.sh`, `CMakeLists.txt`, source code, etc.) when that path:
- points outside the repo itself (e.g. `/mnt/...`, `/home/<specific-user>/...`, `/opt/...`),
- names a username, hostname, or other machine identity,
- is an SDK/toolchain location that varies by OS or install method (CUDA,
  TensorRT, a venv's `site-packages`, a compiler binary).

A path is fine to hardcode when it's relative to `PROJ_ROOT`/`$REPO` (repo-internal,
same on every checkout) or is a location guaranteed by a standard, portable tool
(e.g. `sys.executable`, `shutil.which(...)`).

## How to fix it

1. Read the value from an environment variable, with the *current* hardcoded
   value kept as the fallback default - this way behavior is unchanged for
   whoever already relies on it, and only machines that opt in via `.env` see
   different behavior.
2. Document the variable in `.env.example` (committed) with a comment
   explaining what it's for and what a plausible value looks like. Never
   write real machine-specific values into `.env.example` itself.
3. `.env` (the real, machine-local file) is already gitignored - never commit it.
4. In Python (`dodo.py`, `python/`), call `load_dotenv()` from `python/utils`
   (reads `.env` at repo root via `os.environ.setdefault`, so real env vars
   still take precedence) before reading the variable with `os.environ.get`.
5. In bash scripts, source `.env` directly (it's valid `KEY=VALUE` shell
   syntax) guarded with `set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a`,
   then reference values as `"${VAR:-default}"`.

### Example (Python / dodo.py)

**Bad:**
```python
cmake_cmd += "-DALPHAZERO_TENSORRT_INCLUDE_DIR=/mnt/storage/users/z1201659/tensorrt_sdk/include "
```

**Good:**
```python
from python.utils import load_dotenv
load_dotenv()
TENSORRT_INCLUDE_DIR = os.environ.get(
    "ALPHAZERO_TENSORRT_INCLUDE_DIR",
    "/mnt/storage/users/z1201659/tensorrt_sdk/TensorRT-11.1.0.106/include",  # prior default, preserved
)
...
cmake_cmd += f"-DALPHAZERO_TENSORRT_INCLUDE_DIR={TENSORRT_INCLUDE_DIR} "
```

### Example (bash)

**Bad:**
```bash
source /mnt/storage/users/z1201659/.14_ml_venv/bin/activate
```

**Good:**
```bash
set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
VENV_ACTIVATE="${ALPHAZERO_HIST_VENV_ACTIVATE:-/mnt/storage/users/z1201659/.14_ml_venv/bin/activate}"
source "$VENV_ACTIVATE"
```

## When you find an existing hardcoded path

If you're touching a file and notice a pre-existing hardcoded machine-specific
path unrelated to your current task, proactively flag it to the user rather
than silently leaving it (or silently changing it) - it's an easy fix but
worth a one-line heads-up before you spend time on a broader refactor.
