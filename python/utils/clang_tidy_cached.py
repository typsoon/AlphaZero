import os
import sys
import glob
import subprocess
import json
import concurrent.futures
from pathlib import Path


def main():
    proj_root = Path(sys.argv[1]).resolve()
    build_dir = Path(sys.argv[2]).resolve()

    # 1. Gather all files to check
    excludes = [
        "build",
        "pybind11",
        "node_modules",
        "_deps",
        "libtorch",
        ".git",
        "vcpkg",
        "vcpkg_installed",
        "AlphaZero",
    ]

    cpp_files = []
    for ext in ["*.cpp", "*.hpp", "*.h"]:
        for f in proj_root.rglob(ext):
            parts = f.relative_to(proj_root).parts
            if any(exc in parts for exc in excludes):
                continue
            cpp_files.append(f)

    # 2. Load Cache
    cache_file = build_dir / ".clang_tidy_cache.json"
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception:
            pass

    # 3. Find files that need linting
    to_lint = []
    for f in cpp_files:
        mtime = f.stat().st_mtime
        f_str = str(f)
        if cache.get(f_str) != mtime:
            to_lint.append(f)

    if not to_lint:
        print("All files are up to date! clang-tidy passed.")
        return 0

    print(f"Linting {len(to_lint)} out of {len(cpp_files)} files...")

    # 4. Find omp include path
    env_root = os.path.dirname(os.path.dirname(sys.executable))
    omp_paths = glob.glob(f"{env_root}/lib/gcc/*/*/include")
    extra_arg = f"--extra-arg=-isystem{omp_paths[0]}" if omp_paths else ""

    # 5. Run clang-tidy in parallel
    failed = False
    new_cache = cache.copy()

    def run_tidy(f):
        cmd = ["clang-tidy", "-p", str(build_dir)]
        if extra_arg:
            cmd.append(extra_arg)
        cmd.append(str(f))

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return f, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_tidy, f) for f in to_lint]
        for future in concurrent.futures.as_completed(futures):
            f, result = future.result()
            if result.returncode == 0:
                new_cache[str(f)] = f.stat().st_mtime
            else:
                failed = True
                print(f"FAIL: {f.relative_to(proj_root)}")
                print(result.stdout)

    # 6. Save cache
    with open(cache_file, "w") as out:
        json.dump(new_cache, out)

    if failed:
        print("clang-tidy failed on one or more files.")
        return 1

    print("clang-tidy passed on all changed files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
