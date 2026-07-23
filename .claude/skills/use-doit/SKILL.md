---
name: Use Doit
description: Enforces the use of `doit` for building, testing, linting, and formatting within the repository instead of invoking direct build commands like cmake or npm directly.
---

# Use Doit

When working in this repository, you must prioritize using the `doit` task runner for all development workflows instead of raw commands (e.g., `cmake --build`, `make`, `pytest`, `npm test`).

## Guidelines
1. **Discover Tasks**: Use `doit list` to see all available tasks.
2. **Build and Test**: Use tasks like `doit build`, `doit test_cpp`, `doit test_python`, or `doit test_ts`.
3. **Linting and Formatting**: Use `doit check_all`, `doit format`, or specific tasks like `doit lint`.
4. **Environment**: Ensure the proper conda environment is activated before running `doit` tasks: `source ~/miniconda3/bin/activate && conda activate mpum-big-project && doit <task>`.
