---
name: enforce-doit-check-all
description: Enforces running `doit check_all` after making code changes to ensure all formatting, linting, and tests pass.
---

# Enforce doit check_all

This skill ensures that the codebase remains in a healthy state by mandating the execution of the project's comprehensive test and linting suite.

## Guidelines

1. **Trigger Condition**: Whenever you make changes to the codebase (Python, C++, TypeScript, CMake, etc.), you MUST run `doit check_all` before declaring your task complete.
2. **Execution**: Use your `run_command` tool to execute `doit check_all` in the root directory of the repository (`/home/piotrek/Playground/AlphaZeroCurrentFeature`).
3. **Handling Failures**:
   - If `doit check_all` fails, you must investigate the error output.
   - Fix any formatting issues, linting errors, or test failures before retrying.
   - Do not stop until `doit check_all` passes successfully.
