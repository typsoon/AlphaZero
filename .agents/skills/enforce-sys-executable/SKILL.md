---
name: Enforce sys.executable
description: Ensures that Python subprocesses are invoked using sys.executable instead of the generic 'python' command to avoid picking up the wrong Python interpreter.
---

# Enforce sys.executable

When writing Python scripts, `dodo.py` tasks, or any build system code that needs to spawn another Python process (e.g., using `subprocess` or a shell command string), NEVER use the raw `python` command.

### Why?
Using `python` relies on the environment's `$PATH`, which can lead to unpredictable behavior if the user is running the code inside a virtual environment, a conda environment, or a systemd service. 

### How to Fix
Always use `sys.executable` to guarantee that the subprocess uses the exact same Python interpreter as the parent process.

**Bad:**
```python
"actions": ["python -m pytest"]
```

**Good:**
```python
import sys
"actions": [f"{sys.executable} -m pytest"]
```

**Bad:**
```python
import subprocess
subprocess.run(["python", "script.py"])
```

**Good:**
```python
import subprocess
import sys
subprocess.run([sys.executable, "script.py"])
```
