---
name: enforce-switch-statements
description: Enforces the use of switch/match statements instead of long if-elif chains.
---

# Enforce Switch Statements

To maintain clean and readable code, you MUST adhere to the following rule across the entire codebase:

When checking a single variable against 3 or more discrete values or conditions, **always** use a `switch` statement (in C++, TypeScript, or JavaScript) or a `match` statement (in Python 3.10+) rather than chaining long `if`, `else if`, and `else` blocks.

## C++ / TypeScript / JavaScript
Instead of:
```cpp
if (val == 1) { ... }
else if (val == 2) { ... }
else if (val == 3) { ... }
else { ... }
```
Use:
```cpp
switch (val) {
  case 1: ... break;
  case 2: ... break;
  case 3: ... break;
  default: ... break;
}
```

## Python
Instead of:
```python
if val == 1:
    ...
elif val == 2:
    ...
elif val == 3:
    ...
else:
    ...
```
Use Structural Pattern Matching (Python 3.10+):
```python
match val:
    case 1:
        ...
    case 2:
        ...
    case 3:
        ...
    case _:
        ...
```

If you encounter long `if`/`elif` chains checking the same variable across the codebase, proactively refactor them into `switch` or `match` statements.
