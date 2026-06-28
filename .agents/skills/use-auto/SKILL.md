---
name: Use Auto
description: Enforces the use of `auto` for variable declarations in C++ to avoid implicit conversions and duplicating type names.
---

# Use Auto

When writing C++ code, prefer using the `auto` keyword for variable declarations where the type is obvious or when initializing from a cast or array subscript, in order to avoid unintended implicit conversions (e.g., `signed char` to `int`) and to keep the code clean.

## Examples
- Use `auto p = current_board[i][j];` instead of `int p = current_board[i][j];`
- Use `auto r1 = static_cast<int8_t>(i);` instead of `int8_t r1 = static_cast<int8_t>(i);`
