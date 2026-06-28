---
name: Use rg and fd
description: Enforces the use of ripgrep (rg) over grep, and fd over find for searching and finding files in the terminal.
---

# Instructions

When interacting with the terminal to find files or search through file contents:
- **ALWAYS** prefer `rg` (ripgrep) instead of `grep` or `egrep` for text search.
- **ALWAYS** prefer `fd` instead of `find` for finding files by name or type.

These tools are faster, respect `.gitignore` by default, and generally provide a much more readable output.
