---
name: Use rg and fd
description: Enforces the use of ripgrep (rg) over grep, and fd over find for searching and finding files in the terminal.
---

# Instructions

When interacting with the terminal to find files or search through file contents:

- **ALWAYS** use `rg` (ripgrep) instead of `grep`, `egrep`, or `fgrep` for text search.
- **ALWAYS** use `fd` instead of `find` for finding files by name or type.
- **NEVER** propose a `run_command` that invokes `grep` or `find` — replace them with `rg` and `fd` respectively.
- Prefer `rg` via `run_command` over the built-in `grep_search` tool for any search that benefits from richer output (e.g. multiline context, multiple file types, pipe into other tools).

## Quick reference

| Task | ❌ Don't use | ✅ Use instead |
|---|---|---|
| Search text in files | `grep -r "foo" .` | `rg "foo"` |
| Case-insensitive search | `grep -ri "foo" .` | `rg -i "foo"` |
| Show N lines of context | `grep -n -A2 "foo"` | `rg -n -A2 "foo"` |
| Search specific file types | `grep -r --include="*.cpp"` | `rg "foo" -g "*.cpp"` |
| Find files by name | `find . -name "*.cpp"` | `fd -e cpp` |
| Find files by type | `find . -type f -name "*.py"` | `fd -e py` |
| Find directory | `find . -type d -name build` | `fd -t d build` |

Both tools respect `.gitignore` by default and are significantly faster than their GNU counterparts.
