---
name: enforce-logging
description: Enforces the use of proper logging libraries (spdlog, python logging) instead of raw print or std::cout statements across the codebase.
---

# Enforce Logging

This skill ensures that all output to the console goes through structured logging rather than raw print functions.

## Guidelines

1. **C++**: 
   - **DO NOT** use `std::cout`, `std::cerr`, or `printf`.
   - **DO** use the `spdlog` library.
   - Example: Include `<spdlog/spdlog.h>` and use `spdlog::info("Message: {}", value);`

2. **Python**: 
   - **DO NOT** use `print()`.
   - **DO** use the standard `logging` module.
   - Example: `import logging` and `logging.info(f"Message: {value}")`

3. **TypeScript/JavaScript**: 
   - **DO NOT** use `console.log()`, `console.error()`, etc.
   - **DO** use the application's configured logger (e.g., `server.log.info()` for Fastify).

4. **Proactive Refactoring**: If you are modifying a file or function and notice existing `print()`, `std::cout`, or `console.log()` statements, proactively upgrade them to use the appropriate logging framework.
