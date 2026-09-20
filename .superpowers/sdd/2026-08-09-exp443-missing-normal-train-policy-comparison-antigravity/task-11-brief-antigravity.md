# Exp443 Task 11 brief — fixed CLI and final implementation verification

## Base and scope

- BASE: `ee362b79`
- Create the fixed `readiness`, `new`, `resume`, `inspect`, and `evaluate`
  command surface plus the repository-independent wrapper.
- Add no worker, policy, alpha, kernel, seed, path, or scientific override.
- Keep `evaluation_antigravity` absent from all import paths except the
  `evaluate` branch after a successful frozen-publication inspection.
- Bind `resume` to one explicit basename matching
  `attempt_[0-9]{4}_antigravity`; reject absolute paths and traversal.
- Do not run the official experiment or open official TEST labels in this task.

## Code-map questions before modification

1. What calls it? No mapped caller exists yet. This task creates the wrapper
   and CLI as the first external caller of the grouped Exp443 module.
2. What does it affect? The grouped module reads the pinned SQLite source,
   imports shared create-only artifact helpers, writes checkpoint/progress and
   frozen publication/evaluation artifacts, and participates in all five
   mapped Exp443 flows.
3. Which tests cover it? Eighteen existing Exp443 test modules cover the
   grouped module. Task 11 adds `test_cli_antigravity.py` and focused explicit
   resume-target behavior coverage.

## TDD and verification order

1. Write RED CLI/import/wrapper and explicit-resume tests.
2. Implement the smallest CLI, wrapper, and runtime resume-target seam.
3. Run focused tests, then complete Exp443 and inherited Exp249 regressions.
4. Run compileall, diff checks, the real Policy A control, and the full
   one-worker/11-worker parity script.
5. Update the runbook, catalog, research state, implementation report, and all
   three code-map files together.
6. Require a fresh Sol High read-only review before any official run.
