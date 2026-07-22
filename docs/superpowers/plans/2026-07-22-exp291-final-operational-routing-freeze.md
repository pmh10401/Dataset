# Exp291 Final Operational Routing Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze Exp290 automatic Hard/normal decisions with approved Exp272 review lanes into one label-free final routing artifact.

**Architecture:** A pure routing function merges hash-verified Exp290 and Exp272 rows and enforces lane disjointness. Prediction is frozen before a separate evaluator reports autonomous and human-assisted scopes.

**Tech Stack:** Python 3.14, JSONL/SHA256, NumPy, CSV, unittest.

## Global Constraints

- Exp290/Exp272 inputs are immutable and hash-verified.
- Recovery Priority remains research-only.
- TEST labels are evaluation-only.
- No threshold, source, or candidate selection changes.

### Task 1: Routing contract

- [ ] Write failing tests for final guard subtraction, lane preservation, and overlap rejection.
- [ ] Implement the pure routing function and prediction schema validator.
- [ ] Run tests and commit.

### Task 2: Freeze and evaluation pipeline

- [ ] Write failing tests for input hash verification and autonomous/review metric separation.
- [ ] Implement prediction freeze, evaluation, summaries, manifests, and wrapper.
- [ ] Run tests and commit.

### Task 3: Dashboard and official run

- [ ] Add a failing focused Exp291 registration test.
- [ ] Register runner/dashboard, run tests, commit, restart, queue, and execute.
- [ ] Independently verify counts, overlaps, hashes, and metrics.

### Task 4: Report

- [ ] Write `RESULT_REPORT.md` in the Exp291 output root.
- [ ] Record completed plan state and commit.
