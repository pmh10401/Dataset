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

- [x] Write failing tests for final guard subtraction, lane preservation, and overlap rejection.
- [x] Implement the pure routing function and prediction schema validator.
- [x] Run tests and commit.

### Task 2: Freeze and evaluation pipeline

- [x] Write failing tests for input hash verification and autonomous/review metric separation.
- [x] Implement prediction freeze, evaluation, summaries, manifests, and wrapper.
- [x] Run tests and commit.

### Task 3: Dashboard and official run

- [x] Add a failing focused Exp291 registration test.
- [x] Register runner/dashboard, run tests, commit, restart, queue, and execute.
- [x] Independently verify counts, overlaps, hashes, and metrics.

### Task 4: Report

- [x] Write `RESULT_REPORT.md` in the Exp291 output root.
- [x] Record completed plan state and commit.

## Execution Result

- Official run completed for 1,117 datasets with zero errors.
- Automatic Hard: 2,218 alerts, 1,915 TP, 303 FP, 86.339% precision, 0.608983 mean dataset F1.
- Automatic exact-normal: 35 alerts removed, all 35 were FP and none were TP.
- Default human review: 881 candidates; Recovery Priority 137 remains research-only.
- All seven focused tests passed; lane overlap and source partition errors were zero.
