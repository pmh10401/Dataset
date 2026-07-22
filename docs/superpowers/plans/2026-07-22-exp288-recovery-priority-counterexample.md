# Exp288 Recovery Priority Counterexample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why Exp287 Recovery Priority contains 32 stable false positives by comparing them to nearest TRAIN-normal traces.

**Architecture:** A read-only retrospective script verifies the Exp286 prediction hash, selects the frozen alpha 0.05 Recovery Priority candidates, then loads TEST labels and waveform values for diagnostics. It emits dataset-level comparisons, representative waveform CSVs, summary tables, and no prediction artifact.

**Tech Stack:** Python 3.14, NumPy, SciPy Wasserstein distance, SQLite read-only loader, CSV/JSON, unittest.

### Task 1: Counterexample metrics

- [x] Implement deterministic nearest-normal correlation matching and fixed category assignment.
- [x] Test constant traces, category boundaries, and finite feature metrics.

### Task 2: Retrospective audit

- [x] Verify Exp286 SHA256 before reading labels.
- [x] Reproduce exactly 11 TP and 32 FP in the frozen Recovery Priority subset.
- [x] Save all comparisons and ten representative waveform pairs.

### Task 3: Dashboard and report

- [x] Register Exp288 as retrospective diagnostic with Hard unchanged.
- [x] Run sequentially, independently verify counts, and write `RESULT_REPORT.md`.

## Execution Result

- Completed: 2026-07-22 10:21:06 KST
- Coverage: 1,117 / 1,117 datasets
- Errors: 0
- Frozen candidates: 43 = 11 TP + 32 FP
- FP normal-shape matches: 23 / 32 (71.875%)
- Hard policy changed: no
- Decision: diagnostic only; no distance-based exclusion rule promoted
