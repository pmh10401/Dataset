# Exp300 Independent Evidence Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether independent MultiROCKET, structural-pattern, and change-point score sources can identify precise residual candidates in the 251 large datasets.

**Architecture:** Hash-pinned frozen predictions supply `kernels_1024_seed_intersection`, `structural_combined_440`, and `change_point_combined_33` candidates. Prediction computes label-free 2-of-3 and 3-of-3 consensus, removes current Exp292 Hard, freezes the result, and only then evaluates TEST labels.

**Tech Stack:** Python 3, NumPy, JSONL, CSV, unittest.

## Global Constraints

- Primary alpha is `0.005`; other frozen alphas are sensitivity outputs only.
- Current Exp292 automatic Hard remains unchanged.
- No family name, TEST length policy, TEST label, or TEST-selected threshold is used in prediction.
- Results are retrospective counterfactual diagnostics, not deployment selection.

---

### Task 1: Consensus logic and tests
- [ ] Write failing tests for exact 2-of-3/3-of-3 sets, Hard exclusion, index bounds, evaluation arithmetic and alpha separation.
- [ ] Implement `build_consensus_row`, evaluation and summary functions.
- [ ] Verify focused tests pass.

### Task 2: Frozen pipeline
- [ ] Hash-pin Exp250, Exp254, Exp255 and Exp298 inputs.
- [ ] Implement separate prediction and evaluation phases with prediction SHA verification.
- [ ] Produce detail, summary, manifest and Korean-readable report artifacts.

### Task 3: Queue, Vercel and full replay
- [ ] Add a failing Exp300 dashboard registration test.
- [ ] Register Exp300 with expected coverage 251 and add a queue command.
- [ ] Run through the sequential runner, independently verify all set and TP/FP arithmetic, and confirm Vercel completion visibility.
