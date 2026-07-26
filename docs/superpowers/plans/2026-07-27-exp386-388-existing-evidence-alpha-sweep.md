# Exp386-388 Existing Evidence Alpha Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Exp357 at alpha 0.005 and run a provenance-locked six-alpha coverage sweep.

**Architecture:** A runtime extractor checks out the exact historical source closure into a temporary directory and verifies original SHA256 values. Isolated subprocesses override only the in-memory alpha and output root, then execute the original label-free prediction and post-freeze evaluation phases.

**Tech Stack:** Python 3, Git archive, NumPy, SQLite, JSONL, CSV, SHA256, unittest.

## Global Constraints

- Alpha grid is exactly `(0.005, 0.01, 0.02, 0.05, 0.10, 0.20)`.
- Historical commit is exactly `930b8ce8927d18a0101c647d26e21c2cd20be800`.
- `n_train < 5` remains calibration abstention.
- Existing Exp357-359 outputs are never overwritten.
- TEST metrics cannot select an alpha.

---

### Task 1: Provenance-locked historical runtime

**Files:**
- Create: `experiments/exp386_388/historical_runtime.py`
- Create: `experiments/exp386_388/test_exp386_388.py`

- [ ] Test alpha formatting, historical source extraction, SHA256 verification,
  prediction field guards, and subprocess command construction.
- [ ] Implement extraction with `git archive`, verify the original source hash
  mapping, and execute Exp357/358/359 in an isolated subprocess.

### Task 2: Exp386 exact replay

**Files:**
- Create: `experiments/exp386_388/run_experiment_386_exact_replay.py`

- [ ] Execute alpha 0.005.
- [ ] Compare Exp357 prediction SHA256 and every dataset's evidence indices with
  the original artifact.
- [ ] Exit with code 2 on any mismatch.

### Task 3: Exp387 alpha sweep

**Files:**
- Create: `experiments/exp386_388/run_experiment_387_alpha_sweep.py`

- [ ] Execute 0.01, 0.02, 0.05, 0.10, and 0.20 sequentially.
- [ ] Write dataset-level coverage and a frozen input/output manifest.
- [ ] Preserve all alpha variants without selecting one.

### Task 4: Exp388 comparison and queue

**Files:**
- Create: `experiments/exp386_388/run_experiment_388_alpha_comparison.py`
- Create: `docs/EXP386_388_PREREGISTRATION_KO.md`
- Create: `docs/EXP386_388_RUNBOOK_KO.md`
- Modify: `run_rank_experiments_sequential.py`

- [ ] Independently aggregate coverage, Hard confidence, and review metrics.
- [ ] Register Exp386, Exp387, and Exp388 in sequential order.
- [ ] Run unit tests and a historical-runtime smoke test.
- [ ] Start the detached sequential queue and stop active monitoring.
