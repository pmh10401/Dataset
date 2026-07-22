# Exp285-287 Synthetic Shape Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TRAIN-only synthetic anomaly-shape verifier and evaluate its frozen review candidates.

**Architecture:** A focused package generates four fixed synthetic anomaly types, performs self-excluding five-fold classifier calibration, freezes candidate p-values, and evaluates only after SHA256 verification. Exp285 audits generation, Exp286 freezes predictions, and Exp287 evaluates them.

**Tech Stack:** Python 3.14, NumPy, scikit-learn StandardScaler/LogisticRegression, existing DB loader, JSONL/CSV/SHA256, unittest.

## Global Constraints

- Fixed seed `20260722`, four exact transforms, maximum 512 TRAIN bases, five folds.
- Logistic regression uses `C=1`, `max_iter=500`; alphas are `0.01/0.02/0.05`.
- `n_train < 100` abstains.
- All 1,117 datasets remain represented; Exp253 Hard is immutable.
- No TEST label, position, family, order, or length policy input.

### Task 1: Generator and classifier

- Create `experiments/exp285_287_synthetic_shape_verifier/model.py` and focused tests.
- Verify deterministic transformations, finite fixed features, and self-excluding folds.
- Implement bounded synthetic training and empirical upper-tail p-values.

### Task 2: Prediction boundary

- Create `prediction.py` and freeze all 1,117 dataset rows.
- Emit Exp285 generation audit and Exp286 prediction/manifest/summary.
- Test forbidden fields, index bounds, deterministic hash, and tamper failure.

### Task 3: Evaluation and execution

- Create `evaluation.py`, `run_pipeline.py`, and a scratch wrapper.
- Register Exp285-287 in the sequential runner and dashboard.
- Run package and prior registration tests.
- Execute all stages sequentially, independently reaggregate, and write `RESULT_REPORT.md`.
