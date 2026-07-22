# Exp282-284 Prototype Distance Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate a TRAIN-only prototype-distance verifier using Mahalanobis, correlation, and spectral Wasserstein evidence.

**Architecture:** `experiments/exp282_284_prototype_distance_verifier/` owns fixed feature extraction, deterministic medoids, five-fold cross-fitted calibration, label-free prediction freeze, and hash-gated evaluation. The existing sequential runner and dashboard execute Exp282 cache, Exp283 prediction, and Exp284 evaluation one stage at a time.

**Tech Stack:** Python 3.14, NumPy, SciPy, scikit-learn LedoitWolf, SQLite read-only loader, JSONL/CSV/SHA256, unittest.

**Execution status (2026-07-22):** Completed for all 1,117 datasets. The best
frozen Exp284 sensitivity variant reached 44.83% precision and was rejected for
Hard expansion and default review routing. Results are recorded at
`/Users/minho/Documents/Dataset/exp282_284_prototype_distance_verifier/RESULT_REPORT.md`.

## Global Constraints

- Population is all 1,117 datasets; candidate universe is frozen Exp272 Shadow plus Priority.
- Exp253 Hard remains unchanged.
- `alpha=0.05`, five folds with seed `20260722`, medoid-count rules, condition limit `1e8`, and three variants are frozen before evaluation.
- Mahalanobis variants are one evidence family and cannot cast duplicate votes.
- No TEST label, position, family, TEST-order, or TEST-length policy input.
- `n_train < 20` is explicit calibration abstention.

### Task 1: Distance engine

**Files:**
- Create: `experiments/exp282_284_prototype_distance_verifier/distance_engine.py`
- Test: `experiments/exp282_284_prototype_distance_verifier/tests/test_distance_engine.py`

- [ ] Test fixed feature dimensionality, deterministic medoids, finite diagonal and shrinkage distances, and Mahalanobis fallback recording.
- [ ] Implement fixed feature extraction, robust scaling, farthest-first medoids, correlation distance, and spectral Wasserstein distance.
- [ ] Implement self-excluding five-fold normal references and empirical p-values.
- [ ] Run focused tests and commit.

### Task 2: Exp282 cache and Exp283 prediction freeze

**Files:**
- Create: `experiments/exp282_284_prototype_distance_verifier/cache.py`
- Create: `experiments/exp282_284_prototype_distance_verifier/prediction.py`
- Test: `experiments/exp282_284_prototype_distance_verifier/tests/test_prediction.py`

- [ ] Test that cache/prediction artifacts contain no label or evaluation fields.
- [ ] Cache distance calibration only for datasets with frozen candidates while emitting all-population manifests.
- [ ] Freeze all three variant index sets with exact input hashes.
- [ ] Test index bounds, lane exclusivity, and prediction tamper detection.

### Task 3: Exp284 evaluation and orchestration

**Files:**
- Create: `experiments/exp282_284_prototype_distance_verifier/evaluation.py`
- Create: `experiments/exp282_284_prototype_distance_verifier/run_pipeline.py`
- Create: `scratch/run_exp282_284_prototype_distance_verifier.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Test: `experiments/exp282_284_prototype_distance_verifier/tests/test_registration.py`

- [ ] Test that evaluation loads labels only after SHA256 verification.
- [ ] Register Exp282-284 with explicit Hard-unchanged and review-only scopes.
- [ ] Run package and prior dashboard tests.
- [ ] Execute the three stages sequentially for all 1,117 datasets.
- [ ] Independently reaggregate candidate TP/FP and record `RESULT_REPORT.md`.
