# Exp163 Alternative-Source Tail Agreement Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Implement with test-first RED-GREEN-REFACTOR cycles.

**Goal:** Use the frozen Exp162 alternative score caches to add conservative Standard-review candidates only where frozen Exp160 has no candidate, while preserving autonomous Hard and Priority lanes exactly.

**Architecture:** A pure tail-threshold core converts each source independently using TRAIN empirical quantiles. A label-free prediction runner validates frozen Exp160 and Exp162 inputs, applies 2-of-3 and 3-of-3 same-index agreement, and freezes 8,936 prediction rows. A separate evaluator verifies the prediction hash before loading TEST labels and reports autonomous and human-assisted metrics separately.

**Tech Stack:** Python 3, NumPy, CSV/JSONL, SQLite label loader reused only in evaluation, `unittest`.

## Global Constraints

- Do not alter Exp160 or Exp162 code or outputs.
- Frozen alphas are `(0.005, 0.01, 0.02, 0.05)`.
- Do not use raw score sums, TEST-derived threshold tuning, top-N, TEST-length budget, family routing, past family TEST performance, or tail replacement.
- Exp163 additions go to Standard review only. Hard and Priority must match Exp160 exactly.
- `n_train < 5` remains complete abstention.
- Prediction must not read TEST labels or include evaluation fields.
- All reported variants remain retrospective sensitivity analyses.

---

### Task 1: Pure Tail And Lane Contracts

**Files:**
- Create: `experiments/exp163/__init__.py`
- Create: `experiments/exp163/alternative_tail_core.py`
- Create: `experiments/exp163/test_exp163_alternative_tail_review.py`

- [ ] Write failing tests for `method="higher"` thresholds, strict `>` ties, finite-only TRAIN handling, 2-of-3 and 3-of-3 agreement, `n_train < 5` abstention, candidate-zero targeting, and Standard-only integration.
- [ ] Run the test and confirm failures are caused by missing behavior.
- [ ] Implement the minimum pure functions to pass.
- [ ] Verify lane disjointness, candidate union, range checking, and Exp160 Hard/Priority preservation.

### Task 2: Frozen Input And Prediction Contracts

**Files:**
- Create: `experiments/exp163/exp163_io.py`
- Create: `experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py`
- Create: `experiments/exp163/test_exp163_structure.py`

- [ ] Write failing tests for frozen SHA checks, 1,117 cache coverage, six required finite arrays, matching row counts, forbidden prediction fields, direct CLI execution, deterministic sorted output, and atomic prediction freeze.
- [ ] Implement `audit`, `predict`, `evaluate`, and `manifest` stages.
- [ ] Prediction reads only Exp160 rows and Exp162 NPZ caches; evaluator verifies frozen prediction SHA before loading labels.
- [ ] Emit exactly 8,936 full prediction rows and fail on missing/invalid cache instead of substituting empty candidates.

### Task 3: Documentation And Antigravity Smoke

**Files:**
- Create: `docs/EXP163_PREREGISTRATION_KO.md`
- Create: `docs/EXP163_RUNBOOK_KO.md`
- Generate: `outputs/exp163_alternative_source_tail_agreement_review_smoke/`

- [ ] Record frozen hashes, source display names, variants, alpha grid, candidate-zero caveat, and no-selection rule.
- [ ] Run `py_compile` and both test modules.
- [ ] Run a 10-dataset audit/predict smoke and inspect prediction rows for forbidden fields, lane overlaps, union/count/range errors, and Hard/Priority invariance.
- [ ] Do not run smoke evaluation until Codex review is complete.

### Task 4: Codex Review And Invariance Verification

**Files:**
- Modify only Exp163 files when a failing regression test proves a defect.
- Generate: `outputs/exp163_alternative_source_tail_agreement_review/14_invariance_test_log.txt`
- Generate: `outputs/exp163_alternative_source_tail_agreement_review/15_invariance_test_report.md`

- [ ] Review all prediction call paths for label, family, dataset-name policy, TEST-length budget, and old final-index reuse.
- [ ] Add regression tests before each fix.
- [ ] Verify source thresholds use TRAIN arrays only and raw source scales are never combined.
- [ ] Verify cache/input tamper detection, row-order invariance, output determinism, candidate-empty gate flag, and exact baseline Hard/Priority preservation.

### Task 5: Full Prediction Freeze

**Files:**
- Generate under: `outputs/exp163_alternative_source_tail_agreement_review/`

- [ ] Run full audit over 1,117 datasets.
- [ ] Run full label-free prediction and verify 8,936 rows, zero errors, zero missing datasets, and two variants for every dataset/alpha.
- [ ] Independently reaggregate lane overlap, union/count/range, target gate, source agreement, cache hash, and baseline invariance checks.
- [ ] Freeze prediction SHA256 and configuration hash before any TEST label evaluation.

### Task 6: Separate Evaluation And Reporting

**Files:**
- Generate result CSVs, summary CSV/Markdown, candidate-level diff, workload summary, `VERIFICATION_REPORT.md`, `MANIFEST.csv`, and `MANIFEST.md` under the Exp163 output directory.

- [ ] Verify prediction SHA, then run TEST-label evaluation.
- [ ] Independently reaggregate all variant/alpha metrics.
- [ ] Report autonomous Hard metrics separately from Standard/Priority and human-assisted combined metrics.
- [ ] Confirm Hard/Priority TP, FP, F1, and indices remain identical to Exp160.
- [ ] Report every preregistered alpha and variant without selecting a primary or optimal result.
- [ ] State that results are retrospective, candidate-empty-gated, not end-to-end strict TRAIN-only, and not prospectively validated.
