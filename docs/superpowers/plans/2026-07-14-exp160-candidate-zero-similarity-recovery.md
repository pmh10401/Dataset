# Exp160 Candidate-Zero Similarity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover review candidates for datasets where the frozen Exp152 policy produced no candidate, using only label-free baseline output and TRAIN-normal score-similarity pooling while preserving every autonomous Hard alert and Priority review decision.

**Architecture:** Reuse the frozen Exp152 prediction rows, Exp153-157 TRAIN OOF score cache, and the reviewed Exp158/159 similarity descriptor and donor-selection helpers. For each dataset/alpha with `n_train >= 5` and zero frozen baseline candidates, calculate a 2-of-4 similarity-pooling candidate set and add only previously unseen indices to Standard review. Keep prediction and TEST-label evaluation in separate phases and freeze prediction by SHA256 before evaluation.

**Tech Stack:** Python 3, NumPy, CSV/JSONL, SQLite read-only evaluation, `unittest`.

## Global Constraints

- Experiment ID: `experiment_160_candidate_zero_similarity_recovery`.
- Output directory: `outputs/exp160_candidate_zero_similarity_recovery/`.
- Fixed sensitivity grid: `ALPHAS = (0.005, 0.01, 0.02, 0.05)`; no alpha selection from TEST metrics.
- Internal source keys remain unchanged for reproducibility, but user-facing labels are `시간 패턴 점수`, `주파수 변화 점수`, `반복 구조 점수`, `다중 커널 점수`.
- Frozen baseline: Exp152 B2 prediction and manifest; never overwrite Exp152 or Exp158/159 artifacts.
- Target gate: per dataset/alpha, frozen Exp152 `candidate_count == 0`, `n_train >= 5`, descriptor complete, pool available.
- `n_train < 5` remains complete calibration abstention.
- Baseline candidate-empty is a label-free prediction-output gate, but it uses TEST-derived scores indirectly. Record `uses_test_candidate_empty_gate=1`; do not describe Exp160 as end-to-end strict TRAIN-only.
- Donor selection uses TRAIN OOF scores only, excludes the target dataset, uses no family/dataset-name policy branch, and requires at least 200 finite references for each score source.
- Candidate rule: at least two of four score sources have empirical conformal `p <= alpha` at the same index.
- Similarity-only indices enter Standard review only. Hard and Priority lanes must be exact baseline matches.
- Prediction must not read TEST labels or contain TP, FP, FN, precision, recall, F1, or combined metrics.
- Evaluation may read TEST labels only after verifying frozen prediction SHA256.
- All results are retrospective counterfactual sensitivity, not prospective or deployment validation.

---

### Task 1: Preregister Contract and User-Facing Names

**Files:**
- Create: `docs/EXP160_PREREGISTRATION_KO.md`
- Create: `docs/EXP160_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: Exp152 and Exp158/159 frozen artifact paths and SHA256 contracts.
- Produces: exact experiment contract, commands, output schema, and interpretation limits.

- [ ] **Step 1:** Document the four source labels and internal keys.
- [ ] **Step 2:** Document the candidate-zero gate, including `uses_test_candidate_empty_gate=1` and why this is not end-to-end strict TRAIN-only.
- [ ] **Step 3:** Freeze all constants and lane rules before evaluation.
- [ ] **Step 4:** Record exact unit, smoke, full prediction, and evaluation commands.

### Task 2: Add Pure Candidate-Zero Helpers with TDD

**Files:**
- Create: `experiments/exp160/__init__.py`
- Create: `experiments/exp160/candidate_zero_core.py`
- Create: `experiments/exp160/test_exp160_candidate_zero_recovery.py`

**Interfaces:**
- Consumes: parsed Exp152 baseline lanes and similarity candidate indices.
- Produces: `is_candidate_zero_target(...)`, `integrate_candidate_zero_review(...)`, source display-name mapping, and deterministic canonical row hashing.

- [ ] **Step 1:** Write failing tests for target eligibility, n_train abstention, Hard/Priority preservation, Standard-only addition, lane exclusivity, and out-of-range rejection.
- [ ] **Step 2:** Run the focused tests and verify they fail before implementation.
- [ ] **Step 3:** Implement the minimal pure helpers without DB or label access.
- [ ] **Step 4:** Run focused tests and verify they pass.

### Task 3: Build Audit/Predict/Evaluate Runner

**Files:**
- Create: `experiments/exp160/run_experiment_160_candidate_zero_similarity_recovery.py`
- Create: `experiments/exp160/test_exp160_structure.py`

**Interfaces:**
- Consumes: Exp152 frozen prediction, Exp152 manifest, Exp153-157 score cache, and reviewed helpers from `experiments/exp158_159/similarity_pool_core.py`.
- Produces: audit CSV, prediction JSONL, prediction manifest, evaluation CSV, summary CSV/Markdown, and run manifest.

- [ ] **Step 1:** Write AST/structure tests proving prediction functions do not accept `y_test`, do not select labels, and do not use family or dataset names for policy branching.
- [ ] **Step 2:** Implement audit and predict with full donor universe and optional target-only smoke limit.
- [ ] **Step 3:** Assert 1,117 x 4 frozen baseline coverage, no duplicate rows, index validity, lane union, and baseline Hard/Priority exact preservation.
- [ ] **Step 4:** Freeze prediction hash and implement evaluation as a separate hash-verifying phase.
- [ ] **Step 5:** Emit source internal key and Korean display name together in audit/report output.

### Task 4: Verify Implementation Before Full Run

**Files:**
- Create: `outputs/exp160_candidate_zero_similarity_recovery_smoke/`

**Interfaces:**
- Consumes: implemented Exp160 files.
- Produces: passing compile/tests and a 10-target smoke artifact set.

- [ ] **Step 1:** Run `python3 -m py_compile experiments/exp160/*.py`.
- [ ] **Step 2:** Run both Exp160 unittest modules.
- [ ] **Step 3:** Run audit and prediction for 10 target datasets while retaining all eligible donors.
- [ ] **Step 4:** Inspect prediction JSONL for forbidden TEST/evaluation fields, lane overlap, missing candidates, and target self-pooling.
- [ ] **Step 5:** Run smoke evaluation only after prediction SHA256 is frozen and verified.

### Task 5: Codex Review Gate and Full Retrospective Run

**Files:**
- Create: `outputs/exp160_candidate_zero_similarity_recovery/`
- Create: `outputs/exp160_candidate_zero_similarity_recovery/VERIFICATION_REPORT.md`

**Interfaces:**
- Consumes: reviewed Antigravity implementation and smoke evidence.
- Produces: full 1,117 x 4 prediction/evaluation and independent verification report.

- [ ] **Step 1:** Review implementation against every global constraint and add regression tests for each defect found.
- [ ] **Step 2:** Run full audit and prediction only after all tests pass.
- [ ] **Step 3:** Verify 4,468 prediction rows, zero missing datasets/errors, deterministic hash, and exact baseline Hard/Priority preservation.
- [ ] **Step 4:** Freeze the prediction SHA256, then run TEST-label evaluation.
- [ ] **Step 5:** Independently reaggregate added Standard TP/FP, candidate-zero reduction, workload distribution, and train-size-bin diagnostics.
- [ ] **Step 6:** Report all alpha variants without selecting a primary or optimal value.
