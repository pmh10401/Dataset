# Exp186-200 Review, Shadow Promotion, and Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, register, and sequentially run fifteen frozen retrospective experiments covering review quality, shadow promotion safety, and detection coverage.

**Architecture:** Reuse the frozen Exp137 and Exp177-185 artifacts through a new focused `experiments/exp186_200` package. Prediction phases operate only on frozen candidate/score inputs and TRAIN-normal guards, then write hash-locked JSONL; evaluation phases verify hashes before reading TEST labels. A sequential wrapper exposes one experiment ID per stage while sharing deterministic caches.

**Tech Stack:** Python 3 standard library, NumPy, SQLite, unittest, existing rank queue/dashboard infrastructure.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-15-exp186-200-review-shadow-coverage-design.md` exactly.
- Preserve Exp137 autonomous Hard indices byte-for-byte.
- Use exactly 639 datasets with `train_normal_count >= 20`; no silent exclusions.
- Prediction code must not accept or read TEST labels.
- TEST metrics must not choose any alpha, source, guard, tier, or promotion rule.
- Exp191-195 are `shadow_hard_diagnostic`, never autonomous alerts.
- Run one queued stage at a time; use seven workers only inside a stage.
- Do not modify or overwrite Exp137 or Exp177-185 outputs.

---

### Task 1: Core policy algebra and TRAIN-only guards

**Files:**
- Create: `experiments/exp186_200/__init__.py`
- Create: `experiments/exp186_200/core.py`
- Create: `experiments/exp186_200/test_exp186_200_core.py`

**Interfaces:**
- Consumes: frozen index sets and TRAIN OOF p-value arrays.
- Produces: `ReviewTiers`, `agreement_tiers()`, `stable_dataset_guard()`, `conformal_count_cap()`, `apply_count_cap()`, and `validate_disjoint_indices()`.

- [ ] **Step 1: Write failing policy tests**

```python
def test_three_tiers_are_disjoint_and_complete():
    tiers = agreement_tiers({1, 2, 3}, {2, 3, 4})
    assert tiers.high == {2, 3}
    assert tiers.standard == {1}
    assert tiers.exploratory == {4}
    assert tiers.all_indices == {1, 2, 3, 4}

def test_count_cap_uses_train_counts_only():
    assert conformal_count_cap([0, 0, 1, 1, 2], alpha=0.01) >= 0

def test_stability_requires_four_of_five_folds():
    assert stable_dataset_guard([True, True, True, True, False])
    assert not stable_dataset_guard([True, True, True, False, False])
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_core.py`

Expected: import or symbol failures because `core.py` is absent.

- [ ] **Step 3: Implement deterministic core functions**

Use frozen set algebra only. `conformal_count_cap()` must implement the
one-sided finite-sample empirical quantile with `ceil((n + 1) * (1-alpha))`,
clamped to valid sorted-count indices. `apply_count_cap()` sorts candidates by
the frozen evidence tuple supplied by the caller, then by ascending index.

- [ ] **Step 4: Run core tests**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_core.py`

Expected: all tests pass.

- [ ] **Step 5: Commit core implementation**

```bash
git add experiments/exp186_200
git commit -m "feat: add exp186-200 policy core"
```

### Task 2: Frozen input loader and prediction/evaluation boundary

**Files:**
- Create: `experiments/exp186_200/artifacts.py`
- Create: `experiments/exp186_200/test_exp186_200_structure.py`

**Interfaces:**
- Consumes: Exp137 CSV, Exp177 policy JSON, Exp178/180/181/183 prediction manifests and JSONL, Exp153-157 score cache manifest.
- Produces: `FrozenInputs`, `load_frozen_inputs()`, `freeze_predictions()`, `load_verified_predictions()`, and `load_test_labels_for_evaluation()`.

- [ ] **Step 1: Write failing boundary tests**

```python
def test_prediction_payload_rejects_evaluation_fields():
    with self.assertRaises(ValueError):
        validate_prediction_payload({"dataset": "x", "candidate_indices": [], "hard_tp": 0})

def test_predict_signature_has_no_y_test():
    self.assertNotIn("y_test", inspect.signature(predict_dataset).parameters)

def test_hash_tamper_is_rejected():
    self.assertRaises(ValueError, load_verified_predictions, changed_jsonl, manifest)
```

- [ ] **Step 2: Run structure tests and confirm failures**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_structure.py`

- [ ] **Step 3: Implement hash-verified loaders**

Every source manifest must match its JSONL SHA256. Validate 639 unique dataset
keys and reject missing or duplicated rows. Keep `load_test_labels_for_evaluation`
in an evaluation-only function that is never called by prediction entry points.

- [ ] **Step 4: Run boundary tests**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_structure.py`

- [ ] **Step 5: Commit artifact boundary**

```bash
git add experiments/exp186_200
git commit -m "feat: freeze exp186-200 artifact boundary"
```

### Task 3: Exp186-190 review prediction stages

**Files:**
- Create: `experiments/exp186_200/run_experiment_186_200.py`
- Modify: `experiments/exp186_200/test_exp186_200_core.py`

**Interfaces:**
- Consumes: `FrozenInputs` and core policy functions.
- Produces: stage-specific prediction JSONL and manifests for Exp186-190.

- [ ] **Step 1: Add exact candidate-rule tests**

Test the five rules from the design with synthetic sets, including Exp190 tier
mutual exclusion and Exp189 cap determinism. Assert every payload uses
`final_lane` values from `high_confidence_review`, `standard_review`, or
`exploratory_review` only.

- [ ] **Step 2: Run the new tests and confirm failures**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_core.py`

- [ ] **Step 3: Implement `predict_review_stage(stage, inputs)`**

Map stages exactly:

```python
REVIEW_STAGE_RULES = {
    "186": "conservative_and_canonical_knn_005",
    "187": "conservative_and_dual_canonical_005",
    "188": "conservative_with_4of5_train_stability",
    "189": "conservative_with_train_conformal_count_cap",
    "190": "three_tier_conservative_canonical",
}
```

Write one payload per dataset and tier/configuration, with evidence fields and
no TEST metrics.

- [ ] **Step 4: Run core and structure tests**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_core.py experiments/exp186_200/test_exp186_200_structure.py`

- [ ] **Step 5: Commit review prediction stages**

```bash
git add experiments/exp186_200
git commit -m "feat: add exp186-190 review stages"
```

### Task 4: Exp191-195 shadow promotion stages

**Files:**
- Modify: `experiments/exp186_200/run_experiment_186_200.py`
- Modify: `experiments/exp186_200/test_exp186_200_core.py`

**Interfaces:**
- Consumes: review-stage evidence and frozen TRAIN OOF support.
- Produces: shadow-only prediction payloads for Exp191-195.

- [ ] **Step 1: Add shadow namespace tests**

Assert every Exp191-195 payload has
`final_lane="shadow_hard_diagnostic"`, `autonomous_alert=False`, and cannot
modify or serialize Exp137 Hard indices as output Hard indices.

- [ ] **Step 2: Run tests and confirm failures**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_core.py`

- [ ] **Step 3: Implement the five frozen shadow rules**

```python
SHADOW_STAGE_RULES = {
    "191": "exp186_shadow",
    "192": "exp187_shadow",
    "193": "existing3_plus_canonical_knn_005",
    "194": "exp191_plus_4of5_train_stability",
    "195": "exp191_plus_zero_train_candidate_guard",
}
```

- [ ] **Step 4: Verify shadow tests pass**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_core.py`

- [ ] **Step 5: Commit shadow stages**

```bash
git add experiments/exp186_200
git commit -m "feat: add exp191-195 shadow diagnostics"
```

### Task 5: Exp196-200 coverage and independent reaggregation

**Files:**
- Modify: `experiments/exp186_200/run_experiment_186_200.py`
- Create: `experiments/exp186_200/independent_reaggregate.py`
- Modify: `experiments/exp186_200/test_exp186_200_structure.py`

**Interfaces:**
- Consumes: frozen review predictions and label-delayed evaluation rows.
- Produces: Exp196-199 prediction files and Exp200 full frontier report.

- [ ] **Step 1: Add coverage-rule and reaggregation tests**

Test exact union/intersection rules, evidence-origin preservation, TRAIN-only
cap input signature, summary arithmetic, and the prohibition on `best`,
`optimal`, or `recommended` selection fields.

- [ ] **Step 2: Run tests and confirm failures**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_structure.py`

- [ ] **Step 3: Implement Exp196-199 prediction and evaluation**

Map rules exactly as preregistered. Evaluation must verify prediction SHA256,
then load labels and emit candidate TP/FP/precision/recall/F1 and workload
quantiles. Keep shadow and review metric scopes separate.

- [ ] **Step 4: Implement Exp200 independent arithmetic**

Read stage result CSVs, independently recompute totals and overlaps, compare
them with stage summaries, and fail on mismatch. Emit all rows without policy
selection.

- [ ] **Step 5: Run all package tests**

Run: `python3 -m unittest experiments/exp186_200/test_exp186_200_core.py experiments/exp186_200/test_exp186_200_structure.py`

- [ ] **Step 6: Commit coverage and reaggregation**

```bash
git add experiments/exp186_200
git commit -m "feat: add exp196-200 coverage frontier"
```

### Task 6: Sequential queue and dashboard registration

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Create: `scratch/test_exp186_200_queue_contract.py`

**Interfaces:**
- Consumes: `run_experiment_186_200.py --stage N` for N 186 through 200.
- Produces: fifteen ordered queue definitions and dashboard catalog rows.

- [ ] **Step 1: Write failing queue tests**

Assert exactly fifteen IDs, strict numeric order, expected dataset count 639,
one common runner script, no duplicate IDs, and dashboard catalog presence.

- [ ] **Step 2: Run queue test and confirm failure**

Run: `python3 -m unittest scratch/test_exp186_200_queue_contract.py`

- [ ] **Step 3: Register queue and dashboard metadata**

Add Exp186-200 without changing earlier queue definitions. Each command passes
one `--stage` value and uses `RANK_EXPERIMENT_WORKERS=7`.

- [ ] **Step 4: Run queue and dashboard tests**

Run: `python3 -m unittest scratch/test_exp186_200_queue_contract.py scratch/test_dashboard_state_contract.py scratch/test_rank_dashboard_operational.py`

- [ ] **Step 5: Commit registration**

```bash
git add run_rank_experiments_sequential.py serve_rank_dashboard.py scratch/test_exp186_200_queue_contract.py
git commit -m "feat: register exp186-200 sequential batch"
```

### Task 7: Contract review, smoke test, and launch

**Files:**
- Create: `outputs/exp186_200_validation/00_code_contract_review.md`
- Generated outside Git: `/Users/minho/Documents/Dataset/exp186_200_validation/*`

**Interfaces:**
- Consumes: completed implementation and frozen inputs.
- Produces: test logs, smoke outputs, queue state, and detached sequential run.

- [ ] **Step 1: Run compilation and unit tests**

```bash
python3 -m py_compile experiments/exp186_200/*.py
python3 -m unittest \
  experiments/exp186_200/test_exp186_200_core.py \
  experiments/exp186_200/test_exp186_200_structure.py \
  scratch/test_exp186_200_queue_contract.py \
  scratch/test_dashboard_state_contract.py
```

Expected: all tests pass. Any failure blocks smoke and full launch.

- [ ] **Step 2: Run a 10-dataset smoke batch**

Run each phase into `outputs/exp186_200_validation_smoke`, verify prediction
rows contain no labels or metrics, verify lane disjointness, index ranges,
hashes, and deterministic rerun hashes.

- [ ] **Step 3: Review Gemini implementation before full execution**

Codex checks every stage against the design, records deviations in
`00_code_contract_review.md`, and fixes violations before labels are evaluated.

- [ ] **Step 4: Register and start the full sequential queue**

Use the repository's existing queue reset/enqueue mechanism for only Exp186-200,
then launch one detached sequential runner. Do not start multiple stage runners.

- [ ] **Step 5: Verify dashboard and process state**

```bash
python3 dashboardctl.py status
curl -fsS http://127.0.0.1:8765/api/status
```

Expected: Exp186 is the sole running item, Exp187-200 are queued, and the
dashboard process PID matches `/api/healthz`.

- [ ] **Step 6: Stop agent monitoring**

After process and dashboard verification, leave the batch detached. The user
will monitor progress and request analysis after completion.
