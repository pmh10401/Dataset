# Exp253 Guarded Standard-to-Hard Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible retrospective Exp253 that promotes frozen Exp236 Standard review candidates when either frozen Exp249 MiniROCKET 1024 or 4096 strict seed-intersection evidence agrees.

**Architecture:** A read-only loader validates complete frozen artifacts and hashes. A label-free prediction module performs deterministic set routing and confidence grading, then freezes JSONL output. A separate evaluator verifies that hash before loading TEST labels and is checked by independent reaggregation.

**Tech Stack:** Python 3.14, standard library `json/csv/hashlib`, NumPy, existing TEST-label evaluation loader, `unittest`, local dashboard.

## Global Constraints

- Experiment ID: `experiment_253_guarded_standard_to_hard_expansion`.
- Output: `/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion`.
- Required coverage: 1,117 unique datasets for every frozen source.
- Promotion: `baseline_standard & (mini1024 | mini4096)` at alpha `0.005`.
- Exp240/242/245/247 evidence changes confidence grade only.
- Prediction must not access TEST labels, anomaly positions, family, or TEST-length policy inputs.
- Outside-lane candidates and Priority review are never promoted.
- Mark `selected_from_prior_test_review=true`, `retrospective=true`, `deployment_eligible=false`.
- Never overwrite Exp236, Exp239-250, or Exp252 artifacts.

---

### Task 1: Frozen Input Contract

**Files:**
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/__init__.py`
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/config.py`
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/inputs.py`
- Test: `experiments/exp253_guarded_standard_to_hard_expansion/tests/test_inputs.py`

**Interfaces:**
- Produces: `FrozenDatasetInputs` dataclass with baseline lanes and six evidence sets.
- Produces: `load_frozen_inputs() -> dict[str, FrozenDatasetInputs]`.
- Produces: `audit_inputs(output_root: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing input tests**

```python
def test_required_mini_variants_are_frozen():
    assert MINI_VARIANTS == {
        "mini1024": ("kernels_1024_seed_intersection", 0.005),
        "mini4096": ("kernels_4096_seed_intersection", 0.005),
    }

def test_rejects_out_of_range_index():
    with self.assertRaisesRegex(ValueError, "outside TEST range"):
        validate_indices("demo", 3, {3})
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp253_guarded_standard_to_hard_expansion.tests.test_inputs -v`

Expected: import failure because Exp253 modules do not exist.

- [ ] **Step 3: Implement exact paths, variants, alpha, hashes, and coverage checks**

```python
EXPERIMENT_ID = "experiment_253_guarded_standard_to_hard_expansion"
ALPHA = 0.005
MINI_VARIANTS = {
    "mini1024": ("kernels_1024_seed_intersection", ALPHA),
    "mini4096": ("kernels_4096_seed_intersection", ALPHA),
}

def validate_indices(dataset: str, n_test: int, values: set[int]) -> None:
    if any(index < 0 or index >= n_test for index in values):
        raise ValueError(f"candidate outside TEST range: dataset={dataset}")
```

Reject duplicate or missing rows, compare each source's key set with Exp236, validate all indices, and record SHA256 for every input.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m unittest experiments.exp253_guarded_standard_to_hard_expansion.tests.test_inputs -v`

Expected: all tests pass.

```bash
git add experiments/exp253_guarded_standard_to_hard_expansion
git commit -m "feat: add Exp253 frozen input contract"
```

### Task 2: Label-Free Routing

**Files:**
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/prediction.py`
- Test: `experiments/exp253_guarded_standard_to_hard_expansion/tests/test_prediction.py`

**Interfaces:**
- Consumes: `FrozenDatasetInputs`.
- Produces: `route_dataset(inputs: FrozenDatasetInputs) -> dict[str, object]`.
- Produces: `predict_all(output_root: Path, names: tuple[str, ...]) -> tuple[Path, dict]`.

- [ ] **Step 1: Write failing set-routing tests**

```python
def test_broad_promotion_uses_either_mini_scale():
    row = fixture(standard={2, 3, 4}, mini1024={2}, mini4096={3})
    result = route_dataset(row)
    assert result["promoted_indices"] == [2, 3]
    assert result["final_standard_indices"] == [4]

def test_outside_and_priority_candidates_are_not_promoted():
    row = fixture(standard={2}, priority={5}, mini1024={2, 5, 9})
    result = route_dataset(row)
    assert result["promoted_indices"] == [2]
    assert result["final_priority_indices"] == [5]
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp253_guarded_standard_to_hard_expansion.tests.test_prediction -v`

Expected: prediction module is missing.

- [ ] **Step 3: Implement promotion and confidence grades**

```python
promoted = inputs.standard & (inputs.mini1024 | inputs.mini4096)
both_mini = inputs.mini1024 & inputs.mini4096
independent = inputs.ordinal | inputs.complexity | inputs.fourier | inputs.masked
grade_s = promoted & both_mini & independent
grade_a = (promoted & both_mini) - grade_s
grade_b = promoted - both_mini
```

Assert final Hard, Standard, and Priority lanes are disjoint and `promoted <= baseline_standard`. Sort rows and indices, write JSONL atomically, and freeze SHA256.

- [ ] **Step 4: Add leakage and determinism tests**

```python
def test_prediction_signature_has_no_labels():
    params = inspect.signature(predict_all).parameters
    assert "y_test" not in params and "labels" not in params

def test_prediction_has_no_metrics():
    forbidden = {"tp", "fp", "f1", "true_anomalies", "y_test"}
    assert not forbidden.intersection(route_dataset(fixture()))
```

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest experiments.exp253_guarded_standard_to_hard_expansion.tests.test_prediction -v`

Expected: all tests pass.

```bash
git add experiments/exp253_guarded_standard_to_hard_expansion
git commit -m "feat: add Exp253 guarded promotion routing"
```

### Task 3: Separated Evaluation and Reaggregation

**Files:**
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/evaluation.py`
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/independent_reaggregate.py`
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/reporting.py`
- Test: `experiments/exp253_guarded_standard_to_hard_expansion/tests/test_evaluation.py`

**Interfaces:**
- Produces: `evaluate(output_root: Path) -> tuple[list[dict], dict]`.
- Produces: `reaggregate(output_root: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing hash and zero-denominator tests**

```python
def test_prediction_tamper_is_rejected(tmp_path):
    prediction, manifest = frozen_fixture(tmp_path)
    prediction.write_text(prediction.read_text() + "{}\n")
    with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
        verify_prediction_manifest(prediction, manifest)

def test_empty_selection_metrics():
    assert metrics(set(), {1})["f1"] == 0.0
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp253_guarded_standard_to_hard_expansion.tests.test_evaluation -v`

Expected: evaluation functions are missing.

- [ ] **Step 3: Implement post-freeze evaluation**

Verify prediction SHA256 before calling `load_test_labels_for_evaluation`. Calculate baseline Hard, final Hard, remaining Standard, unchanged Priority, promoted, and S/A/B metrics. Write `results.csv`, `summary.csv`, `lane_transitions.csv`, and `confidence_grades.csv`.

```python
summary["incremental_precision_gate_80pct"] = int(
    summary["promoted_precision"] >= 0.80
)
summary["selected_from_prior_test_review"] = 1
summary["retrospective"] = 1
summary["deployment_eligible"] = 0
```

- [ ] **Step 4: Implement independent reaggregation**

Read only `results.csv`; recompute count, TP, FP, precision, recall, and mean F1 for every lane and grade; compare at absolute tolerance `1e-12`; fail on mismatch.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m unittest experiments.exp253_guarded_standard_to_hard_expansion.tests.test_evaluation -v`

Expected: all tests pass.

```bash
git add experiments/exp253_guarded_standard_to_hard_expansion
git commit -m "feat: add Exp253 separated evaluation"
```

### Task 4: Pipeline and Dashboard

**Files:**
- Create: `experiments/exp253_guarded_standard_to_hard_expansion/run_pipeline.py`
- Create: `scratch/run_exp253_guarded_expansion.py`
- Modify: `serve_rank_dashboard.py`
- Test: `experiments/exp253_guarded_standard_to_hard_expansion/tests/test_pipeline.py`

**Interfaces:**
- CLI stages: `audit`, `predict`, `evaluate`, `all`.
- Dashboard ID: `experiment_253_guarded_standard_to_hard_expansion`.

- [ ] **Step 1: Write failing CLI/dashboard tests**

```python
def test_dashboard_registers_exp253():
    assert "experiment_253_guarded_standard_to_hard_expansion" in Path(
        "serve_rank_dashboard.py"
    ).read_text()

def test_pipeline_exposes_separate_stages():
    choices = parser()._option_string_actions["--stage"].choices
    assert set(choices) == {"audit", "predict", "evaluate", "all"}
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp253_guarded_standard_to_hard_expansion.tests.test_pipeline -v`

Expected: pipeline or dashboard registration is missing.

- [ ] **Step 3: Implement pipeline and status**

Write `status.json`, canonical sequential state, heartbeat, and external live-run state. Report this as frozen-artifact replay, not model training.

- [ ] **Step 4: Run smoke test**

```bash
python3 -m experiments.exp253_guarded_standard_to_hard_expansion.run_pipeline \
  --stage all --dataset-limit 10 \
  --output-root /Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion_smoke
```

Expected: 10 datasets, zero errors, no lane overlap, hash verification and independent reaggregation pass.

- [ ] **Step 5: Run all tests and commit**

Run: `python3 -m unittest discover -s experiments/exp253_guarded_standard_to_hard_expansion/tests -v`

Expected: all tests pass.

```bash
git add experiments/exp253_guarded_standard_to_hard_expansion scratch/run_exp253_guarded_expansion.py serve_rank_dashboard.py
git commit -m "feat: add Exp253 pipeline and dashboard status"
```

### Task 5: Full Retrospective Replay

**Files:**
- Generate: `/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion/*`

**Interfaces:**
- Consumes all Task 1-4 artifacts.
- Produces immutable prediction, evaluation, report, and manifest artifacts.

- [ ] **Step 1: Run audit and prediction**

```bash
python3 -m experiments.exp253_guarded_standard_to_hard_expansion.run_pipeline --stage audit
python3 -m experiments.exp253_guarded_standard_to_hard_expansion.run_pipeline --stage predict
```

Expected: 1,117 datasets, zero errors, deterministic SHA256, and no TEST labels or metrics in predictions.

- [ ] **Step 2: Evaluate only after prediction freeze**

Run: `python3 -m experiments.exp253_guarded_standard_to_hard_expansion.run_pipeline --stage evaluate`

Expected retrospective reference: 137 promoted, 126 TP, 11 FP. Report any difference and do not change routing.

- [ ] **Step 3: Verify everything**

```bash
python3 -m unittest discover -s experiments/exp253_guarded_standard_to_hard_expansion/tests -v
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 4: Confirm report limitations**

`RESULT_REPORT.md` must say in Korean that Exp253 is retrospective, selected after prior TEST review, not deployment eligible, and not prospective equipment validation.

- [ ] **Step 5: Commit final verification references**

```bash
git add experiments/exp253_guarded_standard_to_hard_expansion serve_rank_dashboard.py scratch/run_exp253_guarded_expansion.py
git commit -m "test: verify Exp253 guarded alert expansion"
```
