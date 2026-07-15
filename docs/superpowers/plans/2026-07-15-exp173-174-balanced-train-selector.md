# Exp173/174 Balanced TRAIN-only Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one source-diverse, fold-stable policy under the existing simultaneous TRAIN-normal 1% risk guard and evaluate that policy once as a retrospective review-only result.

**Architecture:** A pure selector core validates the 60-policy TRAIN audit, computes deterministic five-fold stability, and freezes one `2of3` policy without TEST input. A phase-separated runner writes and hashes the policy and label-free predictions before a separate evaluation phase reads TEST labels; an independent reaggregator verifies every reported metric.

**Tech Stack:** Python 3, standard-library `csv`, `hashlib`, `json`, `sqlite3`, `statistics`, NumPy, existing `ExperimentProgress`, existing Exp169/170 artifacts.

## Global Constraints

- Aggregate TRAIN-normal risk target is exactly `0.01`.
- Confidence is one-sided `0.95` with Bonferroni correction over exactly 60 policies.
- Source agreement selectable set is exactly `source_config == "2of3"`.
- Fold count is exactly `5`; fold assignment is `int(SHA256(dataset)[:16], 16) % 5`.
- Choose the largest eligible alpha before comparing fold stability.
- Rank remaining pools by fold-rate population standard deviation ascending, maximum fold rate ascending, mean reference size descending, then frozen pool order.
- Never use TEST labels, TEST positions, TEST length, family names, prior family TEST performance, or Exp170 TEST metrics for selection.
- Exp174 remains a retrospective research review lane; autonomous Hard alert metrics are not evaluated.
- Do not modify Exp169, Exp170, Exp171, or Exp172 artifacts.
- A worse Exp174 result must be reported without changing any frozen constant.

---

### Task 1: Implement the deterministic balanced selector core

**Files:**
- Create: `experiments/exp173_174/__init__.py`
- Create: `experiments/exp173_174/selector_core.py`
- Create: `experiments/exp173_174/test_exp173_174_selector_core.py`

**Interfaces:**
- Consumes: Exp169 global-summary rows and dataset-audit rows as `list[dict]`.
- Produces: `deterministic_fold(dataset: str, folds: int = 5) -> int`.
- Produces: `select_balanced_train_policy(global_rows: list[dict], dataset_rows: list[dict], *, risk_target: float = 0.01, confidence: float = 0.95, expected_dataset_count: int = 316) -> dict`.

- [ ] **Step 1: Write failing tests for deterministic folds and row-order invariance**

```python
def test_fold_is_deterministic_and_in_range():
    fold = deterministic_fold("demo", folds=5)
    assert fold == deterministic_fold("demo", folds=5)
    assert 0 <= fold < 5

def test_selector_is_invariant_to_input_order():
    first = select_balanced_train_policy(global_rows, dataset_rows)
    second = select_balanced_train_policy(global_rows[::-1], dataset_rows[::-1])
    assert first["selected"] == second["selected"]
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run: `python3 -m unittest experiments/exp173_174/test_exp173_174_selector_core.py`

Expected: FAIL because `experiments.exp173_174.selector_core` does not exist.

- [ ] **Step 3: Implement grid validation, risk eligibility, fold statistics, and deterministic ranking**

```python
def deterministic_fold(dataset: str, folds: int = 5) -> int:
    if folds < 2:
        raise ValueError("folds must be at least two")
    digest = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % folds

ranked = sorted(
    largest_alpha_rows,
    key=lambda row: (
        row["fold_rate_pstdev"],
        row["fold_rate_max"],
        -row["mean_reference_size"],
        POOL_PRIORITY.index(row["pool_policy"]),
    ),
)
```

The implementation must reuse or exactly match Exp171's simultaneous Wilson calculation, validate all 60 keys and 316 unique datasets per key, and emit fold counts, fold denominators, fold rates, eligibility reason, selection rank, and `test_metrics_used=0` for every candidate.

- [ ] **Step 4: Add tests for all contract branches**

Cover incomplete grids, duplicate datasets, risk above `0.01`, resolution blockers, exclusion of non-`2of3` rows, largest-alpha precedence, fold-standard-deviation ordering, all tie-breakers, and explicit abstention.

- [ ] **Step 5: Run the selector tests**

Run: `python3 -m unittest experiments/exp173_174/test_exp173_174_selector_core.py`

Expected: all tests PASS.

- [ ] **Step 6: Commit the selector core**

```bash
git add experiments/exp173_174
git commit -m "feat: add Exp173 balanced train selector"
```

### Task 2: Add phase-separated freeze, prediction, and evaluation runner

**Files:**
- Create: `experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py`
- Create: `experiments/exp173_174/test_exp173_174_structure.py`

**Interfaces:**
- Consumes: Exp169 audit files and Exp170 frozen label-free prediction JSONL.
- Produces: `run_audit(output_dir: Path) -> dict`.
- Produces: `run_prediction(output_dir: Path, dataset_limit: int | None = None) -> dict`.
- Produces: `run_evaluation(output_dir: Path) -> dict`.

- [ ] **Step 1: Write failing AST and behavior tests for phase separation**

```python
def test_prediction_path_does_not_load_test_labels():
    source = inspect.getsource(runner.run_prediction)
    assert "_load_test_labels" not in source
    assert "SOURCE_TEST_METRICS" not in source

def test_contract_freezes_balance_rule():
    config = runner.frozen_contract()["config"]
    assert config["risk_target"] == 0.01
    assert config["source_agreement"] == "2of3"
    assert config["fold_count"] == 5
    assert config["test_metric_selection"] is False
```

- [ ] **Step 2: Run the structure tests and verify the missing-runner failure**

Run: `python3 -m unittest experiments/exp173_174/test_exp173_174_structure.py`

Expected: FAIL because the runner module does not exist.

- [ ] **Step 3: Implement the phase-separated runner using Exp171/172 conventions**

Use these fixed IDs and output paths:

```python
AUDIT_EXPERIMENT_ID = "experiment_173_balanced_train_selector_freeze"
EVALUATION_EXPERIMENT_ID = "experiment_174_balanced_frozen_review"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/exp173_174_balanced_train_selector"
```

Required files:

```text
01_contract.json
02_exp173_policy_candidates.csv
03_exp173_frozen_policy.json
04_exp174_frozen_predictions.jsonl
05_prediction_manifest.json
06_exp174_dataset_metrics.csv
07_exp174_global_summary.csv
10_completion.json
full_run.log
```

The prediction row must contain no label, TP, FP, FN, precision, recall, F1, or combined-F1 field. Evaluation must verify both policy and prediction hashes before opening the database and must report review scope only.

- [ ] **Step 4: Add hash-tamper, forbidden-field, coverage, and scope tests**

Cover policy tampering, prediction tampering, incomplete 316-row prediction coverage, duplicate datasets, out-of-range indices, label-free row declarations, and the absence of autonomous Hard metrics.

- [ ] **Step 5: Run selector and structure tests together**

Run:

```bash
python3 -m unittest \
  experiments/exp173_174/test_exp173_174_selector_core.py \
  experiments/exp173_174/test_exp173_174_structure.py
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the runner**

```bash
git add experiments/exp173_174
git commit -m "feat: add Exp173-174 frozen policy runner"
```

### Task 3: Add independent verification and research-scope documentation

**Files:**
- Create: `experiments/exp173_174/independent_reaggregate_exp173_174.py`
- Create: `experiments/exp173_174/test_exp173_174_reaggregation.py`
- Create: `docs/EXP173_174_PREREGISTRATION_KO.md`
- Create: `docs/EXP173_174_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: frozen policy, prediction manifest, Exp174 dataset metrics, global summary, and completion JSON.
- Produces: `11_independent_reaggregation.json` and `13_invariance_report.md` after execution.

- [ ] **Step 1: Write failing tests for independent metric and hash verification**

```python
def test_reaggregation_rejects_summary_mismatch():
    summary["fp"] += 1
    with self.assertRaisesRegex(ValueError, "summary mismatch"):
        verify_outputs(output_dir)
```

Also test 316-dataset coverage, unique dataset names, candidate-index bounds, TP+FP equality with review count, exact policy-key equality, and prediction hash verification.

- [ ] **Step 2: Run the reaggregation test and verify failure**

Run: `python3 -m unittest experiments/exp173_174/test_exp173_174_reaggregation.py`

Expected: FAIL because the reaggregation module does not exist.

- [ ] **Step 3: Implement read-only independent reaggregation**

The script must recompute review candidates, TP, FP, micro precision, mean recall, mean F1, candidate-zero datasets, zero-F1 datasets, and candidate workload quantiles. It must compare every recomputed value with `07_exp174_global_summary.csv` and write mismatch count zero only when all checks pass.

- [ ] **Step 4: Write Korean preregistration and runbook documents**

Document the `1%` aggregate guard, `2of3` agreement, five-fold stability rule, fixed tie-breaks, phase commands, review-only scope, retrospective limitation, already-observed Exp170 benchmark, and the prohibition against retuning after Exp174.

- [ ] **Step 5: Run all Exp173/174 tests**

Run: `python3 -m unittest discover -s experiments/exp173_174 -p 'test_*.py'`

Expected: all tests PASS.

- [ ] **Step 6: Commit verification and docs**

```bash
git add experiments/exp173_174 docs/EXP173_174_PREREGISTRATION_KO.md docs/EXP173_174_RUNBOOK_KO.md
git commit -m "test: add Exp173-174 independent verification"
```

### Task 4: Register Exp173/174 in the sequential queue and dashboard

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_queue_safety.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: runner experiment IDs and output paths from Task 2.
- Produces: one-at-a-time queue entries and historical dashboard rows for Exp173 and Exp174.

- [ ] **Step 1: Write failing queue and dashboard registration tests**

```python
def test_queue_catalog_includes_exp173_174_artifacts():
    by_id = {item["id"]: item for item in runner.EXPERIMENTS}
    assert by_id["experiment_173_balanced_train_selector_freeze"]["args"] == ["--phase", "audit"]
    assert by_id["experiment_174_balanced_frozen_review"]["args"] == ["--phase", "predict-evaluate"]
```

Dashboard assertions must verify `autonomous_metric_scope == "not_evaluated"`, `review_metric_scope == "retrospective_research_review_candidate"`, and the correct result paths.

- [ ] **Step 2: Run queue and dashboard tests and verify missing-entry failures**

Run:

```bash
python3 scratch/test_rank_queue_safety.py
python3 scratch/test_rank_dashboard_operational.py
```

Expected: FAIL on missing Exp173/174 catalog entries.

- [ ] **Step 3: Add sequential queue entries**

Exp173 must run `--phase audit` and Exp174 must run `--phase predict-evaluate`. Both entries use `expected_datasets=316`, the new output directory, and the same `full_run.log`. Do not start either experiment during registration.

- [ ] **Step 4: Add dashboard current and historical catalog entries**

Expose the frozen selection rule, TRAIN-only selection scope, review-only metric scope, detail/summary files, and artifacts. Preserve all previous experiment rows.

- [ ] **Step 5: Run queue, dashboard, and full Exp173/174 tests**

Run:

```bash
python3 scratch/test_rank_queue_safety.py
python3 scratch/test_rank_dashboard_operational.py
python3 -m unittest discover -s experiments/exp173_174 -p 'test_*.py'
```

Expected: all tests PASS.

- [ ] **Step 6: Commit queue and dashboard registration**

```bash
git add run_rank_experiments_sequential.py serve_rank_dashboard.py \
  scratch/test_rank_queue_safety.py scratch/test_rank_dashboard_operational.py
git commit -m "feat: register Exp173-174 balanced selector"
```

### Task 5: Execute Exp173 then Exp174 and freeze the evidence

**Files:**
- Create through execution: `outputs/exp173_174_balanced_train_selector/*`
- Create after analysis: `outputs/exp173_174_balanced_train_selector/09_result_analysis.md`
- Create after verification: `outputs/exp173_174_balanced_train_selector/MANIFEST.csv`

**Interfaces:**
- Consumes: all code and frozen source artifacts from Tasks 1-4.
- Produces: one auditable Exp173 policy and one retrospective Exp174 review result.

- [ ] **Step 1: Verify no overlapping experiment process is active**

Run:

```bash
python3 run_rank_experiments_sequential.py status
pgrep -af 'run_experiment_|run_rank_experiments_sequential' || true
```

Expected: no active experiment except the read-only dashboard.

- [ ] **Step 2: Run a separate 10-dataset smoke prediction/evaluation**

Run:

```bash
SMOKE=outputs/exp173_174_balanced_train_selector_smoke
python3 experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py --phase audit --output-dir "$SMOKE"
python3 experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py --phase predict --dataset-limit 10 --output-dir "$SMOKE"
python3 experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py --phase evaluate --output-dir "$SMOKE"
```

Expected: 10 unique label-free predictions, zero forbidden fields, zero index errors, and a hash-verified evaluation.

- [ ] **Step 3: Run Exp173 audit through the sequential queue**

Run: `python3 run_rank_experiments_sequential.py run experiment_173_balanced_train_selector_freeze`

Expected: a frozen policy selected or an explicit abstention. Do not run Exp174 after abstention.

- [ ] **Step 4: Verify the frozen selector before TEST evaluation**

Confirm the selected row is `2of3`, the simultaneous upper bound is at most `0.01`, the selected alpha is the largest eligible `2of3` alpha, and its fold-stability ranking is one. Record policy and candidate-table SHA256 values.

- [ ] **Step 5: Run Exp174 once through the sequential queue**

Run: `python3 run_rank_experiments_sequential.py run experiment_174_balanced_frozen_review`

Expected: 316 dataset rows, no prediction errors, and no change to the Exp173 policy hash.

- [ ] **Step 6: Run independent verification and the complete test suite**

Run:

```bash
python3 experiments/exp173_174/independent_reaggregate_exp173_174.py
python3 -m unittest discover -s experiments/exp173_174 -p 'test_*.py'
python3 scratch/test_rank_queue_safety.py
python3 scratch/test_rank_dashboard_operational.py
git diff --check
```

Expected: mismatch count zero, all tests PASS, and no whitespace errors.

- [ ] **Step 7: Write unchanged retrospective interpretation and manifest**

Compare Exp174 with Exp172 on review candidates, TP, FP, precision, mean recall, mean F1, candidate-zero datasets, and workload distribution. Explicitly state that Exp170 TEST outcomes were already historically visible, Exp174 is not an independent holdout, and no Hard alert or deployment claim is supported.

- [ ] **Step 8: Commit generated evidence**

```bash
git add -f outputs/exp173_174_balanced_train_selector/*.csv \
  outputs/exp173_174_balanced_train_selector/*.json \
  outputs/exp173_174_balanced_train_selector/*.jsonl \
  outputs/exp173_174_balanced_train_selector/*.md \
  outputs/exp173_174_balanced_train_selector/*.log
git commit -m "results: add Exp173-174 balanced policy evaluation"
```
