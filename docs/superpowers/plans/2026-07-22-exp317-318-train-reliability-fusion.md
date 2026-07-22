# Exp317-318 TRAIN-normal Reliability Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Exp317 TRAIN-normal source reliability audit and Exp318 frozen reliability-weighted FN-recovery Shadow evaluation without changing Exp291 autonomous Hard alerts.

**Architecture:** Exp317 reads the frozen Exp309-315 source artifacts and calculates source reliability from disjoint TRAIN-normal reference/audit splits only. Exp318 consumes the hash-frozen Exp317 artifact and source TEST p-values, creates weighted independent-group Shadow candidates without labels, freezes prediction hashes, and loads TEST labels only in a separate evaluation phase.

**Tech Stack:** Python 3.14, NumPy, standard-library CSV/JSON/hashlib, SQLite evaluation loader, `unittest`, existing sequential runner and read-only dashboard.

## Global Constraints

- Evaluation population is the frozen Exp291 1,117-dataset population.
- Source alpha grid is exactly `(0.005, 0.01, 0.02, 0.05)`.
- Random seed is exactly `20260722`.
- Wilson bound is one-sided 95% with `z = 1.6448536269514722`.
- Reliability uses five deterministic 50/50 TRAIN-normal reference/audit splits.
- Fewer than 20 TRAIN-normal source scores causes source-level abstention.
- Exp318 share thresholds are exactly `(0.50, 0.67, 0.75)` and all are sensitivity variants.
- Exp318 requires at least three available groups and at least two supporting groups.
- Exp291 Hard and Exp290 TRAIN raw exact-normal candidates are excluded.
- Prediction must not read TEST labels, anomaly positions, family names, past family TEST performance, or TEST length for routing.
- TEST metrics must not select alpha, share threshold, source, weight formula, or guard.
- Exp291 autonomous Hard remains unchanged; Exp318 is retrospective research Shadow only.
- Tail replacement and autonomous promotion of Priority review are prohibited.

---

### Task 1: Reliability mathematics and frozen contracts

**Files:**
- Create: `experiments/exp317_318_train_reliability_fusion/__init__.py`
- Create: `experiments/exp317_318_train_reliability_fusion/config.py`
- Create: `experiments/exp317_318_train_reliability_fusion/reliability.py`
- Create: `experiments/exp317_318_train_reliability_fusion/contracts.py`
- Create: `experiments/exp317_318_train_reliability_fusion/tests/__init__.py`
- Test: `experiments/exp317_318_train_reliability_fusion/tests/test_reliability.py`
- Test: `experiments/exp317_318_train_reliability_fusion/tests/test_contracts.py`

**Interfaces:**
- Consumes: numeric TRAIN-normal source scores and immutable JSON-compatible rows.
- Produces: `deterministic_reference_audit_splits`, `empirical_upper_pvalues`, `wilson_upper_one_sided`, `audit_source_reliability`, `freeze_jsonl`, and `verify_frozen_jsonl`.

- [ ] **Step 1: Write failing reliability tests**

```python
def test_reliability_decreases_when_normal_false_alerts_increase():
    stable = audit_source_reliability(np.arange(100.0), alpha=0.05, seed=20260722)
    noisy = audit_source_reliability(
        np.r_[np.arange(80.0), np.repeat(10_000.0, 20)],
        alpha=0.05,
        seed=20260722,
    )
    assert noisy.weight < stable.weight

def test_reference_and_audit_are_disjoint_and_deterministic():
    first = deterministic_reference_audit_splits(25, seed=20260722, splits=5)
    second = deterministic_reference_audit_splits(25, seed=20260722, splits=5)
    assert first == second
    assert all(set(reference).isdisjoint(audit) for reference, audit in first)
```

- [ ] **Step 2: Run tests and confirm missing imports fail**

Run:

```bash
python3 -m unittest experiments.exp317_318_train_reliability_fusion.tests.test_reliability
```

Expected: `ModuleNotFoundError` or missing-function failure.

- [ ] **Step 3: Implement fixed reliability formulas**

```python
WILSON_Z = 1.6448536269514722

def empirical_upper_pvalues(reference, audit):
    reference = np.asarray(reference, dtype=float)
    audit = np.asarray(audit, dtype=float)
    return np.asarray([
        (1 + np.count_nonzero(reference >= value)) / (len(reference) + 1)
        for value in audit
    ])

def wilson_upper_one_sided(false_alerts, total, z=WILSON_Z):
    if total <= 0:
        raise ValueError("Wilson bound requires a positive total")
    p = false_alerts / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return min(1.0, (center + radius) / denominator)
```

`audit_source_reliability` must use five shuffled disjoint splits, take the maximum Wilson upper bound, set `n_audit_min`, and calculate:

```python
weight = max(0.0, -np.log10(max(max_wilson_upper, 1.0 / (n_audit_min + 1))))
```

- [ ] **Step 4: Implement hash-frozen JSONL contracts**

Reject rows containing `y_test`, `tp`, `fp`, `precision`, `recall`, `f1`, or `combined_f1`. Write sorted JSON lines atomically, store row count and SHA256 in a manifest, and verify both before reading downstream.

- [ ] **Step 5: Run focused and package tests**

```bash
python3 -m unittest \
  experiments.exp317_318_train_reliability_fusion.tests.test_reliability \
  experiments.exp317_318_train_reliability_fusion.tests.test_contracts
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp317_318_train_reliability_fusion
git commit -m "feat: add TRAIN-normal reliability contracts"
```

### Task 2: Exp317 source audit

**Files:**
- Create: `experiments/exp317_318_train_reliability_fusion/source_loader.py`
- Create: `experiments/exp317_318_train_reliability_fusion/exp317_audit.py`
- Test: `experiments/exp317_318_train_reliability_fusion/tests/test_exp317_audit.py`

**Interfaces:**
- Consumes: Exp309, 310, 311, 313, 314, and 315 frozen prediction JSONL and manifests.
- Produces: `run_exp317_audit(output_root: Path, dataset_limit: int | None) -> dict` and one frozen reliability row per dataset/source/alpha.

- [ ] **Step 1: Write failing source coverage and abstention tests**

```python
def test_exp317_abstains_below_twenty_scores():
    row = audit_one_source("D", "discord", 0.05, list(range(19)))
    assert row["available"] is False
    assert row["abstention_reason"] == "insufficient_train_reference"

def test_source_loader_rejects_coverage_mismatch():
    with self.assertRaisesRegex(ValueError, "coverage mismatch"):
        validate_source_dataset_sets({"discord": {"A"}, "svdd": {"A", "B"}}, expected=2)
```

- [ ] **Step 2: Run the focused test and verify failure**

```bash
python3 -m unittest experiments.exp317_318_train_reliability_fusion.tests.test_exp317_audit
```

- [ ] **Step 3: Implement verified source loading**

Load each source with its existing manifest verifier. Assert exact equality with the Exp291 dataset-name set and record every source prediction SHA256. Do not load the SQLite label column or call `load_test_labels_for_evaluation`.

- [ ] **Step 4: Implement Exp317 rows and summaries**

Each reliability row must include:

```python
{
    "experiment_id": "experiment_317_train_normal_source_reliability_audit",
    "dataset": dataset,
    "source": source,
    "method_group": method_group,
    "alpha": alpha,
    "n_train_scores": len(scores),
    "available": available,
    "abstention_reason": reason,
    "split_false_alert_counts": counts,
    "split_audit_counts": totals,
    "max_wilson_upper": upper,
    "n_audit_min": n_audit_min,
    "weight": weight,
    "contains_test_labels": False,
    "contains_evaluation_metrics": False,
}
```

Write `exp317_frozen_reliability.jsonl`, `exp317_reliability_manifest.json`, `exp317_reliability_audit.csv`, and `exp317_summary.csv`.

- [ ] **Step 5: Verify ten-dataset audit smoke**

```bash
python3 - <<'PY'
from pathlib import Path
from experiments.exp317_318_train_reliability_fusion.exp317_audit import run_exp317_audit
run_exp317_audit(Path('/tmp/exp317_reliability_smoke'), dataset_limit=10)
PY
```

Expected: ten distinct datasets, six sources times four alphas per dataset, no TEST or evaluation fields, and a valid manifest hash.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp317_318_train_reliability_fusion
git commit -m "feat: add Exp317 TRAIN-normal reliability audit"
```

### Task 3: Exp318 weighted independent-group prediction

**Files:**
- Create: `experiments/exp317_318_train_reliability_fusion/exp318_prediction.py`
- Test: `experiments/exp317_318_train_reliability_fusion/tests/test_exp318_prediction.py`

**Interfaces:**
- Consumes: verified Exp317 reliability rows, verified Exp309-315 source rows, Exp291 Hard indices, and Exp290 exact-normal indices.
- Produces: `predict_exp318(output_root: Path, dataset_limit: int | None) -> dict` and frozen label-free prediction rows.

- [ ] **Step 1: Write failing weighted-fusion tests**

```python
def test_representation_sources_count_as_one_group():
    weights = {"contrastive": 0.4, "svdd": 0.8, "discord": 0.6}
    support = {"contrastive", "svdd", "discord"}
    result = weighted_group_support(weights, support, SOURCE_GROUPS)
    assert result.supporting_group_count == 2
    assert result.supporting_weight == 1.4

def test_candidate_requires_two_supporting_and_three_available_groups():
    assert qualifies(0.75, available_groups=3, supporting_groups=2, threshold=0.67)
    assert not qualifies(1.0, available_groups=2, supporting_groups=2, threshold=0.67)
    assert not qualifies(0.9, available_groups=5, supporting_groups=1, threshold=0.67)
```

- [ ] **Step 2: Run focused test and verify failure**

```bash
python3 -m unittest experiments.exp317_318_train_reliability_fusion.tests.test_exp318_prediction
```

- [ ] **Step 3: Implement independent-group weighted support**

Available group weight is the maximum available source weight in that group. Supporting group weight is the maximum weight among sources in that group that support the candidate. Calculate the share only when available group weight sum is positive.

- [ ] **Step 4: Implement the three frozen sensitivity variants**

For every alpha, emit candidate indices for `share_0.50`, `share_0.67`, and `share_0.75`. Require three available groups and two supporting groups, then subtract Exp291 Hard and Exp290 exact-normal indices.

Prediction rows must contain support groups, weighted share, abstention reason, and candidate indices, but no TEST labels or metrics.

- [ ] **Step 5: Verify ten-dataset prediction smoke and hash tamper failure**

```bash
python3 - <<'PY'
from pathlib import Path
from experiments.exp317_318_train_reliability_fusion.exp318_prediction import predict_exp318
predict_exp318(
    Path('/tmp/exp318_reliability_smoke'),
    reliability_root=Path('/tmp/exp317_reliability_smoke'),
    dataset_limit=10,
)
PY
```

Expected: ten prediction rows, twelve alpha/share variants per row, zero forbidden fields, and manifest verification success. A one-byte prediction edit must make verification fail.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp317_318_train_reliability_fusion
git commit -m "feat: add Exp318 reliability-weighted prediction"
```

### Task 4: Label-separated evaluation and independent reaggregation

**Files:**
- Create: `experiments/exp317_318_train_reliability_fusion/evaluation.py`
- Create: `experiments/exp317_318_train_reliability_fusion/reporting.py`
- Create: `experiments/exp317_318_train_reliability_fusion/independent_reaggregate.py`
- Test: `experiments/exp317_318_train_reliability_fusion/tests/test_evaluation.py`

**Interfaces:**
- Consumes: hash-verified Exp318 prediction and TEST labels loaded after verification.
- Produces: `exp318_results.csv`, `exp318_summary.csv`, evaluation manifest, result report, and independent reaggregation report.

- [ ] **Step 1: Write failing metric and scope tests**

```python
def test_summary_keeps_shadow_scope_and_counts():
    summary = summarize([{"candidate_count": 3, "tp": 2, "fp": 1}])
    assert summary["precision"] == 2 / 3
    assert summary["automatic_policy_changed"] is False
    assert summary["human_assisted_only"] is True
```

- [ ] **Step 2: Run focused test and verify failure**

```bash
python3 -m unittest experiments.exp317_318_train_reliability_fusion.tests.test_evaluation
```

- [ ] **Step 3: Implement prediction-first evaluation**

Call `verify_frozen_jsonl` before any label loader. Report all twelve variants without ranking one as selected. Include TP outside Hard, TP outside all existing lanes, zero-Hard-F1 recovery, workload distribution, abstention counts, and exact candidate overlap with corresponding Exp316 alpha variants.

- [ ] **Step 4: Implement independent CSV reaggregation**

Read only `exp318_results.csv`, group by variant, recompute datasets, candidates, TP, FP, precision, outside-existing TP, and zero-Hard recovery, then compare with `exp318_summary.csv`. Exit nonzero on any mismatch.

- [ ] **Step 5: Complete ten-dataset evaluation smoke**

```bash
python3 - <<'PY'
from pathlib import Path
from experiments.exp317_318_train_reliability_fusion.evaluation import evaluate_exp318
evaluate_exp318(Path('/tmp/exp318_reliability_smoke'))
PY
python3 experiments/exp317_318_train_reliability_fusion/independent_reaggregate.py \
  --output-root /tmp/exp318_reliability_smoke
```

Expected: evaluation manifest records `labels_loaded_after_prediction_freeze=true`, and independent reaggregation matches.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp317_318_train_reliability_fusion
git commit -m "feat: evaluate Exp318 reliability fusion"
```

### Task 5: CLI, sequential queue, and dashboard registration

**Files:**
- Create: `experiments/exp317_318_train_reliability_fusion/pipeline.py`
- Create: `scratch/run_exp317_318_train_reliability_fusion.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: Task 2-4 audit, prediction, and evaluation functions.
- Produces: `--experiment 317|318`, phase controls, atomic `status.json`, sequential queue command `queue-exp317-318`, and dashboard rows.

- [ ] **Step 1: Add failing queue and dashboard registration tests**

```python
def test_exp317_318_are_registered_as_research_shadow():
    assert runner_item(317)["expected_datasets"] == 1117
    assert runner_item(318)["expected_datasets"] == 1117
    assert dashboard_item(317)["comparison_eligible"] is False
    assert dashboard_item(318)["autonomous_metric_scope"] == "Exp291 Hard unchanged"
```

- [ ] **Step 2: Implement CLI and atomic status publishing**

Exp317 supports `audit`; Exp318 supports `predict`, `evaluate`, and `all`. Every phase writes `status.json` with experiment id, phase, done, expected, current dataset, elapsed seconds, ETA, error count, and updated Unix time. Exceptions write `status=failed` before re-raising.

- [ ] **Step 3: Register the sequential batch**

Add exactly two entries in order:

```python
EXP317_318_BATCH = [
    "experiment_317_train_normal_source_reliability_audit",
    "experiment_318_train_reliability_weighted_fusion",
]
```

Both run with one CPU process because they reuse frozen artifacts and do not train models. Add `queue-exp317-318` without altering other queues.

- [ ] **Step 4: Register dashboard metadata**

Both experiments use `history_status="diagnostic_only"`, `comparison_eligible=False`, `autonomous_metric_scope="Exp291 Hard unchanged"`, and explicit retrospective Shadow descriptions. Point the dashboard to each atomic `status.json`, result CSV, summary CSV, and log.

- [ ] **Step 5: Run operational tests**

```bash
python3 -m py_compile \
  experiments/exp317_318_train_reliability_fusion/*.py \
  scratch/run_exp317_318_train_reliability_fusion.py
python3 scratch/test_rank_dashboard_operational.py
git diff --check
```

Expected: compilation and dashboard operational tests pass with no whitespace errors.

- [ ] **Step 6: Commit scoped files**

```bash
git add experiments/exp317_318_train_reliability_fusion \
  scratch/run_exp317_318_train_reliability_fusion.py \
  run_rank_experiments_sequential.py serve_rank_dashboard.py \
  scratch/test_rank_dashboard_operational.py
git commit -m "feat: queue and monitor Exp317-318"
```

If the three existing runner/dashboard files contain unrelated unstaged user changes, do not include those unrelated hunks in the commit.

### Task 6: Full verification and sequential launch

**Files:**
- Verify: `/Users/minho/Documents/Dataset/exp317_train_normal_source_reliability_audit/`
- Verify: `/Users/minho/Documents/Dataset/exp318_train_reliability_weighted_fusion/`

**Interfaces:**
- Consumes: completed implementation and frozen Exp309-315 artifacts.
- Produces: full Exp317/318 artifacts and a live dashboard-visible sequential run.

- [ ] **Step 1: Run all package tests**

```bash
python3 -m unittest discover \
  -s experiments/exp317_318_train_reliability_fusion/tests \
  -p 'test_*.py'
```

Expected: all tests pass.

- [ ] **Step 2: Repeat clean smoke in unique temporary directories**

Run Exp317 audit, Exp318 predict, Exp318 evaluate, and independent reaggregation on ten datasets. Confirm source coverage, forbidden-field checks, and hashes before full launch.

- [ ] **Step 3: Confirm no active queue process**

```bash
python3 run_rank_experiments_sequential.py status
pgrep -af 'run_rank_experiments_sequential.py run-queue|run_exp317_318_train_reliability_fusion.py'
```

Expected: no active conflicting experiment.

- [ ] **Step 4: Queue and launch sequentially**

```bash
python3 run_rank_experiments_sequential.py queue-exp317-318
python3 - <<'PY'
import subprocess
from pathlib import Path
root = Path('/private/tmp/timeseries-exp265-dashboard-demo')
log = Path('/Users/minho/Documents/Dataset/rank_experiments_sequential_launcher.log')
with log.open('ab') as handle:
    process = subprocess.Popen(
        ['python3', 'run_rank_experiments_sequential.py', 'run-queue'],
        cwd=root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(process.pid)
PY
```

Launch the queue runner detached, then verify only one active experiment and seven-second-or-newer atomic status updates. Do not continuously monitor after launch.

- [ ] **Step 5: Verify dashboard truthfulness**

Compare the active experiment `status.json` with `http://127.0.0.1:8765/api/status`. Experiment id, done, expected, progress percent, and current dataset must match.

- [ ] **Step 6: Final post-run verification**

After both runs complete, independently reaggregate Exp318 and report every alpha/share sensitivity. Do not name a primary, optimal, or deployment variant from this retrospective TEST set.
