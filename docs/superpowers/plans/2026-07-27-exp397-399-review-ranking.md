# Exp397-399 Review Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the frozen 806 Exp292 standard-review candidates with Exp393 TRAIN-normal grammar evidence without changing any alert or review candidate set.

**Architecture:** A label-free prediction stage verifies Exp292 and Exp393 artifacts, builds deterministic rankings for four preregistered variants, and freezes the ordered candidate rows. A separate evaluation stage verifies the prediction SHA before reading TEST labels, then calculates fixed-workload ranking metrics. An independent reaggregator verifies every summary from candidate-level output.

**Tech Stack:** Python 3 standard library, NumPy, SQLite read-only evaluation, CSV/JSONL/SHA256 artifacts, `unittest`.

## Global Constraints

- Dataset coverage is exactly 1,117 and the standard-review candidate universe is exactly 806.
- Existing automatic Hard, Priority review, and Standard review sets never change.
- Grammar alphas are exactly `(0.005, 0.01, 0.02, 0.05)`.
- Global workloads are exactly `(25, 50, 100, 200)`.
- TEST labels are unavailable to prediction and loaded only after prediction SHA verification.
- No variant, alpha, or workload is selected from TEST metrics.
- Dataset/family names cannot control ranking policy; dataset name and candidate index are deterministic tie-breaks only.
- Results are retrospective and not deployment-eligible.

---

### Task 1: Deterministic ranking core

**Files:**
- Create: `experiments/exp397_399/__init__.py`
- Create: `experiments/exp397_399/contracts.py`
- Create: `experiments/exp397_399/ranking.py`
- Create: `experiments/exp397_399/test_ranking.py`

**Interfaces:**
- Consumes: Exp292 row dictionaries and Exp393 prediction row dictionaries.
- Produces: `build_candidate_features(...)`, `rank_candidates(...)`, and immutable config constants used by all later tasks.

- [ ] **Step 1: Write failing ranking tests**

```python
def test_tier_preserving_never_promotes_lower_tier():
    rows = [
        candidate("A", 0, tier=1, composite_p=0.9),
        candidate("A", 1, tier=2, composite_p=0.001),
    ]
    ranked = rank_candidates(rows, "tier_preserving_grammar", 0.005)
    assert [row["candidate_index"] for row in ranked] == [0, 1]


def test_additive_grammar_promotes_at_most_one_tier():
    row = candidate("A", 1, tier=3, composite_p=0.001)
    ranked = rank_candidates([row], "additive_grammar_family", 0.005)
    assert ranked[0]["effective_tier"] == 2
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp397_399.test_ranking
```

Expected: import failure because `experiments.exp397_399.ranking` does not exist.

- [ ] **Step 3: Implement minimal ranking contracts**

```python
ALPHAS = (0.005, 0.01, 0.02, 0.05)
WORKLOADS = (25, 50, 100, 200)
VARIANTS = (
    "exp292_baseline",
    "tier_preserving_grammar",
    "additive_grammar_family",
    "grammar_only_control",
)
```

`rank_candidates` must use explicit tuple sort keys and append a 1-based `rank`. Grammar support is a single family for promotion even when several primitive components agree.

- [ ] **Step 4: Run ranking tests and verify GREEN**

Run:

```bash
python3 -m unittest experiments.exp397_399.test_ranking
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp397_399
git commit -m "feat: add frozen review ranking core"
```

### Task 2: Label-free Exp397 artifact

**Files:**
- Create: `experiments/exp397_399/artifact_io.py`
- Create: `experiments/exp397_399/prediction.py`
- Create: `experiments/exp397_399/test_prediction.py`

**Interfaces:**
- Consumes: verified Exp292 JSONL and Exp393 prediction JSONL/manifest.
- Produces: `exp397_frozen_review_ranking.jsonl` and `exp397_prediction_manifest.json`.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_prediction_contains_no_test_metrics():
    rows = build_prediction(exp292_fixture(), exp393_fixture())
    forbidden = {"y_test", "tp", "fp", "precision", "f1"}
    assert all(forbidden.isdisjoint(row) for row in rows)


def test_every_variant_has_identical_candidate_universe():
    rows = build_prediction(exp292_fixture(), exp393_fixture())
    universes = candidate_universes(rows)
    assert len({tuple(sorted(value)) for value in universes.values()}) == 1
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m unittest experiments.exp397_399.test_prediction
```

Expected: import failure because prediction modules do not exist.

- [ ] **Step 3: Implement artifact verification and freezing**

Implementation requirements:

```python
prediction_rows = build_prediction(exp292_rows, exp393_rows)
assert dataset_count == 1117
assert unique_candidate_count == 806
assert all_lanes_unchanged
write_jsonl_atomic(prediction_path, prediction_rows)
write_manifest(prediction_sha256=sha256_file(prediction_path))
```

The loader must verify Exp393 against its manifest and record SHA256 for Exp292, Exp393 prediction, Exp393 manifest, config, and code files.

- [ ] **Step 4: Run prediction tests and verify GREEN**

```bash
python3 -m unittest experiments.exp397_399.test_prediction
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp397_399
git commit -m "feat: freeze Exp397 review rankings"
```

### Task 3: Locked Exp398 evaluation

**Files:**
- Create: `experiments/exp397_399/evaluation.py`
- Create: `experiments/exp397_399/test_evaluation.py`

**Interfaces:**
- Consumes: verified Exp397 prediction artifact and TEST labels loaded afterward.
- Produces: candidate-level evaluation, fixed-workload summaries, and dataset diagnostics.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_precision_at_k_uses_first_k_candidates():
    ranked = [
        {"rank": 1, "is_true_anomaly": True},
        {"rank": 2, "is_true_anomaly": False},
    ]
    assert precision_at_k(ranked, 1)["precision_at_k"] == 1.0


def test_evaluation_verifies_prediction_before_labels():
    source = inspect.getsource(run_evaluation)
    assert source.index("load_verified_prediction") < source.index("load_test_labels")
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m unittest experiments.exp397_399.test_evaluation
```

Expected: import failure because evaluation module does not exist.

- [ ] **Step 3: Implement locked evaluation**

For each variant, alpha, and K, write:

```python
{
    "variant": variant,
    "alpha": alpha,
    "workload_k": k,
    "reviewed": min(k, 806),
    "tp_at_k": tp,
    "fp_at_k": fp,
    "precision_at_k": tp / reviewed,
    "review_tp_capture_rate": tp / total_standard_review_tp,
    "precision_lift_vs_exp292": precision - baseline_precision,
}
```

Also calculate `per_dataset_top1` without changing the candidate universe. The evaluation output, not prediction, may contain labels and metrics.

- [ ] **Step 4: Run evaluation tests and verify GREEN**

```bash
python3 -m unittest experiments.exp397_399.test_evaluation
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp397_399
git commit -m "feat: evaluate Exp398 fixed review workloads"
```

### Task 4: Exp399 audit, CLI, and independent reaggregation

**Files:**
- Create: `experiments/exp397_399/reporting.py`
- Create: `experiments/exp397_399/independent_reaggregate_exp397_399.py`
- Create: `experiments/exp397_399/run_experiment_397_399_pipeline.py`
- Create: `experiments/exp397_399/test_pipeline.py`
- Create: `docs/EXP397_399_PREREGISTRATION_KO.md`
- Create: `docs/EXP397_399_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: Task 1-3 modules.
- Produces: `predict`, `evaluate`, `audit`, and `all` CLI stages, status updates, reports, claims, and consistency result.

- [ ] **Step 1: Write failing CLI and consistency tests**

```python
def test_direct_script_imports_from_repository_root():
    result = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_independent_reaggregate_detects_modified_summary():
    mismatches = compare_summary(candidate_rows, modified_summary)
    assert mismatches
```

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m unittest experiments.exp397_399.test_pipeline
```

Expected: import or CLI failure because pipeline modules do not exist.

- [ ] **Step 3: Implement CLI and reporting**

CLI:

```bash
python3 experiments/exp397_399/run_experiment_397_399_pipeline.py \
  --stage all \
  --output-dir /Users/minho/Documents/Dataset/exp397_399_review_ranking
```

`status.json` phases are `predict`, `evaluate`, `audit`, and `completed`. Claims must be separated into verified, conditional, and prohibited sections.

- [ ] **Step 4: Run all Exp397-399 tests**

```bash
python3 -m py_compile experiments/exp397_399/*.py
python3 -m unittest \
  experiments.exp397_399.test_ranking \
  experiments.exp397_399.test_prediction \
  experiments.exp397_399.test_evaluation \
  experiments.exp397_399.test_pipeline
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp397_399 docs/EXP397_399_PREREGISTRATION_KO.md docs/EXP397_399_RUNBOOK_KO.md
git commit -m "feat: add Exp397-399 review ranking pipeline"
```

### Task 5: Smoke verification and sequential queue

**Files:**
- Modify: `run_rank_experiments_sequential.py`

**Interfaces:**
- Consumes: completed Exp397-399 CLI.
- Produces: one queue entry and a full 1,117-dataset retrospective run.

- [ ] **Step 1: Add a failing registry test**

Add a structural assertion that `experiment_397_399_review_ranking` exists once, uses `--stage all`, expects 1,117 datasets, and points to `exp399_workload_summary.csv`.

- [ ] **Step 2: Run the structural test and verify RED**

```bash
python3 -m unittest experiments.exp397_399.test_pipeline
```

Expected: registry assertion fails.

- [ ] **Step 3: Register one sequential experiment**

```python
{
    "id": "experiment_397_399_review_ranking",
    "script": ROOT / "experiments/exp397_399/run_experiment_397_399_pipeline.py",
    "args": ["--stage", "all", "--output-dir", str(EXP397_399_ROOT)],
    "expected_datasets": 1117,
    "archive_outputs": False,
}
```

- [ ] **Step 4: Run smoke and verify invariants**

```bash
python3 experiments/exp397_399/run_experiment_397_399_pipeline.py \
  --stage all \
  --dataset-limit 20 \
  --output-dir /Users/minho/Documents/Dataset/exp397_399_review_ranking_smoke
```

Verify no labels in prediction, identical candidate universes, zero lane changes, output hashes, and independent reaggregation consistency.

- [ ] **Step 5: Run final checks and commit**

```bash
python3 -m unittest experiments.exp397_399.test_ranking experiments.exp397_399.test_prediction experiments.exp397_399.test_evaluation experiments.exp397_399.test_pipeline
git diff --check
git add run_rank_experiments_sequential.py experiments/exp397_399
git commit -m "chore: queue Exp397-399 review ranking"
```

- [ ] **Step 6: Queue and start once**

```bash
python3 run_rank_experiments_sequential.py queue experiment_397_399_review_ranking
python3 run_rank_experiments_sequential.py run-queue
```

Start the queue in one detached session, verify the initial `status.json`, then stop continuous monitoring.
