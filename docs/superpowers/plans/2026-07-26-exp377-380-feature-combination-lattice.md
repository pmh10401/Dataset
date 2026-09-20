# Exp377-380 Feature Combination Lattice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate every pre-registered combination of four independent TRAIN-only evidence families and compare review, hard-expansion, confidence-tier, and replacement pipeline structures without TEST-driven policy selection.

**Architecture:** Exp377 reads and hash-verifies Exp357 evidence plus Exp358/359 frozen lanes, generates one label-free master prediction containing 32 deterministic rules, and freezes it once. Exp378 and Exp379 reuse that exact frozen prediction for review-overlap and pipeline-structure evaluation; Exp380 performs read-only Pareto and stability aggregation over the preceding outputs.

**Tech Stack:** Python 3 standard library, SQLite read-only evaluation, JSONL, CSV, SHA256, `unittest`, existing sequential experiment queue and dashboard status contract.

## Global Constraints

- DB remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Output root is `/Users/minho/Documents/Dataset/exp377_380_feature_combination_lattice`.
- Existing Exp357-359 and Exp373-376 source/result files are read-only.
- Exactly four independent families are used: `trend`, `dynamics`, `spectral`, `normal_distance`.
- Exactly 15 non-empty subsets and 32 `k-of-m` rules are generated before TEST evaluation.
- Prediction code does not read TEST labels, anomaly positions, TP, FP, F1, family performance, or dataset-name policies.
- Missing or resolution-blocked evidence is not a negative vote.
- Exact-normal indices cannot become new candidates.
- Automatic alert and human-assisted review metrics remain separate.
- All results are retrospective counterfactual and are not deployment or prospective validation.
- TEST results do not select a rule, threshold, alpha, feature, or pipeline structure.

---

### Task 1: Deterministic rule lattice and availability semantics

**Files:**
- Create: `experiments/exp377_380/__init__.py`
- Create: `experiments/exp377_380/feature_combination_core.py`
- Create: `experiments/exp377_380/test_feature_combination_lattice.py`

**Interfaces:**
- Produces: `RuleSpec`, `generate_rule_specs()`, `collapse_family_evidence()`, `evaluate_rule()`
- Consumes: Exp357 evidence-row dictionaries

- [ ] **Step 1: Write failing tests for the 15 subsets and 32 rules**

```python
def test_generate_rule_specs_is_complete_and_deterministic(self):
    first = generate_rule_specs()
    second = generate_rule_specs()
    self.assertEqual(first, second)
    self.assertEqual(len({rule.families for rule in first}), 15)
    self.assertEqual(len(first), 32)
    self.assertEqual(len({rule.rule_id for rule in first}), 32)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice.RuleGenerationTests
```

Expected: import failure because `feature_combination_core.py` does not exist.

- [ ] **Step 3: Implement immutable rule specifications and deterministic generation**

```python
@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    families: tuple[str, ...]
    k: int


def generate_rule_specs(
    families: Sequence[str] = INDEPENDENT_FAMILIES,
) -> tuple[RuleSpec, ...]:
    rules = []
    for size in range(1, len(families) + 1):
        for subset in combinations(families, size):
            for k in range(1, size + 1):
                rules.append(RuleSpec(f"{'_'.join(subset)}__{k}of{size}", subset, k))
    return tuple(rules)
```

- [ ] **Step 4: Write failing tests for family collapse and availability**

Tests must prove:

- `trend`, `spectral`, and `normal_distance` inherit source `available`.
- `dynamics` is available only when at least two derivative sources are available.
- `dynamics` candidates require at least two available derivative votes.
- unavailable evidence contributes no positive vote and records a reason.

- [ ] **Step 5: Run availability tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice.AvailabilityTests
```

Expected: failure because `collapse_family_evidence()` is missing.

- [ ] **Step 6: Implement family collapse and rule evaluation**

`collapse_family_evidence(row)` returns:

```python
{
    "indices": dict[str, list[int]],
    "available": dict[str, bool],
    "reason": dict[str, str],
}
```

`evaluate_rule(spec, family_state)` returns:

```python
{
    "indices": list[int],
    "selected_family_count": int,
    "available_family_count": int,
    "fully_observable": bool,
    "resolution_blocked": bool,
    "unavailable_families": list[str],
}
```

`resolution_blocked` is true only when fewer than `spec.k` selected families are available. A positive index requires at least `spec.k` positive family votes.

- [ ] **Step 7: Run Task 1 tests**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice.RuleGenerationTests experiments.exp377_380.test_feature_combination_lattice.AvailabilityTests
```

Expected: all Task 1 tests pass.

### Task 2: Prediction rows, pipeline structures, and safety guards

**Files:**
- Modify: `experiments/exp377_380/feature_combination_core.py`
- Modify: `experiments/exp377_380/test_feature_combination_lattice.py`

**Interfaces:**
- Produces: `build_master_prediction_row()`, `apply_pipeline_structures()`, `validate_prediction_row()`, `freeze_predictions()`, `load_frozen_predictions()`
- Consumes: matching Exp357, Exp358, and Exp359 rows

- [ ] **Step 1: Write failing tests for the four set structures**

With `H={1,2}`, `R={2,3,4}`, and `exact={4}`, assert:

```python
self.assertEqual(result["review_addition_indices"], [3])
self.assertEqual(result["hard_expansion_indices"], [1, 2, 3])
self.assertEqual(result["baseline_additionally_confirmed_indices"], [2])
self.assertEqual(result["baseline_without_additional_confirmation_indices"], [1])
self.assertEqual(result["replacement_indices"], [2, 3])
```

- [ ] **Step 2: Run the structure test and verify RED**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice.PipelineStructureTests
```

Expected: failure because `apply_pipeline_structures()` is missing.

- [ ] **Step 3: Implement structure set operations**

`apply_pipeline_structures(rule_indices, baseline_indices, exact_normal_indices)` must return the five arrays above and never remove a baseline alert because extra evidence is absent.

- [ ] **Step 4: Write failing tests for row alignment and forbidden fields**

Tests must prove:

- dataset and `n_test` must match across Exp357/358/359.
- master rows contain 32 rule records.
- `review_union_indices`, `standard_review_indices`, `priority_review_indices`, and all Exp359 Shadow lanes are preserved.
- all indices are in range.
- exact-normal indices are absent from review addition, hard expansion additions, and replacement.
- prediction keys do not contain labels or evaluation metrics.

- [ ] **Step 5: Run row-validation tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice.PredictionContractTests
```

Expected: failure because the prediction builder is missing.

- [ ] **Step 6: Implement prediction building and SHA256 freezing**

`build_master_prediction_row(exp357_row, exp358_row, exp359_row)` stores:

- raw family indices and availability
- baseline and exact-normal indices
- Exp359 review/Shadow lane indices
- 32 rule results
- five structure index arrays per rule
- `contains_test_labels=False`
- `contains_evaluation_metrics=False`

The freeze loader verifies file SHA256, row count, unique datasets, sorted order, and every row contract.

- [ ] **Step 7: Write and pass tamper-detection tests**

Mutate one byte after freezing and assert `load_frozen_predictions()` raises `ValueError`.

- [ ] **Step 8: Run all core tests**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice
```

Expected: all core tests pass.

### Task 3: Exp377-379 prediction and evaluation runners

**Files:**
- Create: `experiments/exp377_380/feature_combination_reporting.py`
- Create: `experiments/exp377_380/run_feature_combination_experiment.py`
- Create: `experiments/exp377_380/run_experiment_377_combination_lattice.py`
- Create: `experiments/exp377_380/run_experiment_378_review_overlap.py`
- Create: `experiments/exp377_380/run_experiment_379_pipeline_structures.py`
- Modify: `experiments/exp377_380/test_feature_combination_lattice.py`

**Interfaces:**
- Produces: frozen master prediction, detail CSVs, summary CSVs, manifests, status JSONs
- Consumes: Task 2 master-row functions and read-only TEST labels after prediction freeze

- [ ] **Step 1: Write failing reporting tests**

Tests must verify:

- micro precision and recall use summed TP/FP/FN.
- mean F1 is dataset macro mean.
- automatic and review scopes are never combined.
- full-population and common-support summaries are separate.
- no-candidate and resolution-blocked rows use the frozen zero-denominator rules.

- [ ] **Step 2: Run reporting tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice.ReportingTests
```

Expected: import failure for `feature_combination_reporting`.

- [ ] **Step 3: Implement read-only evaluation helpers**

Implement:

```python
evaluate_indices(indices, labels) -> dict[str, int | float]
summarize_detail_rows(detail_rows, group_fields) -> list[dict]
write_csv(path, rows) -> None
```

Exp377 reports all 32 replacement/standalone rule candidates. Exp378 reports new review additions and overlap with every frozen Exp359 review/Shadow lane. Exp379 reports hard expansion, baseline additional-confirmation tier, baseline without additional confirmation, and replacement.

- [ ] **Step 4: Write failing source-contract tests**

Use temporary JSONL/manifest fixtures to prove the runner rejects:

- source SHA mismatch
- source coverage other than the requested dataset limit
- mismatched dataset order or `n_test`
- label-bearing prediction sources

- [ ] **Step 5: Implement the shared CLI**

CLI:

```bash
python3 experiments/exp377_380/run_feature_combination_experiment.py \
  --experiment 377 \
  --dataset-limit 10 \
  --output-root /tmp/exp377_380_smoke
```

Exp377 creates the master prediction. Exp378/379 load and hash-check the same master prediction. For isolated wrapper execution, Exp378/379 fail clearly when Exp377 master artifacts are missing.

- [ ] **Step 6: Implement thin wrappers**

Each wrapper inserts the repository root into `sys.path` when directly executed and calls the shared runner with its fixed experiment number.

- [ ] **Step 7: Run all package tests and compilation**

Run:

```bash
python3 -m py_compile experiments/exp377_380/*.py
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice
```

Expected: compilation succeeds and all tests pass.

### Task 4: Exp380 Pareto and stability audit

**Files:**
- Create: `experiments/exp377_380/run_experiment_380_pareto_stability.py`
- Modify: `experiments/exp377_380/feature_combination_reporting.py`
- Modify: `experiments/exp377_380/test_feature_combination_lattice.py`

**Interfaces:**
- Produces: `exp380_pareto.csv`, `exp380_rule_stability.csv`, `exp380_summary.md`, manifest and status
- Consumes: frozen Exp377 master prediction and Exp377-379 detail CSVs

- [ ] **Step 1: Write failing tests for exact and tolerance matching**

Test exact Jaccard plus one-to-one greedy matching at tolerances 1, 3, and 5. A candidate can match at most one reference index.

- [ ] **Step 2: Write failing tests for Pareto dominance**

A rule is dominated only when another rule has no worse TP gain, no greater FP gain, and no greater review workload, with at least one strict improvement. Scope and support population must match.

- [ ] **Step 3: Run Exp380 unit tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice.StabilityTests
```

Expected: failures because matching and Pareto helpers are missing.

- [ ] **Step 4: Implement stability and Pareto helpers**

Produce:

- exact-index Jaccard
- tolerance ±1/±3/±5 match rates
- per-dataset candidate count mean/median/p90/p95/max
- added TP/FP versus Exp358
- zero-F1 changes
- full-population/common-support split
- Pareto membership as descriptive output only

- [ ] **Step 5: Implement the read-only Exp380 runner**

The runner verifies all input hashes from Exp377-379 evaluation manifests before aggregation and does not read prediction sources anew.

- [ ] **Step 6: Run the full package tests**

Run:

```bash
python3 -m unittest experiments.exp377_380.test_feature_combination_lattice
```

Expected: all tests pass.

### Task 5: Documentation, smoke test, sequential queue, and full execution

**Files:**
- Create: `docs/EXP377_380_PREREGISTRATION_KO.md`
- Create: `docs/EXP377_380_RUNBOOK_KO.md`
- Modify: `run_rank_experiments_sequential.py`

**Interfaces:**
- Produces: dashboard-visible queue entries and reproducible execution instructions
- Consumes: Tasks 1-4 wrappers and output contracts

- [ ] **Step 1: Write the preregistration and runbook**

Copy the fixed constraints, 32-rule enumeration, availability semantics, four structures, TEST-selection prohibition, output paths, and interpretation limits from the approved design.

- [ ] **Step 2: Run the 10-dataset smoke sequence**

Run:

```bash
SMOKE=/Users/minho/Documents/Dataset/exp377_380_feature_combination_lattice_smoke
rm -rf "$SMOKE"
python3 experiments/exp377_380/run_experiment_377_combination_lattice.py --dataset-limit 10 --output-root "$SMOKE"
python3 experiments/exp377_380/run_experiment_378_review_overlap.py --dataset-limit 10 --output-root "$SMOKE"
python3 experiments/exp377_380/run_experiment_379_pipeline_structures.py --dataset-limit 10 --output-root "$SMOKE"
python3 experiments/exp377_380/run_experiment_380_pareto_stability.py --dataset-limit 10 --output-root "$SMOKE"
```

Expected:

- 10 unique datasets
- 32 rule records per dataset
- no index overlap with exact-normal additions
- errors 0
- prediction JSONL contains no TEST label or metric fields

- [ ] **Step 3: Add four queue entries with minimal patching**

Add Exp377, Exp378, Exp379, and Exp380 in order. Each entry points to its wrapper, detail/summary output, stdout log, and status JSON under the fixed output root. Do not reformat or stage unrelated queue-runner changes.

- [ ] **Step 4: Run queue safety and dashboard tests**

Run:

```bash
python3 -m unittest scratch.test_rank_queue_safety scratch.test_rank_dashboard_operational
```

Expected: tests pass and the queue lists Exp377 before Exp378, Exp379, and Exp380.

- [ ] **Step 5: Start one sequential queue runner**

Verify no overlapping experiment runner exists, then start the established queue command with one active experiment at a time. Do not launch four experiments in parallel.

- [ ] **Step 6: Verify full completion**

Check:

- 1,117 unique datasets
- 32 rules per dataset
- errors 0
- all four status files completed
- input and output SHA256 recorded
- Exp377 master prediction reused unchanged by Exp378/379
- no TEST-derived deployment selection

- [ ] **Step 7: Independently reaggregate key totals**

Use a read-only one-off script to recompute each rule/structure TP, FP, FN, precision, recall, mean F1, blocked count, and common-support count from detail CSVs. Compare to all summary CSVs and record any mismatch without modifying rules.
