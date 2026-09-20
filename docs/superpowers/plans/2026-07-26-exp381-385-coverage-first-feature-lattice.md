# Exp381-385 Coverage-First Feature Lattice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and expand TRAIN-only evidence coverage, evaluate each new score alone and in every pre-registered combination, and compare safe pipeline placements without TEST-driven policy selection.

**Architecture:** Exp381 freezes the population, cache, provenance, and calibration-resolution contract. Exp382 reads the verified Exp162 score cache to produce label-free local-tail and similarity-pooling candidates; Exp383 audits TRAIN-only dependence and defines raw versus collapsed evidence families; Exp384 generates exhaustive combination lattices and TRAIN pseudo-normal risk diagnostics; Exp385 reuses the frozen predictions for review, hard-expansion, confidence-tier, and replacement structures.

**Tech Stack:** Python 3, NumPy, SQLite read-only access, JSONL, CSV, NPZ, SHA256, `unittest`, existing sequential queue and dashboard.

## Global Constraints

- DB remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Exp162 cache root remains `/Users/minho/Documents/timesries project/outputs/exp162_full_coverage_alternative_score_cache`.
- Output root is `/Users/minho/Documents/Dataset/exp381_385_coverage_first_feature_lattice`.
- Existing Exp357-359, Exp373-380, and Exp162 artifacts are read-only.
- Alpha grid is exactly `(0.005, 0.01, 0.02, 0.05)`.
- `n_train < 5` always produces `calibration_abstained`, with no candidate index.
- TEST labels and anomaly positions are unavailable to prediction functions.
- Dataset/family names cannot alter calibration, source selection, threshold, budget, or routing.
- TEST results cannot select alpha, source, correlation threshold, rule, risk target, or pipeline structure.
- Automatic alert and human-assisted review metrics remain separate.
- All outputs are retrospective counterfactual, not prospective or deployment validation.

---

### Task 1: Exp381 population, provenance, and cache audit

**Files:**
- Create: `experiments/exp381_385/__init__.py`
- Create: `experiments/exp381_385/coverage_audit_core.py`
- Create: `experiments/exp381_385/run_experiment_381_coverage_provenance_audit.py`
- Create: `experiments/exp381_385/test_coverage_first_lattice.py`

**Interfaces:**
- Produces: `DatasetContract`, `alpha_resolution()`, `audit_exp162_cache()`, Exp381 CSV/JSON/Markdown/status outputs
- Consumes: read-only SQLite metadata, Exp357 frozen evidence, Exp162 manifest/contract/NPZ files

- [ ] **Step 1: Write failing tests for alpha resolution and TRAIN bins**

```python
def test_alpha_resolution_uses_finite_sample_minimum(self):
    self.assertEqual(alpha_resolution(4, 0.05), "abstained_n_train_lt5")
    self.assertEqual(alpha_resolution(19, 0.05), "attainable")
    self.assertEqual(alpha_resolution(19, 0.02), "resolution_blocked")
    self.assertEqual(alpha_resolution(199, 0.005), "attainable")
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python3 -m unittest experiments.exp381_385.test_coverage_first_lattice.CoverageContractTests
```

Expected: import failure because the audit core does not exist.

- [ ] **Step 3: Implement deterministic TRAIN-size and resolution contracts**

`DatasetContract` contains dataset, n_train, n_test, train_bin, minimum_attainable_p, and alpha-specific status. The minimum p-value is `1 / (n_train + 1)` for `n_train >= 5`.

- [ ] **Step 4: Write failing tests for NPZ manifest integrity**

Temporary fixtures must prove rejection of:

- file SHA mismatch
- missing one of six score arrays
- non-finite score
- TRAIN or TEST array length mismatch
- duplicate or missing dataset

- [ ] **Step 5: Implement `audit_exp162_cache()`**

Required arrays:

```python
REQUIRED_ARRAYS = (
    "spectral_train", "spectral_test",
    "shapelet_train", "shapelet_test",
    "interval_train", "interval_test",
)
```

Verify 1,117 rows, unique datasets, config hash, DB hash, file hash, finite arrays, and SQLite lengths.

- [ ] **Step 6: Implement and run Exp381**

Outputs:

- `exp381_dataset_coverage.csv`
- `exp381_alpha_resolution.csv`
- `exp381_exp357_provenance.json`
- `exp381_exp162_cache_audit.csv`
- `exp381_summary.md`
- `exp381_manifest.json`
- `exp381_status.json`

The report explicitly records that Exp357 source code/raw score provenance is unavailable if the repository search remains empty.

- [ ] **Step 7: Verify Exp381**

```bash
python3 -m py_compile experiments/exp381_385/*.py
python3 -m unittest experiments.exp381_385.test_coverage_first_lattice
```

Stop before Exp382 if cache coverage or integrity is incomplete.

### Task 2: Exp382 alternative-score calibration and 12-rule lattice

**Files:**
- Create: `experiments/exp381_385/alternative_score_core.py`
- Create: `experiments/exp381_385/run_experiment_382_alternative_score_lattice.py`
- Modify: `experiments/exp381_385/test_coverage_first_lattice.py`

**Interfaces:**
- Produces: `calibrate_local_empirical()`, `generate_rule_specs()`, label-free Exp382 prediction JSONL
- Consumes: Exp381 verified cache manifest and Exp162 NPZ score arrays

- [ ] **Step 1: Write failing tests for strict empirical tails**

```python
def test_local_tail_is_strict_and_abstains_below_five(self):
    self.assertEqual(calibrate_local_empirical([1, 2, 3, 4], [10], 0.05).status,
                     "calibration_abstained")
    result = calibrate_local_empirical([0, 1, 2, 3, 4], [4, 5], 0.05)
    self.assertEqual(result.indices, [1])
```

The test score equal to the threshold is not a candidate.

- [ ] **Step 2: Write failing tests for three-source exhaustive rules**

Three sources must generate 7 non-empty subsets and 12 `k-of-m` rules.

- [ ] **Step 3: Implement local empirical calibration**

For each source and alpha store threshold, threshold rank, distinct TRAIN score count, candidate indices, n_train, and abstention status. Never call this exact conformal.

- [ ] **Step 4: Implement similarity-pooling adapter**

Reuse the Exp158-160 TRAIN-only donor-selection contract:

- target exclusion
- TRAIN descriptor only
- minimum pooled reference 200
- all four alphas reported
- `n_train < 5` abstention
- candidate output marked `human_review_only`

If required frozen descriptor inputs cannot be reproduced with verified hashes, mark the pooling variant blocked rather than approximating it.

- [ ] **Step 5: Build and freeze Exp382 predictions**

Prediction row contains source candidates and 12 combination rules for each calibration variant and alpha. Freeze JSONL before loading TEST labels.

- [ ] **Step 6: Evaluate after prediction freeze**

Report full population, eligible population, TRAIN bins, source-alone results, every combination, candidate workload, TP/FP, precision, mean F1, zero-F1, and abstention. Do not select a variant.

### Task 3: Exp383 TRAIN-only dependence audit

**Files:**
- Create: `experiments/exp381_385/source_dependence_core.py`
- Create: `experiments/exp381_385/run_experiment_383_source_dependence_audit.py`
- Modify: `experiments/exp381_385/test_coverage_first_lattice.py`

**Interfaces:**
- Produces: TRAIN-only source correlation matrices and frozen family maps
- Consumes: verified Exp162 TRAIN score arrays and frozen Exp357 availability

- [ ] **Step 1: Write failing tests for rank correlation and family collapse**

Tests cover tied ranks, constant arrays, missing source, and deterministic connected-component collapse.

- [ ] **Step 2: Pre-register correlation sensitivities**

Report absolute Spearman thresholds `0.70`, `0.85`, and `0.95`. Do not designate one as primary from TEST results.

- [ ] **Step 3: Implement per-dataset and global TRAIN correlation**

Normalize per dataset using TRAIN median/MAD before pooled correlation. TEST scores are never used for correlation.

- [ ] **Step 4: Freeze source-family maps**

Produce raw-source and threshold-specific independent-family maps with config SHA256. A missing source is `unavailable`, not a negative vote.

- [ ] **Step 5: Generate Exp383 audit outputs**

- source-pair correlation distribution
- collapsed-family counts
- source availability
- common-support size
- raw versus collapsed vote-count differences

Exp383 is diagnostic and does not evaluate TEST performance to choose a map.

### Task 4: Exp384 exhaustive seven-source lattice and TRAIN risk

**Files:**
- Create: `experiments/exp381_385/exhaustive_lattice_core.py`
- Create: `experiments/exp381_385/run_experiment_384_exhaustive_lattice_risk.py`
- Modify: `experiments/exp381_385/test_coverage_first_lattice.py`

**Interfaces:**
- Produces: 448 raw rules, independent-family rules, pseudo-normal risk outputs
- Consumes: Exp357 frozen four-source candidates, Exp382 three-source candidates, Exp383 frozen family maps

- [ ] **Step 1: Write failing seven-source rule-count test**

```python
rules = generate_all_k_of_m_rules(SEVEN_SOURCES)
self.assertEqual(len(rules), 7 * 2 ** 6)
self.assertEqual(len(rules), 448)
```

- [ ] **Step 2: Write failing availability-aware voting tests**

Positive votes require available evidence. If available sources are fewer than `k`, mark the rule resolution-blocked. Missing evidence cannot remove a baseline alert.

- [ ] **Step 3: Implement raw and collapsed lattices**

Generate all 448 raw rules for the alpha-compatible source set and every independent-family rule for each Exp383 sensitivity map. Preserve deterministic names and order.

- [ ] **Step 4: Implement TRAIN pseudo-normal risk**

Cross-fitted TRAIN-normal candidates produce false alerts per 100 runs and one-sided 95% Wilson upper bounds. Report risk targets `0.5`, `1`, and `2` per 100 runs independently.

- [ ] **Step 5: Freeze predictions before evaluation**

Prediction rows contain no TEST metrics. After hash freeze, evaluate all rules and retain both full-population and common-support summaries.

- [ ] **Step 6: Verify no policy selection**

All rules and risk targets remain in output. `selected_for_deployment` is false everywhere.

### Task 5: Exp385 pipeline structures, smoke, queue, and full run

**Files:**
- Create: `experiments/exp381_385/run_experiment_385_pipeline_structures.py`
- Create: `docs/EXP381_385_PREREGISTRATION_KO.md`
- Create: `docs/EXP381_385_RUNBOOK_KO.md`
- Modify: `run_rank_experiments_sequential.py`

**Interfaces:**
- Produces: pipeline comparison CSVs/manifests/status and dashboard-visible queue entries
- Consumes: frozen Exp384 predictions and TRAIN-risk records

- [ ] **Step 1: Write failing pipeline set-operation tests**

Verify:

- baseline Hard is preserved
- review additions exclude baseline and exact-normal
- hard expansion is baseline union risk-eligible additions
- confirmation is a badge, not a baseline filter
- replacement is research-only

- [ ] **Step 2: Implement four structures**

Keep autonomous Hard and human review summaries separate. Risk targets remain separate sensitivity variants.

- [ ] **Step 3: Write preregistration and runbook**

Document provenance limitations, every fixed constant, output hashes, retrospective scope, and TEST-selection prohibition.

- [ ] **Step 4: Run package tests and 10-dataset smoke**

```bash
python3 -m py_compile experiments/exp381_385/*.py
python3 -m unittest experiments.exp381_385.test_coverage_first_lattice
```

Smoke must include at least one dataset from each TRAIN bin; do not use only the first ten sorted datasets.

- [ ] **Step 5: Add sequential queue entries**

Register Exp381 through Exp385 in order with expected dataset coverage 1,117 and status/detail/summary paths. Preserve unrelated queue changes.

- [ ] **Step 6: Execute one experiment at a time**

Start one queue runner. Do not run Exp381-385 concurrently.

- [ ] **Step 7: Independently reaggregate**

Recompute coverage, abstention, candidate count, TP, FP, FN, precision, mean F1, zero-F1, TRAIN risk, and lane transitions from detail outputs. Verify every summary and manifest hash.
