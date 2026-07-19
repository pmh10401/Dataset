# Exp226-235 Hard Alert Quality Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one restartable program that reads SQLite data, reproduces Exp137, runs ten independent TRAIN-only quality experiments, freezes label-free predictions, evaluates afterward, and independently reaggregates every result.

**Architecture:** A shared foundation owns database access, preprocessing, contracts, and immutable score caches. Ten experiment modules consume frozen `DatasetBundle` and `ScoreBundle` objects but cannot access labels or database connections. The orchestrator runs one experiment at a time, records checkpoints/dashboard state, freezes prediction hashes, and invokes a separate evaluator only after prediction completion.

**Tech Stack:** Python 3.11+, SQLite, NumPy, SciPy/scikit-learn already present in the workspace, CSV/JSON/JSONL, `unittest`, existing Exp137/Exp153/Exp177/Exp219/Exp224 artifacts.

## Global Constraints

- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`, read-only.
- External output: `/Users/minho/Documents/Dataset/exp226_235_hard_alert_quality/`.
- Repository output: `outputs/exp226_235_hard_alert_quality/`.
- Prediction code never accepts or reads TEST labels, anomaly positions, family performance, or TEST metrics.
- Dataset/family names cannot branch policy; dataset name is a lookup key only.
- Exp137 original code and results remain immutable.
- Frozen score artifacts are reused only after SHA-256, config, seed, and coverage verification.
- Every sensitivity grid is fully reported; TEST results select no operating value.
- Exp234 is research-review only and Exp235 is shadow-export only.
- Every stage is deterministic, restartable, and visible through an atomic status JSON.

---

### Task 1: Contracts, read-only database, and TRAIN-only preprocessing

**Files:**
- Create: `experiments/exp226_235_hard_alert_quality/__init__.py`
- Create: `experiments/exp226_235_hard_alert_quality/config.py`
- Create: `experiments/exp226_235_hard_alert_quality/contracts.py`
- Create: `experiments/exp226_235_hard_alert_quality/database.py`
- Create: `experiments/exp226_235_hard_alert_quality/preprocessing.py`
- Create: `experiments/exp226_235_hard_alert_quality/tests/__init__.py`
- Create: `experiments/exp226_235_hard_alert_quality/tests/test_foundation.py`

**Interfaces:**
- Produces frozen dataclasses `FrozenConfig`, `DatasetBundle`, `ScoreBundle`, and `PredictionRow`.
- Produces `audit_database(path) -> DatabaseAudit`.
- Produces label-free `load_values_bundle(name, path) -> DatasetBundle`.
- Produces separate `load_test_labels_for_evaluation(name, path) -> np.ndarray`.
- Produces `sanitize_series`, `train_target_length`, and `align_bundle`.

- [ ] **Step 1: Write failing information-boundary tests**

```python
def test_prediction_bundle_has_no_label_field():
    assert "y_test" not in DatasetBundle.__dataclass_fields__
    assert "labels" not in DatasetBundle.__dataclass_fields__

def test_values_loader_query_never_selects_label():
    source = inspect.getsource(load_values_bundle).lower()
    assert "label" not in source

def test_target_length_uses_train_rows_only():
    assert train_target_length([np.arange(8), np.arange(10)]) == 9
```

- [ ] **Step 2: Run RED**

Run: `python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_foundation`

Expected: import failure because the package does not exist.

- [ ] **Step 3: Implement contracts and database functions**

Open SQLite with URI `file:<path>?mode=ro`. Values queries select blobs and `instance_index` ordered by `instance_index`; only the evaluation function may select labels. Audit records DB SHA-256, schema, dataset count, ordering, and coverage.

- [ ] **Step 4: Implement TRAIN-only sanitization and alignment**

Replace NaN/Inf using finite values, interpolate to median TRAIN length clipped to `[8, 2048]`, and reject empty/non-finite output. TEST length cannot affect target length.

- [ ] **Step 5: Run GREEN and commit**

```bash
python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_foundation
git add experiments/exp226_235_hard_alert_quality
git commit -m "feat: add hard-alert pipeline foundation"
```

### Task 2: Immutable score cache and exact Exp137 replay

**Files:**
- Create: `experiments/exp226_235_hard_alert_quality/score_cache.py`
- Create: `experiments/exp226_235_hard_alert_quality/baseline_exp137.py`
- Create: `experiments/exp226_235_hard_alert_quality/tests/test_cache_baseline.py`

**Interfaces:**
- Produces `build_or_verify_score_bundle(dataset, config) -> ScoreBundle`.
- Produces `load_exp137_baseline() -> dict[str, BaselineLanes]`.
- Produces `verify_exp137_exact_replay(baseline) -> BaselineReplayReport`.

- [ ] **Step 1: Write failing hash and replay tests**

```python
def test_cache_rejects_modified_npz(tmp_path):
    path, manifest = write_example_cache(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with self.assertRaisesRegex(ValueError, "SHA-256"):
        load_verified_score_bundle(path, manifest)

def test_exp137_reference_totals_are_exact():
    report = verify_exp137_exact_replay(load_exp137_baseline())
    assert (report.datasets, report.hard_alerts) == (1117, 2005)
    assert (report.hard_tp, report.hard_fp) == (1691, 314)
```

- [ ] **Step 2: Run RED**

Run: `python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_cache_baseline`

Expected: missing cache/baseline functions.

- [ ] **Step 3: Implement immutable score manifest**

Map frozen sources to operational names: temporal random kernel, frequency image, texture shape, canonical normal state, and fixed dictionary shape. Store TRAIN OOF and TEST vectors with provenance. Missing coverage, vector mismatch, seed/config mismatch, or hash mismatch aborts.

- [ ] **Step 4: Implement exact baseline replay**

Parse all Exp137 lane indices, verify disjointness/range against values-only TEST lengths, reproduce per-dataset lane hashes, and assert global totals.

- [ ] **Step 5: Run GREEN and commit**

```bash
python3 -m unittest discover -s experiments/exp226_235_hard_alert_quality/tests -p 'test_*.py'
git add experiments/exp226_235_hard_alert_quality
git commit -m "feat: add immutable scores and Exp137 replay"
```

### Task 3: Exp226-229 risk and confidence modules

**Files:**
- Create: `experiments/exp226_235_hard_alert_quality/exp226_train_risk.py`
- Create: `experiments/exp226_235_hard_alert_quality/exp227_independent_confirmation.py`
- Create: `experiments/exp226_235_hard_alert_quality/exp228_margin_confidence.py`
- Create: `experiments/exp226_235_hard_alert_quality/exp229_correlation_aware_agreement.py`
- Create: `experiments/exp226_235_hard_alert_quality/tests/test_exp226_229.py`

**Interfaces:**
- Every module exports `predict_dataset(bundle, scores, baseline, config) -> list[PredictionRow]`.
- Exp226 emits TRAIN pseudo-run risk only.
- Exp227 emits dual/single/unconfirmed tiers.
- Exp228 emits every frozen margin variant.
- Exp229 emits every TRAIN-OOF correlation-weight variant.

- [ ] **Step 1: Write Exp226 failing tests**

Test Exp215 fold-concatenated p-value restoration, Exp224 fold-local p-values, original TRAIN-index alignment, and lane partition equality.

- [ ] **Step 2: Run RED, implement Exp226, run GREEN**

Run: `python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_exp226_229.Exp226Tests`.

- [ ] **Step 3: Write and implement Exp227 confirmation tests**

```python
def test_confirmation_tiers_partition_baseline_hard():
    tiers = confirmation_tiers({1, 2, 3}, {1, 2}, {1, 3})
    assert tiers.dual == {1}
    assert tiers.single == {2, 3}
    assert tiers.unconfirmed == set()
```

The tier union must equal original Exp137 Hard; Exp227 cannot add candidates.

- [ ] **Step 4: Write and implement Exp228/229 tests**

Exp228 margin is `(score - threshold) / max(abs(threshold), 1e-12)` plus distance from frozen conformal alpha. Exp229 derives Spearman correlation from aligned TRAIN OOF vectors only and rejects APIs accepting TEST vectors for weighting.

- [ ] **Step 5: Run GREEN and commit**

```bash
python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_exp226_229
git add experiments/exp226_235_hard_alert_quality
git commit -m "feat: add Exp226-229 quality audits"
```

### Task 4: Exp230-233 stability and calibration modules

**Files:**
- Create: `experiments/exp226_235_hard_alert_quality/exp230_seed_stability.py`
- Create: `experiments/exp226_235_hard_alert_quality/exp231_normal_mode_calibration.py`
- Create: `experiments/exp226_235_hard_alert_quality/exp232_calibration_reliability_gate.py`
- Create: `experiments/exp226_235_hard_alert_quality/exp233_conformal_fdr.py`
- Create: `experiments/exp226_235_hard_alert_quality/tests/test_exp230_233.py`

**Interfaces:**
- Exp230 reports support across frozen seed/kernel vectors.
- Exp231 reports all fixed TRAIN-local calibration variants.
- Exp232 emits `eligible`, `review_only`, or `abstain`.
- Exp233 applies Benjamini-Hochberg to conformal p-values for every frozen FDR level.

- [ ] **Step 1: Write RED tests and implement Exp230**

Support is invariant to seed-result input order; every stability count is reported.

- [ ] **Step 2: Write RED tests and implement Exp231**

Deterministic TRAIN-only clustering fits references before assigning TEST values. Label/family mutations cannot affect candidates.

- [ ] **Step 3: Write RED tests and implement Exp232**

Gate inputs are `n_train`, attainable p-value, OOF Wilson upper, OOF stability, and TRAIN score drift only. Passing `n_test`, family, or historical performance must fail.

- [ ] **Step 4: Write RED tests and implement Exp233**

```python
def test_bh_rejections_are_monotone():
    p = np.array([0.001, 0.01, 0.2, 0.8])
    assert bh_indices(p, q=0.05) == {0, 1}
    assert bh_indices(p, q=0.10) >= bh_indices(p, q=0.05)
```

- [ ] **Step 5: Run GREEN and commit**

```bash
python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_exp230_233
git add experiments/exp226_235_hard_alert_quality
git commit -m "feat: add Exp230-233 reliability modules"
```

### Task 5: Exp234 research rescue and Exp235 shadow export

**Files:**
- Create: `experiments/exp226_235_hard_alert_quality/exp234_precision_first_rescue.py`
- Create: `experiments/exp226_235_hard_alert_quality/exp235_shadow_export.py`
- Create: `experiments/exp226_235_hard_alert_quality/tests/test_exp234_235.py`

**Interfaces:**
- Exp234 emits `research_rescue_indices`, disjoint from every Exp137 lane and never autonomous.
- Exp235 transforms frozen evidence into a label-free shadow schema without evaluation.

- [ ] **Step 1: Write RED safety tests and implement Exp234**

Test minimum independent-source agreement, complete Exp137 exclusion, no family branch, and forced `autonomous_alert=False`. Report all source-count variants without selection.

- [ ] **Step 2: Write RED schema tests and implement Exp235**

Allowed fields are dataset key, candidate index, evidence, p-values, margins, reliability, hashes, and timestamp. TP/FP/F1/labels/anomaly positions are forbidden. Input manifests require SHA verification.

- [ ] **Step 3: Run GREEN and commit**

```bash
python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_exp234_235
git add experiments/exp226_235_hard_alert_quality
git commit -m "feat: add Exp234 rescue and Exp235 shadow export"
```

### Task 6: Connected runner, evaluator, checkpoints, and dashboard

**Files:**
- Create: `experiments/exp226_235_hard_alert_quality/run_pipeline.py`
- Create: `experiments/exp226_235_hard_alert_quality/reporting.py`
- Create: `experiments/exp226_235_hard_alert_quality/independent_reaggregate.py`
- Create: `experiments/exp226_235_hard_alert_quality/tests/test_pipeline.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Stages: `audit`, `cache`, `baseline`, `predict`, `freeze`, `evaluate`, `reaggregate`, `report`, `all`.
- Selectors: `--experiments 226-235`, `--dataset-limit`, `--workers`, `--resume`, `--reset`.
- Status: `outputs/exp226_235_hard_alert_quality/pipeline_status.json`.

- [ ] **Step 1: Write RED stage-order and hash tests**

Evaluation refuses missing/unfrozen/altered predictions. Prediction workers cannot import the label loader.

- [ ] **Step 2: Implement numeric registry and one-active execution**

Checkpoint rows include dataset, experiment, config hash, input hash, status, duration, and error. Resume only on exact hash match.

- [ ] **Step 3: Write RED dashboard tests and implement atomic status**

Expose active experiment, current dataset, complete/pending counts, heartbeat, PID, error, and artifact links. Keep Exp137 and Exp219-225 history visible. Use temporary-file replacement and treat stale heartbeat separately from failure.

- [ ] **Step 4: Implement evaluator and independent reaggregation**

Evaluation verifies prediction SHA before loading labels and keeps Hard metrics separate from review/research metrics. Reaggregation cannot import runner summary functions and must match all counts.

- [ ] **Step 5: Run GREEN and commit**

```bash
python3 -m unittest experiments.exp226_235_hard_alert_quality.tests.test_pipeline
python3 -m unittest scratch.test_rank_dashboard_operational
git add experiments/exp226_235_hard_alert_quality serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "feat: connect quality experiments and dashboard"
```

### Task 7: Smoke, full run, reports, and verification

**Files:**
- Create: `outputs/exp226_235_hard_alert_quality/SMOKE_REPORT.md`
- Create: `outputs/exp226_235_hard_alert_quality/RESULT_ANALYSIS.md`
- Create: `outputs/exp226_235_hard_alert_quality/MANIFEST.csv`
- Create: `outputs/exp226_235_hard_alert_quality/MANIFEST.md`

- [ ] **Step 1: Run unit and neighboring regression tests**

```bash
python3 -m unittest discover -s experiments/exp226_235_hard_alert_quality/tests -p 'test_*.py'
python3 -m unittest scratch/test_experiment_137_operational_triage.py experiments/exp224/test_core.py experiments/exp225/test_exp225_fixed_dictionary_confirmation.py
```

Expected: all pass before smoke.

- [ ] **Step 2: Run 10-dataset smoke**

```bash
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline --stage all --experiments 226-235 --dataset-limit 10 --workers 2 --reset
```

Expected: ten datasets per applicable experiment, zero errors, no forbidden prediction fields, exact independent aggregation.

- [ ] **Step 3: Audit smoke independently**

Hash all inputs/outputs, verify partitions/ranges and smoke baseline replay, and record exact commands and results in `SMOKE_REPORT.md`.

- [ ] **Step 4: Run full suite**

```bash
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline --stage all --experiments 226-235 --workers 7 --reset
```

Expected: 1,117 baseline datasets, zero silent exclusions, one active experiment, complete checkpoint/manifest coverage.

- [ ] **Step 5: Produce comparison without policy selection**

Report Exp137 reference, every variant, Hard TP/FP/precision/recall/mean F1, review workload, paired differences, zero-F1, and TRAIN-size/length diagnostics. Mark all results retrospective and Exp235 non-evaluative.

- [ ] **Step 6: Run final verification**

```bash
git diff --check
python3 -m unittest discover -s experiments/exp226_235_hard_alert_quality/tests -p 'test_*.py'
python3 -m experiments.exp226_235_hard_alert_quality.independent_reaggregate --verify-manifest /Users/minho/Documents/Dataset/exp226_235_hard_alert_quality/manifest/MANIFEST.json
```

Expected: no diff errors, all tests pass, every independent total and SHA matches.

- [ ] **Step 7: Commit and push**

```bash
git add experiments/exp226_235_hard_alert_quality outputs/exp226_235_hard_alert_quality serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "results: record Exp226-235 hard-alert quality suite"
git push origin experiment/exp177-185-batch
```
