# Exp266 Outside-Lane FN Recovery Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every Exp253/Exp265 lane while creating and evaluating label-free FN-recovery Shadow and Priority candidates outside the current 3,092-candidate universe.

**Architecture:** Add an isolated `experiments/exp266_fn_recovery_shadow` package. It verifies frozen Exp249/253/264 artifacts, groups MiniROCKET variants as one evidence family, freezes label-free recovery predictions, audits the same rule on TRAIN-normal OOF scores, and reads TEST labels only after prediction hash verification.

**Tech Stack:** Python 3, NumPy, SQLite, CSV/JSONL, `unittest`, existing score caches, existing rank dashboard.

## Global Constraints

- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`, read-only.
- Output: `/Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow`.
- Population: exactly 1,117 datasets; partial coverage is failure.
- Existing lane totals remain Hard 2,253, Shadow 2, Priority 31, Standard 806.
- Alpha `0.005`, seed `20260721`, full-run workers `7`.
- Evidence groups: kernel, statistical, structural, change point, masked reconstruction.
- MiniROCKET 1,024 and 4,096 seed-intersection variants count as one kernel group.
- Three or more groups: `RecoveryShadow`; exactly two: `RecoveryPriority`; zero or one: no alert.
- Quality-flagged outside candidates: `DataQualityReview` before agreement routing.
- No TEST-length budget, top-N, candidate cap K, adjacent-index merge, family branch, tail rule, or TEST-driven selection.
- Prediction receives no TEST labels, anomaly positions, or evaluation metrics.
- Exp266 does not change autonomous Hard alerts.
- Supabase/Vercel lane migration is deferred until TRAIN-risk approval and a separate user-approved plan.

---

## File Map

```text
experiments/exp266_fn_recovery_shadow/
├── __init__.py
├── config.py
├── contracts.py
├── inputs.py
├── routing.py
├── prediction.py
├── train_risk.py
├── evaluation.py
├── reporting.py
├── independent_reaggregate.py
├── run_pipeline.py
└── tests/
    ├── __init__.py
    ├── test_inputs.py
    ├── test_routing.py
    ├── test_prediction_boundary.py
    ├── test_train_risk.py
    ├── test_evaluation.py
    └── test_pipeline.py
scratch/run_exp266_fn_recovery_shadow.py
scratch/test_rank_dashboard_operational.py
serve_rank_dashboard.py
```

## Task 1: Frozen Configuration and Input Audit

**Files:**
- Create: `experiments/exp266_fn_recovery_shadow/__init__.py`
- Create: `experiments/exp266_fn_recovery_shadow/config.py`
- Create: `experiments/exp266_fn_recovery_shadow/contracts.py`
- Create: `experiments/exp266_fn_recovery_shadow/inputs.py`
- Create: `experiments/exp266_fn_recovery_shadow/tests/__init__.py`
- Create: `experiments/exp266_fn_recovery_shadow/tests/test_inputs.py`

**Interfaces:**
- Consumes: Exp253, Exp264, and Exp249 frozen prediction artifacts and manifests.
- Produces: `FrozenDatasetInputs`, `FrozenRecoveryInputs`, `load_frozen_inputs()`, `00_contract_audit.json`, and `01_source_manifest.csv`.

- [ ] **Step 1: Write failing contract tests**

```python
class FrozenInputTests(unittest.TestCase):
    def test_exact_variants_are_frozen(self):
        self.assertEqual(ALPHA, 0.005)
        self.assertEqual(MINI_VARIANTS, (
            "kernels_1024_seed_intersection",
            "kernels_4096_seed_intersection",
        ))
        self.assertEqual(EVIDENCE_GROUPS, (
            "kernel", "statistical", "structural",
            "change_point", "reconstruction",
        ))

    def test_partial_population_is_rejected(self):
        with self.assertRaisesRegex(InputContractError, "1,117"):
            validate_population(("demo",))
```

- [ ] **Step 2: Verify the tests fail before implementation**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_inputs -v
```

Expected: module import failure.

- [ ] **Step 3: Implement immutable config and contracts**

```python
EXPERIMENT = 266
EXPERIMENT_ID = "experiment_266_outside_lane_fn_recovery_shadow"
ALPHA = 0.005
SEED = 20260721
EXPECTED_DATASETS = 1117
RISK_TARGET = 0.01
MINIMUM_TRAIN = 5
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow")
EXP253_PREDICTION = Path("/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion/frozen_predictions.jsonl")
EXP264_PREDICTION = Path("/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/frozen_predictions.jsonl")
MINIROCKET_PREDICTION = Path("/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/frozen_predictions.jsonl")
MINI_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/cache")
STAT_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp252_sktime_statistical_features_safe_hurst/cache")
STRUCT_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp254_sktime_structural_features/cache")
CHANGE_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp255_change_point_features/cache")
RECON_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp257_mask_ratio_timeseries_sweep/cache")
MINI_VARIANTS = (
    "kernels_1024_seed_intersection",
    "kernels_4096_seed_intersection",
)
EVIDENCE_GROUPS = (
    "kernel", "statistical", "structural",
    "change_point", "reconstruction",
)

@dataclass(frozen=True)
class FrozenDatasetInputs:
    dataset: str
    n_train: int
    n_test: int
    hard: frozenset[int]
    shadow: frozenset[int]
    priority: frozenset[int]
    standard: frozenset[int]
    evidence: Mapping[str, frozenset[int]]
    available_groups: frozenset[str]
    quality_flags: frozenset[int]
```

- [ ] **Step 4: Implement strict loading and grouping**

```python
def group_kernel_candidates(rows_1024, rows_4096):
    if set(rows_1024) != set(rows_4096):
        raise InputContractError("MiniROCKET population mismatch")
    return {
        dataset: frozenset(rows_1024[dataset]["candidate_indices"])
                 | frozenset(rows_4096[dataset]["candidate_indices"])
        for dataset in sorted(rows_1024)
    }

def validate_population(names):
    if len(names) != EXPECTED_DATASETS or len(set(names)) != EXPECTED_DATASETS:
        raise InputContractError("frozen population must contain 1,117 unique datasets")
```

Derive baseline lanes exactly:

```python
hard = frozenset(exp253_row["final_hard_indices"])
shadow = frozenset(exp264_row["promoted_indices"])
priority = frozenset(exp264_row["final_priority_indices"])
standard = frozenset(exp264_row["final_standard_indices"])
```

Require global totals `2253/2/31/806`, union `3092`, exact SHA-256 matches, and in-range indices. OOF cache manifests must state query-excluded cross-fitting and the exact frozen source seed/config. Source absence due the frozen reconstruction `n_train >= 100` contract is `unavailable`; unexpected missing rows are fatal.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_inputs -v
git add experiments/exp266_fn_recovery_shadow
git commit -m "feat(exp266): freeze recovery input contracts"
```

Expected: Task 1 tests pass and no prior result file changes.

## Task 2: Label-Free Outside-Lane Routing

**Files:**
- Create: `experiments/exp266_fn_recovery_shadow/routing.py`
- Create: `experiments/exp266_fn_recovery_shadow/tests/test_routing.py`

**Interfaces:**
- Consumes: `FrozenDatasetInputs`.
- Produces: `route_recovery(inputs: FrozenDatasetInputs) -> RecoveryPrediction`.

- [ ] **Step 1: Write failing routing tests**

```python
def example_inputs(evidence, quality_flags=frozenset()):
    return FrozenDatasetInputs(
        dataset="demo", n_train=100, n_test=8,
        hard=frozenset({0}), shadow=frozenset({1}),
        priority=frozenset({2}), standard=frozenset({3}),
        evidence={name: frozenset(evidence.get(name, set())) for name in EVIDENCE_GROUPS},
        available_groups=frozenset(EVIDENCE_GROUPS),
        quality_flags=frozenset(quality_flags),
    )

def test_routes_only_outside_existing_union():
    inputs = example_inputs({"kernel": {0, 4}, "statistical": {4},
                             "structural": {4}})
    row = route_recovery(inputs)
    assert row.recovery_shadow == frozenset({4})
    assert row.hard == frozenset({0})

def test_quality_review_precedes_agreement_lane():
    row = route_recovery(example_inputs(
        {"kernel": {6}, "statistical": {6}, "structural": {6}},
        quality_flags={6},
    ))
    assert row.data_quality_review == frozenset({6})
    assert not row.recovery_shadow
```

- [ ] **Step 2: Verify failure**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_routing -v
```

Expected: `route_recovery` is missing.

- [ ] **Step 3: Implement routing**

```python
@dataclass(frozen=True)
class RecoveryPrediction:
    dataset: str
    hard: frozenset[int]
    shadow: frozenset[int]
    priority: frozenset[int]
    standard: frozenset[int]
    recovery_shadow: frozenset[int]
    recovery_priority: frozenset[int]
    data_quality_review: frozenset[int]
    support_by_index: Mapping[int, int]

def route_recovery(inputs: FrozenDatasetInputs) -> RecoveryPrediction:
    existing = inputs.hard | inputs.shadow | inputs.priority | inputs.standard
    outside = {
        group: indices - existing
        for group, indices in inputs.evidence.items()
        if group in inputs.available_groups
    }
    support = Counter(index for indices in outside.values() for index in indices)
    quality = frozenset(index for index in support if index in inputs.quality_flags)
    recovery_shadow = frozenset(
        index for index, count in support.items() if count >= 3 and index not in quality
    )
    recovery_priority = frozenset(
        index for index, count in support.items() if count == 2 and index not in quality
    )
    prediction = RecoveryPrediction(
        dataset=inputs.dataset,
        hard=inputs.hard, shadow=inputs.shadow,
        priority=inputs.priority, standard=inputs.standard,
        recovery_shadow=recovery_shadow,
        recovery_priority=recovery_priority,
        data_quality_review=quality,
        support_by_index=MappingProxyType(dict(sorted(support.items()))),
    )
    validate_recovery_prediction(inputs, prediction)
    return prediction
```

`validate_recovery_prediction` compares all four baseline sets for exact equality, checks all seven lanes for overlap, requires integer indices, and rejects indices outside `[0, n_test)`.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_routing -v
git add experiments/exp266_fn_recovery_shadow/routing.py experiments/exp266_fn_recovery_shadow/tests/test_routing.py
git commit -m "feat(exp266): route outside-lane recovery reviews"
```

Expected: routing tests pass.

## Task 3: Freeze Prediction Artifacts

**Files:**
- Create: `experiments/exp266_fn_recovery_shadow/prediction.py`
- Create: `experiments/exp266_fn_recovery_shadow/tests/test_prediction_boundary.py`

**Interfaces:**
- Consumes: `load_frozen_inputs()` and `route_recovery()`.
- Produces: `predict_all(output_root, dataset_limit=None)` and `03_frozen_predictions.jsonl`.

- [ ] **Step 1: Write failing boundary tests**

```python
def prediction_row(dataset):
    return {
        "dataset": dataset,
        "hard_indices": [0], "shadow_indices": [],
        "priority_indices": [], "standard_indices": [],
        "recovery_shadow_indices": [1], "recovery_priority_indices": [],
        "data_quality_review_indices": [],
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }

def example_inputs_with_n_test(n_test):
    return FrozenDatasetInputs(
        dataset="demo", n_train=100, n_test=n_test,
        hard=frozenset({0}), shadow=frozenset(), priority=frozenset(),
        standard=frozenset(),
        evidence={
            "kernel": frozenset({1}), "statistical": frozenset({1}),
            "structural": frozenset({1}), "change_point": frozenset(),
            "reconstruction": frozenset(),
        },
        available_groups=frozenset(EVIDENCE_GROUPS),
        quality_flags=frozenset(),
    )

def test_prediction_has_no_evaluation_fields(tmp_path):
    path = freeze_rows(tmp_path, [prediction_row("a")])
    row = json.loads(path.read_text().splitlines()[0])
    forbidden = {"y_test", "tp", "fp", "fn", "precision", "recall", "f1"}
    assert not forbidden.intersection(key.lower() for key in row)
    assert row["contains_test_labels"] is False

def test_hash_is_input_order_invariant(tmp_path):
    left = freeze_rows(tmp_path / "left", [prediction_row("b"), prediction_row("a")])
    right = freeze_rows(tmp_path / "right", [prediction_row("a"), prediction_row("b")])
    assert sha256_file(left) == sha256_file(right)

def test_appending_unselected_test_rows_keeps_prefix_routing():
    short = route_recovery(example_inputs_with_n_test(8))
    long = route_recovery(example_inputs_with_n_test(12))
    assert short.recovery_shadow == long.recovery_shadow

def test_prediction_api_has_no_label_parameter():
    assert "y_test" not in inspect.signature(predict_all).parameters
```

- [ ] **Step 2: Verify failure**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_prediction_boundary -v
```

Expected: prediction functions are missing.

- [ ] **Step 3: Implement atomic sorted JSONL freeze**

```python
def freeze_rows(output_root, rows):
    ordered = sorted(rows, key=lambda row: str(row["dataset"]))
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in ordered
    )
    path = output_root / "03_frozen_predictions.jsonl"
    atomic_write(path, payload)
    atomic_json(output_root / "04_prediction_manifest.json", {
        "experiment_id": EXPERIMENT_ID,
        "rows": len(ordered),
        "datasets": len(ordered),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
    })
    return path
```

Rows include immutable baseline lanes, recovery lanes, support sources, source availability, quality state, and input/config hashes.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_prediction_boundary -v
git add experiments/exp266_fn_recovery_shadow/prediction.py experiments/exp266_fn_recovery_shadow/tests/test_prediction_boundary.py
git commit -m "feat(exp266): freeze label-free recovery predictions"
```

Expected: boundary and deterministic-hash tests pass.

## Task 4: TRAIN-Normal OOF Combined-Policy Risk Audit

**Files:**
- Create: `experiments/exp266_fn_recovery_shadow/train_risk.py`
- Create: `experiments/exp266_fn_recovery_shadow/tests/test_train_risk.py`

**Interfaces:**
- Consumes: cache arrays keyed by `sha256(dataset)[:20]` from Exp249, Exp252, Exp254, Exp255, and Exp257.
- Produces: `audit_train_risk()` and `02_train_normal_risk_audit.csv`.

- [ ] **Step 1: Write failing leakage and abstention tests**

```python
def test_loo_pvalue_excludes_query():
    scores = np.array([1.0, 2.0, 10.0])
    assert loo_tail_pvalues(scores).tolist() == [1.0, 2 / 3, 1 / 3]

def test_n_train_below_five_abstains():
    arrays = {
        "kernel_1024": np.ones((2, 4)),
        "kernel_4096": np.ones((2, 4)),
        "statistical": np.ones(4),
        "structural": np.ones(4),
        "change_point": np.ones(4),
    }
    row = audit_dataset_train_risk("demo", n_train=4, source_arrays=arrays)
    assert row.abstained is True
    assert row.normal_runs == 0
```

- [ ] **Step 2: Verify failure**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_train_risk -v
```

Expected: risk functions are missing.

- [ ] **Step 3: Implement leave-one-out tail p-values and seed stability**

```python
def loo_tail_pvalues(scores):
    values = np.asarray(scores, dtype=float)
    result = np.ones(values.shape[0], dtype=float)
    for index, query in enumerate(values):
        reference = np.delete(values, index)
        finite = reference[np.isfinite(reference)]
        if np.isfinite(query) and finite.size:
            result[index] = (1 + np.count_nonzero(finite >= query)) / (finite.size + 1)
    return result

def seed_intersection_flags(scores_by_seed, alpha):
    flags = [loo_tail_pvalues(seed_scores) <= alpha for seed_scores in scores_by_seed]
    return np.logical_and.reduce(flags)
```

Use exact arrays:

```python
def by_seed_flags(archive, key):
    return seed_intersection_flags(archive[key], ALPHA)

kernel = by_seed_flags(mini_npz, "kernels_1024__train_by_seed") \
       | by_seed_flags(mini_npz, "kernels_4096__train_by_seed")
statistical = loo_tail_pvalues(stat_npz["combined_40__train_oof"]) <= ALPHA
structural = loo_tail_pvalues(struct_npz["structural_combined_440__train_oof"]) <= ALPHA
change_point = loo_tail_pvalues(change_npz["change_point_combined_33__train_oof"]) <= ALPHA
reconstruction = (
    by_seed_flags(recon_npz, "mask_025__train_by_seed")
    if n_train >= 100 else None
)
```

Require equal TRAIN array lengths and document that the calibration reference combines query-excluded OOF scores produced by different fold models. Count `>=3` and `==2` false review requests and calculate one-sided 95% Wilson upper bounds against `0.01`. Never open the TEST split or label column.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_train_risk -v
git add experiments/exp266_fn_recovery_shadow/train_risk.py experiments/exp266_fn_recovery_shadow/tests/test_train_risk.py
git commit -m "feat(exp266): audit recovery policy on TRAIN normal OOF scores"
```

Expected: risk tests pass; mismatched cache arrays fail explicitly.

## Task 5: Post-Freeze Evaluation, Reporting, and Independent Reaggregation

**Files:**
- Create: `experiments/exp266_fn_recovery_shadow/evaluation.py`
- Create: `experiments/exp266_fn_recovery_shadow/reporting.py`
- Create: `experiments/exp266_fn_recovery_shadow/independent_reaggregate.py`
- Create: `experiments/exp266_fn_recovery_shadow/tests/test_evaluation.py`

**Interfaces:**
- Consumes: frozen prediction and verified manifest hash.
- Produces: `05_dataset_results.csv`, `06_global_summary.csv`, `07_source_agreement_diagnostics.csv`, `08_invariance_report.md`, `09_result_report.md`, `10_independent_reaggregation.json`, and `MANIFEST.csv`.

- [ ] **Step 1: Write failing tamper and metric-separation tests**

```python
def write_frozen_fixture(root):
    prediction = root / "03_frozen_predictions.jsonl"
    prediction.write_text(json.dumps({
        "dataset": "demo", "hard_indices": [0],
        "recovery_shadow_indices": [1], "recovery_priority_indices": [],
        "contains_test_labels": False, "contains_evaluation_metrics": False,
    }, sort_keys=True) + "\n")
    manifest = root / "04_prediction_manifest.json"
    manifest.write_text(json.dumps({"prediction_sha256": sha256_file(prediction)}))
    return prediction, manifest

def test_evaluation_refuses_tampered_prediction(self):
    with tempfile.TemporaryDirectory() as directory:
        prediction, manifest = write_frozen_fixture(Path(directory))
        prediction.write_text(prediction.read_text() + "{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            evaluate_predictions(prediction, manifest)

def test_recovery_review_never_changes_hard(self):
    summary = aggregate([{
        "hard_alerts": 1, "baseline_hard_alerts": 1,
        "hard_tp": 1, "hard_fp": 0,
        "recovery_shadow_count": 1,
        "recovery_shadow_tp": 1, "recovery_shadow_fp": 0,
    }])
    self.assertEqual(summary["hard_alert_delta"], 0)
    self.assertEqual(summary["hard_alerts"], summary["baseline_hard_alerts"])
    self.assertEqual(summary["recovery_shadow_count"], 1)
```

- [ ] **Step 2: Verify failure**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_evaluation -v
```

Expected: evaluation functions are missing.

- [ ] **Step 3: Implement read-only TEST-label evaluation**

```python
def labels_for_dataset(connection, dataset):
    rows = connection.execute(
        """SELECT i.instance_index, CAST(i.label AS INTEGER)
           FROM instances i JOIN datasets d ON d.id=i.dataset_id
           WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index""",
        (dataset,),
    )
    return {int(index): int(label) for index, label in rows}

def lane_metrics(indices, labels):
    selected = set(indices)
    tp = sum(labels[index] for index in selected)
    fp = len(selected) - tp
    return {"count": len(selected), "tp": tp, "fp": fp,
            "precision": tp / len(selected) if selected else 0.0}
```

Open SQLite with `mode=ro`; verify prediction SHA-256 before the first label query. Report Hard and human-assisted lanes separately.

- [ ] **Step 4: Implement independent summary verification**

```python
def numerically_equal(stored, recomputed):
    try:
        return math.isclose(float(stored), float(recomputed), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return stored == recomputed

def verify_summary(results_path, summary_path):
    recomputed = aggregate_csv(results_path)
    stored = next(csv.DictReader(summary_path.open(newline="")))
    mismatches = {
        key: {"stored": stored[key], "recomputed": recomputed[key]}
        for key in recomputed if not numerically_equal(stored[key], recomputed[key])
    }
    return {"passed": not mismatches, "mismatches": mismatches}
```

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_evaluation -v
git add experiments/exp266_fn_recovery_shadow/evaluation.py experiments/exp266_fn_recovery_shadow/reporting.py experiments/exp266_fn_recovery_shadow/independent_reaggregate.py experiments/exp266_fn_recovery_shadow/tests/test_evaluation.py
git commit -m "feat(exp266): evaluate and reaggregate recovery lanes"
```

Expected: tamper detection, metric separation, and reaggregation tests pass.

## Task 6: Resumable Pipeline, Queue, and Local Dashboard

**Files:**
- Create: `experiments/exp266_fn_recovery_shadow/run_pipeline.py`
- Create: `experiments/exp266_fn_recovery_shadow/tests/test_pipeline.py`
- Create: `scratch/run_exp266_fn_recovery_shadow.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: Task 1-5 stage functions.
- Produces: `audit|predict|risk|evaluate|report|all` CLI, queue state, heartbeat, dashboard catalog.

- [ ] **Step 1: Write failing CLI and dashboard tests**

```python
def test_cli_exposes_every_stage():
    parser = build_parser()
    for stage in ("audit", "predict", "risk", "evaluate", "report", "all"):
        assert parser.parse_args(["--stage", stage]).stage == stage

def test_dashboard_registers_exp266():
    info = dash.EXPERIMENTS["experiment_266_outside_lane_fn_recovery_shadow"]
    assert info["summary_csv"].name == "06_global_summary.csv"
```

- [ ] **Step 2: Verify failure**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_pipeline -v
python3 scratch/test_rank_dashboard_operational.py
```

Expected: Exp266 CLI and dashboard registration are missing.

- [ ] **Step 3: Implement ordered stages**

```python
STAGE_ORDER = ("audit", "predict", "risk", "evaluate", "report")

def run_stage(stage, output_root, dataset_limit):
    actions = {
        "audit": lambda: audit_inputs(output_root, dataset_limit),
        "predict": lambda: predict_all(output_root, dataset_limit),
        "risk": lambda: write_train_risk_audit(output_root, dataset_limit),
        "evaluate": lambda: evaluate_predictions(output_root),
        "report": lambda: write_reports(output_root),
    }
    actions[stage]()

def run_all(output_root, dataset_limit):
    for stage in STAGE_ORDER:
        run_stage(stage, output_root, dataset_limit)
```

Status JSON records stage, processed/expected, current dataset, elapsed time, recent rate, ETA, warnings, errors, PID, workers, output path, and log path. Process datasets by ascending `n_train + n_test`, then sequence length, then name.

- [ ] **Step 4: Implement sequential runner and dashboard entry**

```python
command = [
    sys.executable, "-m", "experiments.exp266_fn_recovery_shadow.run_pipeline",
    "--stage", "all", "--workers", "7", "--output-root", str(OUTPUT_ROOT),
]

EXPERIMENTS["experiment_266_outside_lane_fn_recovery_shadow"] = {
    "label": "Exp266 - lane 밖 FN 복구 Shadow",
    "detail_csv": OUTPUT_ROOT / "05_dataset_results.csv",
    "summary_csv": OUTPUT_ROOT / "06_global_summary.csv",
    "status_json": OUTPUT_ROOT / "status.json",
    "stdout_log": OUTPUT_ROOT / "full.log",
}
```

The runner acquires the existing global sequential lock, forwards SIGTERM, writes `queued/running/completed/failed`, and never converts missing datasets into success.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest experiments.exp266_fn_recovery_shadow.tests.test_pipeline -v
python3 scratch/test_rank_dashboard_operational.py
git add experiments/exp266_fn_recovery_shadow/run_pipeline.py experiments/exp266_fn_recovery_shadow/tests/test_pipeline.py scratch/run_exp266_fn_recovery_shadow.py serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "feat(exp266): add resumable runner and dashboard status"
```

Expected: pipeline and dashboard suites pass.

## Task 7: Smoke, Full Run, and Result Verification

**Files:**
- Generate: `/Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow/*`
- Generate: `outputs/exp266_outside_lane_fn_recovery_shadow/*`
- Modify code only when a reproduced test exposes a defect.

**Interfaces:**
- Consumes: complete Exp266 package.
- Produces: verified smoke, 1,117-dataset run, reports, and dashboard-visible state.

- [ ] **Step 1: Run syntax and complete unit tests**

```bash
python3 -m py_compile experiments/exp266_fn_recovery_shadow/*.py scratch/run_exp266_fn_recovery_shadow.py
python3 -m unittest discover -s experiments/exp266_fn_recovery_shadow/tests -v
python3 scratch/test_rank_dashboard_operational.py
```

Expected: zero failures and errors.

- [ ] **Step 2: Run isolated 10-dataset smoke**

```bash
python3 -m experiments.exp266_fn_recovery_shadow.run_pipeline \
  --stage all --dataset-limit 10 --workers 2 \
  --output-root /Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow_smoke
```

Expected: 10 prediction rows, forbidden fields 0, overlaps 0, index errors 0, and reaggregation pass.

- [ ] **Step 3: Run full experiment sequentially**

```bash
python3 scratch/run_exp266_fn_recovery_shadow.py
```

Expected: one active process, 1,117/1,117 coverage, errors 0, terminal state `completed`.

- [ ] **Step 4: Verify immutable baseline and risk contract**

```bash
python3 -m experiments.exp266_fn_recovery_shadow.independent_reaggregate \
  --output-root /Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow
```

Expected:

```text
Hard=2253, Shadow=2, Priority=31, Standard=806
baseline lane delta=0
prediction label fields=0
prediction evaluation fields=0
missing datasets=0
index errors=0
lane overlaps=0
independent reaggregation=PASS
```

Also run label-permutation, anomaly-position mutation, dataset display-name remapping, input-row permutation, and repeat-hash invariance checks. Write each check with `pass/fail`, input hash, output hash, and command in `08_invariance_report.md`.

Record whether `>=3` and `==2` policies meet the one-sided 95% Wilson upper-bound target `0.01`. Do not change alpha, source groups, agreement counts, or target after reading results.

- [ ] **Step 5: Commit repository-sized summaries only**

```bash
git add outputs/exp266_outside_lane_fn_recovery_shadow
git commit -m "results(exp266): record FN recovery shadow validation"
```

Do not add DB files, caches, large raw JSONL, logs, checkpoints, or `.superpowers/brainstorm/`.

## Deferred Prospective Integration Gate

Do not migrate Supabase or deploy new Vercel lanes in this plan. Start a separate design only when Exp266-B has complete coverage, the 3+ policy meets the frozen TRAIN-normal risk contract, existing lanes are unchanged, a label-free prospective feed exists, and the user explicitly approves the remote schema change.
