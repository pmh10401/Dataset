# Exp449 Residual FN Method Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-use, TRAIN-only method-qualification experiment that evaluates fixed ROCKET/Hydra representations on the eight Exp448 residual-FN target families and qualifies a candidate only when global FN falls below 1,988 while global FP remains at or below 314.

**Architecture:** Exp449 descriptor-pins the immutable Exp448 result and source database, audits the fixed aeon 1.4.0 method surface, transforms TRAIN/TEST values without labels, applies one shared five-fold LocalGap score head, and freezes every candidate prediction before opening one ordered label session. A source-free replay validates the final publication, while a sanitized aggregate snapshot feeds the existing owner-only dashboard without coupling dashboard availability to scientific execution.

**Tech Stack:** Python 3.14.6; aeon 1.4.0; NumPy 2.3.5; scikit-learn 1.7.2; SQLite read-only descriptor boundary; `unittest`; canonical JSON/JSONL; owner-only Sites dashboard; git; versioned HTML/JSON/lock code map.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-12-exp449-residual-fn-method-qualification-antigravity-design.md`, SHA256 `7af33c92f80ed05be40b10112ece018d157bcc4493e1ecb11bca357c3db114e6`.
- Experiment identity is `Exp449`; schema is `exp449`; kind is `residual_fn_method_qualification`; run identity is `exp449-residual-fn-method-qualification-antigravity-20260812`.
- Run root is `/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812`; fixed failure audit path is `/Users/mac/Documents/Dataset/exp449_failure_audit_antigravity.json`.
- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Exp448 remains byte-immutable. Never rerun or modify its official lifecycle, root, artifacts, truth, or published dashboard payload.
- Baseline is exact Exp448 stable-26: TP 1,930; FP 314; FN 1,988; TN 191,668; F1 `0.6264199935`.
- A candidate qualifies only when global FN is below 1,988, global FP is at most 314, global F1 is at least `0.6264199935`, and target-family FN decreases.
- The target families and candidate pairs are exact and ordered as specified in the approved design. Do not add a method, family, seed, alpha, threshold, or hyperparameter after observing TEST results.
- Performance candidates are exactly `minirocket_local_gap`, `multirocket_local_gap`, `hydra_local_gap`, and `hydra_multirocket_local_gap`. Classifier-only literature methods remain compatibility evidence, not anomaly-score candidates.
- Calibration is exact: seeds `(20260721, 20261721)`, elementwise median TRAIN-OOF and TEST scores across seeds, empirical upper-tail p-values, alpha `0.005`, and candidate condition `p <= 0.005`. No other alpha is a performance candidate.
- Transform settings are exact: MiniROCKET 4,096 requested kernels; MultiROCKET 1,024 requested kernels, 32 maximum dilations, four features per kernel, no normalization; Hydra eight kernels and 64 groups. Hydra+MultiROCKET uses the same Hydra and MultiROCKET settings.
- Prediction is label-free. Freeze all candidate predictions and their inventory before the sole ordered TEST-label session.
- The sole label session queries exactly the 48 frozen target datasets in Exp448 order and consumes exactly 72,300 rows. The remaining 1,069 datasets are represented only by immutable Exp448 aggregate confusion.
- Official commands are single-use and ordered: readiness, predict, evaluate, inspect. Any nonzero command stops Exp449 permanently; do not retry, resume, or override.
- The fixed prediction interpreter is `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python`. Do not alter its package set.
- The aeon environment has no MLX or mlx-metal. Record the failed availability gate and retain the official CPU transformers; do not install packages or rewrite them for GPU.
- CPU uses at most seven outer dataset workers and every transformer uses `n_jobs=1`. Never run duplicate heavy processes.
- Dashboard data is aggregate-only. Never export raw TEST labels, instance indices, internal dataset names/lists, per-instance rows, hashes, local paths, logs, database details, checkpoints, or secrets.
- Dashboard publication failure changes no scientific artifact and authorizes no experiment rerun.
- At every code-changing task start, compare the current tree with the versioned map lock and answer callers, effects, and tests. When the boundary changes, commit `codemap_exp449_antigravity.html/json/lock` with the same task.
- TDD is mandatory: observe the focused RED, implement the minimum GREEN, rerun focused tests, run specified predecessor regressions, validate the map, then commit.
- A fresh commitment-boundary review is required after every bounded task. A fresh final review must return `ship` on the exact clean candidate before official commands.

---

## File Structure

### Governance and map

- `docs/EXP449_PREREGISTRATION_antigravity_KO.md`: frozen method matrix, thresholds, seeds, label boundary, and no-promotion contract.
- `docs/EXP449_RUNBOOK_antigravity_KO.md`: exact commands, stop rules, dashboard checkpoints, and recovery prohibition.
- `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`: append-only implementation and official evidence summary.
- `docs/codemap/codemap_exp449_antigravity.html`: human-viewable versioned map.
- `docs/codemap/codemap_exp449_antigravity.json`: machine-readable nodes, edges, flows, evidence, and unknowns.
- `docs/codemap/codemap_exp449_antigravity.lock`: functional commit, clean-tree state, and deterministic fingerprints.
- `scratch/build_codemap_exp449_antigravity.py`: deterministic map triplet generator/validator.

### Product package

- `experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py`: fixed identity, baseline, target-family matrix, method matrix, seeds, score settings, environment, and document hashes.
- `experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py`: exact typed immutable authorities and result records.
- `experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py`: canonical bytes, create-only writes, inventories, and artifact maps.
- `experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py`: exact aeon implementation/interface/source audit.
- `experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py`: descriptor-pinned Exp448 publication and metric authority.
- `experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py`: descriptor-pinned label-free database bundles and one-use label session.
- `experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py`: four fixed aeon representation adapters.
- `experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py`: exact shared LocalGap model.
- `experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py`: deterministic five-fold OOF representation-to-score pipeline.
- `experiments/exp449_residual_fn_method_qualification_antigravity/mlx_gate_antigravity.py`: explicit MLX availability/parity/speed decision record.
- `experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py`: target-family prediction freezer and manifest.
- `experiments/exp449_residual_fn_method_qualification_antigravity/progress_antigravity.py`: durable progress/PID/event state and sanitized status payload.
- `experiments/exp449_residual_fn_method_qualification_antigravity/bootstrap_antigravity.py`: fixed paired dataset bootstrap.
- `experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py`: sole label snapshot and counterfactual metrics.
- `experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py`: source-free semantic replay of the final publication.
- `experiments/exp449_residual_fn_method_qualification_antigravity/dashboard_export_antigravity.py`: strict aggregate-only status/result projection.
- `experiments/exp449_residual_fn_method_qualification_antigravity/readiness_antigravity.py`: clean reviewed commit and preflight gate.
- `experiments/exp449_residual_fn_method_qualification_antigravity/cli_antigravity.py`: exact four-command lazy CLI.
- `scratch/run_exp449_residual_fn_method_qualification_antigravity.py`: outside-CWD launcher with seven thread environment variables set to `1` before imports.

### Tests

- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_config_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_contracts_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_artifacts_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_method_audit_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_predecessor_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_sources_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_representations_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_scoring_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_mlx_gate_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_prediction_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_progress_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_evaluation_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_replay_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_dashboard_export_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_readiness_antigravity.py`
- `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_cli_antigravity.py`

---

### Task 1: Governance Freeze and Reproducible Exp449 Code Map

**Files:**
- Create: `docs/EXP449_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP449_RUNBOOK_antigravity_KO.md`
- Create: `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`
- Create: `docs/codemap/codemap_exp449_antigravity.html`
- Create: `docs/codemap/codemap_exp449_antigravity.json`
- Create: `docs/codemap/codemap_exp449_antigravity.lock`
- Create: `scratch/build_codemap_exp449_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`

**Interfaces:**
- Consumes: approved design SHA `7af33c92f80ed05be40b10112ece018d157bcc4493e1ecb11bca357c3db114e6`; Exp448 truth at current clean HEAD.
- Produces: final design/preregistration/runbook SHA bindings; planned Exp449 truth with label access `0`; deterministic `build_codemap_exp449_antigravity.py --source-commit <commit>`.

- [ ] **Step 1: Confirm the clean starting boundary and current map convention**

Run:

```bash
git status --short
git rev-parse HEAD
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.lock >/dev/null
```

Expected: empty status; HEAD descends from `e6192f636a0a1e7db9649705621cb8730cd18ae0`; Exp448 map and lock parse; lock remains the normal clean-functional-parent/map-child binding.

- [ ] **Step 2: Write the failing governance test**

```python
class Exp449GovernanceTest(unittest.TestCase):
    def test_planned_truth_and_global_constraints_are_exact_antigravity(self):
        prereg = Path("docs/EXP449_PREREGISTRATION_antigravity_KO.md").read_text()
        self.assertIn("global_fp_max = 314", prereg)
        self.assertIn("global_fn_must_be_below = 1988", prereg)
        self.assertIn("official_label_sessions = 1", prereg)
        state = strict_json_antigravity(Path("RESEARCH_STATE.json"))
        self.assertEqual(state["active_experiment"], "Exp449")
        self.assertEqual(state["exp449"]["label_access_count"], 0)
        self.assertFalse(state["exp449"]["official_readiness_started"])
        self.assertEqual(state["pending_candidate"], EXPECTED_PENDING_CANDIDATE)
```

- [ ] **Step 3: Run the test and observe RED**

Run:

```bash
python3 -m unittest experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_governance_antigravity -v
```

Expected: FAIL because the Exp449 governance documents and ledger records do not exist.

- [ ] **Step 4: Write the frozen governance documents and ledger records**

Use these exact preregistered values:

```text
families = Crop, ElectricDevices, FordA, FordB, Yoga, ECG5000, ChlorineConcentration, StarLightCurves
candidates = minirocket_local_gap, multirocket_local_gap, hydra_local_gap, hydra_multirocket_local_gap
crossfit_folds = 5
knn_neighbors = 3
max_train_reference = 1000
query_chunk_size = 128
seeds = 20260721, 20261721
alpha = 0.005
seed_aggregation = elementwise_median
calibration = empirical_upper_pvalue_lte_alpha
minirocket_requested_kernels = 4096
multirocket_requested_kernels = 1024
multirocket_max_dilations_per_kernel = 32
multirocket_features_per_kernel = 4
multirocket_normalise = false
hydra_kernels = 8
hydra_groups = 64
global_fn_must_be_below = 1988
global_fp_max = 314
global_f1_min = 0.6264199935
official_label_sessions = 1
promotion = prohibited
routing = prohibited
```

The planned ledger record must use status `planned_not_started`, null prediction/evaluation/decision hashes, label access `0`, and must not change champion or pending candidate.

- [ ] **Step 5: Implement the deterministic map generator**

Expose:

```python
def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    """Return JSON graph, lock, and self-contained HTML for the same graph."""

def validate_map_antigravity(graph: object, lock: object, html: str) -> None:
    """Reject missing evidence, unresolved refs, >20 primary nodes, and fingerprint drift."""
```

The initial map must mark product boundaries as unknown future relations, not existing calls. It must include governance, Exp448 predecessor, future method audit, future score/prediction, future evaluation/replay, future CLI, and dashboard presentation nodes.

- [ ] **Step 6: Compute final document hashes and regenerate the triplet together**

Run:

```bash
sha256sum docs/superpowers/specs/2026-08-12-exp449-residual-fn-method-qualification-antigravity-design.md docs/EXP449_PREREGISTRATION_antigravity_KO.md docs/EXP449_RUNBOOK_antigravity_KO.md
python3 scratch/build_codemap_exp449_antigravity.py --source-commit "$(git rev-parse HEAD)"
```

Write the measured hashes into the preregistration, runbook, report, and ledgers. Do not invent a reviewed implementation commit.

- [ ] **Step 7: Run governance and map GREEN checks**

Run:

```bash
python3 -m unittest experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_governance_antigravity -v
python3 scratch/build_codemap_exp449_antigravity.py --validate-only
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
git diff --check
```

Expected: all pass; map HTML embeds the exact JSON graph; no source edge claims product implementation exists.

- [ ] **Step 8: Commit Task 1**

```bash
git add docs/EXP449_PREREGISTRATION_antigravity_KO.md docs/EXP449_RUNBOOK_antigravity_KO.md docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock scratch/build_codemap_exp449_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py EXPERIMENT_CATALOG.json RESEARCH_STATE.json CHAMPION_LEDGER.json
git commit -m "docs(exp449): freeze residual FN qualification governance"
```

---

### Task 2: Candidate Registry, Contracts, Artifacts, and Official Method Audit

**Files:**
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_contracts_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_artifacts_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_method_audit_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: Task 1 document hashes and fixed aeon interpreter.
- Produces: `FROZEN_CONFIG_ANTIGRAVITY`; `CANDIDATE_MATRIX_ANTIGRAVITY`; immutable `MethodAuditAntigravity`; canonical artifact helpers.

- [ ] **Step 1: Answer map questions before source edits**

Run `python3 scratch/build_codemap_exp449_antigravity.py --validate-only`, then record in the Task 2 report: callers are future prediction/readiness; effects are candidate inventory and environment authority; tests are the four Task 2 modules.

- [ ] **Step 2: Write RED tests for exact registry and types**

```python
def test_candidate_matrix_is_exact_antigravity(self):
    self.assertEqual(
        CANDIDATE_MATRIX_ANTIGRAVITY,
        (
            ("Crop", ("minirocket_local_gap", "multirocket_local_gap")),
            ("ElectricDevices", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
            ("FordA", ("minirocket_local_gap", "hydra_local_gap")),
            ("FordB", ("multirocket_local_gap", "hydra_local_gap")),
            ("Yoga", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
            ("ECG5000", ("multirocket_local_gap", "hydra_local_gap")),
            ("ChlorineConcentration", ("minirocket_local_gap", "multirocket_local_gap")),
            ("StarLightCurves", ("multirocket_local_gap", "hydra_local_gap")),
        ),
    )

def test_bool_is_not_an_integer_metric_antigravity(self):
    with self.assertRaisesRegex(ValueError, "fn must be an exact integer"):
        ConfusionAntigravity(tp=1, fp=0, fn=True, tn=3)
```

Artifact tests must reject duplicate JSON keys, non-canonical bytes, symlinks, an extra inventory leaf, an existing destination, and a bool where an exact integer is required.

- [ ] **Step 3: Run RED**

```bash
python3 -m unittest \
  experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_config_antigravity \
  experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_contracts_antigravity \
  experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_artifacts_antigravity \
  experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_method_audit_antigravity -v
```

Expected: import errors because the four source modules are absent.

- [ ] **Step 4: Implement immutable contracts and canonical artifacts**

Define exact frozen/slots dataclasses:

```python
@dataclass(frozen=True, slots=True)
class MethodCandidateAntigravity:
    name: str
    representation: str
    seeds: tuple[int, ...]
    alphas: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class MethodAuditAntigravity:
    name: str
    status: Literal["score_compatible", "classifier_only", "unavailable"]
    module: str
    symbol: str
    package_version: str | None
    source_sha256: str | None
    methods: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ConfusionAntigravity:
    tp: int
    fp: int
    fn: int
    tn: int

@dataclass(frozen=True, slots=True)
class DatasetBundleAntigravity:
    dataset: str
    train_normal: np.ndarray
    test_values: np.ndarray
    target_length: int

@dataclass(frozen=True, slots=True)
class DatasetBaselineAntigravity:
    dataset_order: int
    family: str
    stable_candidates: frozenset[int]
    confusion: ConfusionAntigravity

@dataclass(frozen=True, slots=True)
class CandidateMetricAntigravity:
    family: str
    method: str
    target_confusion: ConfusionAntigravity
    global_confusion: ConfusionAntigravity
    target_fn_delta: int
    global_f1: float
    qualified: bool
```

Expose `canonical_json_bytes_antigravity`, `canonical_jsonl_bytes_antigravity`, `strict_json_bytes_antigravity`, `write_create_only_antigravity`, and `artifact_map_bytes_antigravity`.

- [ ] **Step 5: Implement the exact method audit**

Audit these symbols from the fixed interpreter without importing them in readiness or inspect:

```python
AUDIT_SPECS_ANTIGRAVITY = (
    ("MiniRocket", "aeon.transformations.collection.convolution_based", "score_compatible"),
    ("MultiRocket", "aeon.transformations.collection.convolution_based", "score_compatible"),
    ("HydraTransformer", "aeon.transformations.collection.convolution_based", "score_compatible"),
    ("WEASEL_V2", "aeon.classification.dictionary_based", "classifier_only"),
    ("DrCIFClassifier", "aeon.classification.interval_based", "classifier_only"),
    ("HIVECOTEV2", "aeon.classification.hybrid", "classifier_only"),
    ("InceptionTimeClassifier", "aeon.classification.deep_learning", "classifier_only"),
    ("TSChiefClassifier", "aeon.classification.hybrid", "unavailable"),
)
```

Require Python 3.14.6, aeon 1.4.0, NumPy 2.3.5, scikit-learn 1.7.2, and SHA256 of each available class source file. Reject a class whose public methods differ from the frozen audit.

- [ ] **Step 6: Run GREEN twice and update the map**

Run the four-module command from Step 3 twice. Regenerate the map against the clean functional source snapshot and require real method-audit nodes/edges/tests, no stale unknown relation for Task 2, exact HTML/JSON equality, valid references, and matching fingerprints.

- [ ] **Step 7: Commit Task 2**

```bash
git add experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_config_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_contracts_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_artifacts_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_method_audit_antigravity.py docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock
git commit -m "feat(exp449): freeze method qualification contracts"
```

---

### Task 3: Descriptor-Pinned Exp448 and Label-Free Data Authority

**Files:**
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_predecessor_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_sources_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: Exp448 immutable 11-leaf evaluation; Exp446 immutable stable-26 prediction publication; source DB SHA/inode/nlink; Task 2 contracts.
- Produces: `capture_exp448_authority_antigravity() -> Exp448AuthorityAntigravity`; `load_target_bundle_antigravity(dataset: str) -> DatasetBundleAntigravity`; `open_test_labels_once_antigravity(...)` for Task 6 only.

- [ ] **Step 1: Map preflight and read-only authority check**

Verify current map, Exp448 root inventory, DB SHA/inode/nlink, and absence of the Exp449 root and failure audit. Do not issue label SQL.

- [ ] **Step 2: Write adversarial predecessor RED tests**

```python
def test_target_family_authority_is_exact_antigravity(self):
    authority = capture_exp448_authority_antigravity(root=self.fixture_root)
    self.assertEqual(authority.population, 1117)
    self.assertEqual(authority.target_dataset_count, 48)
    self.assertEqual(authority.target_family_fn, (246, 200, 88, 75, 63, 56, 56, 53))
    self.assertEqual(authority.baseline_confusion, ConfusionAntigravity(1930, 314, 1988, 191668))

def test_leaf_swap_after_read_is_rejected_antigravity(self):
    with self.assertRaisesRegex(ValueError, "Exp448 leaf identity changed"):
        capture_with_swap_antigravity(self.fixture_root, "dataset_metrics_antigravity.jsonl")
```

Cover symlink leaf/root, hard-linked leaf, extra/missing leaf, duplicate JSON key, same-size overwrite, directory replacement, false dataset order, wrong family prefix membership, and baseline sum mismatch.

- [ ] **Step 3: Write label-firewall and DB RED tests**

```python
def test_values_bundle_never_selects_label_antigravity(self):
    connection = RecordingConnectionAntigravity()
    load_target_bundle_antigravity("Crop_normal_1", connection_factory=lambda _: connection)
    self.assertFalse(any("label" in sql.lower() for sql in connection.sql))

def test_label_session_requires_frozen_prediction_authority_antigravity(self):
    with self.assertRaisesRegex(ValueError, "prediction authority is invalid"):
        open_test_labels_once_antigravity(forged_prediction_authority_antigravity())
```

Cover database SHA mismatch, inode replacement, nlink >1, path symlink, metadata drift, invalid exact TEXT label type, noncontiguous index, query count drift, and truthful close failure.

- [ ] **Step 4: Run RED**

```bash
python3 -m unittest \
  experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_predecessor_antigravity \
  experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_sources_antigravity -v
```

Expected: import errors for absent modules.

- [ ] **Step 5: Implement one descriptor transaction for Exp448**

Pin the Exp448 root and every required leaf with `O_NOFOLLOW`, require regular single-link leaves, read bytes from retained descriptors, parse strict canonical JSON/JSONL, recompute the artifact map, validate 1,117 ordered datasets and the eight fixed prefix groups, then revalidate descriptor and namespace identities before return. Independently call Exp448's source-free `capture_exp446_publication_antigravity()` twice, require distinct but value-identical authorities, and bind the exact stable-26 candidate coordinates used to construct each `DatasetBaselineAntigravity`.

Return only immutable values:

```python
@dataclass(frozen=True, slots=True)
class Exp448AuthorityAntigravity:
    root_identity: tuple[int, int, int, int, int, int, int]
    artifact_map_sha256: str
    evaluation_sha256: str
    decision_sha256: str
    population: int
    target_dataset_count: int
    target_dataset_order: tuple[int, ...]
    target_baseline_rows: tuple[DatasetBaselineAntigravity, ...]
    baseline_confusion: ConfusionAntigravity
    target_family_fn: tuple[int, ...]
```

- [ ] **Step 6: Implement label-free values and closure-private labels**

Use read-only SQLite descriptors and exact query allowlists. `load_target_bundle_antigravity` may read only values/split/index required to construct z-normalized TRAIN-normal and TEST-value matrices. It must not select label. `open_test_labels_once_antigravity` is closure-private, accepts only the exact frozen prediction authority, validates exact string `'0'/'1'`, counts one completion per ordered target dataset, and closes before evaluation artifacts are computed.

- [ ] **Step 7: Run GREEN twice, real pre-label capture once, and regress Exp448**

Run the focused pair twice. Then call only `capture_exp448_authority_antigravity` and label-free target bundle loading against the real source. Assert DB nlink returns to 1 and Exp449 root remains absent. Run Exp448 source/replay focused tests without invoking its CLI.

- [ ] **Step 8: Update map and commit Task 3**

Regenerate map nodes/edges for predecessor and source authority, validate fingerprints, then commit:

```bash
git add experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_predecessor_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_sources_antigravity.py docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock
git commit -m "feat(exp449): pin residual FN source authority"
```

---

### Task 4: Four Fixed Representations, Shared LocalGap Scoring, and MLX Decision

**Files:**
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/mlx_gate_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_representations_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_scoring_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_mlx_gate_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: `DatasetBundleAntigravity`, `MethodCandidateAntigravity`, fixed method audit.
- Produces: `make_representation_antigravity(name, seed) -> RepresentationAntigravity`; `score_candidate_antigravity(...) -> CandidateScoresAntigravity`; `assess_mlx_antigravity() -> MlxDecisionAntigravity`.

- [ ] **Step 1: Write representation and scoring RED tests**

```python
def test_all_four_representations_are_finite_and_repeatable_antigravity(self):
    for name in EXPECTED_NAMES:
        first = score_candidate_antigravity(self.train, self.test, candidate(name), seed=20260721)
        second = score_candidate_antigravity(self.train, self.test, candidate(name), seed=20260721)
        np.testing.assert_array_equal(first.train_oof, second.train_oof)
        np.testing.assert_array_equal(first.test_scores, second.test_scores)
        self.assertTrue(np.isfinite(first.test_scores).all())

def test_hydra_multirocket_concatenation_order_is_exact_antigravity(self):
    actual = representation.transform(self.rows)
    expected = np.concatenate((scaled_hydra(self.rows), scaled_multirocket(self.rows)), axis=1)
    np.testing.assert_array_equal(actual, expected)
```

Cover wrong seed, feature dimension drift, nonfinite output, bool numeric fields, inconsistent fold dimensions, reference-sampling order, and fewer than five TRAIN rows.

- [ ] **Step 2: Write MLX RED test**

```python
def test_missing_mlx_is_a_recorded_cpu_fallback_antigravity(self):
    decision = assess_mlx_antigravity(package_probe=lambda _: None)
    self.assertEqual(decision.backend, "cpu")
    self.assertEqual(decision.reason, "mlx_and_mlx_metal_absent_in_fixed_environment")
    self.assertFalse(decision.parity_run)
```

- [ ] **Step 3: Run RED**

Run the three Task 4 test modules with the fixed aeon interpreter. Expected: imports fail because modules are absent.

- [ ] **Step 4: Implement the exact representation adapters**

Expose a small protocol:

```python
class RepresentationAntigravity(Protocol):
    def fit_transform(self, rows: np.ndarray) -> np.ndarray: ...
    def transform(self, rows: np.ndarray) -> np.ndarray: ...
```

Use `MiniRocket`, `MultiRocket`, and `HydraTransformer` from aeon 1.4.0 with `n_jobs=1` and exact seed. For `hydra_multirocket_local_gap`, reproduce the official aeon 1.4.0 order: Hydra transform → `_SparseScaler`; MultiROCKET transform → `StandardScaler`; concatenate Hydra first, MultiROCKET second. Never instantiate or fit `RidgeClassifierCV`.

Use MiniROCKET 4,096 requested kernels; MultiROCKET 1,024 requested kernels, 32 maximum dilations, four features per kernel, and `normalise=False`; Hydra eight kernels and 64 groups. Do not expose these settings as CLI options.

- [ ] **Step 5: Implement exact LocalGap and five-fold scoring**

Port the Exp249 semantics into antigravity-named code, retaining `KNN_NEIGHBORS=3`, `MAX_TRAIN_REFERENCE=1000`, `QUERY_CHUNK_SIZE=128`, and deterministic five-fold `crossfit_plan`. Add a source-semantic regression that compares the Exp449 LocalGap output with `experiments.exp249_250_official_mini_multirocket.local_gap.LocalGapModel` on the same arrays.

Return exact immutable score and backend contracts:

```python
@dataclass(frozen=True, slots=True)
class CandidateScoresAntigravity:
    train_oof: np.ndarray
    test_scores: np.ndarray
    feature_dimension: int
    seed: int

@dataclass(frozen=True, slots=True)
class MlxDecisionAntigravity:
    backend: Literal["cpu", "mlx"]
    reason: str
    parity_run: bool
    candidate_indices_identical: bool | None
    speedup: float | None
```

- [ ] **Step 6: Implement MLX preflight record**

The fixed environment probe must report `mlx` and `mlx-metal` absent and select CPU without package installation. Keep the full parity/speed fields in the contract so a future environment could only select MLX after exact candidate-index parity, finite scores, tolerance compliance, and speedup `>=1.1`.

- [ ] **Step 7: Run GREEN twice and a bounded real smoke**

Run all three focused modules twice. Then run one label-free representative dataset per representation, sequentially, recording time, feature dimension, finite status, candidate SHA, and DB nlink. Do not run the full 48-dataset matrix yet.

- [ ] **Step 8: Update map and commit Task 4**

```bash
git add experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/mlx_gate_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_representations_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_scoring_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_mlx_gate_antigravity.py docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock
git commit -m "feat(exp449): add fixed representation score sources"
```

---

### Task 5: Frozen Prediction Lifecycle, Durable Progress, and Sanitized Live Status

**Files:**
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/progress_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/dashboard_export_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_prediction_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_progress_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_dashboard_export_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: fixed target authority, four score adapters, canonical artifacts.
- Produces: `predict_antigravity(reviewed_commit: str) -> FrozenPredictionAuthorityAntigravity`; `snapshot_status_antigravity(...) -> Mapping[str, object>`; `public_status_antigravity(...) -> Mapping[str, object>`.

- [ ] **Step 1: Write exact inventory and no-label RED tests**

```python
def test_prediction_inventory_has_sixteen_family_method_pairs_antigravity(self):
    authority = predict_with_dependencies_antigravity(self.dependencies)
    self.assertEqual(authority.family_method_pairs, 16)
    self.assertEqual(authority.dataset_units, 96)
    self.assertEqual(authority.label_access_count, 0)
    self.assertEqual(authority.candidate_names, EXPECTED_ORDERED_CANDIDATES)

def test_public_status_rejects_sensitive_fields_antigravity(self):
    for key in ("dataset", "instance_index", "sha256", "path", "log", "database"):
        with self.subTest(key=key), self.assertRaises(ValueError):
            public_status_antigravity({**valid_status(), key: "forbidden"})
```

The 96 units are 48 target datasets × exactly two preregistered candidates per family. Test exact dataset order from Exp448, candidate order from config, and no duplicates.

- [ ] **Step 2: Write failure-ordering RED tests**

Cover worker exception, scorer exception, nonfinite score, cache write failure, progress write failure, event/text failure, PID close failure, candidate SHA mismatch, predecessor drift after candidate 0/47/95, and persistent status-channel failure. Check exact-next error evidence, inactive PID when writable, unchanged completed candidate bytes, no frozen prediction authority, and no label calls.

- [ ] **Step 3: Run RED**

Run Task 5's three modules. Expected: absent-module import errors.

- [ ] **Step 4: Implement deterministic dataset scheduling and prediction bytes**

Use up to seven outer `ProcessPoolExecutor` workers, submit units in frozen sequence, and commit results only in sequence order. Each dataset-method result must contain exact TRAIN-only config, target length, train/test counts, the two frozen seed score identities, and `contains_test_labels=false`.

For each dataset-method unit, compute both frozen seeds, take the elementwise median of TRAIN-OOF scores and TEST scores, then call `empirical_candidates(train_median, test_median, 0.005)`. The returned `p <= 0.005` indices are the sole binary prediction. Do not compute performance candidates at any other alpha or use seed intersections.

- [ ] **Step 5: Implement durable progress and prediction authority**

Attempt inventory:

```text
context_antigravity.json
progress_antigravity.json
progress_antigravity.jsonl
progress_antigravity.txt
pid_antigravity.json
candidates_antigravity/
prediction_manifest_antigravity.json
```

Each candidate file is create-only canonical JSON. The final prediction manifest contains exact 96-file inventory, per-file SHA, aggregate SHA, method audit SHA, source identity, MLX decision, and label access `0`. Freeze it before evaluation.

Return the exact authority:

```python
@dataclass(frozen=True, slots=True)
class FrozenPredictionAuthorityAntigravity:
    reviewed_commit: str
    family_method_pairs: int
    dataset_units: int
    candidate_names: tuple[str, ...]
    inventory: tuple[str, ...]
    leaf_sha256: tuple[tuple[str, str], ...]
    aggregate_sha256: str
    label_access_count: int
```

- [ ] **Step 6: Implement the sanitized dashboard status projection**

Use exact status fields:

```python
STATUS_FIELDS_ANTIGRAVITY = (
    "schema", "experiment_id", "phase", "status", "reviewed_commit",
    "target_family_count", "candidate_pair_count", "dataset_unit_total",
    "dataset_unit_completed", "method_family", "backend", "label_access_started",
)
```

No scientific module imports network or dashboard code. The controller reads this local aggregate snapshot and publishes owner-only status at task start, after each family boundary, on failure, and on completion.

- [ ] **Step 7: Run GREEN twice and a synthetic 96-unit lifecycle**

Use small dependency-injected matrices so all 96 units finish quickly. Assert two runs are byte-identical, progress counts 0→96, status contains exactly 12 fields, and label access stays false.

- [ ] **Step 8: Update map and commit Task 5**

```bash
git add experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/progress_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/dashboard_export_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_prediction_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_progress_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_dashboard_export_antigravity.py docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock
git commit -m "feat(exp449): freeze label-free method predictions"
```

---

### Task 6: One-Session Counterfactual Evaluation and Source-Free Replay

**Files:**
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/bootstrap_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_evaluation_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_replay_antigravity.py`
- Modify: `experiments/exp449_residual_fn_method_qualification_antigravity/dashboard_export_antigravity.py`
- Modify: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_dashboard_export_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: immutable Exp448 authority, frozen prediction authority, one closure-private ordered label session.
- Produces: `evaluate_antigravity(reviewed_commit: str) -> FrozenEvaluationAntigravity`; `inspect_antigravity() -> FrozenEvaluationAntigravity`; `public_result_antigravity(...) -> Mapping[str, object>`.

- [ ] **Step 1: Write exact boundary RED tests**

```python
def test_qualification_edges_are_exact_antigravity(self):
    self.assertTrue(qualifies_antigravity(confusion(1931, 314, 1987, 191668), target_fn_delta=-1))
    self.assertFalse(qualifies_antigravity(confusion(1931, 315, 1987, 191667), target_fn_delta=-1))
    self.assertFalse(qualifies_antigravity(confusion(1930, 314, 1988, 191668), target_fn_delta=0))

def test_one_session_evaluates_fixed_inventory_antigravity(self):
    result = evaluate_with_dependencies_antigravity(self.dependencies)
    self.assertEqual(result.label_session_count, 1)
    self.assertEqual(result.candidate_count, 16)
    self.assertEqual(result.label_query_count, 48)
    self.assertEqual(result.label_row_count, 72300)
    self.assertEqual(self.dependencies.open_labels_calls, 1)
```

Also test exact integer types, 48 target datasets, row counts, duplicate/missing candidate, candidate order, family membership, Exp448 non-target byte preservation, and exact global confusion conservation.

- [ ] **Step 2: Write failure and semantic replay RED tests**

Cover label open/query/close failures, source drift after close/compute/precommit, partial/existing/symlink publication, audit write failure, leaf symlink, directory replacement, duplicate keys, noncanonical JSON, reordered candidate metric rows, bool dataset order, wrong baseline, forged target delta, forged global confusion, recomputed artifact map after semantic tamper, and dashboard forbidden fields.

- [ ] **Step 3: Run RED**

Run evaluation, replay, and dashboard export tests. Expected: evaluation/replay import errors and missing result projection.

- [ ] **Step 4: Implement fixed paired dataset bootstrap**

Use NumPy `Generator(PCG64(44920260812))`, exactly 20,000 paired draws over the 48 target datasets, `int64` indices, `endpoint=False`, and linear 2.5/97.5 percentiles. Record the draw SHA from the final generated bytes in preregistration before official evaluation. Bootstrap reports FN delta, FP delta, and F1 delta but never changes qualification.

- [ ] **Step 5: Implement one-session counterfactual evaluation**

For each frozen candidate, compute its target confusion from the same label snapshot. Subtract the exact Exp448 target baseline row and add the candidate target row to the fixed global baseline. Require TP+FN and FP+TN conservation and derive F1 from exact confusion. `qualifies_antigravity` must enforce all four gates without coercion.

- [ ] **Step 6: Write the exact final publication**

Final `publication_antigravity/` inventory is exactly:

```text
context_antigravity.json
compatibility_antigravity.json
candidate_registry_antigravity.json
frozen_predictions_antigravity.jsonl
prediction_manifest_antigravity.json
evaluation_antigravity.json
candidate_metrics_antigravity.jsonl
family_metrics_antigravity.jsonl
bootstrap_antigravity.json
decision_antigravity.json
dashboard_snapshot_antigravity.json
artifact_sha256_antigravity.json
```

Build in same-filesystem staging, fsync leaves/directories, rename without replacement, reopen with descriptors, replay all semantics, then return success. On failure write the fixed create-only sibling audit and never publish a valid root.

- [ ] **Step 7: Implement source-free replay and aggregate dashboard result**

Replay imports no `sqlite3`, sources, aeon, evaluator, network, or dashboard client. It descriptor-pins all 12 leaves, validates exact inventory and canonical bytes, rebuilds candidate/global confusion, qualification gates, bootstrap draw SHA, and artifact map.

Both evaluate and replay return the same exact authority:

```python
@dataclass(frozen=True, slots=True)
class FrozenEvaluationAntigravity:
    reviewed_commit: str
    candidate_count: int
    qualified_count: int
    label_session_count: int
    label_query_count: int
    label_row_count: int
    candidate_metrics: tuple[CandidateMetricAntigravity, ...]
    artifact_map_sha256: str
    decision_sha256: str
    routing_research_eligible: bool
```

The final public payload contains only schema, experiment, baseline aggregate, candidate aggregate rows, family labels, qualification booleans, deterministic status, and runtime summaries. Reject every forbidden field recursively.

- [ ] **Step 8: Run GREEN twice and predecessor regressions**

Run evaluation/replay/dashboard tests twice. Run Task 2–5 focused suites once, Exp448 evaluation/replay/CLI once, and Exp249 scoring tests once. Use synthetic labels only; do not touch the real label session or official Exp449 root.

- [ ] **Step 9: Update map and commit Task 6**

```bash
git add experiments/exp449_residual_fn_method_qualification_antigravity/bootstrap_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/dashboard_export_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_evaluation_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_replay_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_dashboard_export_antigravity.py docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock
git commit -m "feat(exp449): evaluate frozen FN counterfactuals"
```

---

### Task 7: Closed CLI, Readiness, Full Verification, and Clean Candidate Truth

**Files:**
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/readiness_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/cli_antigravity.py`
- Create: `scratch/run_exp449_residual_fn_method_qualification_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_readiness_antigravity.py`
- Create: `experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_cli_antigravity.py`
- Modify: `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: all Task 1–6 boundaries.
- Produces: exact commands `readiness --reviewed-commit`, `predict --reviewed-commit`, `evaluate --reviewed-commit`, and bare `inspect`; truth status `implementation_verified_awaiting_official_readiness`.

- [ ] **Step 1: Write closed-surface RED tests**

```python
def test_parser_accepts_only_four_commands_antigravity(self):
    parser = build_parser_antigravity()
    for argv in (("readiness", "--reviewed-commit", COMMIT), ("predict", "--reviewed-commit", COMMIT), ("evaluate", "--reviewed-commit", COMMIT), ("inspect",)):
        parser.parse_args(argv)
    for argv in (("new",), ("resume",), ("retry",), ("inspect", "--root", "/tmp/x"), ("readiness", "--reviewed-c", COMMIT)):
        with self.assertRaises(SystemExit):
            parser.parse_args(argv)
```

Add fresh subprocess tests showing help/readiness/inspect leave aeon, SQLite label session, evaluator, network, and dashboard modules unloaded; evaluate checks clean HEAD before importing the evaluator; predict checks clean HEAD before importing aeon.

- [ ] **Step 2: Write readiness RED matrix**

Table-drive wrong/dirty commit, wrong Python/package versions, changed docs, stale map, wrong Exp448 bytes, wrong DB SHA/inode/nlink, existing/symlink run root, existing failure audit, changed method source SHA, candidate inventory drift, MLX decision drift, and pending/champion mutation. Every gate must fail before root creation and label access.

- [ ] **Step 3: Run RED**

Run readiness and CLI tests. Expected: missing modules/launcher.

- [ ] **Step 4: Implement readiness and lazy CLI**

Use `argparse.ArgumentParser(allow_abbrev=False)`. Require lowercase 40-hex commits. Readiness verifies clean HEAD exact equality, document hashes, map fingerprints, Exp448/DB/method authorities, CPU fallback, root/failure-audit absence, and label access `0`. CLI imports only the selected command boundary.

Launcher must set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, and `AEON_NUM_THREADS` to `1` before importing the CLI, and must work from `/tmp`.

- [ ] **Step 5: Run focused GREEN twice**

Run readiness and CLI tests twice, including outside-CWD subprocess and import firewalls.

- [ ] **Step 6: Run full verification sequentially with retained logs**

Use each experiment's frozen interpreter and one process at a time:

```bash
EXP449_PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
PREDECESSOR_PY=/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python
"$EXP449_PY" -m unittest discover -s experiments/exp449_residual_fn_method_qualification_antigravity/tests -p 'test_*_antigravity.py'
"$EXP449_PY" -m unittest discover -s experiments/exp449_residual_fn_method_qualification_antigravity/tests -p 'test_*_antigravity.py'
"$EXP449_PY" -m unittest discover -s experiments/exp249_250_official_mini_multirocket/tests -p 'test_*.py'
"$PREDECESSOR_PY" -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*.py'
"$PREDECESSOR_PY" -m unittest discover -s experiments/exp445_exp444_publication_source_authority_repair_antigravity/tests -p 'test_*.py'
"$PREDECESSOR_PY" -m unittest discover -s experiments/exp446_exp445_publication_materialization_repair_antigravity/tests -p 'test_*.py'
"$PREDECESSOR_PY" -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
"$PREDECESSOR_PY" -m unittest discover -s experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests -p 'test_*.py'
```

Retain exact stdout/stderr and exit sentinels under the ignored SDD task directory. Do not infer success from a detached process; keep the session ID and wait for the exact exit code.

- [ ] **Step 7: Verify frozen external state and dashboard source boundary**

Confirm DB SHA/inode/nlink, Exp448 root SHA/inventory, Exp449 root/failure audit absence, label access `0`, clean separate Sites project, owner-only access, and unchanged Exp448 public payload. Validate a synthetic `exp449.public-method-qualification.v1` snapshot but do not publish official results.

- [ ] **Step 8: Record clean-candidate truth and regenerate map**

Set Exp449 status to `implementation_verified_awaiting_official_readiness`, record the exact candidate commit and test counts, retain null official hashes and label access `0`, and leave champion/pending unchanged. Regenerate map with real CLI/readiness routes and no unknown implementation relations.

- [ ] **Step 9: Commit Task 7 and obtain fresh final review**

```bash
git add experiments/exp449_residual_fn_method_qualification_antigravity/readiness_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/cli_antigravity.py scratch/run_exp449_residual_fn_method_qualification_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_readiness_antigravity.py experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_cli_antigravity.py docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md EXPERIMENT_CATALOG.json RESEARCH_STATE.json CHAMPION_LEDGER.json docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock
git commit -m "feat(exp449): close method qualification candidate"
```

Request a fresh final review on the exact clean commit. Proceed only if the verdict is `ship` with zero Critical and Important findings.

---

### Task 8: Single-Use Official Lifecycle, Truth Closeout, and Owner-Only Dashboard

**Files:**
- Modify: `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify on failure only: `FAILURE_LEDGER.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`
- Publish through existing owner-only Sites project: sanitized status/result snapshot only; no scientific source files.

**Interfaces:**
- Consumes: clean `ship` candidate, fixed CLI, absent root/audit, owner-only dashboard project.
- Produces: immutable Exp449 publication or immutable failure audit; exact truth/map closeout; owner-only aggregate dashboard snapshot.

- [ ] **Step 1: Reconfirm authorization and external preflight**

Verify exact reviewed commit, clean tree, fresh `ship` verdict, DB SHA/inode/nlink, Exp448 identity, Exp449 root/failure-audit absence, method source SHAs, fixed environment, label access `0`, and owner-only site access. Stop before any command if one check fails.

- [ ] **Step 2: Publish the preregistered readiness status snapshot**

Use `data-analytics:build-dashboard` and the Sites artifact specification. Publish only the exact 12-field status payload with phase `readiness`, label access false, target-family count 8, candidate-pair count 16, dataset-unit total 96, and source freshness timestamp. Confirm owner-only access. Dashboard failure stops presentation work but does not consume a scientific command.

- [ ] **Step 3: Run official readiness exactly once**

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python
REVIEWED_COMMIT=$(git rev-parse HEAD)
"$PY" scratch/run_exp449_residual_fn_method_qualification_antigravity.py readiness --reviewed-commit "$REVIEWED_COMMIT"
```

Capture exact output and exit. If nonzero, record immutable failure truth, publish aggregate failure status, skip Steps 4–8, and never retry.

- [ ] **Step 4: Run official prediction exactly once**

```bash
"$PY" scratch/run_exp449_residual_fn_method_qualification_antigravity.py predict --reviewed-commit "$REVIEWED_COMMIT"
```

Publish sanitized progress at each completed family boundary from the local status snapshot. Never publish dataset names or candidate indices. If nonzero, preserve the attempt/failure audit, publish aggregate failure status, skip evaluate/inspect, and never retry.

- [ ] **Step 5: Independently verify prediction freeze before labels**

Descriptor-pin all 96 candidate leaves and the prediction manifest; recompute aggregate SHA; verify label access `0`, exact candidate/family inventory, deterministic seed outputs, CPU fallback record, DB identity, and Exp448 identity. A mismatch permanently stops Exp449 before evaluation.

- [ ] **Step 6: Run official evaluation exactly once**

```bash
"$PY" scratch/run_exp449_residual_fn_method_qualification_antigravity.py evaluate --reviewed-commit "$REVIEWED_COMMIT"
```

Capture exact output and exit. If nonzero, preserve the unique failure audit, publish aggregate failure status, skip inspect, and never retry.

- [ ] **Step 7: Run source-free inspect exactly once after success**

```bash
"$PY" scratch/run_exp449_residual_fn_method_qualification_antigravity.py inspect
```

Require exact equality with evaluate output and no SQLite/source/aeon/evaluator/network imports. Any inspect failure records truth and stops dashboard result publication; it does not authorize evaluation retry.

- [ ] **Step 8: Independently recompute final authority**

Read the 12 final leaves through retained descriptors, recompute artifact-map SHA, candidate/global confusion, FN/FP/F1 gates, target FN deltas, qualification decisions, bootstrap draw SHA, method audit, prediction inventory, and public snapshot allowlist. Confirm DB nlink 1 and Exp448 bytes unchanged.

- [ ] **Step 9: Commit official truth and synchronized map**

On success record status `official_method_qualification_complete_awaiting_routing_discussion`, exact command counts/exits, one label session, artifact hashes, candidate qualification table, and `routing_research_eligible`. Keep champion/pending/promotion unchanged. On failure record the exact stopped phase/error/audit identity and null success hashes. Regenerate the code map against the same truth commit and commit the triplet with the truth files.

```bash
git add docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md EXPERIMENT_CATALOG.json RESEARCH_STATE.json CHAMPION_LEDGER.json FAILURE_LEDGER.json docs/codemap/codemap_exp449_antigravity.html docs/codemap/codemap_exp449_antigravity.json docs/codemap/codemap_exp449_antigravity.lock
git commit -m "docs(exp449): close official method qualification truth"
```

- [ ] **Step 10: Publish and validate the owner-only result snapshot**

Only after successful truth closeout, publish `exp449.public-method-qualification.v1` to the existing owner-only research observatory. The default view must show baseline FN/FP/F1, number of qualified candidates, family-method qualification table, global FN/FP deltas, runtime/backend, and routing eligibility. Reconcile every displayed number with the immutable publication, verify custom owner-only access, record the Sites version and snapshot timestamp, and state that it is a published snapshot rather than a live database connection.

- [ ] **Step 11: Obtain final read-only truth review and stop**

Review exact truth commit, immutable root/audit, DB and Exp448 identities, code map, dashboard payload/access, and champion/pending invariants. Do not run commands, query labels, or modify artifacts during review. Stop for user discussion. Start Exp450 design only when at least one candidate qualified and the user explicitly approves routing research.

---

## Self-Review Checklist

- Every approved design section maps to at least one task: governance (Task 1), compatibility (Task 2), source authority (Task 3), score/MLX (Task 4), freeze/dashboard status (Task 5), one-session evaluation/replay (Task 6), closed candidate (Task 7), official truth/dashboard (Task 8).
- Exact type names and signatures are consistent across tasks: `Exp448AuthorityAntigravity`, `MethodCandidateAntigravity`, `MethodAuditAntigravity`, `FrozenPredictionAuthorityAntigravity`, `FrozenEvaluationAntigravity`, and `ConfusionAntigravity`.
- The plan contains no hyperparameter or candidate choice made after TEST results.
- All eight target families receive exactly two score-compatible candidates; classifier-only methods remain audit evidence.
- FP 314/315 and FN 1,987/1,988 boundaries have explicit tests.
- The fixed environment's absent MLX packages produce a truthful CPU fallback without installation.
- Dashboard status and results are aggregate-only, owner-only, source-backed snapshots and cannot alter scientific execution.
- Every new filename contains `antigravity`; no `__init__.py` is created.
- Every code-changing task includes the synchronized Exp449 HTML/JSON/lock triplet in the same commit.
- Official label access is exactly one session after all 96 candidate units freeze; any failure forbids retry.
