# Exp472 Makima Upsampling Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one isolated Makima-upsampled condition E, compute it for the frozen 1,117 datasets, and publish a paper-ready E-versus-D comparison while reusing verified Exp471 evidence.

**Architecture:** Exp450's dataset-local scorer gains one keyword-only alignment-function injection point while its existing A--D entry point and default alignment remain unchanged and byte-parity tested. Exp472 supplies the Makima aligner through that narrow interface, writes one durable final-score shard per dataset, then combines the verified E manifest with the immutable Exp471 cache for D, A--C, and six public baselines. A values-only smoke gates the full run; TEST labels remain inaccessible until E and the reused Exp471 authorities both verify.

**Tech Stack:** Python 3.14.6, NumPy, SciPy 1.17.1 `Akima1DInterpolator(method="makima")`, PyTorch 2.13 MPS for the unchanged area-image path, scikit-learn, aeon, SQLite read-only access, `unittest`, canonical JSON/JSONL/CSV/SVG/PNG/HTML artifacts.

**Spec:** `docs/superpowers/specs/2026-08-28-exp472-makima-upsampling-ablation-antigravity-design.md`

## Global Constraints

- Exp471 code, cache, evaluation, report, source root, and original database remain immutable.
- Every new repository and runtime filename contains `antigravity`.
- Frozen database SHA-256 is `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Full population is exactly 1,117 datasets and 195,900 TEST rows; smoke uses the frozen 23-dataset Exp450 inventory.
- Condition E is exactly polyphase downsampling, Makima upsampling, and area/local-mean image construction.
- Condition D is the immutable Exp471 control: polyphase downsampling, PCHIP upsampling, and area/local-mean image construction.
- Equal-length, singleton, downsampling, repair, target-length, features, estimators, seeds, thresholds, routes, policy, and score orientation remain frozen.
- The selected proposed score is `hard_plus_stable`, `float64`, finite, and higher means more anomalous.
- Makima/polyphase remain exact SciPy CPU operations. The existing area-image path uses MPS when available and a frozen CPU fallback otherwise.
- MLX is recorded as assessed but unsuitable for the exact SciPy interpolation path unless a separate exact-parity and 1.1x benchmark succeeds.
- Run one MPS-heavy process at a time. CPU-only execution uses `max(1, logical_cpu_count - 1)` workers and one inner numerical thread.
- TEST labels are not imported before all 1,117 E shards and the reused Exp471 cache independently verify.
- Bootstrap uses 20,000 draws, seed `47220260828`, percentile 95% intervals, and absolute tie tolerance `1e-12`.
- ROC axes are false-positive rate × true-positive rate; precision-recall axes are recall × precision.
- No candidate search, clipping, interpolation fallback, result-dependent rerun, or automatic promotion is permitted.
- The official Python is `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`.
- Because that environment has no `pytest`, repository tests run with `python -m unittest`; Numba cache is directed to a writable Exp472 runtime cache during tests.

## File Structure

- `experiments/exp472_makima_upsampling_ablation_antigravity/config_antigravity.py`: frozen roots, identities, method order, backend assessment, bootstrap seed, and worker policy.
- `experiments/exp472_makima_upsampling_ablation_antigravity/contracts_antigravity.py`: condition, run context, score shard, manifest, and evaluation/report authority types.
- `experiments/exp472_makima_upsampling_ablation_antigravity/transforms_antigravity.py`: exact polyphase/Makima alignment and label-free shape diagnostics.
- `experiments/exp472_makima_upsampling_ablation_antigravity/scoring_antigravity.py`: thin condition-E adapter over the byte-parity-tested Exp450 alignment-injection boundary.
- `experiments/exp472_makima_upsampling_ablation_antigravity/artifacts_antigravity.py`: create-only E shards, failure records, manifest, progress log, and scorer-free verification.
- `experiments/exp472_makima_upsampling_ablation_antigravity/runner_antigravity.py`: smoke/full input loading, one-dataset final score composition, two-wave recovery, and backend-safe scheduling.
- `experiments/exp472_makima_upsampling_ablation_antigravity/authority_antigravity.py`: read-only Exp471 D/context authority and E-to-D TEST-coordinate binding.
- `experiments/exp472_makima_upsampling_ablation_antigravity/evaluation_antigravity.py`: one-time label access, A--E/public-baseline metrics, upsampling subset, paired bootstrap, and curve tables.
- `experiments/exp472_makima_upsampling_ablation_antigravity/reporting_antigravity.py`: deterministic paper tables, figures, Korean explanation, and hash-bound report manifest.
- `experiments/exp472_makima_upsampling_ablation_antigravity/cli_antigravity.py`: closed `readiness`, `smoke`, `run`, `status`, `verify`, `evaluate`, and `report` commands.
- `scratch/run_exp472_makima_upsampling_ablation_antigravity.py`: fixed launcher with no alternative roots or scientific parameters.
- `scratch/generate_exp472_codemap_antigravity.py`: complete Exp472 callers/effects/tests code-map generator.
- `docs/runbooks/exp472_makima_upsampling_ablation_antigravity.md`: exact operator commands and artifact interpretation.

---

### Task 1: Create the Exp472 Code Map, Identity, and Contracts

**Files:**
- Create: `scratch/generate_exp472_codemap_antigravity.py`
- Create together: `docs/codemap/codemap_exp472_antigravity.html`
- Create together: `docs/codemap/codemap_exp472_antigravity.json`
- Create together: `docs/codemap/codemap_exp472_antigravity.lock`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/config_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/contracts_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_config_contracts_antigravity.py`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`

**Interfaces:**
- Consumes: the approved Exp472 design, `FROZEN_EXP450_CONFIG.smoke_datasets`, Exp471 root constants, and `FrozenPaperConfig`.
- Produces: `CONDITION_E_ANTIGRAVITY`, `Exp472BackendDecisionAntigravity`, `Exp472RunContextAntigravity`, `Exp472ScoreShardAntigravity`, `Exp472ScoreManifestAntigravity`, frozen roots, and a synchronized code-map triplet.

- [ ] **Step 1: Generate the initial map before scientific source edits**

Implement the generator from the Exp471 AST-based generator with version constants changed to Exp472. Its tracked set must include the design, Exp450 transforms/imaging/scoring/runner, Exp471 cache/evaluation/reporting/source-authority modules, paper data/label modules, and every existing Exp472 file:

```python
VERSION_ANTIGRAVITY = "exp472"
TRIPLET_ANTIGRAVITY = tuple(
    REPOSITORY_ROOT / "docs" / "codemap" / f"codemap_exp472_antigravity.{suffix}"
    for suffix in ("html", "json", "lock")
)

def main(argv: Sequence[str] | None = None) -> int:
    check = tuple(sys.argv[1:] if argv is None else argv) == ("--check",)
    paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
    payload = build_codemap_payload_antigravity(
        paths, ast_callers_antigravity(paths)
    )
    triplet = render_triplet_antigravity(
        payload, module_fingerprints_antigravity(paths)
    )
    if check:
        require_existing_triplet_equal_antigravity(triplet)
    else:
        write_triplet_together_antigravity(triplet)
    return 0
```

Run:

```bash
python3 scratch/generate_exp472_codemap_antigravity.py
python3 scratch/generate_exp472_codemap_antigravity.py --check
```

Expected: all three files exist; JSON nodes answer what calls Exp450 scoring, what Exp471 evaluation affects, and which tests cover each boundary.

- [ ] **Step 2: Write failing frozen-identity tests**

```python
class Exp472ConfigContractsTests(unittest.TestCase):
    def test_condition_e_is_the_only_new_condition(self):
        self.assertEqual(
            (
                CONDITION_E_ANTIGRAVITY.condition_id,
                CONDITION_E_ANTIGRAVITY.downsampling,
                CONDITION_E_ANTIGRAVITY.upsampling,
                CONDITION_E_ANTIGRAVITY.image_mode,
            ),
            ("E", "polyphase", "makima", "area"),
        )

    def test_population_roots_and_seed_are_frozen(self):
        self.assertEqual(FROZEN_EXP472_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_EXP472_CONFIG.test_row_count, 195900)
        self.assertEqual(len(FROZEN_EXP472_CONFIG.smoke_datasets), 23)
        self.assertEqual(FROZEN_EXP472_CONFIG.bootstrap_seed, 47220260828)
        self.assertTrue(all("antigravity" in part.name for part in FROZEN_EXP472_CONFIG.roots))
```

- [ ] **Step 3: Confirm RED**

Run:

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_config_contracts_antigravity -v
```

Expected: import failure for missing Exp472 configuration/contracts.

- [ ] **Step 4: Implement frozen types and roots**

```python
@dataclass(frozen=True, slots=True)
class Exp472ConditionAntigravity:
    condition_id: str = "E"
    downsampling: str = "polyphase"
    upsampling: str = "makima"
    image_mode: str = "area"

    def __post_init__(self) -> None:
        if (self.condition_id, self.downsampling, self.upsampling, self.image_mode) != (
            "E", "polyphase", "makima", "area"
        ):
            raise ValueError("Exp472 condition differs from the approved design")

@dataclass(frozen=True, slots=True)
class Exp472ScoreShardAntigravity:
    dataset: str
    test_identity: Mapping[str, object]
    scores: np.ndarray
    alerts: frozenset[int]
    upsampled_rows: int
    diagnostic: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.scores)
        if values.dtype != np.dtype("float64") or values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Exp472 score shard differs")
        if frozenset(np.flatnonzero(values >= 0.5).tolist()) != self.alerts:
            raise ValueError("Exp472 score threshold differs")
```

Freeze:

```python
EXP472_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp472_makima_upsampling_ablation_antigravity_20260828")
EXP472_SMOKE_ROOT_ANTIGRAVITY = EXP472_ROOT_ANTIGRAVITY / "smoke_antigravity"
EXP472_OFFICIAL_ROOT_ANTIGRAVITY = EXP472_ROOT_ANTIGRAVITY / "official_antigravity"
EXP472_EVALUATION_ROOT_ANTIGRAVITY = EXP472_ROOT_ANTIGRAVITY / "evaluation_antigravity"
EXP472_REPORT_ROOT_ANTIGRAVITY = EXP472_ROOT_ANTIGRAVITY / "report_antigravity"
EXP471_REUSE_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_antigravity_20260824")
BOOTSTRAP_SEED_ANTIGRAVITY = 47220260828
BOOTSTRAP_DRAWS_ANTIGRAVITY = 20_000
```

The backend record must contain PyTorch, SciPy, MLX, mlx-metal, MPS availability, selected tensor backend, logical CPUs, `max(1, logical - 1)`, one inner thread, and the exact MLX fallback reason from the spec.

- [ ] **Step 5: Update active research records without editing old experiment entries**

Set `active_experiment` to `Exp472`, state to `implementation_planned`, and point the active design/plan fields at the two Exp472 documents. Append one Exp472 catalog record; retain every Exp471 value unchanged.

- [ ] **Step 6: Regenerate the map and run GREEN**

```bash
python3 scratch/generate_exp472_codemap_antigravity.py
python3 scratch/generate_exp472_codemap_antigravity.py --check
python3 -m json.tool docs/codemap/codemap_exp472_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp472_antigravity.lock >/dev/null
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_config_contracts_antigravity -v
```

Expected: all tests pass and the map check reports no byte drift.

- [ ] **Step 7: Commit**

```bash
git add RESEARCH_STATE.json EXPERIMENT_CATALOG.json experiments/exp472_makima_upsampling_ablation_antigravity/config_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/contracts_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_config_contracts_antigravity.py scratch/generate_exp472_codemap_antigravity.py docs/codemap/codemap_exp472_antigravity.html docs/codemap/codemap_exp472_antigravity.json docs/codemap/codemap_exp472_antigravity.lock
git commit -m "feat(exp472): freeze makima ablation identity"
```

---

### Task 2: Implement Exact Makima Alignment and Diagnostics

**Files:**
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/transforms_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_transforms_antigravity.py`
- Modify together: `docs/codemap/codemap_exp472_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: finite `float32` source rows and a positive target length.
- Produces: `resample_series_exp472_antigravity(values, target_len) -> np.ndarray`, `align_rows_exp472_antigravity(rows, target_len, diagnostics=None) -> np.ndarray`, `shape_diagnostic_exp472_antigravity(source, output) -> Mapping[str, object]`, `count_upsampled_rows_exp472_antigravity(values) -> int`, `dataset_shape_diagnostic_exp472_antigravity(values) -> Mapping[str, object]`, and `test_identity_exp472_antigravity(values) -> Mapping[str, object]`.

- [ ] **Step 1: Write RED tests for every route**

```python
def test_makima_is_used_only_for_true_upsampling(self):
    row = np.asarray([0.0, 1.0, -0.5, 2.0], dtype=np.float32)
    with patch.object(transforms.scipy.interpolate, "Akima1DInterpolator", wraps=scipy.interpolate.Akima1DInterpolator) as makima:
        up = transforms.resample_series_exp472_antigravity(row, 13)
        same = transforms.resample_series_exp472_antigravity(row, 4)
        down = transforms.resample_series_exp472_antigravity(row, 2)
    self.assertEqual(makima.call_count, 1)
    self.assertEqual(makima.call_args.kwargs, {"method": "makima", "extrapolate": False})
    np.testing.assert_array_equal(same, row)
    self.assertEqual((up.shape, down.shape), ((13,), (2,)))

def test_equal_singleton_and_downsampling_equal_exp450_d(self):
    for row, target in ((np.array([2.5], np.float32), 9), (np.arange(8, dtype=np.float32), 8), (np.arange(12, dtype=np.float32), 5)):
        actual = transforms.resample_series_exp472_antigravity(row, target)
        expected = resample_series_exp450_antigravity(row, target, mode="polyphase_pchip")
        np.testing.assert_array_equal(actual, expected)

def test_two_point_makima_is_finite_deterministic_and_endpoint_exact(self):
    row = np.asarray([-2.0, 3.0], dtype=np.float32)
    first = transforms.resample_series_exp472_antigravity(row, 17)
    second = transforms.resample_series_exp472_antigravity(row, 17)
    self.assertEqual(first.tobytes(), second.tobytes())
    self.assertTrue(np.isfinite(first).all())
    self.assertEqual((float(first[0]), float(first[-1])), (-2.0, 3.0))

def test_shape_diagnostics_cover_preregistered_fixture_families(self):
    fixtures = {
        "constant": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "monotone": np.asarray([0.0, 0.3, 1.0], dtype=np.float32),
        "peaked": np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        "oscillatory": np.asarray([0.0, 1.0, -1.0, 1.0, 0.0], dtype=np.float32),
    }
    for name, row in fixtures.items():
        with self.subTest(name=name):
            output = transforms.resample_series_exp472_antigravity(row, 31)
            diagnostic = transforms.shape_diagnostic_exp472_antigravity(row, output)
            self.assertGreaterEqual(diagnostic["overshoot"], 0.0)
            self.assertGreaterEqual(diagnostic["undershoot"], 0.0)
            self.assertGreaterEqual(diagnostic["output_extrema"], 0)
```

- [ ] **Step 2: Run RED**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_transforms_antigravity -v
```

Expected: missing transform API failure.

- [ ] **Step 3: Implement the exact transform**

```python
def resample_series_exp472_antigravity(values: np.ndarray, target_len: int) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_len <= 0 or row.size == 0 or not np.isfinite(row).all():
        raise ValueError("Exp472 resampling requires finite nonempty input and positive target")
    if row.size == target_len:
        return row.astype(np.float32, copy=False)
    if row.size == 1:
        return np.full(target_len, row[0], dtype=np.float32)
    if target_len < row.size:
        divisor = math.gcd(row.size, target_len)
        result = scipy.signal.resample_poly(row, target_len // divisor, row.size // divisor)
    else:
        source_x = np.linspace(0.0, 1.0, row.size, dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        result = scipy.interpolate.Akima1DInterpolator(
            source_x, row, method="makima", extrapolate=False
        )(target_x)
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (target_len,) or not np.isfinite(output).all():
        raise ValueError("Exp472 resampler output differs")
    if target_len > row.size and (output[0] != row[0] or output[-1] != row[-1]):
        raise ValueError("Exp472 Makima endpoints differ")
    return output
```

`shape_diagnostic_exp472_antigravity` returns source/output min/max, positive overshoot above the source range, negative undershoot below it, source/output extrema counts based on derivative sign changes, and deterministic SHA-256 values. It does not clip or reject finite overshoot.

- [ ] **Step 4: Run GREEN and refresh the map**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_transforms_antigravity -v
python3 scratch/generate_exp472_codemap_antigravity.py
python3 scratch/generate_exp472_codemap_antigravity.py --check
```

- [ ] **Step 5: Commit**

```bash
git add experiments/exp472_makima_upsampling_ablation_antigravity/transforms_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_transforms_antigravity.py docs/codemap/codemap_exp472_antigravity.html docs/codemap/codemap_exp472_antigravity.json docs/codemap/codemap_exp472_antigravity.lock
git commit -m "feat(exp472): add exact makima alignment"
```

---

### Task 3: Build the Isolated Dataset-Local E Scorer

**Files:**
- Modify: `experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/scoring_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_scoring_antigravity.py`
- Modify together: `docs/codemap/codemap_exp472_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: `RawDataset`, `CONDITION_E_ANTIGRAVITY`, the Exp472 aligner, the unchanged Exp450 area-image functions, and frozen paper selectors.
- Produces: `score_dataset_condition_e_exp472_antigravity(values, backend) -> Exp450DatasetConditionResult` and `compose_final_score_exp472_antigravity(values, result, backend) -> Exp472ScoreShardAntigravity`.

- [ ] **Step 1: Write RED tests for factor isolation and label firewall**

```python
def test_e_routes_all_alignment_through_exp472_and_keeps_area_images(self):
    with patch.object(scoring, "align_rows_exp472_antigravity", wraps=align_rows_exp472_antigravity) as align, patch.object(scoring, "imaging_score_pair_exp450_antigravity", wraps=imaging_score_pair_exp450_antigravity) as image:
        scoring.score_dataset_condition_e_exp472_antigravity(RAW_FIXTURE, BACKEND_CPU)
    self.assertGreater(align.call_count, 0)
    self.assertTrue(all(call.kwargs["image_mode"] == "area" for call in image.call_args_list))

def test_scoring_module_has_no_label_import(self):
    source = Path(scoring.__file__).read_text(encoding="utf-8")
    self.assertNotIn("label_access", source)
    self.assertNotIn("load_dataset_labels", source)

def test_downsample_only_e_equals_exp450_d(self):
    e_result = scoring.score_dataset_condition_e_exp472_antigravity(DOWNSAMPLE_FIXTURE, BACKEND_CPU)
    d_result = score_dataset_condition_exp450_antigravity(DOWNSAMPLE_FIXTURE, CONDITION_D, BACKEND_CPU)
    self.assertEqual(canonical_result_bytes_antigravity(e_result), canonical_result_bytes_antigravity(d_result))
```

- [ ] **Step 2: Run RED**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_scoring_antigravity -v
```

- [ ] **Step 3: Add one alignment-function injection boundary without changing A--D**

In the Exp450 scorer, add a private keyword-only `align_rows_override` argument to the four functions that currently perform row alignment. Centralize the selection in one helper; all existing calls omit the override and therefore execute the original `align_rows_exp450_antigravity(..., mode=condition.length_mode)` expression:

```python
AlignmentRowsAntigravity = Callable[[Sequence[np.ndarray], int], np.ndarray]

def _align_rows_selected_antigravity(rows, target_len, condition, align_rows_override):
    if align_rows_override is None:
        return align_rows_exp450_antigravity(
            rows, target_len, mode=condition.length_mode
        )
    return np.asarray(align_rows_override(rows, target_len), dtype=np.float32)
```

Add `score_dataset_condition_with_alignment_exp450_antigravity(values, condition, backend, *, align_rows_override)` as the only new shared entry point. It threads the override through Hard sources, MiniROCKET, and legacy-route evidence. Keep `score_dataset_condition_exp450_antigravity(values, condition, backend)` unchanged in signature and make it delegate with `align_rows_override=None`.

Add a retained Exp450 test that serializes A--D fixture results before and after the refactor and requires identical canonical bytes. Add an Exp472 test that passes `align_rows_exp472_antigravity` and proves only true upsampling differs from D.

The Exp472 public entry point is fixed to:

```python
CONDITION_E_ANTIGRAVITY = Exp472ConditionAntigravity()

def score_dataset_condition_e_exp472_antigravity(
    values: RawDataset,
    backend: Exp472BackendDecisionAntigravity,
) -> Exp450DatasetConditionResult:
    return score_dataset_condition_with_alignment_exp450_antigravity(
        values,
        CONDITION_E_ANTIGRAVITY,
        backend,
        align_rows_override=align_rows_exp472_antigravity,
    )
```

- [ ] **Step 4: Compose the final policy score per dataset**

```python
def compose_final_score_exp472_antigravity(values, result, backend):
    raw = {values.dataset: values}
    sources = {values.dataset: result.sources}
    routes = {values.dataset: result.route}
    hard = {values.dataset: result.route.exp253.hard}
    low = score_low_frequency(raw, hard)
    stability = classify_stability(low, raw)
    surface = build_hard_evidence_surface(sources, routes, raw)
    policies = build_policies(surface, low, stability, raw, routes)
    policy = next(item for item in policies.policies if item.name == "hard_plus_stable")
    scores = np.asarray(policy.scores[values.dataset], dtype=np.float64)
    return Exp472ScoreShardAntigravity(
        dataset=values.dataset,
        test_identity=test_identity_exp472_antigravity(values),
        scores=scores,
        alerts=policy.alerts[values.dataset],
        upsampled_rows=count_upsampled_rows_exp472_antigravity(values),
        diagnostic=dataset_shape_diagnostic_exp472_antigravity(values),
    )
```

Add a fixture test proving per-dataset composition equals full-map composition for the same two datasets and condition E.

- [ ] **Step 5: Run GREEN, refresh map, and commit**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_scoring_antigravity -v
python3 scratch/generate_exp472_codemap_antigravity.py
python3 scratch/generate_exp472_codemap_antigravity.py --check
git add experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/scoring_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_scoring_antigravity.py docs/codemap/codemap_exp472_antigravity.html docs/codemap/codemap_exp472_antigravity.json docs/codemap/codemap_exp472_antigravity.lock
git commit -m "feat(exp472): add isolated condition e scorer"
```

---

### Task 4: Add Durable Shards, Progress, Resume, and Two-Wave Execution

**Files:**
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/runner_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_artifacts_runner_antigravity.py`
- Modify together: `docs/codemap/codemap_exp472_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: frozen catalog entries and `compose_final_score_exp472_antigravity`.
- Produces: `publish_score_shard_exp472_antigravity`, `verify_score_shard_exp472_antigravity`, `publish_manifest_exp472_antigravity`, `verify_manifest_exp472_antigravity`, `status_exp472_antigravity`, and `run_score_wave_exp472_antigravity`.

- [ ] **Step 1: Write RED lifecycle tests**

```python
def test_create_only_shard_is_idempotent_for_same_bytes_and_rejects_drift(self):
    first = publish_score_shard_exp472_antigravity(ROOT, CONTEXT, ENTRY, SHARD)
    second = publish_score_shard_exp472_antigravity(ROOT, CONTEXT, ENTRY, SHARD)
    self.assertEqual(first.read_bytes(), second.read_bytes())
    with self.assertRaisesRegex(ValueError, "different bytes"):
        publish_score_shard_exp472_antigravity(ROOT, CONTEXT, ENTRY, replace(SHARD, scores=OTHER))

def test_resume_schedules_only_missing_and_failed_coordinates(self):
    pending = pending_datasets_exp472_antigravity(CATALOG, completed={"A", "C"}, failures={"B"})
    self.assertEqual(pending, ("B", "D"))

def test_status_reports_completed_failed_remaining_rate_and_eta(self):
    status = status_exp472_antigravity(ROOT, now_monotonic=40.0)
    self.assertEqual(set(status), {"schema", "completed", "failed", "remaining", "total", "rate_per_second", "eta_seconds", "manifest_complete", "latest_progress"})
```

- [ ] **Step 2: Run RED**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_artifacts_runner_antigravity -v
```

- [ ] **Step 3: Implement canonical create-only artifacts**

Use schema `exp472-score-shard-antigravity.v1`. Each shard binds database, catalog, run-context and code-closure digests; condition E; backend; dataset and TEST-row identity; `float64` scores; alerts; upsampled-row count; shape diagnostics; and its own SHA-256. Publish with `O_CREAT|O_EXCL`, fsync the file and parent directory, and accept an existing file only when bytes are identical.

```python
def score_shard_record_exp472_antigravity(context, shard):
    bare = {
        "schema": "exp472-score-shard-antigravity.v1",
        "context_sha256": context.sha256,
        "database_sha256": context.database_sha256,
        "catalog_sha256": context.catalog_sha256,
        "condition": {"id": "E", "downsampling": "polyphase", "upsampling": "makima", "image_mode": "area"},
        "backend": context.backend.artifact_record(),
        "dataset": shard.dataset,
        "test_identity": dict(shard.test_identity),
        "score_dtype": "float64",
        "scores": [float(value) for value in shard.scores],
        "alerts": sorted(shard.alerts),
        "upsampled_rows": shard.upsampled_rows,
        "diagnostic": dict(shard.diagnostic),
    }
    return {**bare, "shard_sha256": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}
```

The terminal `exp472_score_manifest_antigravity.json` contains exactly 1,117 ordered descriptors and is published only after every shard is independently reopened and verified. No manifest is written when a failure remains.

- [ ] **Step 4: Implement backend-safe scheduling and progress**

```python
workers = 1 if context.backend.tensor_backend == "mps" else context.backend.cpu_workers
for wave in (1, 2):
    pending = pending_datasets_exp472_antigravity(catalog, completed, failures if wave == 2 else frozenset())
    outcomes = execute_datasets_exp472_antigravity(pending, workers=workers, score=score_one)
    for outcome in outcomes:
        append_progress_exp472_antigravity(
            root,
            completed=len(completed),
            failed=len(failures),
            remaining=total - len(completed),
            elapsed_seconds=monotonic() - started,
            dataset=outcome.dataset,
            wave=wave,
            status=outcome.status,
        )
```

MPS execution is strictly sequential. CPU execution uses a process pool with the frozen worker count and one inner numerical thread. A failed dataset is recorded and processing continues; only failed/unattempted datasets enter wave two.

- [ ] **Step 5: Run GREEN, refresh map, and commit**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_artifacts_runner_antigravity -v
python3 scratch/generate_exp472_codemap_antigravity.py
python3 scratch/generate_exp472_codemap_antigravity.py --check
git add experiments/exp472_makima_upsampling_ablation_antigravity/artifacts_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/runner_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_artifacts_runner_antigravity.py docs/codemap/codemap_exp472_antigravity.html docs/codemap/codemap_exp472_antigravity.json docs/codemap/codemap_exp472_antigravity.lock
git commit -m "feat(exp472): add resumable condition e execution"
```

---

### Task 5: Bind Immutable Exp471 Evidence and Build Evaluation

**Files:**
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/authority_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_authority_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap_exp472_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: verified Exp472 E manifest, verified Exp471 cache `11,170 = 10,775 scored + 395 abstained + 0 failed`, frozen TEST labels, and Exp471 method order.
- Produces: `Exp472ReuseAuthorityAntigravity`, `evaluate_exp472_antigravity(...) -> Exp472EvaluationAntigravity`, canonical metric tables, denominator audits, and curve coordinates.

- [ ] **Step 1: Write RED authority and metric tests**

```python
def test_reuse_requires_exact_exp471_terminal_counts(self):
    authority = verify_exp471_reuse_exp472_antigravity(EXP471_ROOT)
    self.assertEqual((authority.total, authority.scored, authority.abstained, authority.failed), (11170, 10775, 395, 0))
    self.assertEqual((authority.dataset_count, authority.common_support), (1117, 922))

def test_e_and_d_test_coordinates_must_match_exactly(self):
    with self.assertRaisesRegex(ValueError, "TEST coordinate"):
        bind_e_to_exp471_exp472_antigravity(E_MANIFEST, replace(D_AUTHORITY, coordinate_sha256="0" * 64))

def test_curve_axes_and_method_order_are_fixed(self):
    evaluation = evaluate_exp472_antigravity(FIXTURE_E, FIXTURE_EXP471, FIXTURE_LABELS)
    self.assertEqual(evaluation.method_order, ("A", "B", "C", "D", "E", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d"))
    self.assertEqual(evaluation.curves["roc"]["x_name"], "false_positive_rate")
    self.assertEqual(evaluation.curves["pr"]["x_name"], "recall")
```

- [ ] **Step 2: Run RED**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_authority_evaluation_antigravity -v
```

- [ ] **Step 3: Implement read-only reuse authority**

Reconstruct the frozen Exp471 source session with its existing read-only API, open the complete cache, read and verify the completed evaluation/report, and bind:

```python
@dataclass(frozen=True, slots=True)
class Exp472ReuseAuthorityAntigravity:
    exp471_source_authority_sha256: str
    exp471_cache_manifest_sha256: str
    exp471_evaluation_sha256: str
    exp471_report_manifest_sha256: str
    database_sha256: str
    coordinate_inventory_sha256: str
    method_order: tuple[str, ...]
    total: int
    scored: int
    abstained: int
    failed: int
    dataset_count: int
    common_support: int
```

The E manifest database digest, ordered datasets, TEST counts, indices, blob hashes, and row digests must match the D cache shards exactly. Any mismatch stops before label access.

- [ ] **Step 4: Implement one-time evaluation**

Load TEST labels only after both authorities verify. Compute per-dataset AP/AUROC, macro and micro summaries, paired E-minus-D bootstrap intervals, win/tie/loss counts, full-population denominators, and the label-independent subset where `upsampled_rows > 0`. Use the fixed seed and draws from configuration.

```python
def paired_e_minus_d_exp472_antigravity(e_values, d_values, *, seed=47220260828):
    delta = np.asarray(e_values, dtype=np.float64) - np.asarray(d_values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, delta.size, size=(20_000, delta.size))
    means = delta[draws].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "mean_delta": float(delta.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "wins": int(np.count_nonzero(delta > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
        "losses": int(np.count_nonzero(delta < -1e-12)),
    }
```

Write canonical ROC and PR coordinate tables before figures. Validate monotone x coordinates, `[0,1]` bounds, ROC endpoints `(0,0)` and `(1,1)`, and PR recall direction. Preserve typed abstentions for public baselines; do not substitute scores.

- [ ] **Step 5: Run GREEN, refresh map, and commit**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_authority_evaluation_antigravity -v
python3 scratch/generate_exp472_codemap_antigravity.py
python3 scratch/generate_exp472_codemap_antigravity.py --check
git add experiments/exp472_makima_upsampling_ablation_antigravity/authority_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/evaluation_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_authority_evaluation_antigravity.py docs/codemap/codemap_exp472_antigravity.html docs/codemap/codemap_exp472_antigravity.json docs/codemap/codemap_exp472_antigravity.lock
git commit -m "feat(exp472): bind exp471 evidence and evaluate makima"
```

---

### Task 6: Add Deterministic Reporting and the Closed CLI

**Files:**
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/reporting_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/cli_antigravity.py`
- Create: `experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_reporting_cli_antigravity.py`
- Create: `scratch/run_exp472_makima_upsampling_ablation_antigravity.py`
- Create: `docs/runbooks/exp472_makima_upsampling_ablation_antigravity.md`
- Modify together: `docs/codemap/codemap_exp472_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: frozen configuration, E runtime state, reuse authority, and evaluation artifact.
- Produces: closed commands, live status, deterministic paper artifacts, and `Exp472ReportAuthorityAntigravity`.

- [ ] **Step 1: Write RED CLI and report tests**

```python
def test_status_is_values_only_and_reports_eta(self):
    output = io.StringIO()
    self.assertEqual(main_antigravity(("status",), stdout=output), 0)
    value = json.loads(output.getvalue())
    self.assertEqual(value["experiment"], "Exp472")
    self.assertNotIn("label_access", sys.modules)
    self.assertEqual(set(value["paths"]), {"root", "smoke", "official", "evaluation", "report", "progress"})

def test_evaluate_refuses_incomplete_e_manifest(self):
    with self.assertRaisesRegex(ValueError, "1,117"):
        main_antigravity(("evaluate",), runtime=INCOMPLETE_RUNTIME)

def test_report_contains_e_vs_d_and_correct_axes(self):
    authority = write_exp472_report_antigravity(EVALUATION_FIXTURE, ROOT, WORKER_FIXTURE)
    html = (ROOT / "exp472_paper_report_antigravity.html").read_text(encoding="utf-8")
    self.assertIn("Makima", html)
    self.assertIn("PCHIP", html)
    self.assertIn("False-positive rate", html)
    self.assertIn("Recall", html)
    self.assertEqual(len(authority.leaf_sha256), 15)
    self.assertNotIn("exp472_report_manifest_antigravity.json", authority.leaf_sha256)
```

- [ ] **Step 2: Run RED**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_reporting_cli_antigravity -v
```

- [ ] **Step 3: Implement deterministic report leaves**

Publish exactly these 15 leaves before the completion marker:

```python
REPORT_LEAVES_ANTIGRAVITY = (
    "exp472_paper_report_antigravity.html",
    "exp472_paper_summary_ko_antigravity.md",
    "exp472_methods_appendix_antigravity.md",
    "exp472_aggregate_metrics_antigravity.csv",
    "exp472_per_dataset_metrics_antigravity.csv",
    "exp472_paired_e_minus_d_antigravity.csv",
    "exp472_upsampling_subset_antigravity.csv",
    "exp472_shape_diagnostics_antigravity.csv",
    "exp472_denominator_audit_antigravity.csv",
    "exp472_roc_coordinates_antigravity.csv",
    "exp472_pr_coordinates_antigravity.csv",
    "exp472_roc_curve_antigravity.svg",
    "exp472_pr_curve_antigravity.svg",
    "exp472_curves_antigravity.png",
    "exp472_backend_provenance_antigravity.json",
)
```

After those 15 content leaves are durable, publish `exp472_report_manifest_antigravity.json`; it binds all 15 leaf hashes, the Exp472 evaluation digest, E manifest digest, Exp471 reuse authority digest, code-map lock digest, backend/worker record, and report closure digest. The separate completion marker binds the manifest hash, so no artifact contains a recursive self-hash. HTML and Korean summary state improvement, degradation, or no decisive difference according to the frozen interval; they never claim equivalence merely because an interval crosses zero.

- [ ] **Step 4: Implement closed commands and runbook**

`readiness` checks roots/backends/identities without labels. `smoke` and `run` execute only E. `status` is read-only. `verify` independently verifies E and Exp471 authority. `evaluate` requires both terminal authorities. `report` only consumes a persisted evaluation and never starts score generation or label evaluation.

The launcher is exactly:

```python
from experiments.exp472_makima_upsampling_ablation_antigravity.cli_antigravity import main_antigravity

if __name__ == "__main__":
    raise SystemExit(main_antigravity())
```

The runbook lists the exact official-Python commands, expected terminal counts, log paths, and rules forbidding restart after an unexplained failure.

- [ ] **Step 5: Run GREEN, refresh map, and commit**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_reporting_cli_antigravity -v
python3 scratch/generate_exp472_codemap_antigravity.py
python3 scratch/generate_exp472_codemap_antigravity.py --check
git add experiments/exp472_makima_upsampling_ablation_antigravity/reporting_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/cli_antigravity.py experiments/exp472_makima_upsampling_ablation_antigravity/tests/test_reporting_cli_antigravity.py scratch/run_exp472_makima_upsampling_ablation_antigravity.py docs/runbooks/exp472_makima_upsampling_ablation_antigravity.md docs/codemap/codemap_exp472_antigravity.html docs/codemap/codemap_exp472_antigravity.json docs/codemap/codemap_exp472_antigravity.lock
git commit -m "feat(exp472): add report and closed command surface"
```

---

### Task 7: Run Complete Implementation Verification

**Files:**
- Modify: any Exp472 file only when a failing test proves a defect.
- Modify together after fixes: `docs/codemap/codemap_exp472_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: Tasks 1--6.
- Produces: a clean, fully tested Exp472 implementation commit ready for values-only smoke.

- [ ] **Step 1: Run all Exp472 tests in a writable cache environment**

```bash
mkdir -p /private/tmp/exp472_numba_cache_antigravity
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest discover -s experiments/exp472_makima_upsampling_ablation_antigravity/tests -p 'test_*_antigravity.py' -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run retained boundary suites that do not depend on absent Exp450 runtime roots**

```bash
NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity PYTHONDONTWRITEBYTECODE=1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_transforms_backend_antigravity experiments.exp450_resampling_image_ablation_antigravity.tests.test_imaging_antigravity experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_config_contracts_antigravity experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_cache_artifacts_antigravity -v
```

Expected: zero failures and zero errors. Existing Exp450 root-presence tests are excluded because the completed historical roots intentionally exist.

- [ ] **Step 3: Validate map, filenames, diff, and source isolation**

```bash
python3 scratch/generate_exp472_codemap_antigravity.py --check
git diff --check codex/exp471-exp467-evaluation-recovery-antigravity...HEAD
git diff --name-only codex/exp471-exp467-evaluation-recovery-antigravity...HEAD
```

Expected: no changed path under the Exp471 namespace; every added filename contains `antigravity`; the only pre-implementation document is the already committed design/plan.

- [ ] **Step 4: Commit verification repairs if any were required**

```bash
git add experiments/exp472_makima_upsampling_ablation_antigravity scratch/generate_exp472_codemap_antigravity.py scratch/run_exp472_makima_upsampling_ablation_antigravity.py docs/codemap/codemap_exp472_antigravity.html docs/codemap/codemap_exp472_antigravity.json docs/codemap/codemap_exp472_antigravity.lock docs/runbooks/exp472_makima_upsampling_ablation_antigravity.md
git commit -m "test(exp472): verify makima experiment pipeline"
```

If no repair was needed, do not create an empty commit.

---

### Task 8: Run and Verify the 23-Dataset Values-Only Smoke

**Files:**
- Runtime only under: `/Users/mac/Documents/Dataset/exp472_makima_upsampling_ablation_antigravity_20260828/smoke_antigravity`

**Interfaces:**
- Consumes: the verified implementation and frozen smoke inventory.
- Produces: 23 verified E shards, one smoke manifest, progress history, backend record, and shape diagnostics without TEST labels.

- [ ] **Step 1: Run readiness and record backend/worker selection**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py readiness
```

Expected: database and Exp471 authority match, SciPy 1.17.1 supports Makima, MPS selection is recorded when available, MLX assessment is present, and worker counts follow policy.

- [ ] **Step 2: Run smoke with live progress**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py smoke
```

Expected: 23 completed, 0 failed, 0 remaining; every output is finite and exact-length; no label module is imported.

- [ ] **Step 3: Independently verify smoke and diagnostics**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py verify --mode smoke
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py status --mode smoke
```

Expected: the manifest binds all 23 shards; overshoot/extrema diagnostics are present and no diagnostic value was used as a promotion selector.

- [ ] **Step 4: Freeze the smoke decision**

Append a research-state record containing the smoke manifest SHA-256, backend, completed/failed counts, and `full_run_authorized=true` only when all operational gates pass. Commit only repository state, never runtime score data.

```bash
git add RESEARCH_STATE.json
git commit -m "chore(exp472): record successful makima smoke"
```

---

### Task 9: Run the 1,117-Dataset Condition E Experiment

**Files:**
- Runtime only under: `/Users/mac/Documents/Dataset/exp472_makima_upsampling_ablation_antigravity_20260828/official_antigravity`

**Interfaces:**
- Consumes: successful smoke authority and immutable repository commit.
- Produces: exactly 1,117 verified E shards, zero unresolved failures, one terminal manifest, and complete progress/ETA evidence.

- [ ] **Step 1: Start the fixed full command once**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py run
```

Expected: progress reports completed, failed, remaining, rate, and ETA. A prior verified shard is reused; it is never recomputed.

- [ ] **Step 2: Monitor read-only**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py status --mode full
```

Expected terminal state: 1,117 completed, 0 failed, 0 remaining, `manifest_complete=true`. If the process exits unexpectedly, preserve the exact last progress and failure record; do not restart until the cause is identified.

- [ ] **Step 3: Verify the complete E authority**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py verify --mode full
```

Expected: all ordered dataset and TEST identities match; all scores are finite `float64`; alerts equal `scores >= 0.5`; the manifest and code closure verify.

---

### Task 10: Evaluate Once, Publish the Report, and Verify Final Authority

### Task 8 amendment: actual-trace v3 closure

This amendment supersedes v2/raw-replay trace guidance without deleting its
history.  The Exp450 shared gate predicates emit an ordered `(phase_tag,
source_tag)` plan before observer calls.  Exp472 must observe that exact plan,
poison failures, record completion only after the exact scorer returns, and
bind the sealed trace to deterministic raw-input and canonical-result digests.
IPC validates the complete primitive v3 diagnostic before typed shard
reconstruction; booleans and floats are invalid for ordinal and extrema/count
fields.  Composition remains fail-closed when no trace is supplied.

### Task 8 amendment: v3 report/evaluation consumption

Evaluation and reporting accept only `exp472-shape-diagnostic-antigravity.v3`
through the public strict validator. They reject v2/raw replay before opening
labels or publishing leaves. The diagnostic leaf is call-level: every actual
invocation records phase/source, its own target length, routes, Makima indices,
and input/output evidence. `upsampled_rows` is the unique verified
`(TRAIN|TEST,row_index)` slot count used for subset membership;
`makima_row_invocation_count` is separately published as repeated exposure
count.

### Task 8 amendment: durable evaluation authority and dataset identity binding

This amendment supersedes any implicit in-memory evaluation handoff. The only
production evaluation handoff is the canonical regular-file artifact
`evaluation_antigravity/exp472_evaluation_antigravity.json`; it remains outside
the report's exact 15-leaf inventory. `evaluate` first reopens verified Exp471
and E authority, returns an identical persisted artifact without labels when it
matches, otherwise evaluates once and publishes create-or-identical then
reopens. `report` reopens only Exp471/E and that persisted evaluation; it must
not call evaluation or import the label module. Before leaf publication it
matches every trace by dataset identity (not digest lookup): the exact dataset
set, trace dataset key, diagnostic SHA-256, unique `upsampled_rows`, and
repeated `makima_row_invocation_count` must agree. Missing, duplicate, extra,
swapped, noncanonical, nonregular, symlinked, or digest-drifting inputs fail
closed. Call-level CSV rows exist only for actual invocations; a sealed zero
call trace contributes no placeholder row while its dataset summary remains
authoritative.

### Task 8 amendment: persisted evaluation repair

The terminal E authority is exactly the verified
`manifest_sha256` field, never a second hash of a JSON object containing that
self-digest. A persisted evaluation is accepted only after canonical-byte,
outer-digest, authority, full 1,117-dataset × 11-method scientific inventory,
subset, denominator, paired-statistic, curve, and diagnostic-summary checks.
`evaluate` takes an advisory OS claim before rechecking the artifact. Only the
claim holder may open TEST labels; all waiters reopen the holder's verified
record. Publication writes and fsyncs a same-directory temporary file, creates
the final name with atomic no-clobber linking, fsyncs the directory, removes the
temporary file, and never replaces an existing final artifact. Status exposes
only untrusted presence flags rather than treating a pathname as completion.

### Task 8 amendment: private label-evidence replay anchor

The public evaluation artifact is schema v3 and binds the digest of the private,
0600, create-only `exp472_private_label_evidence_antigravity.json` file. The
private evidence is canonical JSON with duplicate-key and symlink rejection; it
contains only ordered binary TEST-label vectors plus exact Exp471 authority, E
terminal-manifest, context, and dataset-order bindings. It is excluded from the
15 public report leaves, report manifest, and Git-tracked sources. Production
reopen verifies Exp471 and every E shard again, reconstructs all scores, loads
the private evidence without the source label loader, and canonical-compares the
entire recomputed scientific evaluation. A crash after private evidence but
before public evaluation resumes from that evidence without reopening labels.
This detects accidental, corrupt, or coherent public-result rewrites while the
private evidence and existing score authorities remain intact. Simultaneous
privileged replacement of every trust anchor is out of scope without external
signing or WORM storage.

### SDD progress ledger — private evidence replay ruling

Implementation evidence is `a0ee23e2` plus the follow-up replay-boundary
change. The accepted public evaluation/report path is v3 only: immutable
Exp471/E terminal score authority is re-opened with the private label-evidence
commitment and every scientific section is rebuilt before report publication.
The private file remains outside Git and the 15 public report leaves. This
protects accidental, corrupt, and coherent public evaluation/marker rewrites
while the existing score authorities and 0600 create-only private evidence
remain intact; replacing every anchor requires external signing/WORM and is
outside this local authority model. Verification evidence: Exp472 128/128
tests completed in 429.453 seconds; Exp450 transform/imaging boundary 17/17
passed, with only six MPS-unavailable skips in each applicable suite.

**Files:**
- Runtime only under: `/Users/mac/Documents/Dataset/exp472_makima_upsampling_ablation_antigravity_20260828/evaluation_antigravity`
- Runtime only under: `/Users/mac/Documents/Dataset/exp472_makima_upsampling_ablation_antigravity_20260828/report_antigravity`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`

**Interfaces:**
- Consumes: terminal E manifest and verified Exp471 reuse authority.
- Produces: one evaluation artifact, 15 report leaves, E-versus-D conclusion, and final research-state evidence.

- [ ] **Step 1: Evaluate after the label firewall opens**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py evaluate
```

Expected: 1,117 full datasets, a separately reported actual-upsampling subset, A--E plus all six public baselines, paired E-minus-D statistics, denominator audits, and correct curve coordinates.

- [ ] **Step 2: Publish the deterministic report**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py report
```

Expected: all 15 content leaves, the report manifest, and the completion marker exist; the marker binds the manifest that binds the content leaves to the evaluation, E manifest, Exp471 reuse authority, code map, and worker/backend record.

- [ ] **Step 3: Run final read-only verification**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py verify --mode full
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python scratch/run_exp472_makima_upsampling_ablation_antigravity.py status --mode full
```

Expected: 1,117 E shards, zero failures, evaluation complete, report complete, 15 verified report content leaves, one verified report manifest, and no live Exp472 process.

- [ ] **Step 4: Record the frozen conclusion and commit**

Update Exp472 state with the E manifest, Exp471 reuse authority, evaluation, and report digests; exact macro/micro AP/AUROC; paired confidence intervals; upsampling-subset size; and the interpretation required by the spec. Mark Exp472 complete only after every digest reopens successfully.

```bash
git add RESEARCH_STATE.json EXPERIMENT_CATALOG.json
git commit -m "docs(exp472): record makima ablation result"
```
