# Exp450 Resampling and Image Construction Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a four-condition Exp450 ablation that replaces linear time-series alignment and image resizing independently, then publish continuous-score AP/AUROC, ROC, and precision-recall comparisons against all frozen public baselines.

**Architecture:** Exp450 is an isolated namespace layered over immutable paper contracts. Pure resampling and MPS image operations feed variant-aware dataset-local Hard-source, MiniROCKET, and legacy-route scoring; verified dataset × condition checkpoints feed the unchanged global low-frequency, stability, and route-preserving score stages. Four frozen score manifests must complete before the one permitted TEST-label evaluation and paper report.

**Tech Stack:** Python 3.14.6, NumPy, SciPy, PyTorch 2.13 MPS, scikit-learn, aeon, SQLite read-only data access, `unittest`, Matplotlib, canonical JSON/CSV/SVG/PNG/HTML artifacts.

**Spec:** `docs/superpowers/specs/2026-08-20-exp450-resampling-image-ablation-antigravity-design.md`

## Global Constraints

- Exp449 code, score roots, reports, and versioned code map remain immutable.
- Every newly created repository and runtime filename contains `antigravity`.
- Frozen database: `/Users/mac/Documents/Dataset/univariate_ts.db`, SHA-256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Full population: exactly 1,117 datasets and 195,900 TEST rows; smoke population: the frozen 23-dataset Exp449 inventory.
- Condition order is exactly `A`, `B`, `C`, `D`; selected policy is exactly `hard_plus_stable`.
- Existing internal nonfinite repair and target-length selection rules remain unchanged.
- MPS is canonical for tensor-eligible Exp450 operations whenever readiness reports it available; no CPU--MPS numerical parity test is added.
- PyTorch, MLX, and MPS availability are recorded. MLX is assessed but unused.
- A run chooses its tensor backend once. It never silently switches after a failure.
- Only one MPS-heavy job runs at a time. CPU-only stages use `max(1, logical_cpu_count - 1)` workers and one inner numerical thread.
- Persist one verified checkpoint per dataset × condition; retry failed and never-attempted jobs in one second bounded wave.
- TEST labels remain inaccessible until all four complete 1,117-dataset score manifests independently verify.
- Bootstrap uses 20,000 draws, seed `45020260820`, percentile 95% intervals, and absolute tie tolerance `1e-12`.
- ROC axes are false-positive rate × true-positive rate. Precision-recall axes are recall × precision.
- No hyperparameter search, result-dependent rerun, automatic promotion, or unrelated security/operations redesign.

---

### Task 1: Freeze Exp450 Identity, Contracts, State, and Initial Code Map

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/config_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_config_contracts_antigravity.py`
- Create: `scratch/generate_exp450_codemap_antigravity.py`
- Create together: `docs/codemap/codemap_exp450_antigravity.html`
- Create together: `docs/codemap/codemap_exp450_antigravity.json`
- Create together: `docs/codemap/codemap_exp450_antigravity.lock`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`

**Interfaces:**
- Consumes: `FROZEN_EXP449_CONFIG.smoke_datasets`, `FrozenPaperConfig`, approved design path.
- Produces: `Exp450Condition`, `Exp450BackendDecision`, `Exp450DatasetConditionResult`, `Exp450ConditionScore`, `FROZEN_EXP450_CONFIG`, and a complete Exp450 code-map triplet.

- [ ] **Step 1: Write the failing identity and naming tests**

```python
class Exp450ConfigTests(unittest.TestCase):
    def test_four_conditions_are_exact_factorial(self):
        self.assertEqual(
            tuple((c.condition_id, c.length_mode, c.image_mode) for c in FROZEN_EXP450_CONFIG.conditions),
            (
                ("A", "linear", "linear"),
                ("B", "polyphase_pchip", "linear"),
                ("C", "linear", "area"),
                ("D", "polyphase_pchip", "area"),
            ),
        )

    def test_population_statistics_and_roots_are_frozen(self):
        self.assertEqual(FROZEN_EXP450_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_EXP450_CONFIG.test_row_count, 195900)
        self.assertEqual(len(FROZEN_EXP450_CONFIG.smoke_datasets), 23)
        self.assertTrue(all("antigravity" in root.name for root in FROZEN_EXP450_CONFIG.roots))
```

- [ ] **Step 2: Run the test and confirm the missing-module RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_config_contracts_antigravity -v
```

Expected: import failure for the absent Exp450 config/contracts modules.

- [ ] **Step 3: Implement the frozen configuration and typed contracts**

```python
@dataclass(frozen=True, slots=True)
class Exp450Condition:
    condition_id: str
    length_mode: str
    image_mode: str

    def __post_init__(self) -> None:
        allowed = {
            "A": ("linear", "linear"),
            "B": ("polyphase_pchip", "linear"),
            "C": ("linear", "area"),
            "D": ("polyphase_pchip", "area"),
        }
        if allowed.get(self.condition_id) != (self.length_mode, self.image_mode):
            raise ValueError("Exp450 condition differs from the approved factorial")

@dataclass(frozen=True, slots=True)
class Exp450BackendDecision:
    tensor_backend: str
    torch_version: str
    mps_available: bool
    mlx_available: bool
    logical_cpu_count: int
    cpu_workers: int
    inner_numerical_threads: int = 1

    def artifact_record(self) -> dict[str, object]:
        return {
            "tensor_backend": self.tensor_backend,
            "torch_version": self.torch_version,
            "mps_available": self.mps_available,
            "mlx_available": self.mlx_available,
            "logical_cpu_count": self.logical_cpu_count,
            "cpu_workers": self.cpu_workers,
            "inner_numerical_threads": self.inner_numerical_threads,
        }

@dataclass(frozen=True, slots=True)
class Exp450DatasetConditionResult:
    dataset: str
    condition_id: str
    sources: HardSourceBundle
    route: HardRouteResult

@dataclass(frozen=True, slots=True)
class Exp450ConditionScore:
    condition: Exp450Condition
    raw: Mapping[str, RawDataset]
    sources: Mapping[str, HardSourceBundle]
    routes: Mapping[str, HardRouteResult]
    low_frequency: LowFrequencyResult
    stability: StabilityPartition
    policies: PolicyBundle
    backend: Exp450BackendDecision
    stage_seconds: Mapping[str, float]
```

Freeze these roots:

```python
SMOKE_ROOT = Path("/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_antigravity_20260820")
OFFICIAL_ROOT = Path("/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_official_antigravity_20260820")
EVALUATION_ROOT = Path("/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_evaluation_antigravity_20260820")
REPORT_ROOT = Path("/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_report_antigravity_20260820")
```

- [ ] **Step 4: Declare Exp450 without changing Exp449 history**

Update `RESEARCH_STATE.json` to set `active_experiment` to `Exp450`, status to `design_approved_implementation_planned`, point `active_documents.design` and `.plan` at the Exp450 files, and add an `exp450_resampling_image_ablation_antigravity` record with every execution/evaluation flag false. Append Exp450 to `EXPERIMENT_CATALOG.json`; do not edit prior experiment records.

- [ ] **Step 5: Generate the initial Exp450 map from factual files**

Implement `generate_exp450_codemap_antigravity.py` so it inspects existing Exp450 and retained paper modules, derives Python callers from AST imports, and writes the JSON/HTML/lock triplet together:

```python
def main(argv: Sequence[str] | None = None) -> int:
    check = tuple(sys.argv[1:] if argv is None else argv) == ("--check",)
    paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
    payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
    generated = render_triplet_antigravity(payload, module_fingerprints_antigravity(paths))
    if check:
        require_existing_triplet_equal_antigravity(generated)
    else:
        write_triplet_together_antigravity(generated)
    return 0
```

The initial map must answer callers, effects, and tests for current paper preprocessing, imaging, Hard-source, MiniROCKET, legacy-route, Exp449 evaluation utilities, and new Exp450 files. It must never include a nonexistent path or overwrite Exp449 files.

- [ ] **Step 6: Run GREEN and validate the triplet**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_config_contracts_antigravity -v
python3 scratch/generate_exp450_codemap_antigravity.py --check
python3 -m json.tool docs/codemap/codemap_exp450_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp450_antigravity.lock >/dev/null
```

Expected: all tests pass; HTML embedded JSON equals the standalone JSON; lock fingerprints match tracked bytes.

- [ ] **Step 7: Commit Task 1**

```bash
git add RESEARCH_STATE.json EXPERIMENT_CATALOG.json experiments/exp450_resampling_image_ablation_antigravity/config_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_config_contracts_antigravity.py scratch/generate_exp450_codemap_antigravity.py docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): freeze interpolation ablation identity"
```

---

### Task 2: Implement Pure Length Transforms and MPS-First Backend Selection

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/transforms_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/backend_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_transforms_backend_antigravity.py`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: `Exp450Condition`, `Exp450BackendDecision`, frozen `resample_series_paper` for the control.
- Produces: `resample_series_exp450_antigravity`, `align_rows_exp450_antigravity`, `adaptive_average_1d_antigravity`, `adaptive_average_2d_antigravity`, `assess_exp450_backend_antigravity`.

- [ ] **Step 1: Write RED tests for exact control, PCHIP/polyphase rules, pooling, and backend choice**

```python
def test_control_resampler_is_byte_exact_to_exp449(self):
    row = np.asarray([0.0, 3.0, -1.0, 2.0], dtype=np.float32)
    actual = resample_series_exp450_antigravity(row, 11, mode="linear")
    expected = resample_series_paper(row, 11)
    self.assertEqual(actual.tobytes(), expected.tobytes())

def test_hybrid_upsampling_preserves_monotone_shape(self):
    row = np.asarray([0.0, 0.2, 0.7, 1.0], dtype=np.float32)
    result = resample_series_exp450_antigravity(row, 31, mode="polyphase_pchip")
    self.assertEqual(result.shape, (31,))
    self.assertTrue(np.all(np.diff(result) >= 0.0))
    self.assertGreaterEqual(result.min(), 0.0)
    self.assertLessEqual(result.max(), 1.0)

def test_mps_is_selected_without_cpu_parity_when_available(self):
    with patch.object(backend_antigravity, "_probe_antigravity", return_value={
        "torch_version": "2.13.0", "mps_available": True, "mlx_available": True,
        "logical_cpu_count": 12,
    }):
        decision = assess_exp450_backend_antigravity()
    self.assertEqual(decision.tensor_backend, "mps")
    self.assertEqual(decision.cpu_workers, 11)
```

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_transforms_backend_antigravity -v
```

Expected: import failure for absent transform/backend APIs.

- [ ] **Step 3: Implement the exact hybrid resampler**

```python
def resample_series_exp450_antigravity(values: np.ndarray, target_len: int, *, mode: str) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_len <= 0 or not len(row):
        raise ValueError("Exp450 resampling requires nonempty input and positive target")
    if mode == "linear":
        return resample_series_paper(row, target_len)
    if mode != "polyphase_pchip":
        raise ValueError("unknown Exp450 length mode")
    if len(row) == target_len:
        return row.astype(np.float32, copy=False)
    if len(row) == 1:
        return np.full(target_len, row[0], dtype=np.float32)
    if target_len < len(row):
        divisor = math.gcd(len(row), target_len)
        result = scipy.signal.resample_poly(row, target_len // divisor, len(row) // divisor)
    else:
        source = np.linspace(0.0, 1.0, len(row), dtype=np.float64)
        target = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        result = scipy.interpolate.PchipInterpolator(source, row, extrapolate=False)(target)
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (target_len,) or not np.isfinite(output).all():
        raise ValueError("Exp450 resampler output differs")
    return output
```

Implement row alignment by stacking this function. Implement adaptive pooling with `torch.nn.functional.adaptive_avg_pool1d/2d`; move only the result back to a CPU NumPy `float32` array after `torch.mps.synchronize()`.

- [ ] **Step 4: Implement backend assessment with no parity gate**

```python
def assess_exp450_backend_antigravity() -> Exp450BackendDecision:
    observed = _probe_antigravity()
    selected = "mps" if observed["mps_available"] else "cpu"
    count = int(observed["logical_cpu_count"])
    return Exp450BackendDecision(
        tensor_backend=selected,
        torch_version=str(observed["torch_version"]),
        mps_available=bool(observed["mps_available"]),
        mlx_available=bool(observed["mlx_available"]),
        logical_cpu_count=count,
        cpu_workers=max(1, count - 1),
        inner_numerical_threads=1,
    )
```

Record stage decisions `adaptive_pool=mps`, `recurrence_distance=mps`, `polyphase=cpu`, `pchip=cpu`, `sklearn_aeon=cpu` when MPS is available.

- [ ] **Step 5: Run GREEN on CPU fixtures and the one MPS completion test**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_transforms_backend_antigravity -v
```

Expected: all tests pass; the MPS test checks finite output and exact shape only.

- [ ] **Step 6: Refresh map and commit**

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
git add experiments/exp450_resampling_image_ablation_antigravity/transforms_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/backend_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_transforms_backend_antigravity.py docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): add hybrid resampling and MPS pooling"
```

---

### Task 3: Implement Variant-Aware Imaging Without Linear Resizing

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/imaging_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_imaging_antigravity.py`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: adaptive pooling APIs from Task 2 and frozen Exp55/56 config, PCA, GLCM, scaling, and KNN helpers.
- Produces: `spectrogram_images_exp450_antigravity`, `recurrence_images_exp450_antigravity`, `imaging_score_pair_exp450_antigravity`.

- [ ] **Step 1: Write RED tests for control identity and area-mode semantics**

```python
def test_linear_imaging_delegates_to_frozen_exp449(self):
    with patch.object(imaging_antigravity, "imaging_score_pair_paper", return_value=(np.array([1.0]), np.array([2.0]))) as frozen:
        actual = imaging_score_pair_exp450_antigravity(TRAIN, TEST, EXP55_CONFIG_PAPER, image_mode="linear", device="cpu")
    frozen.assert_called_once()
    self.assertEqual(actual[1].tolist(), [2.0])

def test_area_recurrence_is_symmetric_and_not_a_resized_finished_matrix(self):
    images = recurrence_images_exp450_antigravity(TRAIN, 32, device="cpu")
    self.assertEqual(images.shape, (len(TRAIN), 32, 32))
    np.testing.assert_allclose(images, images.transpose(0, 2, 1), rtol=0.0, atol=0.0)

def test_area_spectrogram_has_exact_shape_and_finite_values(self):
    images = spectrogram_images_exp450_antigravity(TRAIN, 32, device="cpu")
    self.assertEqual(images.shape, (len(TRAIN), 32, 32))
    self.assertTrue(np.isfinite(images).all())
```

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_imaging_antigravity -v
```

Expected: missing imaging module/API failure.

- [ ] **Step 3: Implement area spectrogram and recurrence paths**

```python
def recurrence_images_exp450_antigravity(matrix: np.ndarray, size: int, *, device: str) -> np.ndarray:
    fixed = adaptive_average_1d_antigravity(np.asarray(matrix, dtype=np.float32), size, device=device)
    tensor = torch.as_tensor(fixed, device=device, dtype=torch.float32)
    distance = torch.abs(tensor[:, :, None] - tensor[:, None, :])
    scale = torch.quantile(distance.flatten(1), 0.75, dim=1).clamp_min(1e-6)
    images = torch.exp(-distance / scale[:, None, None])
    if device == "mps":
        torch.mps.synchronize()
    return images.detach().cpu().numpy().astype(np.float32, copy=False)
```

Build the pre-resize spectrogram with the existing window, hop, padding, FFT, and `log1p(abs(...))` rules, then call `adaptive_average_2d_antigravity`. Do not call `np.interp` in area mode.

- [ ] **Step 4: Preserve downstream imaging semantics**

For `image_mode="linear"`, call `imaging_score_pair_paper` directly. For `image_mode="area"`, use the new image functions, then the frozen `_glcm_features_paper`, `_pca_pair_paper`, `scale_feature_pair_paper`, and exact KNN distance rules from `imaging_paper.py`. Record the source module SHA-256 in Exp450 provenance.

- [ ] **Step 5: Run GREEN on CPU and MPS**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_imaging_antigravity -v
```

Expected: all tests pass; MPS test checks shape/finite/symmetry, not CPU equality.

- [ ] **Step 6: Refresh map and commit**

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
git add experiments/exp450_resampling_image_ablation_antigravity/imaging_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_imaging_antigravity.py docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): add area pooled imaging variants"
```

---

### Task 4: Build Dataset-Local A--D Scientific Scoring

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_scoring_antigravity.py`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: Tasks 1--3, immutable paper Hard-source/MiniROCKET/legacy-route types and pure selector functions.
- Produces: `score_hard_sources_exp450_antigravity`, `score_minirocket_exp450_antigravity`, `build_legacy_route_evidence_exp450_antigravity`, `replay_hard_route_exp450_antigravity`, `score_dataset_condition_exp450_antigravity`.

- [ ] **Step 1: Write RED tests for A control, factor isolation, and label firewall**

```python
def test_condition_a_delegates_to_exact_exp449_science(self):
    expected_sources = object()
    expected_route = object()
    with patch.object(scoring_antigravity, "score_hard_sources", return_value=expected_sources), \
         patch.object(scoring_antigravity, "score_minirocket_confirmations", return_value=object()), \
         patch.object(scoring_antigravity, "replay_hard_route", return_value=expected_route):
        result = score_dataset_condition_exp450_antigravity(RAW, CONDITION_A, BACKEND_CPU)
    self.assertIs(result.sources, expected_sources)
    self.assertIs(result.route, expected_route)

def test_condition_b_changes_alignment_only(self):
    with patch.object(scoring_antigravity, "align_rows_exp450_antigravity") as align, \
         patch.object(scoring_antigravity, "imaging_score_pair_exp450_antigravity") as image:
        score_dataset_condition_exp450_antigravity(RAW, CONDITION_B, BACKEND_CPU)
    self.assertTrue(all(call.kwargs["mode"] == "polyphase_pchip" for call in align.call_args_list))
    self.assertTrue(all(call.kwargs["image_mode"] == "linear" for call in image.call_args_list))
```

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_scoring_antigravity -v
```

Expected: missing scoring APIs.

- [ ] **Step 3: Implement variant-aware Hard sources**

Keep source names, folds, seeds, thresholds, p-values, abstention, ROCKET, Exp84, canonical, and Block-B rules unchanged. Replace only length alignment and route Exp55/56 through:

```python
imaging_score_pair_exp450_antigravity(
    fit, query, config,
    image_mode=condition.image_mode,
    device=backend.tensor_backend,
)
```

- [ ] **Step 4: Implement variant-aware MiniROCKET and legacy evidence**

Reuse `score_configuration_paper`, seeds, kernel counts, local-gap model, repair rule, target rules, `LegacyRouteEvidence`, `FullFitScoreView`, and selector functions. Replace only alignment and Exp55/56 image calls. Condition A calls frozen public functions directly.

- [ ] **Step 5: Implement variant-aware Hard-route replay**

```python
def replay_hard_route_exp450_antigravity(values, sources, mini, condition, backend):
    if condition.condition_id == "A":
        return replay_hard_route(values, sources, mini)
    evidence = build_legacy_route_evidence_exp450_antigravity(values, condition, backend)
    exp74d, exp89, exp90, exp93 = replay_exp74d_89_90_93(evidence)
    exp137 = replay_exp133_134_135_137(exp93, evidence.confirmation_blocks)
    exp234 = minimum_source_expansion(exp137, sources, alpha=.01)
    exp236 = union_three_source_hard(exp137, exp234)
    promoted = frozenset(set(exp236.standard) & (set(mini.k1024) | set(mini.k4096)))
    exp253 = RouteLanes(frozenset(set(exp236.hard) | set(promoted)), frozenset(set(exp236.standard) - set(promoted)), exp236.priority)
    automatic, removed = exact_normal_guard(exp253.hard, values.train_blobs, values.test_blobs)
    return HardRouteResult(exp74d, exp89, exp90, exp93, exp137, exp234, exp236, promoted, exp253, removed, automatic, len(values.test_rows))
```

- [ ] **Step 6: Run focused GREEN and one capable real-values dataset**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_scoring_antigravity -v
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest paper.equipment_anomaly_detection.tests_paper.test_hard_sources_paper paper.equipment_anomaly_detection.tests_paper.test_minirocket_paper -v
```

The real-values test loads only `ACSF1_normal_0`, asserts A matches the frozen paper source/route bytes, and asserts B--D have finite score vectors with exact TEST length. It does not load labels.

- [ ] **Step 7: Refresh map and commit**

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
git add experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_scoring_antigravity.py docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): score factorial preprocessing variants"
```

---

### Task 5: Add Result-Bound Checkpoints and Two-Wave Recovery

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_artifacts_recovery_antigravity.py`
- Modify: `experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: dataset-local results from Task 4, catalog entries, backend and code-closure identity.
- Produces: `Exp450RunContext`, `write_dataset_condition_success_antigravity`, `read_dataset_condition_success_antigravity`, `append_dataset_condition_failure_antigravity`, `verify_dataset_condition_inventory_antigravity`.

Define the run context exactly once in `contracts_antigravity.py`:

```python
@dataclass(frozen=True, slots=True)
class Exp450RunContext:
    mode: str
    run_root: Path
    database_sha256: str
    catalog_sha256: str
    code_closure: Mapping[str, str]
    backend: Exp450BackendDecision
    config_sha256: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.artifact_record())).hexdigest()
```

- [ ] **Step 1: Write RED round-trip, tamper, and interruption tests**

```python
def test_dataset_condition_checkpoint_round_trips_typed_result(self):
    path = write_dataset_condition_success_antigravity(ROOT, CONTEXT, CATALOG, RAW, RESULT)
    reread = read_dataset_condition_success_antigravity(ROOT, CONTEXT, CATALOG, RAW, CONDITION_A)
    self.assertEqual(canonical_result_bytes_antigravity(reread), canonical_result_bytes_antigravity(RESULT))
    self.assertIn("antigravity", path.name)

def test_tampered_checkpoint_is_rejected(self):
    path = write_dataset_condition_success_antigravity(ROOT, CONTEXT, CATALOG, RAW, RESULT)
    path.write_bytes(path.read_bytes().replace(b'"condition_id":"A"', b'"condition_id":"B"'))
    with self.assertRaisesRegex(ValueError, "checkpoint"):
        read_dataset_condition_success_antigravity(ROOT, CONTEXT, CATALOG, RAW, CONDITION_A)
```

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_artifacts_recovery_antigravity -v
```

Expected: missing artifact APIs.

- [ ] **Step 3: Implement context-bound canonical checkpoint envelopes**

Use the existing typed Task4/5 source and route payload logic, copied into Exp450 with its source file SHA recorded. Each success envelope is exactly:

```python
{
    "schema": "exp450-dataset-condition-success-antigravity.v1",
    "context_sha256": context.sha256,
    "catalog_sha256": context.catalog_sha256,
    "condition_id": result.condition_id,
    "dataset": result.dataset,
    "raw_identity": catalog_entry,
    "backend": context.backend.artifact_record(),
    "payload": {"sources": source_payload, "route": route_payload},
    "payload_sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
}
```

Write create-only paths as `checkpoints_antigravity/{condition_id}_antigravity/{sha256(dataset)}_antigravity/dataset_condition_success_antigravity.json`.

- [ ] **Step 4: Implement append-only failures and two-wave queue reconstruction**

Failure records contain dataset, condition, wave in `{1,2}`, sequential attempt number, exception type/message, context SHA, and record SHA. Success records alone count as completed. Queue order is condition order within catalog order.

- [ ] **Step 5: Run GREEN and byte-identical interrupted/resumed comparison**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_artifacts_recovery_antigravity -v
```

The final test runs a two-dataset fixture uninterrupted and fail-once/resumed in separate roots, then compares every success checkpoint after normalizing only the run-root-derived context field.

- [ ] **Step 6: Refresh map and commit**

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
git add experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_artifacts_recovery_antigravity.py docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): checkpoint each dataset condition"
```

---

### Task 6: Compose Global Scores, Freeze Four Manifests, and Verify Without Labels

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/verifier_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_runner_verifier_antigravity.py`
- Modify: `experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: verified Task 5 checkpoints and unchanged `score_low_frequency`, `classify_stability`, `build_hard_evidence_surface`, `build_policies`.
- Produces: `Exp450Incomplete`, `run_dataset_local_exp450_antigravity`, `compose_condition_score_antigravity`, `freeze_condition_score_antigravity`, `verify_exp450_score_root_antigravity`.

```python
class Exp450Incomplete(RuntimeError):
    def __init__(self, pending: tuple[tuple[str, str], ...]) -> None:
        self.pending = pending
        super().__init__(f"exp450_incomplete pending={len(pending)}")
```

- [ ] **Step 1: Write RED scheduling, global-gate, freeze, and label-firewall tests**

```python
def test_mps_schedule_runs_ab_cpu_then_cd_one_at_a_time(self):
    schedule = build_execution_schedule_antigravity(CATALOG, BACKEND_MPS)
    self.assertEqual(schedule.cpu_conditions, ("A", "B"))
    self.assertEqual(schedule.mps_conditions, ("C", "D"))
    self.assertEqual(schedule.maximum_concurrent_mps_jobs, 1)

def test_global_stages_wait_for_all_dataset_condition_successes(self):
    with patch.object(runner_antigravity, "score_low_frequency", side_effect=AssertionError("global stage called")):
        with self.assertRaisesRegex(Exp450Incomplete, "pending=1"):
            compose_all_conditions_antigravity(INCOMPLETE_CHECKPOINTS, RAW, BACKEND)

def test_four_manifests_verify_before_label_module_can_import(self):
    manifests = freeze_all_conditions_antigravity(SCORES, ROOT, CONTEXT, CATALOG)
    self.assertEqual(tuple(manifests), ("A", "B", "C", "D"))
    self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)
```

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_runner_verifier_antigravity -v
```

Expected: missing runner/verifier APIs.

- [ ] **Step 3: Implement backend-aware local scheduling**

When MPS is selected, finish A/B CPU jobs with `cpu_workers`, close the pool, then run C/D sequentially in the parent process. When CPU is selected, run all conditions through the CPU pool. Persist each success before replenishing the queue; record failures and continue. After wave 2, return a typed incomplete result if any job remains.

- [ ] **Step 4: Compose one global score per condition**

```python
def compose_condition_score_antigravity(condition, raw, results, backend, timings):
    sources = {dataset: results[(dataset, condition.condition_id)].sources for dataset in raw}
    routes = {dataset: results[(dataset, condition.condition_id)].route for dataset in raw}
    hard_lanes = {dataset: route.exp253.hard for dataset, route in routes.items()}
    low = score_low_frequency(raw, hard_lanes)
    stability = classify_stability(low, raw)
    surface = build_hard_evidence_surface(sources, routes, raw)
    policies = build_policies(surface, low, stability, raw, routes)
    return Exp450ConditionScore(condition, raw, sources, routes, low, stability, policies, backend, dict(timings))
```

- [ ] **Step 5: Freeze result-relevant score shards and manifests**

Each condition shard contains dataset TEST identity and all four policy IDs, alert coordinates, `float64` scores, and score masks. The manifest binds ordered dataset descriptors, catalog SHA, condition, backend, context, and checkpoint hashes. The independent verifier imports no scorer or label module and checks exact inventory, hashes, finite vectors, score length, policy order, and `score >= 0.5` alert equivalence.

- [ ] **Step 6: Run GREEN and regression suites**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest discover -s experiments/exp450_resampling_image_ablation_antigravity/tests -p 'test_*_antigravity.py' -v
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper paper.equipment_anomaly_detection.tests_paper.test_stability_paper -v
```

Expected: all tests pass; fresh verifier import leaves scorer and label modules absent.

- [ ] **Step 7: Refresh map and commit**

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
git add experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/verifier_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_runner_verifier_antigravity.py docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): freeze four continuous score roots"
```

---

### Task 7: Evaluate AP/AUROC and Build Correct ROC/PR Curve Tables

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: four verified Exp450 manifests, frozen baseline score roots, and a lazy TEST-label loader.
- Produces: `evaluate_exp450_antigravity`, `metric_tables_exp450_antigravity`, `curve_tables_exp450_antigravity`.

Freeze these baseline roots:

```python
CLASSICAL_ROOT = Path("/Users/mac/Documents/Dataset/paper_public_baselines_classical_v1")
DEEP_SVDD_ROOT = Path("/Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_1d_derived_full_antigravity_20260817")
DROCC_ROOT = Path("/Users/mac/Documents/Dataset/paper_public_baselines_drocc_mps_unverified_antigravity_20260817")
METHOD_ORDER = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")
```

- [ ] **Step 1: Write RED tests for the label gate, metric denominators, axes, and baselines**

```python
def test_labels_are_loaded_only_after_all_four_verifiers_return(self):
    events = []
    def verify(condition):
        events.append(f"verify-{condition}")
        return MANIFESTS[condition]
    def labels(datasets):
        events.append("labels")
        return LABELS
    evaluate_exp450_antigravity(ROOT, baseline_roots=BASELINES, verifier=verify, label_loader=labels)
    self.assertEqual(events, ["verify-A", "verify-B", "verify-C", "verify-D", "labels"])

def test_curve_axes_are_not_swapped(self):
    curves = curve_tables_exp450_antigravity(LABELS, SCORES)
    self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
    self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
    self.assertEqual(curves["precision_recall"]["x_name"], "recall")
    self.assertEqual(curves["precision_recall"]["y_name"], "precision")

def test_all_four_variants_and_six_baselines_are_present(self):
    result = metric_tables_exp450_antigravity(LABELS, ALL_SCORES)
    self.assertEqual(tuple(result.method_order), METHOD_ORDER)
```

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_evaluation_antigravity -v
```

Expected: missing evaluation APIs.

- [ ] **Step 3: Implement frozen score loading before labels**

Verify all four Exp450 manifests and the three public score roots. Require identical database, dataset, and TEST-row identities. Load baseline raw TEST scores for per-dataset metrics and `1 - upper_tail_p_values(TRAIN_OOF, TEST)` for pooled micro ranking, matching the frozen public-baseline protocol. Import the TEST-label loader only after those checks return.

- [ ] **Step 4: Implement metrics and paired statistics**

Use `average_precision_score`, `roc_auc_score`, `roc_curve`, and `precision_recall_curve`. Record per-dataset AP/AUROC or an explicit exclusion reason. Compute macro and micro summaries, B−A/C−A/D−A differences, 20,000 shared bootstrap draws with seed `45020260820`, and `1e-12` win/tie/loss counts.

- [ ] **Step 5: Implement canonical curve tables**

```python
roc_fpr, roc_tpr, roc_threshold = roc_curve(joined_labels, joined_scores)
pr_precision, pr_recall, pr_threshold = precision_recall_curve(joined_labels, joined_scores)
roc_rows = [{"method": method, "false_positive_rate": float(x), "true_positive_rate": float(y), "threshold": float(t)} for x, y, t in zip(roc_fpr, roc_tpr, roc_threshold, strict=True)]
pr_rows = [{"method": method, "recall": float(x), "precision": float(y)} for x, y in zip(pr_recall, pr_precision, strict=True)]
```

Preserve scikit-learn's PR coordinate pairing exactly; do not reverse an axis unless both paired arrays are reversed together for display order.

- [ ] **Step 6: Run GREEN and existing baseline evaluation regressions**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_evaluation_antigravity experiments.paper_public_baselines.tests.test_evaluation_boundary experiments.paper_public_baselines.tests.test_official_comparison_antigravity -v
```

Expected: all tests pass and the label-loader ordering assertion is exact.

- [ ] **Step 7: Refresh map and commit**

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
git add experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_evaluation_antigravity.py docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): evaluate continuous interpolation ablation"
```

---

### Task 8: Add Paper Report, CLI, Runbook, and Full Implementation Verification

**Files:**
- Create: `experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py`
- Create: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_reporting_cli_antigravity.py`
- Create: `scratch/run_exp450_resampling_image_ablation_antigravity.py`
- Create: `docs/EXP450_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP450_RUNBOOK_antigravity_KO.md`
- Create: `docs/EXP450_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: Tasks 1--7.
- Produces: `write_exp450_report_antigravity`, CLI commands `readiness`, `smoke`, `run`, `status`, `verify`, `evaluate`, `report`.

- [ ] **Step 1: Write RED tests for figures, axes, method coverage, and scorer-free status**

```python
def test_report_contains_both_curves_with_correct_axes_and_all_methods(self):
    artifact = write_exp450_report_antigravity(EVALUATION, ROOT)
    html = artifact.html_path.read_text()
    self.assertIn("False-positive rate", html)
    self.assertIn("True-positive rate", html)
    self.assertIn("Recall", html)
    self.assertIn("Precision", html)
    for method in METHOD_ORDER:
        self.assertIn(method, html)

def test_status_imports_no_scorer_or_label_module(self):
    result = subprocess.run([PYTHON, "-B", SCRIPT, "status"], check=True, capture_output=True, text=True)
    self.assertIn('"experiment":"Exp450"', result.stdout)
```

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_reporting_cli_antigravity -v
```

Expected: missing report/CLI APIs.

- [ ] **Step 3: Implement the report writer**

Write canonical metric CSV/JSON, curve CSV/JSON, AP/AUROC tables, paired-difference tables, PNG and SVG figures, and a self-contained HTML report. The report must state that Exp450 is an ablation and makes no automatic promotion claim.

- [ ] **Step 4: Implement the thin CLI**

The CLI maps fixed commands to fixed roots. `status` and `verify` import only artifact/config/verifier code. `evaluate` refuses unless all four full manifests verify. Every long-running command emits one JSON progress line per completed or failed dataset × condition with elapsed time and ETA.

- [ ] **Step 5: Write paper-ready Korean documentation**

`EXP450_PREREGISTRATION` freezes the A--D hypotheses, metrics, axes, and no-selection rule. `EXP450_RUNBOOK` contains exact pinned-interpreter commands and recovery instructions. `EXP450_IMPLEMENTATION_REPORT` lists commits, test counts, backend facts, and leaves result fields explicitly `not_run` until Tasks 9--10.

- [ ] **Step 6: Run all bounded tests and static checks**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest discover -s experiments/exp450_resampling_image_ablation_antigravity/tests -p 'test_*_antigravity.py' -v
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest discover -s experiments/paper_public_baselines/tests -p 'test_*.py' -v
python3 scratch/generate_exp450_codemap_antigravity.py --check
python3 -m py_compile experiments/exp450_resampling_image_ablation_antigravity/*_antigravity.py scratch/run_exp450_resampling_image_ablation_antigravity.py
git diff --check
```

Expected: all commands pass; no production run root is created by tests.

- [ ] **Step 7: Mark implementation ready and commit**

Set Exp450 state to `implementation_ready_awaiting_smoke` with every run/evaluation/label flag false.

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
git add RESEARCH_STATE.json experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py experiments/exp450_resampling_image_ablation_antigravity/tests/test_reporting_cli_antigravity.py scratch/run_exp450_resampling_image_ablation_antigravity.py docs/EXP450_PREREGISTRATION_antigravity_KO.md docs/EXP450_RUNBOOK_antigravity_KO.md docs/EXP450_IMPLEMENTATION_REPORT_antigravity.md docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "feat(exp450): complete ablation execution pipeline"
```

---

### Task 9: Run and Verify the 23-Dataset Values-Only Smoke

**Files:**
- Modify after success: `docs/EXP450_IMPLEMENTATION_REPORT_antigravity.md`
- Modify after success: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`
- Runtime root: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_antigravity_20260820`

**Interfaces:**
- Consumes: implementation-ready Task 8 commit and frozen 23-dataset inventory.
- Produces: four verified smoke manifests and a smoke promotion record; no labels or metrics.

- [ ] **Step 1: Confirm readiness and absent/new root state**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py readiness --mode smoke
```

Expected: database SHA matches, 23 datasets are present, PyTorch 2.13 MPS is available and selected, MLX assessment is recorded, CPU count is 12 and CPU workers are 11, TEST labels are unloaded.

- [ ] **Step 2: Start or resume the smoke**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py smoke 2>&1 | tee /Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_progress_antigravity_20260820.log
```

Expected: 92 dataset × condition successes, no third wave, no label import.

- [ ] **Step 3: Verify all smoke artifacts in a fresh process**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py verify --mode smoke
```

Expected: A--D each contain exactly 23 dataset shards with finite scores and exact TEST row identities.

- [ ] **Step 4: Apply the label-free promotion gate**

Require exact score lengths, fixed image shapes, finite values, unchanged backend, no MPS OOM, all four manifests, and zero label access. Do not inspect AP, AUROC, alerts, or variant superiority.

- [ ] **Step 5: Record smoke evidence and commit**

Update state to `smoke_verified_ready_for_full_1117`, store manifest hashes and backend facts, keep evaluation/label flags false, refresh the map, and commit:

```bash
git add RESEARCH_STATE.json docs/EXP450_IMPLEMENTATION_REPORT_antigravity.md docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "test(exp450): verify 23 dataset interpolation smoke"
```

If the smoke gate fails, stop here, preserve completed same-context checkpoints, fix the defect in a new root if code changes, and do not start Task 10.

---

### Task 10: Run 1,117 Datasets, Evaluate Once, and Publish Paper Artifacts

**Files:**
- Modify: `docs/EXP450_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap_exp450_antigravity.{html,json,lock}`
- Runtime roots: full, evaluation, and report roots frozen in Task 1.

**Interfaces:**
- Consumes: verified smoke promotion, immutable implementation commit, frozen public-baseline score roots.
- Produces: four complete 1,117-dataset score manifests, one label-open evaluation, AP/AUROC/ROC/PR artifacts, and the final Exp450 paper report.

- [ ] **Step 1: Run or resume all dataset-local and global score stages**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py run 2>&1 | tee /Users/mac/Documents/Dataset/exp450_resampling_image_ablation_official_progress_antigravity_20260820.log
```

Expected: 4,468 dataset × condition successes; A/B use CPU worker policy, C/D use one MPS-heavy job at a time; no TEST labels.

- [ ] **Step 2: Verify all four full manifests in a fresh scorer-free process**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py verify --mode full
```

Expected: A--D each contain 1,117 ordered datasets and 195,900 TEST scores; baseline roots match database and TEST identities; label module remains unloaded.

- [ ] **Step 3: Perform the single post-freeze evaluation**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py evaluate
```

Expected: one label-access event after verification; A--D plus six baselines; per-dataset/macro/micro AP and AUROC; paired bootstrap and denominator audit; canonical ROC and PR tables.

- [ ] **Step 4: Generate and verify the paper report**

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py report
```

Expected outputs include:

```text
exp450_metric_summary_antigravity.csv
exp450_dataset_metrics_antigravity.csv
exp450_paired_bootstrap_antigravity.json
exp450_roc_coordinates_antigravity.csv
exp450_precision_recall_coordinates_antigravity.csv
exp450_roc_curve_antigravity.png
exp450_roc_curve_antigravity.svg
exp450_precision_recall_curve_antigravity.png
exp450_precision_recall_curve_antigravity.svg
AI_DRIVEN_EQUIPMENT_INTELLIGENCE_EXP450_ABLATION_ANTIGRAVITY.html
EXP450_RESULTS_SUMMARY_antigravity_KO.md
```

- [ ] **Step 5: Independently validate paper claims**

Recompute aggregate AP/AUROC from the per-dataset table, verify ROC/PR axes and method order, verify A's continuous scores against the frozen Exp449 control where identities match, and confirm no score root changed after label access.

- [ ] **Step 6: Record final evidence and commit only repository documentation**

Set state to `official_ablation_complete_awaiting_discussion`; record all manifest/report hashes, backend, elapsed times, label-access count, metric denominators, and no-promotion status. Refresh the Exp450 triplet and commit:

```bash
git add RESEARCH_STATE.json docs/EXP450_IMPLEMENTATION_REPORT_antigravity.md docs/codemap/codemap_exp450_antigravity.html docs/codemap/codemap_exp450_antigravity.json docs/codemap/codemap_exp450_antigravity.lock
git commit -m "docs(exp450): publish interpolation ablation evidence"
```

Do not change Exp449 artifacts or automatically promote B, C, or D.
