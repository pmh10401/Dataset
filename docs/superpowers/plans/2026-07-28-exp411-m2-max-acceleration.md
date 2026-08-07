# Exp411 M2 Max Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve Exp411's TRAIN-only research semantics and eight-file publication contract while accelerating the 1,117-dataset qualification with 12-thread Numba distance kernels, fold reuse, deterministic numerical safeguards, and resumable dataset checkpoints.

**Architecture:** Keep datasets sequential so only one dataset's waveforms are resident, but introduce an acceleration engine that computes direct float64 RMS distances across 12 Numba threads. Qualification will precompute one identity-to-identity matrix per usable view and reuse it across folds, batch synthetic queries, and recheck numerical boundary cases with NumPy. A separate checkpoint layer will atomically persist fully verified dataset rows and provenance, while the runner will validate complete coverage before invoking the unchanged final publisher.

**Tech Stack:** Python 3.14, NumPy, Numba/llvmlite, SQLite read-only access, `unittest`, macOS `launchctl`, JSON/CSV/SHA256 provenance.

## Global Constraints

- Read only normal TRAIN data; never access TEST split, TEST labels, `labels_blob`, actual anomaly positions, or historical family TEST performance.
- Keep the three views `raw_rms`, `diff_rms`, and `spectrum_rms`, quantiles `0.01 / 0.05 / 0.10`, seed `411`, Exp407 perturbations, gates, folds, and operational lanes unchanged.
- Use direct elementwise float64 RMS; do not use the Gram identity or construct a full `query × fit × length` tensor.
- Numerical distance error versus NumPy direct RMS must be `<= 1e-6`; final representative tags and selected quantile must be identical.
- Use all 12 CPU cores for Numba and set nested BLAS/OpenMP thread counts to `1`.
- Process datasets sequentially and views sequentially; do not keep multiple large datasets or all three full pairwise view matrices resident.
- MLX is excluded; PyTorch MPS is unavailable and is not a canonical execution backend.
- Numba becomes canonical only if warm end-to-end speedup is at least `2.0` on both fixed medium and large TRAIN benchmarks.
- Checkpoint root is `/Users/mac/Documents/Dataset/exp411_near_repeat_qualification.checkpoint`; final output remains `/Users/mac/Documents/Dataset/exp411_near_repeat_qualification`.
- Preserve exactly seven non-manifest artifacts plus `MANIFEST.json`; checkpoints and detailed benchmark metrics are local-only and never final research artifacts.
- Do not start the full 1,117-dataset run until tests, benchmark gates, independent review, and the already-granted full-run authorization are all verified at the implementation commit.

## File Structure

- `experiments/exp411_near_repeat_qualification/acceleration.py`: frozen execution-only policy, Numba kernels, adaptive selection, deterministic parity audit, and boundary-safe distance helpers.
- `experiments/exp411_near_repeat_qualification/checkpoints.py`: canonical dataset-row serialization, provenance validation, atomic checkpoint writes, resume coverage, and progress state.
- `experiments/exp411_near_repeat_qualification/contracts.py`: execution audit dataclasses shared by acceleration, checkpointing, and reporting provenance.
- `experiments/exp411_near_repeat_qualification/qualification.py`: view-at-a-time identity matrix reuse, fold extraction, batched synthetic evaluation, and audit collection.
- `experiments/exp411_near_repeat_qualification/runner.py`: startup gate, resume orchestration, full-coverage verification, benchmark gate consumption, and final publication.
- `experiments/exp411_near_repeat_qualification/benchmark.py`: fixed real-TRAIN benchmark runner and canonical adoption decision.
- `experiments/exp411_near_repeat_qualification/tests/test_acceleration.py`: kernel parity, thread policy, adaptive backend, audit, fold reuse, batching, and boundary tests.
- `experiments/exp411_near_repeat_qualification/tests/test_checkpoints.py`: atomic write, resume, corruption/drift rejection, and coverage tests.
- `experiments/exp411_near_repeat_qualification/tests/test_qualification.py`: baseline-versus-accelerated tag and count parity.
- `experiments/exp411_near_repeat_qualification/tests/test_runner.py`: checkpointed orchestration and fail-closed publication tests.
- `experiments/exp411_near_repeat_qualification/tests/test_benchmark.py`: fixed benchmark universe and adoption gate tests.
- `scratch/run_exp411_near_repeat_qualification.py`: explicit checkpoint and benchmark CLI options.
- `scratch/launch_exp411_full.sh`: validated one-shot `launchctl` job construction with fixed environment and absolute paths.
- `docs/EXP411_RUNBOOK_KO.md`: accelerated validation, benchmark, launch, resume, and result-check commands.

---

### Task 1: Direct 12-Thread RMS Engine and Numerical Audit

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/acceleration.py`
- Modify: `experiments/exp411_near_repeat_qualification/contracts.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_acceleration.py`
- Modify: `experiments/exp411_near_repeat_qualification/tests/test_backends.py`

**Interfaces:**
- Produces: `AccelerationPolicy(numba_threads=12, numpy_cost_max=250_000, audit_pairs=64, synthetic_batch_rows=32, distance_tolerance=1e-6, adoption_speedup=2.0)`.
- Produces: `configure_runtime(policy: AccelerationPolicy) -> Mapping[str, object]`.
- Produces: `full_pairwise_rms(rows: object, *, backend: str) -> np.ndarray`.
- Produces: `minimum_rms(fit: object, query: object, *, backend: str) -> np.ndarray`.
- Produces: `choose_backend(pair_count: int, width: int, policy: AccelerationPolicy) -> str`.
- Produces: `audit_dataset_pairs(rows: np.ndarray, identity_sha256: Sequence[str], policy: AccelerationPolicy) -> DatasetBackendAudit`.
- Produces: `boundary_safe_threshold(fit: np.ndarray, reference: np.ndarray, quantile: float, *, tolerance: float) -> float`.
- Produces: `boundary_safe_minimum(fit: np.ndarray, query: np.ndarray, observed: np.ndarray, thresholds: Sequence[float], *, tolerance: float) -> np.ndarray`, which recomputes values within `1e-6` of any supplied threshold using `cpu_pairwise_rms`.
- Produces: `DatasetBackendAudit(backend, sampled_pairs, max_abs_error, fallback_reason)`.

- [ ] **Step 1: Write failing kernel, policy, and audit tests**

```python
def test_numba_full_matrix_matches_direct_large_offset(self) -> None:
    rows = np.asarray([[1e9, 1e9 + 0.25], [1e9 + 0.5, 1e9 + 0.75]])
    expected = np.sqrt(np.mean((rows[:, None, :] - rows[None, :, :]) ** 2, axis=2))
    observed = full_pairwise_rms(rows, backend="numba_cpu")
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-6)

def test_runtime_uses_twelve_numba_threads_and_one_nested_thread(self) -> None:
    state = configure_runtime(AccelerationPolicy())
    self.assertEqual(state["numba_threads"], 12)
    self.assertEqual(state["nested_threads"], 1)

def test_backend_selection_is_fixed_from_train_shape(self) -> None:
    policy = AccelerationPolicy(numpy_cost_max=100)
    self.assertEqual(choose_backend(5, 4, policy), "numpy_cpu")
    self.assertEqual(choose_backend(6, 4, policy), "numba_cpu")

def test_dataset_audit_is_deterministic_and_falls_back_on_error(self) -> None:
    first = audit_dataset_pairs(rows, digests, AccelerationPolicy(audit_pairs=3))
    second = audit_dataset_pairs(rows, digests, AccelerationPolicy(audit_pairs=3))
    self.assertEqual(first.sampled_pairs, second.sampled_pairs)
    self.assertLessEqual(first.max_abs_error, 1e-6)

def test_quantile_boundary_is_recomputed_by_numpy(self) -> None:
    reference = np.asarray([0.2, 0.3, 0.4, 0.5])
    observed = boundary_safe_threshold(
        rows, reference, 0.05, tolerance=1e-6
    )
    self.assertEqual(observed, cpu_self_excluded_rms(rows).min())
```

- [ ] **Step 2: Run the focused tests and verify missing interfaces fail**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_acceleration \
  experiments.exp411_near_repeat_qualification.tests.test_backends -v
```

Expected: `ImportError` for `acceleration` or missing interface names.

- [ ] **Step 3: Implement the direct parallel kernels and immutable execution policy**

```python
@numba.njit(parallel=True, cache=True)
def _full_pairwise_kernel(rows: np.ndarray) -> np.ndarray:
    count, width = rows.shape
    output = np.empty((count, count), dtype=np.float64)
    for left in numba.prange(count):
        for right in range(count):
            total = 0.0
            for column in range(width):
                delta = rows[left, column] - rows[right, column]
                total += delta * delta
            output[left, right] = math.sqrt(total / width)
    return output

@numba.njit(parallel=True, cache=True)
def _minimum_kernel(fit: np.ndarray, query: np.ndarray) -> np.ndarray:
    output = np.empty(query.shape[0], dtype=np.float64)
    for query_index in numba.prange(query.shape[0]):
        best = np.inf
        for fit_index in range(fit.shape[0]):
            total = 0.0
            for column in range(fit.shape[1]):
                delta = query[query_index, column] - fit[fit_index, column]
                total += delta * delta
            best = min(best, math.sqrt(total / fit.shape[1]))
        output[query_index] = best
    return output
```

Set `numba.set_num_threads(12)`, require `numba.get_num_threads() == 12`, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `"1"` before benchmark work, and record versions and platform. Derive sampled pair indexes only from `sha256(canonical_json_bytes(sorted(identity_sha256)))`; never from computed distances.

- [ ] **Step 4: Implement NumPy boundary rechecks and fallback audit**

```python
def boundary_safe_minimum(
    fit: np.ndarray,
    query: np.ndarray,
    observed: np.ndarray,
    thresholds: Sequence[float],
    *,
    tolerance: float,
) -> np.ndarray:
    result = observed.copy()
    boundary = np.asarray([
        any(abs(float(value) - float(threshold)) <= tolerance
            for threshold in thresholds)
        for value in observed
    ])
    if boundary.any():
        result[boundary] = cpu_pairwise_rms(fit, query[boundary])
    return result
```

Return `backend="numpy_cpu"` with a stable fallback reason when any audited pair exceeds `1e-6`; otherwise preserve the shape-derived backend.
For threshold calibration, locate the `method="lower"` order-statistic identity,
recompute that identity's self-excluded minimum against the fold fit identities
with NumPy direct RMS, and use the NumPy value as the final threshold.

- [ ] **Step 5: Run focused and existing distance tests**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_acceleration \
  experiments.exp411_near_repeat_qualification.tests.test_backends \
  experiments.exp411_near_repeat_qualification.tests.test_distances -v
```

Expected: all tests pass, and large-offset error is `<= 1e-6`.

- [ ] **Step 6: Commit the execution engine**

```bash
git add experiments/exp411_near_repeat_qualification/acceleration.py \
  experiments/exp411_near_repeat_qualification/contracts.py \
  experiments/exp411_near_repeat_qualification/tests/test_acceleration.py \
  experiments/exp411_near_repeat_qualification/tests/test_backends.py
git commit -m "feat: add Exp411 parallel distance engine"
```

### Task 2: View-At-A-Time Fold Reuse and Synthetic Batching

**Files:**
- Modify: `experiments/exp411_near_repeat_qualification/qualification.py`
- Modify: `experiments/exp411_near_repeat_qualification/tests/test_qualification.py`
- Modify: `experiments/exp411_near_repeat_qualification/tests/test_acceleration.py`

**Interfaces:**
- Consumes: Task 1's `full_pairwise_rms`, `minimum_rms`, `choose_backend`, `audit_dataset_pairs`, and `boundary_safe_minimum`.
- Produces: `qualify_dataset(*, dataset: str, family: str, runs: Sequence[TrainRun], backend: str = "numpy_cpu", accelerated: bool = False, backend_audit: list[DatasetBackendAudit] | None = None, parity_trace: list[dict[str, object]] | None = None, config: Exp411FrozenConfig = FROZEN_CONFIG) -> DatasetQualification`.
- Produces: `_precompute_fold_view_state(runs: Sequence[TrainRun], folds: Sequence[IdentityFold], *, policy: AccelerationPolicy, audit: DatasetBackendAudit, config: Exp411FrozenConfig) -> dict[int, dict[int, dict[str, dict[str, object]]]]`, retaining only per-fold thresholds and normal minima after each full view matrix is released.
- Produces: `_batched_synthetic_distances(fit: np.ndarray, query: np.ndarray, *, backend: str, batch_rows: int) -> np.ndarray`, retaining original query/type/order semantics.

- [ ] **Step 1: Add failing fold-reuse, batching, and parity tests**

```python
@patch(
    "experiments.exp411_near_repeat_qualification.qualification.full_pairwise_rms",
    wraps=full_pairwise_rms,
)
def test_each_usable_view_builds_one_identity_matrix(self, matrix_mock) -> None:
    runs = tuple(
        TrainRun(
            dataset="Alpha_normal_0",
            instance_index=index,
            values=tuple(float(index + column * column) for column in range(8)),
            values_sha256=f"{index:064x}",
        )
        for index in range(20)
    )
    qualify_dataset(
        dataset="Alpha_normal_0",
        family="Alpha",
        runs=runs,
        accelerated=True,
    )
    self.assertLessEqual(matrix_mock.call_count, 3)

def test_accelerated_counts_and_trace_match_numpy_baseline(self) -> None:
    runs = tuple(
        TrainRun(
            dataset="Alpha_normal_0",
            instance_index=index,
            values=(0.0, index * 0.1, index * index * 0.02, index * 0.3),
            values_sha256=f"{index:064x}",
        )
        for index in range(6)
    )
    baseline_trace, accelerated_trace = [], []
    baseline = qualify_dataset(
        dataset="Alpha_normal_0", family="Alpha", runs=runs,
        backend="numpy_cpu", parity_trace=baseline_trace,
    )
    accelerated = qualify_dataset(
        dataset="Alpha_normal_0", family="Alpha", runs=runs,
        accelerated=True, parity_trace=accelerated_trace,
    )
    self.assertEqual(accelerated, baseline)
    self.assertEqual(accelerated_trace, baseline_trace)

def test_synthetic_batch_matches_per_query_distance_order(self) -> None:
    expected = np.concatenate([
        minimum_rms(fit, batch[index:index + 1], backend="numpy_cpu")
        for index in range(len(batch))
    ])
    observed = _batched_synthetic_distances(
        fit, batch, backend="numba_cpu", batch_rows=3
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-6)
```

- [ ] **Step 2: Run qualification tests and verify the new behavior fails**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_qualification \
  experiments.exp411_near_repeat_qualification.tests.test_acceleration -v
```

Expected: failures for the missing `accelerated` path and helper functions.

- [ ] **Step 3: Refactor qualification into view-at-a-time precomputation**

For each sequence length, build sorted identity representatives once. For each usable view, scale by the existing MAD rule, call `full_pairwise_rms` once, set the diagonal to infinity only when extracting self-excluded references, and use fold fit/query representative positions to calculate the same lower-quantile thresholds and normal minima as the baseline. Save only:

```python
fold_state[fold_number][length][view_name] = {
    "fit": scaled_fit_for_synthetic_queries,
    "scale": mad,
    "thresholds": {
        key: calibrate_thresholds({view_name: reference}, quantile)[view_name]
        for key, quantile in (("0.01", 0.01), ("0.05", 0.05), ("0.10", 0.10))
    },
    "normal_distance_by_identity": {
        digest: float(distance)
        for digest, distance in zip(query_identity_sha256, normal_minima)
    },
}
```

Delete each full pairwise matrix before advancing to the next view. Keep the existing baseline path callable for parity tests and fallback.

- [ ] **Step 4: Batch Exp407 synthetic queries without changing seed or order**

Generate perturbations with the existing `_derived_seed(dataset, fold_number, query_identity)` and group only the distance calls by fold, view, and perturbation type. Slice batches in original query order using `AccelerationPolicy.synthetic_batch_rows`, then restore results to `(query identity, perturbation type, candidate index)` before calling unchanged `tag_queries`.

- [ ] **Step 5: Recheck threshold boundaries with NumPy and run parity tests**

Use `boundary_safe_threshold` for each fold/view calibration order statistic.
Then use `boundary_safe_minimum` for normal and synthetic minima whenever a
computed value is within `1e-6` of any of the three thresholds for that view.
Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_qualification \
  experiments.exp411_near_repeat_qualification.tests.test_acceleration \
  experiments.exp411_near_repeat_qualification.tests.test_folds -v
```

Expected: accelerated and baseline `DatasetQualification` objects and full parity traces are identical.

- [ ] **Step 6: Commit fold reuse and batching**

```bash
git add experiments/exp411_near_repeat_qualification/qualification.py \
  experiments/exp411_near_repeat_qualification/tests/test_qualification.py \
  experiments/exp411_near_repeat_qualification/tests/test_acceleration.py
git commit -m "feat: reuse Exp411 fold distances"
```

### Task 3: Atomic Dataset Checkpoints and Validated Resume

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/checkpoints.py`
- Modify: `experiments/exp411_near_repeat_qualification/contracts.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_checkpoints.py`

**Interfaces:**
- Produces: `CheckpointProvenance(git_commit, database_snapshot, population_manifest_sha256, frozen_config_sha256, perturbation_contract_sha256, acceleration_contract_sha256)`.
- Produces: `CheckpointStore(root: Path, provenance: CheckpointProvenance, expected_datasets: Sequence[str])`.
- Produces: `CheckpointStore.load_completed() -> Mapping[str, DatasetQualification]`.
- Produces: `CheckpointStore.write_dataset(row, audit, dataset_train_identity_sha256, elapsed_seconds) -> None`.
- Produces: `CheckpointStore.write_progress(current_dataset: str | None) -> None`.
- Produces: `CheckpointStore.require_complete(expected_families: int) -> tuple[DatasetQualification, ...]`.

- [ ] **Step 1: Write failing checkpoint tests**

```python
def test_dataset_checkpoint_round_trips_and_resume_skips_completed(self) -> None:
    store = CheckpointStore(root, provenance, ("Alpha_normal_0",))
    store.write_dataset(row, audit, train_hash, 1.25)
    self.assertEqual(store.load_completed(), {"Alpha_normal_0": row})

def test_provenance_drift_is_rejected(self) -> None:
    store = CheckpointStore(root, provenance, ("Alpha_normal_0",))
    store.write_dataset(row, audit, train_hash, 1.25)
    with self.assertRaisesRegex(ValueError, "provenance"):
        CheckpointStore(
            root,
            replace(provenance, git_commit="f" * 40),
            ("Alpha_normal_0",),
        )

def test_corrupt_or_duplicate_checkpoint_is_rejected(self) -> None:
    checkpoint.write_text("{", encoding="utf-8")
    with self.assertRaisesRegex(ValueError, "checkpoint"):
        store.load_completed()

def test_incomplete_coverage_cannot_publish(self) -> None:
    with self.assertRaisesRegex(ValueError, "coverage"):
        store.require_complete(expected_families=128)
```

- [ ] **Step 2: Run checkpoint tests and verify import failure**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_checkpoints -v
```

Expected: `ImportError` for `checkpoints`.

- [ ] **Step 3: Implement canonical serialization and provenance contracts**

Hash canonical JSON for the frozen config, perturbation contract, acceleration module source/policy, expected dataset list, and each dataset's ordered `(instance_index, values_sha256)` list. A dataset file must contain:

```json
{
  "schema_version": 1,
  "dataset": "Alpha_normal_0",
  "dataset_train_identity_sha256": "<64 hex>",
  "qualification": {},
  "backend_audit": {},
  "elapsed_seconds": 1.25,
  "payload_sha256": "<64 hex>"
}
```

Reject unknown datasets, duplicate dataset names, malformed counters, wrong hashes, non-finite elapsed times, and any root provenance mismatch.

- [ ] **Step 4: Implement atomic writes and progress**

Write bytes to a uniquely named sibling using exclusive creation, flush and `os.fsync`, `os.replace` into `datasets/<encoded-name>.json`, then fsync the directory. Write `progress.json` the same way with completed count, expected count, current dataset, and last completion time. Never write a dataset checkpoint until qualification and its parity audit have completed successfully.

- [ ] **Step 5: Run checkpoint tests**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_checkpoints -v
```

Expected: all checkpoint, drift, corruption, and coverage cases pass.

- [ ] **Step 6: Commit checkpoint support**

```bash
git add experiments/exp411_near_repeat_qualification/checkpoints.py \
  experiments/exp411_near_repeat_qualification/contracts.py \
  experiments/exp411_near_repeat_qualification/tests/test_checkpoints.py
git commit -m "feat: add Exp411 resumable checkpoints"
```

### Task 4: Fixed TRAIN Benchmark and Canonical Adoption Gate

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/benchmark.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_benchmark.py`
- Modify: `experiments/exp411_near_repeat_qualification/config.py`
- Modify: `experiments/exp411_near_repeat_qualification/runner.py`
- Modify: `experiments/exp411_near_repeat_qualification/tests/test_runner.py`

**Interfaces:**
- Consumes: Tasks 1–2 acceleration and qualification interfaces.
- Produces: `run_fixed_train_benchmark(db_path: Path) -> Mapping[str, object]`.
- Produces: `require_acceleration_adoption(report: Mapping[str, object]) -> None`.
- Benchmark datasets are exactly `ACSF1_normal_0`, `HandOutlines_normal_0`, and `StarLightCurves_normal_3`.

- [ ] **Step 1: Write failing benchmark contract tests**

```python
def test_fixed_benchmark_datasets_cannot_drift(self) -> None:
    self.assertEqual(
        BENCHMARK_DATASETS,
        ("ACSF1_normal_0", "HandOutlines_normal_0", "StarLightCurves_normal_3"),
    )

def test_adoption_requires_medium_and_large_two_x_speedup(self) -> None:
    passing = report(medium=2.01, large=2.40, error=1e-7, tags=True, quantile=True)
    require_acceleration_adoption(passing)
    with self.assertRaisesRegex(ValueError, "speedup"):
        require_acceleration_adoption(
            report(medium=1.99, large=2.40, error=1e-7, tags=True, quantile=True)
        )

def test_adoption_rejects_tag_or_quantile_drift(self) -> None:
    with self.assertRaisesRegex(ValueError, "parity"):
        require_acceleration_adoption(
            report(medium=2.1, large=2.4, error=1e-7, tags=False, quantile=True)
        )
```

- [ ] **Step 2: Run benchmark tests and verify missing module failure**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_benchmark -v
```

Expected: `ImportError` for `benchmark`.

- [ ] **Step 3: Implement fixed cold/warm real-TRAIN benchmark**

For each fixed dataset, load only normal TRAIN through the existing loader contract, measure NumPy baseline, reset and measure Numba cold, then measure Numba warm. Record wall seconds, peak RSS, max absolute audit error, trace SHA256, selected quantile, backend/thread environment, Python/NumPy/Numba/llvmlite versions, and input identity SHA256. Do not read TEST and do not alter thresholds based on timings.

- [ ] **Step 4: Implement fail-closed adoption**

Require:

```python
medium["warm_speedup"] >= 2.0
large["warm_speedup"] >= 2.0
max(dataset["max_abs_error"] for dataset in datasets) <= 1e-6
all(dataset["tag_identity"] for dataset in datasets)
all(dataset["selected_quantile_identity"] for dataset in datasets)
runtime["numba_threads"] == 12
runtime["nested_threads"] == 1
```

Keep the research config's existing semantic values unchanged; add execution-only constants through `AccelerationPolicy`, not by changing thresholds or gates in `Exp411FrozenConfig`.

- [ ] **Step 5: Run benchmark and runner unit tests**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_benchmark \
  experiments.exp411_near_repeat_qualification.tests.test_runner -v
```

Expected: all tests pass with mocked timings, and the runner refuses a missing or failed benchmark report.

- [ ] **Step 6: Commit benchmark gate**

```bash
git add experiments/exp411_near_repeat_qualification/benchmark.py \
  experiments/exp411_near_repeat_qualification/config.py \
  experiments/exp411_near_repeat_qualification/runner.py \
  experiments/exp411_near_repeat_qualification/tests/test_benchmark.py \
  experiments/exp411_near_repeat_qualification/tests/test_runner.py
git commit -m "feat: gate Exp411 acceleration by TRAIN benchmark"
```

### Task 5: Resume-Aware Full Runner and Fail-Closed Publication

**Files:**
- Modify: `experiments/exp411_near_repeat_qualification/runner.py`
- Modify: `experiments/exp411_near_repeat_qualification/tests/test_runner.py`
- Modify: `experiments/exp411_near_repeat_qualification/tests/test_structure.py`
- Modify: `scratch/run_exp411_near_repeat_qualification.py`

**Interfaces:**
- Consumes: Tasks 1–4 interfaces.
- Produces: `run_full_qualification(db_path, population_manifest, output_dir, *, checkpoint_dir, benchmark_report) -> Mapping[str, str]`.
- The final renderer and publisher interfaces remain unchanged.

- [ ] **Step 1: Write failing resume orchestration tests**

```python
def test_full_runner_skips_valid_completed_dataset(self) -> None:
    store.write_dataset(alpha_row, alpha_audit, alpha_train_hash, 1.0)
    with patch(
        "experiments.exp411_near_repeat_qualification.runner.qualify_dataset",
        wraps=qualify_dataset,
    ) as qualify:
        run_full_qualification(
            db_path, population_manifest, output,
            checkpoint_dir=root, benchmark_report=passing_report,
        )
    self.assertNotIn(
        "Alpha_normal_0",
        {call.kwargs["dataset"] for call in qualify.call_args_list},
    )

def test_dataset_error_preserves_existing_checkpoints_and_does_not_publish(self) -> None:
    store.write_dataset(alpha_row, alpha_audit, alpha_train_hash, 1.0)
    with patch(
        "experiments.exp411_near_repeat_qualification.runner.qualify_dataset",
        side_effect=RuntimeError("boom"),
    ), self.assertRaisesRegex(RuntimeError, "boom"):
        run_full_qualification(
            db_path, population_manifest, output,
            checkpoint_dir=root, benchmark_report=passing_report,
        )
    self.assertTrue(alpha_checkpoint.exists())
    self.assertFalse(output.exists())

def test_publication_requires_exact_dataset_and_family_coverage(self) -> None:
    with patch(
        "experiments.exp411_near_repeat_qualification.runner.iter_train_runs",
        return_value=iter(()),
    ), self.assertRaisesRegex(ValueError, "coverage"):
        run_full_qualification(
            db_path, population_manifest, output,
            checkpoint_dir=root, benchmark_report=passing_report,
        )
    self.assertFalse(output.exists())
```

Build these tests with a temporary one-dataset SQLite fixture, a manifest whose
hash is injected through `replace(FROZEN_CONFIG, expected_datasets=1,
expected_families=1, database_sha256=sha256_file(db_path),
population_manifest_sha256=sha256_file(population_manifest))`, and a complete
passing benchmark mapping.
Patch only the frozen universe counts/hashes at the runner boundary; do not weaken
the production `FROZEN_CONFIG.validate()` path.

- [ ] **Step 2: Run runner tests and verify new signature/behavior fails**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_runner \
  experiments.exp411_near_repeat_qualification.tests.test_structure -v
```

Expected: failures for checkpoint and benchmark arguments.

- [ ] **Step 3: Integrate startup gate and resume loop**

At startup, require a clean Git commit, pinned DB snapshot, manifest universe, passing fixed benchmark report, 12-thread runtime state, and matching checkpoint provenance. For each streamed dataset:

```python
if dataset in completed:
    require_matching_dataset_train_identity(dataset, runs, checkpoint)
    rows.append(completed[dataset])
    continue
store.write_progress(current_dataset=dataset)
row = qualify_dataset(
    dataset=dataset,
    family=_family_from_dataset(dataset),
    runs=runs,
    accelerated=True,
    backend_audit=audits,
)
store.write_dataset(row, audits[-1], train_identity_hash(runs), elapsed)
rows.append(row)
```

On any exception, leave prior checkpoint files untouched, record no checkpoint for the failed dataset, and propagate the error.

- [ ] **Step 4: Enforce exact coverage before unchanged publication**

Call `store.require_complete(expected_families=128)`, verify the returned dataset ordering equals the sorted population manifest, then aggregate and render the existing seven payloads. The final pre-publish callback must recheck Git, DB snapshot, config/provenance hashes, checkpoint coverage, and output absence immediately before atomic publication.

- [ ] **Step 5: Extend the CLI**

Add required full-mode flags:

```text
--checkpoint-dir PATH
--benchmark-report PATH
```

Smoke mode must reject both. Full mode must reject an existing final output but allow an existing, valid checkpoint directory. Preserve explicit `--full-qualification`.

- [ ] **Step 6: Run integration and structure tests**

Run:

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp411_near_repeat_qualification/tests \
  -p "test_*.py"
```

Expected: all Exp411 tests pass; forbidden TEST/label identifier guards still pass.

- [ ] **Step 7: Commit resume-aware orchestration**

```bash
git add experiments/exp411_near_repeat_qualification/runner.py \
  experiments/exp411_near_repeat_qualification/tests/test_runner.py \
  experiments/exp411_near_repeat_qualification/tests/test_structure.py \
  scratch/run_exp411_near_repeat_qualification.py
git commit -m "feat: resume Exp411 full qualification safely"
```

### Task 6: One-Shot macOS Launch Job and Runbook

**Files:**
- Create: `scratch/launch_exp411_full.sh`
- Modify: `docs/EXP411_RUNBOOK_KO.md`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_launch_contract.py`

**Interfaces:**
- Consumes: Task 5 CLI.
- Produces: a launch script that prints the job label, log paths, and one-shot `launchctl submit` command but never polls.

- [ ] **Step 1: Write a failing static launch-contract test**

```python
def test_launch_script_pins_environment_and_paths(self) -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "NUMBA_NUM_THREADS=12",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "--full-qualification",
        "--checkpoint-dir",
        "--benchmark-report",
        "/private/tmp/exp411-full.stdout.log",
        "/private/tmp/exp411-full.stderr.log",
    ):
        self.assertIn(required, text)
    self.assertNotIn("while ", text)
    self.assertNotIn("sleep ", text)
```

- [ ] **Step 2: Run the test and verify the script is missing**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_launch_contract -v
```

Expected: failure because `scratch/launch_exp411_full.sh` does not exist.

- [ ] **Step 3: Implement the one-shot launcher**

Use fixed absolute worktree, `.venv/bin/python`, DB, manifest, checkpoint, output, benchmark report, stdout, and stderr paths. Export all nested thread variables as `1`, export `NUMBA_NUM_THREADS=12`, validate every input before submitting, reject an existing final output, and call:

```bash
launchctl submit \
  -l "$job_label" \
  -o "/private/tmp/exp411-full.stdout.log" \
  -e "/private/tmp/exp411-full.stderr.log" \
  -- "$python_path" "$runner_path" \
  --full-qualification \
  --db-path "$db_path" \
  --population-manifest "$population_manifest" \
  --checkpoint-dir "$checkpoint_dir" \
  --benchmark-report "$benchmark_report" \
  --output-dir "$output_dir"
```

Do not contain polling or monitoring loops.

- [ ] **Step 4: Update the Korean runbook**

Document dependency verification, unit/regression commands, benchmark generation and inspection, launch, resume after interruption, user-requested progress inspection, manifest verification, stop conditions, high CPU/heat expectation, and the rule that checkpoint/full artifacts remain local. Explicitly state that launch is not monitoring and that no automatic polling is configured.

- [ ] **Step 5: Run launch-contract and CLI help checks**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_launch_contract -v
.venv/bin/python scratch/run_exp411_near_repeat_qualification.py --help
bash -n scratch/launch_exp411_full.sh
```

Expected: test passes, CLI lists checkpoint and benchmark flags, and shell syntax is valid.

- [ ] **Step 6: Commit launch and operations documentation**

```bash
git add scratch/launch_exp411_full.sh \
  docs/EXP411_RUNBOOK_KO.md \
  experiments/exp411_near_repeat_qualification/tests/test_launch_contract.py
git commit -m "docs: add Exp411 accelerated launch runbook"
```

### Task 7: Full Verification, Real TRAIN Benchmark, and Review Gate

**Files:**
- Modify only if verification reveals an implementation defect in Tasks 1–6.
- Generate local-only benchmark report: `/Users/mac/Documents/Dataset/exp411_acceleration_benchmark.json`.

**Interfaces:**
- Consumes all prior tasks.
- Produces a clean implementation commit that is eligible for the authorized one-shot full run.

- [ ] **Step 1: Run all Exp411 tests**

Run:

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp411_near_repeat_qualification/tests \
  -p "test_*.py"
```

Expected: all tests pass with zero errors and failures.

- [ ] **Step 2: Run Exp407 and Exp410 regressions**

Run:

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp403_408/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp410_frozen_hard_family_robustness_audit/tests \
  -p "test_*.py"
```

Expected: all tests pass; Exp407 perturbation contract hash remains pinned.

- [ ] **Step 3: Run static verification**

Run:

```bash
.venv/bin/python -m compileall \
  experiments/exp411_near_repeat_qualification \
  scratch/run_exp411_near_repeat_qualification.py
git diff --check
git status --short --branch
```

Expected: compile succeeds, `git diff --check` prints nothing, and only intentional changes are present before their final commit.

- [ ] **Step 4: Generate the fixed real-TRAIN benchmark once**

Run with `NUMBA_NUM_THREADS=12` and every nested thread variable set to `1`:

```bash
.venv/bin/python scratch/run_exp411_near_repeat_qualification.py \
  --acceleration-benchmark \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --benchmark-report "/Users/mac/Documents/Dataset/exp411_acceleration_benchmark.json"
```

Expected: small/medium/large entries exist; every maximum error is `<=1e-6`; tags and selected quantiles match; medium and large warm speedups are both `>=2.0`.

- [ ] **Step 5: Run representative smoke with the adopted engine**

Run:

```bash
smoke_dir="$(mktemp -d /private/tmp/exp411-smoke.XXXXXX)"
rmdir "$smoke_dir"
.venv/bin/python scratch/run_exp411_near_repeat_qualification.py \
  --representative-smoke \
  --output-dir "$smoke_dir"
```

Expected: seven non-manifest artifacts plus `MANIFEST.json`, all manifest hashes verify, and baseline/canonical tag hashes and selected quantile are identical.

- [ ] **Step 6: Request independent code review**

Review the implementation against:

```text
docs/superpowers/specs/2026-07-28-exp411-m2-max-acceleration-design.md
docs/superpowers/plans/2026-07-28-exp411-m2-max-acceleration.md
```

Require explicit findings for numerical correctness, TRAIN-only isolation, memory bounds, checkpoint provenance, publication coverage, and launch safety. Fix every confirmed issue with a failing regression test first.

- [ ] **Step 7: Record verification and commit any review fixes**

If fixes were necessary:

```bash
git add experiments/exp411_near_repeat_qualification \
  scratch/run_exp411_near_repeat_qualification.py \
  scratch/launch_exp411_full.sh docs/EXP411_RUNBOOK_KO.md
git commit -m "fix: harden Exp411 accelerated execution"
```

Re-run Steps 1–5 after the final fix commit. Do not proceed if the benchmark gate fails; keep NumPy canonical and report the failed criterion.

### Task 8: Start the Authorized Full Run Without Polling

**Files:**
- No repository changes.
- Runtime checkpoint: `/Users/mac/Documents/Dataset/exp411_near_repeat_qualification.checkpoint`
- Runtime output: `/Users/mac/Documents/Dataset/exp411_near_repeat_qualification`
- Runtime logs: `/private/tmp/exp411-full.stdout.log` and `/private/tmp/exp411-full.stderr.log`

**Interfaces:**
- Consumes the clean, reviewed implementation commit and passing benchmark report from Task 7.
- Produces a submitted one-shot macOS job; completion is checked only when the user later requests it.

- [ ] **Step 1: Revalidate launch preconditions**

Run:

```bash
git status --porcelain
git rev-parse HEAD
shasum -a 256 "/Users/mac/Documents/Dataset/univariate_ts.db"
sqlite3 "/Users/mac/Documents/Dataset/univariate_ts.db" "PRAGMA quick_check;"
.venv/bin/python -c 'import json; from pathlib import Path; from experiments.exp411_near_repeat_qualification.benchmark import require_acceleration_adoption; require_acceleration_adoption(json.loads(Path("/Users/mac/Documents/Dataset/exp411_acceleration_benchmark.json").read_text()))'
```

Expected: clean worktree, pinned database SHA256, `ok`, and benchmark adoption succeeds.

- [ ] **Step 2: Submit the one-shot job**

Run:

```bash
bash scratch/launch_exp411_full.sh
```

Expected: the script prints a unique job label and the fixed log/checkpoint/output paths, then exits after successful submission.

- [ ] **Step 3: Stop active monitoring**

Do not poll `launchctl`, logs, `progress.json`, or the output directory. Report only that submission succeeded and provide the paths the user can ask to inspect later.
