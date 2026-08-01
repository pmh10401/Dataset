# Exp421 MPS Batched Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, TRAIN-only Exp421 prototype that can prove or reject a batched MPS execution path for the frozen Exp417 causal multi-horizon workflow without changing the official Exp419 run or any scientific decision rule.

**Architecture:** A new `experiments/exp421_mps_batched_acceleration` package wraps the frozen Exp417 model as a narrow adapter, reuses the exact NumPy scoring and perturbation definitions, and adds immutable calibration/perturbation caches. A predeclared hybrid policy chooses CPU for batch one and MPS only after an isolated benchmark proves finite outputs, coordinate parity, numerical/rank parity, provenance completeness, and at least `1.1x` speedup including transfers and synchronization. A bounded CLI emits only local diagnostic artifacts under a new attempt root; it never calls TEST SQL and never edits Exp417 or Exp419 code.

**Tech Stack:** Python 3.14, PyTorch 2.13.0, NumPy, SQLite read-only TRAIN loader, `unittest`, MPS (`torch.backends.mps`), optional MLX version detection only, canonical SHA-256 provenance, and the existing Exp417 reference modules.

## Global Constraints

- Preserve Exp417 architecture, horizons `(1, ceil(.02L), ceil(.10L))`, Huber loss, AdamW settings, model seeds, perturbation seeds, alpha, persistence, qualification gates, frozen lanes, and publication meaning byte-for-byte in the reference path.
- Load normal TRAIN values only during Exp421; do not issue a TEST query, read TEST labels/locations, use human review, use historical TEST metrics, or choose a backend/threshold/source/budget retrospectively.
- Do not use tail replacement, do not promote Priority review or Shadow candidates, and do not publish predictions, private records, or metrics to GitHub.
- Do not modify `experiments/exp417_causal_multihorizon_shadow/*` or `experiments/exp419_deterministic_cpu_parallel/*`; import their frozen public/reference functions through a new adapter only.
- Candidate batch sizes are exactly `(1, 4, 8, 16, 32)`; CPU is mandatory at batch one and is the fail-closed fallback.
- MPS-heavy work is one process at a time and is never combined with outer multiprocess workers unless a separately approved isolated deterministic benchmark explicitly proves that combination faster.
- MLX is not installed and is not implemented in this prototype; the manifest must record MLX and `mlx-metal` version absence, eligibility result, and the exact fallback reason.
- MPS may be selected only when outputs are finite, candidate coordinates are identical, normalized maximum error is at most `1e-4`, Spearman rank is at least `0.999`, all compared frozen evidence matches, and full-stage CPU-over-MPS speedup including input/output transfer and `torch.mps.synchronize()` is at least `1.1x`.
- Every benchmark and artifact uses a fresh no-clobber attempt directory and records Git commit, database SHA-256, frozen scientific-config digest, device, PyTorch version, MLX/`mlx-metal` versions, batch policy, timings, parity, and fallback evidence.
- The prototype is diagnostic only; no full 1,117-dataset Exp419/Exp417 rerun is part of this plan and the running official Exp419 process remains untouched.

---

## File Map and Boundaries

The implementation is intentionally additive. The new package owns runtime adapters, contracts, caches, policy, benchmark, and manifest serialization. Existing Exp417 modules remain the scientific reference and are not edited.

- Create `experiments/exp421_mps_batched_acceleration/__init__.py`: package version and narrow public exports.
- Create `experiments/exp421_mps_batched_acceleration/config.py`: immutable Exp421 gates, exact batch list, memory/benchmark bounds, and frozen source identifiers.
- Create `experiments/exp421_mps_batched_acceleration/contracts.py`: validated immutable records for batch keys, predictions, calibration entries, perturbation entries, measurements, parity, MLX assessment, policy, and manifest.
- Create `experiments/exp421_mps_batched_acceleration/batch_inference.py`: materialize a serialized `TrainedSeedModel` once, run batched residual inference, and reproduce frozen Huber/AdamW fitting for a bounded batch without changing the reference function.
- Create `experiments/exp421_mps_batched_acceleration/calibration.py`: build and consume a TRAIN OOF calibration cache while preserving `score_horizon` results.
- Create `experiments/exp421_mps_batched_acceleration/perturbations.py`: cache deterministic primary perturbations and seed-keyed secondary spikes using `build_fixed_perturbations`.
- Create `experiments/exp421_mps_batched_acceleration/policy.py`: evaluate gates and return a fail-closed CPU/MPS hybrid policy.
- Create `experiments/exp421_mps_batched_acceleration/manifest.py`: canonical manifest construction, validation, and atomic JSON writing.
- Create `experiments/exp421_mps_batched_acceleration/benchmark.py`: bounded normal-TRAIN loading, exact-length preprocessing, isolated CPU/MPS measurements, parity checks, and resource bounds.
- Create `experiments/exp421_mps_batched_acceleration/tests/`: `unittest` regression tests for every scientific and boundary contract.
- Create `scratch/run_exp421_mps_batched_acceleration.py`: explicit CLI for a fresh local benchmark attempt; it accepts a database path and output root, but has no TEST path.
- Create `docs/EXP421_PREREGISTRATION_KO.md`: Korean preregistration of the diagnostic question, gates, and non-publication status.
- Create `docs/EXP421_RUNBOOK_KO.md`: Korean run, inspection, failure, and rollback instructions.
- Create `SOURCE_MANIFEST.json`: source file paths and SHA-256 digests used by the prototype.

### Shared interfaces

Later tasks use these exact names and types:

```python
# config.py
@dataclass(frozen=True)
class Exp421Config:
    experiment_id: str = "Exp421"
    candidate_batch_sizes: tuple[int, ...] = (1, 4, 8, 16, 32)
    max_normalized_error: float = 1e-4
    min_spearman: float = 0.999
    min_speedup: float = 1.1
    max_benchmark_series: int = 64
    max_batch_bytes: int = 256 * 1024 * 1024
    frozen_model_seeds: tuple[int, ...] = (41701, 41702, 41703)
    frozen_perturbation_seeds: tuple[int, ...] = (41711, 41712, 41713)
```

```python
# batch_inference.py
class FrozenModelAdapter:
    @classmethod
    def from_trained_seed(cls, model: TrainedSeedModel, device: torch.device) -> "FrozenModelAdapter": ...
    def predict_batch(self, channels: np.ndarray) -> Mapping[int, np.ndarray]: ...
    def fit_batch(self, channels: Mapping[int, np.ndarray], *, seed: int,
                  initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
                  fit_indices: Sequence[int], validation_indices: Sequence[int],
                  early_stop_fit_indices: Sequence[int],
                  early_stop_validation_indices: Sequence[int]) -> TrainedSeedModel: ...
```

```python
# calibration.py
@dataclass(frozen=True)
class CalibrationCache: ...
def build_calibration_cache(oof_residuals: Sequence[np.ndarray], *, horizon: int) -> CalibrationCache: ...
def score_query_with_cache(cache: CalibrationCache, query_residual: np.ndarray) -> HorizonScore: ...
```

```python
# perturbations.py
def build_cached_perturbations(dataset: str, instance_index: int, series: np.ndarray,
                                robust_scale: float, *, seed: int) -> Mapping[str, np.ndarray]: ...
```

```python
# policy.py
def choose_backend(measurements: Sequence[BackendMeasurement], *, config: Exp421Config) -> BackendPolicy: ...
```

```python
# benchmark.py
def run_train_only_benchmark(database: Path, output_root: Path, *, config: Exp421Config,
                              dataset_names: Sequence[str] | None = None) -> Exp421Manifest: ...
```

---

### Task 1: Freeze Exp421 configuration and provenance contracts

**Files:**
- Create: `experiments/exp421_mps_batched_acceleration/__init__.py`
- Create: `experiments/exp421_mps_batched_acceleration/config.py`
- Create: `experiments/exp421_mps_batched_acceleration/contracts.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/__init__.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_config.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_contracts.py`

**Interfaces:**
- Consumes: `Exp417FrozenConfig`, `FROZEN_CONFIG`, and the existing `TrainedSeedModel`/`HorizonScore` types.
- Produces: validated `Exp421Config`, `BatchKey`, `BackendMeasurement`, `ParityReport`, `MlxAssessment`, `BackendPolicy`, and `Exp421Manifest` records used by the following tasks.

- [ ] **Step 1: Write failing configuration tests.** Assert the exact batch tuple, gates, frozen seeds, positive resource bounds, and rejection of a changed candidate batch, changed gate, or changed frozen seed.

```python
def test_config_is_exactly_preregistered(self):
    cfg = Exp421Config()
    self.assertEqual(cfg.candidate_batch_sizes, (1, 4, 8, 16, 32))
    self.assertEqual(cfg.max_normalized_error, 1e-4)
    self.assertEqual(cfg.min_spearman, 0.999)
    self.assertEqual(cfg.min_speedup, 1.1)
    self.assertEqual(cfg.frozen_model_seeds, (41701, 41702, 41703))
    self.assertEqual(cfg.frozen_perturbation_seeds, (41711, 41712, 41713))
```

- [ ] **Step 2: Run the focused tests to verify the new names fail.**

Run:

```bash
cd "/Users/mac/Documents/Time Series/.worktrees/exp421-mps-batched-acceleration"
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_config -v
```

Expected: import failure because the new package and records do not yet exist.

- [ ] **Step 3: Implement immutable configuration and records.** Validate finite thresholds, sorted unique batch sizes equal to `(1, 4, 8, 16, 32)`, exact seed tuples, nonnegative timing values, finite parity values, and a manifest state of only `draft`, `passed`, or `failed`. Store arrays as read-only copies and mappings as `MappingProxyType`.

```python
@dataclass(frozen=True)
class BackendMeasurement:
    batch_size: int
    backend: str
    transfer_seconds: float
    compute_seconds: float
    synchronize_seconds: float
    total_seconds: float
    finite: bool
    candidate_coordinates_sha256: str
```

- [ ] **Step 4: Add negative tests for scientific immutability.** Verify that a TEST token in a manifest boundary field, `batch_size=2`, a nonfinite timing, a speedup below `1.1`, a missing MLX fallback reason, and a writable cached array are rejected before serialization.

- [ ] **Step 5: Run the focused tests and commit.**

Run:

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest discover -s experiments/exp421_mps_batched_acceleration/tests -p 'test_config.py' -v
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest discover -s experiments/exp421_mps_batched_acceleration/tests -p 'test_contracts.py' -v
```

Expected: all configuration and contract tests pass.

- [ ] **Step 6: Commit the task.**

```bash
git add experiments/exp421_mps_batched_acceleration
git commit -m "feat(exp421): add frozen acceleration contracts"
```

### Task 2: Add the frozen-model batch adapter with CPU reference parity

**Files:**
- Create: `experiments/exp421_mps_batched_acceleration/batch_inference.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_batch_inference.py`

**Interfaces:**
- Consumes: `TrainedSeedModel`, `CausalMultiHorizonCNN`, frozen state serialization, and the exact Exp417 `predict_residuals` reference.
- Produces: `FrozenModelAdapter.from_trained_seed`, `FrozenModelAdapter.predict_batch`, and `FrozenModelAdapter.fit_batch`.

- [ ] **Step 1: Build a deterministic fixture and write red parity tests.** Construct one valid small `TrainedSeedModel` fixture from a frozen state and two-channel float32 arrays at a valid length. For batch sizes `1`, `4`, `8`, `16`, and `32`, compare each returned horizon residual row with repeated `predict_residuals` calls; assert shape, finiteness, read-only output, and identical candidate-coordinate hashes.

```python
def test_predict_batch_matches_single_series_reference_for_every_batch_size(self):
    for batch_size in (1, 4, 8, 16, 32):
        batch = np.repeat(self.channels[None, ...], batch_size, axis=0)
        actual = self.adapter.predict_batch(batch)
        expected = {
            horizon: np.stack(
                [predict_residuals(self.model, row)[horizon] for row in batch]
            )
            for horizon in self.model.horizons
        }
        for horizon in self.model.horizons:
            np.testing.assert_allclose(actual[horizon], expected[horizon], rtol=0, atol=1e-5)
```

- [ ] **Step 2: Run the parity test to verify it fails.**

Run:

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_batch_inference.FrozenModelAdapterTests.test_predict_batch_matches_single_series_reference_for_every_batch_size -v
```

Expected: failure because `FrozenModelAdapter` is not implemented.

- [ ] **Step 3: Implement single materialization and batched residual inference.** Validate shape `(N, 2, L)`, finite float32 input, `L > max(horizons)`, and bounded allocation. Materialize the network once from the canonical CPU state, move one contiguous batch to the selected device, call the network under `eval()`/`no_grad()`, synchronize only on MPS, copy predictions back once, subtract channel zero targets per horizon, and return read-only float64 residual arrays. Assert every tensor’s device explicitly.

- [ ] **Step 4: Add the exact fit adapter.** Reproduce the existing one-thread DataLoader ordering, Huber loss, AdamW learning rate/weight decay, early-stop patience, epoch bounds, seed initialization, finite-loss checks, and canonical state serialization. The CPU adapter delegates comparison to the frozen fit functions; the MPS adapter uses the same operations and returns a `TrainedSeedModel` whose CPU state hash can be compared.

- [ ] **Step 5: Add fit/state parity tests.** On a fixture with at least five identity groups per early-stop partition, compare stop epoch, training/validation loss tuples within the frozen tolerance, CPU state hash, and final residuals for CPU versus adapter. Test invalid channel rank, invalid horizon length, wrong device assertion, and nonfinite input failure.

- [ ] **Step 6: Run tests and commit.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_batch_inference -v
git add experiments/exp421_mps_batched_acceleration/batch_inference.py experiments/exp421_mps_batched_acceleration/tests/test_batch_inference.py
git commit -m "feat(exp421): add frozen batched model adapter"
```

Expected: CPU reference parity passes for all five batch sizes; the MPS-specific assertions are guarded by `torch.backends.mps.is_available()` and are skipped with an explicit unavailable-device reason rather than silently treated as passing.

### Task 3: Add TRAIN OOF calibration caching with exact score equivalence

**Files:**
- Create: `experiments/exp421_mps_batched_acceleration/calibration.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_calibration.py`

**Interfaces:**
- Consumes: residual sequences from `OOFModelBundle.oof_residuals` and the frozen `score_horizon` function.
- Produces: immutable `CalibrationCache`, `build_calibration_cache`, `score_query_with_cache`, and `calibration_digest`.

- [ ] **Step 1: Write red cache-equivalence tests.** Build finite synthetic residuals with a valid aligned length, call `score_horizon` for several queries, then compare every cached `HorizonScore` field (`horizon`, `score`, `p_value`, `persistence`) and candidate decision at exact floating tolerances. Assert the cache includes no labels or TEST identity fields.

- [ ] **Step 2: Run the focused test and observe the missing implementation.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_calibration -v
```

Expected: import/attribute failure before implementation.

- [ ] **Step 3: Implement the cache from the exact scoring formula.** Store the horizon, pooled median, MAD-derived scale, 99th percentile threshold, normalized OOF arrays, and OOF series scores as read-only arrays/tuples. Compute a canonical digest from little-endian float64 bytes, shapes, horizon, and formula version. `score_query_with_cache` must use the same `_series_score` and `empirical_p_value` semantics as `score_horizon`, with no new threshold or alpha selection.

- [ ] **Step 4: Test cache immutability and malformed input rejection.** Verify a mutated source array cannot alter the cache, a writable cache is rejected, mismatched aligned lengths fail closed, zero scale fails, nonfinite query fails, and a cache digest changes if residual bytes change.

- [ ] **Step 5: Run and commit.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_calibration -v
git add experiments/exp421_mps_batched_acceleration/calibration.py experiments/exp421_mps_batched_acceleration/tests/test_calibration.py
git commit -m "feat(exp421): cache TRAIN OOF calibration"
```

### Task 4: Add immutable TRAIN synthetic perturbation caching

**Files:**
- Create: `experiments/exp421_mps_batched_acceleration/perturbations.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_perturbations.py`

**Interfaces:**
- Consumes: `build_fixed_perturbations`, `FROZEN_CONFIG.perturbation_seeds`, and a processed group’s exact raw scale.
- Produces: `build_cached_perturbations(dataset, instance_index, series, robust_scale, seed)` with immutable arrays keyed by dataset, instance, length, perturbation, and seed.

- [ ] **Step 1: Write red identity and seed-isolation tests.** For every frozen seed, compare the four primary arrays with the reference builder; assert primary arrays are byte-identical across seeds, secondary spikes are deterministic for the same seed and differ for at least two distinct seeds on a suitable fixture, and all arrays preserve length and float32 conversion only at the channel-cache boundary.

- [ ] **Step 2: Run the focused tests before implementation.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_perturbations -v
```

Expected: missing module/function failure.

- [ ] **Step 3: Implement the cache using the frozen builder exactly once per `(dataset, instance_index, length, seed)`.** Reuse the four primary arrays from the first seed; generate only `secondary_spike` for each additional seed; copy to immutable arrays; expose cache keys and a digest without including any TEST field.

- [ ] **Step 4: Test cache reuse, source mutation isolation, malformed keys, and forbidden names.** A source mutation after insertion must not change cached bytes; a dataset/instance/length mismatch must miss rather than return another series; names containing TEST or label fields must be rejected.

- [ ] **Step 5: Run and commit.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_perturbations -v
git add experiments/exp421_mps_batched_acceleration/perturbations.py experiments/exp421_mps_batched_acceleration/tests/test_perturbations.py
git commit -m "feat(exp421): cache deterministic TRAIN perturbations"
```

### Task 5: Implement the fail-closed hybrid backend policy

**Files:**
- Create: `experiments/exp421_mps_batched_acceleration/policy.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_policy.py`

**Interfaces:**
- Consumes: `BackendMeasurement`, `ParityReport`, `MlxAssessment`, and `Exp421Config`.
- Produces: `choose_backend(measurements, config)` and a `BackendPolicy` containing selected backend per batch size, exact reasons, and a `fail_closed` boolean.

- [ ] **Step 1: Write red policy tests.** Cover CPU selection at batch one, MPS selection only when all gates pass, CPU fallback for unavailable MPS, nonfinite output, coordinate mismatch, normalized error above `1e-4`, Spearman below `0.999`, speedup below `1.1`, missing transfer/synchronization timings, and any MLX ineligibility. Assert no input can alter alpha, threshold, seeds, or qualification fields.

- [ ] **Step 2: Run the focused tests and verify failure.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_policy -v
```

Expected: missing policy implementation failure.

- [ ] **Step 3: Implement deterministic policy evaluation.** Validate that one measurement exists for every declared batch size and that total timing equals transfer + compute + synchronization within a fixed tolerance. Calculate CPU-over-MPS speedup from the complete stage; choose MPS only for passing rows with batch size at least four; otherwise record a reason and choose CPU. Reject an incomplete manifest rather than silently assuming a pass.

- [ ] **Step 4: Add MPS/MLX capability assessment.** Record `torch.__version__`, `torch.backends.mps.is_built()`, `torch.backends.mps.is_available()`, device name, `mlx` version, `mlx-metal` version, and the exact reason MLX is ineligible (`package_absent` or `algorithm_rewrite_required`). Never import or install MLX as part of the benchmark.

- [ ] **Step 5: Run and commit.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_policy -v
git add experiments/exp421_mps_batched_acceleration/policy.py experiments/exp421_mps_batched_acceleration/tests/test_policy.py
git commit -m "feat(exp421): add fail-closed backend policy"
```

### Task 6: Add provenance manifest and no-TEST artifact validation

**Files:**
- Create: `experiments/exp421_mps_batched_acceleration/manifest.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_manifest.py`
- Create: `SOURCE_MANIFEST.json`

**Interfaces:**
- Consumes: `Exp421Config`, `BackendPolicy`, `MlxAssessment`, `BackendMeasurement`, source file paths, and the pinned database digest.
- Produces: `build_manifest(...) -> Exp421Manifest`, `validate_manifest(payload) -> Exp421Manifest`, `canonical_manifest_sha256(payload) -> str`, and atomic `write_manifest(path, manifest)`.

- [ ] **Step 1: Write red manifest tests.** Assert required fields include experiment ID, Git commit, database SHA-256, Exp417 config digest, device, PyTorch, MLX/`mlx-metal` status, candidate batch sizes, measurements, parity, policy, fallback reasons, source digests, attempt root, and `train_only=true`. Reject missing fields, non-hex digests, TEST/label/location tokens in SQL or artifact declarations, changed batch list, and output paths inside the official Exp419 root.

- [ ] **Step 2: Run the focused tests before implementation.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_manifest -v
```

Expected: missing module/manifest function failure.

- [ ] **Step 3: Implement canonical serialization and atomic write.** Sort keys, use UTF-8 JSON with stable separators, hash the exact bytes, write to a temporary file in the same fresh attempt root, flush and replace atomically, and re-read/validate after writing. Include the source manifest digest and never include raw series values.

- [ ] **Step 4: Add boundary tests for root isolation and source drift.** A manifest targeting `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel.checkpoint` must fail; a changed source file digest must fail validation; writing twice to the same path must fail unless the caller explicitly supplies a new attempt root.

- [ ] **Step 5: Generate `SOURCE_MANIFEST.json` from only Exp421 package, docs, and scratch source files.** Record path, SHA-256, and Git commit; exclude the database, checkpoints, prediction files, and any private dataset artifact.

- [ ] **Step 6: Run and commit.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_manifest -v
git add experiments/exp421_mps_batched_acceleration/manifest.py experiments/exp421_mps_batched_acceleration/tests/test_manifest.py SOURCE_MANIFEST.json
git commit -m "feat(exp421): add provenance and artifact boundary manifest"
```

### Task 7: Build the bounded normal-TRAIN benchmark

**Files:**
- Create: `experiments/exp421_mps_batched_acceleration/benchmark.py`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_benchmark_boundary.py`

**Interfaces:**
- Consumes: `verified_read_only_connection`, `load_raw_train_values_from_connection`, `preprocess_length_groups`, `FrozenModelAdapter`, `CalibrationCache`, `choose_backend`, and the pinned DB SHA-256.
- Produces: `run_train_only_benchmark(database, output_root, config, dataset_names) -> Exp421Manifest`.

- [ ] **Step 1: Write boundary tests before benchmark code.** Patch the read-only connection with a fixture SQLite database containing only normal TRAIN rows; assert the benchmark requests only the frozen TRAIN loader, limits selected normal TRAIN series to `max_benchmark_series`, uses exact-length preprocessing, creates a fresh output root, and never imports/calls `load_raw_test_values` or a TEST SQL constant.

- [ ] **Step 2: Run the boundary tests to confirm the benchmark is absent.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_benchmark_boundary -v
```

Expected: missing benchmark module/function failure.

- [ ] **Step 3: Implement deterministic TRAIN selection.** Open one verified read-only connection, load normal TRAIN values for explicitly selected dataset names or the first lexicographically sorted population subset, preprocess exact length groups, and choose at most `max_benchmark_series` finite candidate-eligible rows. Record dataset/length/instance identities as hashes only.

- [ ] **Step 4: Implement isolated timing for each exact batch size.** For each representative length/count stratum and each frozen model seed, run CPU and MPS sequentially in one process with warm-up excluded from measured repeats. Include NumPy-to-device transfer, network compute, MPS synchronization, device-to-CPU transfer, residual aggregation, and memory bound checks. Measure a CPU reference and MPS candidate on identical inputs and seeds; do not run any outer worker pool.

- [ ] **Step 5: Compute parity and policy without TEST data.** Compare finite values, normalized max error, rank correlation, candidate-coordinate hashes, model state/stop/loss/OOF/calibration hashes where fitting is included, and perturbation/qualification decisions. Pass all measurements to `choose_backend`; any failed gate records a precise CPU fallback reason.

- [ ] **Step 6: Emit and validate the manifest in a fresh attempt root.** Include benchmark start/end times, process ID, host/device, timings, parity, selected policy, MLX assessment, source digests, and no-clobber output paths. Return the validated `Exp421Manifest` rather than a private record.

- [ ] **Step 7: Run and commit.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_benchmark_boundary -v
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest discover -s experiments/exp421_mps_batched_acceleration/tests -p 'test_*.py' -v
git add experiments/exp421_mps_batched_acceleration/benchmark.py experiments/exp421_mps_batched_acceleration/tests/test_benchmark_boundary.py
git commit -m "feat(exp421): add isolated TRAIN-only benchmark"
```

### Task 8: Add the explicit CLI, preregistration, and Korean runbook

**Files:**
- Create: `scratch/run_exp421_mps_batched_acceleration.py`
- Create: `docs/EXP421_PREREGISTRATION_KO.md`
- Create: `docs/EXP421_RUNBOOK_KO.md`
- Create: `experiments/exp421_mps_batched_acceleration/tests/test_cli.py`

**Interfaces:**
- Consumes: `run_train_only_benchmark`, pinned database path, and an explicit fresh output root.
- Produces: a command-line entry point that exits nonzero on any parity/provenance/boundary failure and prints only the manifest path and summarized gate state.

- [ ] **Step 1: Write red CLI and documentation tests.** Assert `--help` exposes `--database`, `--output-root`, optional `--dataset`, and `--repeats`; an output root is required; an existing official Exp419 root is rejected; `--full-exp419` and any TEST/label option are rejected; and docs state Exp421 is diagnostic, local, TRAIN-only, one MPS-heavy process, and not an official rerun.

- [ ] **Step 2: Run the CLI tests before implementation.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_cli -v
```

Expected: missing CLI/doc tests or entry point failure.

- [ ] **Step 3: Implement argument validation and bounded dispatch.** Resolve the database path without changing it, require an output root that does not exist or is an explicitly empty new directory, pass only selected dataset names to `run_train_only_benchmark`, set `torch.set_num_threads(1)`, and never launch multiprocessing.

- [ ] **Step 4: Write the Korean preregistration.** State the question, frozen Exp417 reference, exact batch candidates, TRAIN-only sampling, parity/runtime/provenance gates, MLX absence/ineligibility, CPU fallback, no TEST selection, no lane/publication change, and the rule that a future full run needs a new experiment number and approval.

- [ ] **Step 5: Write the Korean runbook.** Include preflight DB SHA/quick-check, current official Exp419 isolation, one-process command, expected manifest fields, failure interpretation, safe cancellation, and a clear statement that this prototype must not reuse Exp419 checkpoints or start the 1,117-dataset run.

- [ ] **Step 6: Run and commit.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest experiments.exp421_mps_batched_acceleration.tests.test_cli -v
git add scratch/run_exp421_mps_batched_acceleration.py docs/EXP421_PREREGISTRATION_KO.md docs/EXP421_RUNBOOK_KO.md experiments/exp421_mps_batched_acceleration/tests/test_cli.py
git commit -m "docs(exp421): add preregistration runbook and bounded CLI"
```

### Task 9: Run the complete verification suite and prepare the diagnostic attempt

**Files:**
- Modify only if a test exposes a contract mismatch: `experiments/exp421_mps_batched_acceleration/*` and its tests.
- Do not modify: official Exp419 worktree, Exp417 frozen modules, dataset database, checkpoints, predictions, or lane artifacts.

**Interfaces:**
- Consumes: every implementation task and the approved design/spec.
- Produces: a clean, reviewable Exp421 branch and a ready-to-run command; no full benchmark is launched without a separate user instruction after the suite passes.

- [ ] **Step 1: Run the Exp421 unit suite.**

```bash
cd "/Users/mac/Documents/Time Series/.worktrees/exp421-mps-batched-acceleration"
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest discover -s experiments/exp421_mps_batched_acceleration/tests -p 'test_*.py' -v
```

Expected: all Exp421 tests pass, with only an explicit capability-guarded MPS skip if the device becomes unavailable.

- [ ] **Step 2: Run frozen Exp417 and Exp419 regression tests without starting either experiment.**

```bash
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest discover -s experiments/exp417_causal_multihorizon_shadow/tests -p 'test_*.py'
"/Users/mac/Documents/Time Series/.venv/bin/python" -m unittest discover -s experiments/exp419_deterministic_cpu_parallel/tests -p 'test_*.py'
```

Expected: existing suites retain their prior pass/fail baseline; any changed result blocks completion and is investigated before a benchmark.

- [ ] **Step 3: Rebuild and validate `SOURCE_MANIFEST.json`.** Hash only the committed Exp421 source/docs/CLI files, run `validate_manifest`, and verify `git status --short --branch` contains no generated checkpoint or dataset artifact.

- [ ] **Step 4: Perform static boundary checks.** Search only the new package/CLI for accidental TEST SQL imports, label/ground-truth access, tail replacement, lane promotion, or dynamic alpha/threshold selection; inspect imports to ensure no Exp419 process control or checkpoint reuse.

```bash
rg -n "load_raw_test_values|_TEST_VALUES_SQL|label|ground.?truth|tail replacement|tail_replacement|Exp419|checkpoint" experiments/exp421_mps_batched_acceleration scratch/run_exp421_mps_batched_acceleration.py docs/EXP421_PREREGISTRATION_KO.md docs/EXP421_RUNBOOK_KO.md
```

Expected: only explicit boundary-test strings, documentation prohibitions, and validation error messages; no executable TEST path.

- [ ] **Step 5: Commit the verified branch.**

```bash
git add experiments/exp421_mps_batched_acceleration docs/EXP421_PREREGISTRATION_KO.md docs/EXP421_RUNBOOK_KO.md scratch/run_exp421_mps_batched_acceleration.py SOURCE_MANIFEST.json
git commit -m "test(exp421): verify isolated acceleration prototype"
```

- [ ] **Step 6: Report readiness and stop before the benchmark.** Report the exact branch, commit, test counts, capability state, and the command that would run a fresh bounded TRAIN-only benchmark. Do not execute that command or alter the official Exp419 process until the user explicitly authorizes the diagnostic run.

## Self-review checklist

- The design requirements map to Tasks 1–5: frozen adapter, calibration cache, perturbation cache, hybrid policy, and MLX manifest assessment.
- The boundary and provenance requirements map to Tasks 6–8: no TEST access, source/database/Git digests, fresh attempt roots, Korean preregistration/runbook, and explicit CLI refusal of full Exp419 execution.
- The acceptance gates map to Tasks 2, 3, 5, and 7: finite outputs, identical candidate coordinates, normalized error `1e-4`, Spearman `0.999`, model/stop/loss/OOF/calibration/qualification parity, transfer/synchronization-inclusive speedup `1.1`, memory bound, and one-process isolation.
- No task changes frozen Exp417/Exp419 source, lanes, checkpoints, predictions, or private data.
- All interfaces named in following tasks are defined in the file map or an earlier task; no step relies on an undefined symbol.
- Every task has a red test, an explicit command, a minimal implementation action, a green test, and a commit.
