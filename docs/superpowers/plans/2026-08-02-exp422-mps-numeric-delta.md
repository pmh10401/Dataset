# Exp422 MPS Numeric Delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TRAIN-only Exp422 diagnostic that measures CPU/MPS numeric deltas and end-to-end score parity at the fixed batch sizes 16, 32, 64, 128, and 256 without changing Exp419 or Exp421.

**Architecture:** Exp422 owns a new frozen configuration, contracts, numeric-delta engine, score-parity adapter, manifest validator, and CLI. It reuses only low-level Exp417/Exp421 model, calibration, perturbation, and read-only input primitives; it never mutates Exp421's `FROZEN_CONFIG`. The CLI runs core parity first and runs the bounded TRAIN score stage only when every required core gate passes.

**Tech Stack:** Python 3.14, PyTorch 2.13, NumPy, SQLite read-only connections, existing Exp417 causal CNN and Exp421 low-level adapter, `unittest`, canonical JSON and SHA-256 provenance.

## Global Constraints

- Candidate batch sizes are exactly `(16, 32, 64, 128, 256)`; no old Exp421 batch size is silently added.
- `max_normalized_error` is `1e-4`, `min_spearman` is `0.999`, and `min_speedup` is `1.1`.
- The model seeds are exactly `41701, 41702, 41703`; Exp417 horizons, alpha, persistence, loss, and fold rules remain frozen.
- Inputs are finite normal TRAIN values or deterministic synthetic tensors only; no TEST value, label, anomaly location, ground truth, human review, or historical TEST metric may be loaded.
- SQLite is opened read-only with the frozen database SHA-256 and `PRAGMA quick_check` gate.
- A fresh Exp422 attempt root is mandatory; official Exp419 paths, checkpoint files, and output roots are rejected.
- One MPS-heavy process runs at a time, only after the official Exp419 process has ended successfully.
- MLX is metadata-only for this study; record availability/version and `package_absent` or `algorithm_rewrite_required` without rewriting the algorithm.
- A failed parity or speed gate records the exact fallback reason and never changes an operational backend, threshold, source, lane, or budget.

---

### Task 1: Freeze Exp422 configuration, contracts, and user-facing boundaries

**Files:**
- Create: `experiments/exp422_mps_numeric_delta/__init__.py`
- Create: `experiments/exp422_mps_numeric_delta/config.py`
- Create: `experiments/exp422_mps_numeric_delta/contracts.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/__init__.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_config.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_contracts.py`
- Create: `docs/EXP422_PREREGISTRATION_KO.md`
- Create: `docs/EXP422_RUNBOOK_KO.md`

**Interfaces:**
- `FROZEN_CONFIG: Exp422Config` exposes `candidate_batch_sizes == (16, 32, 64, 128, 256)`, the Exp417-bound tolerances, frozen seeds, and `max_series == 64` / `max_batch_bytes == 256 * 1024 * 1024`.
- `Exp422Config.validate() -> None` rejects any changed field, unsorted batch list, invalid tolerance, or seed mismatch.
- `BatchKey(batch_size: int, model_seed: int, perturbation_seed: int)` validates only Exp422's five batch sizes and Exp417's frozen seeds.
- `TensorDelta`, `TimingEvidence`, `CoreParityReport`, and `StageResult` are immutable records with finite numeric fields, lowercase SHA-256 digests, and explicit `passed`, `failed`, or `skipped` states.

- [ ] **Step 1: Write the failing configuration and contract tests**

```python
def test_exp422_batches_are_exact() -> None:
    assert FROZEN_CONFIG.candidate_batch_sizes == (16, 32, 64, 128, 256)

def test_old_exp421_batches_are_rejected() -> None:
    with pytest.raises(ValueError):
        BatchKey(1, 41701, 41711)

def test_timing_includes_transfer_compute_and_sync() -> None:
    with pytest.raises(ValueError):
        TimingEvidence("mps", 0.2, 0.2, 0.2, 0.1)
```

Run: `python -m unittest discover -s experiments/exp422_mps_numeric_delta/tests -p "test_*.py"`  
Expected: FAIL because the Exp422 package and contracts do not yet exist.

- [ ] **Step 2: Implement the minimal frozen config and immutable contracts**

```python
@dataclass(frozen=True)
class Exp422Config:
    experiment_id: str = "Exp422"
    candidate_batch_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
    max_normalized_error: float = 1e-4
    min_spearman: float = 0.999
    min_speedup: float = 1.1
    max_series: int = 64
    max_batch_bytes: int = 256 * 1024 * 1024
    model_seeds: tuple[int, ...] = (41701, 41702, 41703)
    perturbation_seeds: tuple[int, ...] = (41711, 41712, 41713)
```

Use the existing canonical serialization style and bind `model_seeds` and
`perturbation_seeds` to `experiments.exp417_causal_multihorizon_shadow.config.FROZEN_CONFIG`.

- [ ] **Step 3: Add the Korean preregistration and runbook**

The preregistration must state the exact five batches, two-stage ordering,
TRAIN-only boundary, numeric gates, speed timing definition, MPS-unavailable
fallback, and no operational adoption. The runbook must show only a fresh
Exp422 output command and the Exp419 completion guard; it must not contain a
command that targets an Exp419 root.

- [ ] **Step 4: Run the focused tests and commit**

Run: `python -m unittest discover -s experiments/exp422_mps_numeric_delta/tests -p "test_*.py"`  
Expected: PASS for all config and contract tests.  
Commit: `git add experiments/exp422_mps_numeric_delta docs/EXP422_PREREGISTRATION_KO.md docs/EXP422_RUNBOOK_KO.md && git commit -m "feat(exp422): freeze numeric delta boundaries"`

### Task 2: Implement deterministic core CPU/MPS parity and delta metrics

**Files:**
- Create: `experiments/exp422_mps_numeric_delta/capability.py`
- Create: `experiments/exp422_mps_numeric_delta/core_parity.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_capability.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_core_parity.py`

**Interfaces:**
- `detect_capability() -> DeviceCapability` records PyTorch, MPS built/available, device identity, MLX version/availability, and the fallback reason without touching the database.
- `run_core_parity(config: Exp422Config, capability: DeviceCapability) -> CoreStageResult` evaluates every frozen batch and model seed on deterministic float32 inputs, CPU first and MPS second.
- `compare_tensor(cpu: np.ndarray, mps: np.ndarray, *, name: str) -> TensorDelta` computes finite status, max/mean absolute error, normalized relative error, max ULP, output digests, and the failing index.
- `candidate_coordinate_digest(scores: np.ndarray, alpha: float) -> str` uses stable ordering and the frozen alpha without labels.

- [ ] **Step 1: Write failing tests for capability and metric edge cases**

```python
def test_ulp_metric_is_zero_for_identical_float32() -> None:
    delta = compare_tensor(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32), name="x")
    assert delta.max_ulp == 0
    assert delta.max_normalized_error == 0.0

def test_nonfinite_mps_output_fails_closed() -> None:
    delta = compare_tensor(np.array([1.0], dtype=np.float32), np.array([np.nan], dtype=np.float32), name="x")
    assert not delta.passes

def test_unavailable_mps_has_explicit_reason() -> None:
    capability = DeviceCapability(
        torch_version="2.13.0",
        mps_built=True,
        mps_available=False,
        device_name="Darwin arm64 MPS",
        mlx_version="unavailable",
        mlx_metal_version="unavailable",
        mlx_availability="unavailable",
        fallback_reason="mps_unavailable",
    )
    assert capability.fallback_reason == "mps_unavailable"
```

- [ ] **Step 2: Implement deterministic model/input construction**

Create finite two-channel tensors for exact lengths `(64, 128, 256, 512)` and
the five batch sizes using a local `numpy.random.Generator` seeded by the
frozen model seed. Construct the existing `CausalMultiHorizonCNN` with the
same frozen horizons and deterministic state, without fitting on TEST or
labels. Set CPU threads to one and use `torch.no_grad()` for forward parity.

- [ ] **Step 3: Implement CPU/MPS timing and synchronization**

Measure input transfer, device computation, output transfer, and explicit
`torch.mps.synchronize()` separately. Batch 256 must fail closed on the fixed
memory limit; do not silently reduce it to a smaller batch. MPS exceptions,
unavailable capability, and nonfinite outputs become structured failure rows.

- [ ] **Step 4: Implement delta, ULP, rank, and candidate checks**

Use an ordered float32 integer view for ULP distance, stable sorting for
candidate coordinates, and a tie-aware Spearman implementation. A report passes
only when all output tensors are finite, normalized error is at most `1e-4`,
candidate coordinates match exactly, and Spearman is at least `0.999`.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m unittest discover -s experiments/exp422_mps_numeric_delta/tests -p "test_*.py"`  
Expected: PASS on CPU-only hosts; MPS-specific tests are host-conditional skips.  
Commit: `git add experiments/exp422_mps_numeric_delta && git commit -m "feat(exp422): add core CPU MPS delta audit"`

### Task 3: Implement bounded normal-TRAIN score parity after core gates

**Files:**
- Create: `experiments/exp422_mps_numeric_delta/score_parity.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_score_boundary.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_score_parity.py`

**Interfaces:**
- `select_train_sample(database: Path, config: Exp422Config) -> TrainSample` opens one verified read-only connection, selects the deterministic exact-length normal TRAIN group, and returns no labels or TEST values.
- `run_score_parity(sample: TrainSample, capability: DeviceCapability, config: Exp422Config) -> ScoreStageResult` reuses only Exp417/Exp421 low-level model, calibration, perturbation, and scoring primitives.
- `run_score_parity` returns CPU/MPS forward outputs, residual/calibration digests, candidate-coordinate parity, qualification parity, and per-batch timing for `(16, 32, 64, 128, 256)`.

- [ ] **Step 1: Write boundary tests before database code**

```python
def test_score_selector_rejects_test_tokens() -> None:
    with pytest.raises(ValueError):
        validate_dataset_name("TEST_Adiac")

def test_exp422_does_not_import_exp419_checkpoint_paths() -> None:
    assert "exp419_deterministic_cpu_parallel.checkpoint" not in SCORE_SOURCE

def test_score_stage_is_skipped_when_core_failed() -> None:
    assert decide_stage_two(StageResult.failed("mps_unavailable")) == "skipped"
```

- [ ] **Step 2: Implement deterministic TRAIN-only selection**

Reuse the verified read-only SQLite boundary and exact-length selection rule,
recording dataset name, length, TRAIN count, raw waveform digest, and the
bounded series count. Reject TEST/evaluation tokens and any nonfinite series;
do not issue a TEST query.

- [ ] **Step 3: Reuse CPU-trained frozen state and compare score paths**

Fit the frozen Exp417 CPU reference exactly once for the selected TRAIN group,
build the existing calibration and perturbation evidence, then materialize the
same CPU state on CPU and MPS through the low-level Exp421 adapter. This keeps
training state identical while measuring forward/score numeric differences.
Compare residual, calibration, horizon score, candidate coordinates, rank, and
qualification evidence for every Exp422 batch.

- [ ] **Step 4: Run score-boundary tests and commit**

Run: `python -m unittest discover -s experiments/exp422_mps_numeric_delta/tests -p "test_score_*.py"`  
Expected: PASS with synthetic fixtures; database integration remains bounded and label-free.  
Commit: `git add experiments/exp422_mps_numeric_delta && git commit -m "feat(exp422): add TRAIN score parity stage"`

### Task 4: Add canonical manifest, source trust root, and guarded CLI

**Files:**
- Create: `experiments/exp422_mps_numeric_delta/manifest.py`
- Create: `experiments/exp422_mps_numeric_delta/SOURCE_MANIFEST.json`
- Create: `scratch/run_exp422_mps_numeric_delta.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_manifest.py`
- Create: `experiments/exp422_mps_numeric_delta/tests/test_cli.py`

**Interfaces:**
- `build_manifest(core: CoreStageResult, score: ScoreStageResult | None, provenance: Mapping[str, object]) -> Exp422Manifest` returns a complete immutable payload.
- `validate_manifest(payload: Mapping[str, object]) -> Exp422Manifest` rejects missing provenance, altered batch lists, noncanonical digests, TEST tokens, and inconsistent stage states.
- `write_manifest(path: Path, manifest: Exp422Manifest) -> Path` uses canonical JSON and exclusive creation.
- `require_exp419_finished(pidfile: Path, exitfile: Path) -> None` fails while the official PID is alive, when the exit file is absent, or when the recorded exit code is nonzero.
- CLI arguments are `--database`, `--output-root`, `--exp419-pidfile`, `--exp419-exitfile`, and bounded `--repeats 1..7`.

- [ ] **Step 1: Write manifest and CLI rejection tests**

```python
def test_cli_rejects_existing_official_output_root() -> None:
    result = run_cli("--output-root", "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_official")
    assert result.returncode != 0

def test_cli_rejects_live_exp419_pid(tmp_path: Path) -> None:
    pidfile = tmp_path / "exp419.pid"
    pidfile.write_text(str(os.getpid()))
    with pytest.raises(ValueError, match="Exp419 is still running"):
        require_exp419_finished(pidfile, tmp_path / "exp419.exit")
```

- [ ] **Step 2: Implement canonical provenance and no-clobber validation**

Bind Git commit, database SHA, SQLite quick-check, Exp422 config digest,
source-manifest digest, PyTorch/MPS/MLX metadata, stage results, attempt root,
and fallback reasons. Reject official Exp419 path components and any output root
that is not new or empty.

- [ ] **Step 3: Implement stage ordering in the CLI**

The CLI performs capability and input preflight, runs core parity, and invokes
score parity only when `core.state == "passed"`. It writes one manifest even for
structured failure; it never starts a full-population run and never modifies the
Exp419 checkpoint.

- [ ] **Step 4: Generate and verify the Exp422 source manifest**

List every Exp422 source file and documentation file with SHA-256, bind the
manifest to the Git commit, and add tests that tampering with a declared source
is rejected.

- [ ] **Step 5: Run all focused tests and commit**

Run: `python -m unittest discover -s experiments/exp422_mps_numeric_delta/tests -p "test_*.py"`  
Expected: PASS with host-conditional MPS skips only.  
Commit: `git add experiments/exp422_mps_numeric_delta scratch/run_exp422_mps_numeric_delta.py && git commit -m "feat(exp422): add guarded diagnostic CLI and manifest"`

### Task 5: Full regression and implementation review

**Files:**
- Modify: `.superpowers/sdd/2026-08-02-exp422-mps-numeric-delta/progress.md`
- Modify: `docs/EXP422_PREREGISTRATION_KO.md` only if test-proven wording needs correction
- Modify: `docs/EXP422_RUNBOOK_KO.md` only if the final CLI flags differ from the frozen interface

**Interfaces:**
- The Exp422 test suite remains independently runnable and does not require the database for unit tests.
- Existing Exp421 tests and source trust-root checks remain unchanged and passing.

- [ ] **Step 1: Run Exp422 tests, Exp421 regression, and compile checks**

```bash
/Users/mac/Documents/Time\ Series/.venv/bin/python -m unittest discover \
  -s experiments/exp422_mps_numeric_delta/tests -p "test_*.py"
/Users/mac/Documents/Time\ Series/.venv/bin/python -m unittest discover \
  -s experiments/exp421_mps_batched_acceleration/tests -p "test_*.py"
/Users/mac/Documents/Time\ Series/.venv/bin/python -m compileall -q \
  experiments/exp422_mps_numeric_delta scratch/run_exp422_mps_numeric_delta.py
```

Expected: all Exp422 tests pass, Exp421 baseline remains green, and compileall
produces no output or error.

- [ ] **Step 2: Run static boundary checks**

Search the new package and CLI for `TEST`, `label`, `ground_truth`, `location`,
and official Exp419 path tokens. Confirm these occur only in explicit rejection
guards and tests, never in SQL loaders or output payloads.

- [ ] **Step 3: Commit the verified implementation**

Run `git diff --check`, inspect the source manifest against Git blob hashes, and
commit: `git add . && git commit -m "feat(exp422): complete MPS numeric delta study"`.

### Task 6: Execute after Exp419 completion and produce the review report

**Files:**
- Create outside Git output root: `/Users/mac/Documents/Dataset/exp422_mps_numeric_delta_attempt_<timestamp>/manifest.json`
- Create: `docs/EXP422_RESULTS_ANALYSIS_KO.md` after the bounded execution

**Interfaces:**
- The actual run is permitted only when `require_exp419_finished` observes the official Exp419 PID absent and exit code `0`.
- The result report consumes only the Exp422 manifest and does not use TEST labels or mean combined F1 for a backend decision.

- [ ] **Step 1: Perform final preflight without starting MPS**

Verify the Exp422 branch is clean, the database SHA is
`8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`, SQLite
quick-check is `ok`, the official Exp419 exit file contains `0`, and the new
output root does not exist.

- [ ] **Step 2: Start one bounded Exp422 attempt**

```bash
PYTHON="/Users/mac/Documents/Time Series/.venv/bin/python"
"$PYTHON" scratch/run_exp422_mps_numeric_delta.py \
  --database "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --output-root "/Users/mac/Documents/Dataset/exp422_mps_numeric_delta_attempt_$(date +%Y%m%d-%H%M%S)" \
  --exp419-pidfile "/Users/mac/Documents/Dataset/exp419_official_20260802-001822.pid" \
  --exp419-exitfile "/Users/mac/Documents/Dataset/exp419_official_20260802-001822.exit" \
  --repeats 3
```

Use one process, do not add outer worker parallelism, and preserve any failed
attempt root for diagnosis. Do not retry with different batches or tolerances.

- [ ] **Step 3: Validate and analyze the manifest**

Check the manifest digest, stage states, per-batch errors, ULP distribution,
candidate coordinate equality, score parity, speedup including synchronization,
device capability, and MLX fallback metadata. Write a Korean report that clearly
separates numeric evidence from any operational conclusion.

- [ ] **Step 4: Mark the implementation plan complete**

Update the SDD progress ledger and report that Exp422 is diagnostic evidence
only; no Exp419 or production routing change is authorized by its result.
