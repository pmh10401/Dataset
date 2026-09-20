# Exp438 Frequency-Complexity Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:test-driven-development` for every product-code task and
> `superpowers:verification-before-completion` before each completion claim.
> Execute this plan task-by-task; do not start the official run while Exp437 is
> active.

**Goal:** Build and evaluate an isolated, review-only Exp438 score-source
experiment that uses only normal TRAIN to freeze a transparent high/low FFT
split and amplitude-sensitive permutation entropy (ASPE), then reports three
separate anomaly channels across all 1,117 datasets.

**Architecture:** Exp438 owns a new package, CLI, fixed run root, checkpoints,
logs, predictions, and evaluation. It imports only stable generic boundaries
for canonical JSON, frozen lane inputs, and read-only SQLite value loading; it
does not modify or import Exp437's profile, runtime, checkpoint, or artifacts.
The canonical algorithm is NumPy CPU. Numba, MLX, or MPS can be selected only by
the frozen parity-and-speed readiness gate. TEST labels are loaded lazily by the
evaluation module only after publication hashes are frozen.

**Tech Stack:** Python 3, `unittest`, NumPy, SQLite read-only snapshots,
optional Numba, optional MLX, optional PyTorch MPS, canonical JSON/SHA256,
`ProcessPoolExecutor`, Darwin atomic/no-clobber filesystem operations.

## Frozen Constraints

- Work only in branch `codex/exp438-frequency-complexity-evidence` and worktree
  `/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence`.
- Never modify Exp437 product code, run root, checkpoint, progress, PID,
  prediction, publication, or evaluation.
- While Exp437 is active, allow only documentation, implementation, and bounded
  single-process tests. Do not run Exp438 readiness, accelerator benchmarks, or
  the 1,117-dataset prediction.
- Use only the frozen local database and frozen lane artifacts. Prediction must
  not query TEST labels, anomaly positions, TEST metrics, or historical
  family-level TEST performance.
- TEST labels may be opened only after Exp438 prediction, manifest, and artifact
  hashes are frozen and independently verified.
- Preserve exact raw length. No padding, truncation, interpolation, resampling,
  tail replacement, sibling substitution, or nonfinite-value repair.
- Keep `high_frequency`, `low_frequency`, and `amplitude_complexity` as three
  separate channels. Exp438 contains no fusion or best-channel selection.
- `review_only=true`; `operational_alert_eligible=false`.
- Official CPU workers are always `max(1, logical_cpu_count - 1)`, expected 11
  on this 12-logical-CPU Mac, with one numerical thread per worker.
- Assess MLX before the full run. Adopt Numba/MLX/MPS only with finite output,
  identical candidate coordinates, score error `<=1e-5`, and stage speedup
  `>=1.1x`, including transfer, synchronization, and preregistered compilation
  amortization.
- Run only one accelerator-heavy process. Never combine outer multiprocessing
  with an adopted MLX/MPS scoring path.
- Every newly created filename contains `antigravity`; do not create
  `__init__.py`.
- Long execution records PID, heartbeat, current dataset, TRAIN/TEST sizes and
  lengths, stage/backend, completed/total, success/abstention/error counts,
  rate, elapsed time, ETA, checkpoint, per-channel counts, and latest error in
  atomic JSON plus text/JSONL logs.
- Do not poll an unchanged run. The process owns progress and checkpointing.

## Dependency And Test Impact

Before product code, the synchronized code map must answer:

1. **What calls Exp438?** Only the new scratch wrapper calls the new Exp438 CLI;
   the CLI calls readiness/runtime/publication/evaluation.
2. **What does Exp438 affect?** Only its fixed run parent, its new package,
   Exp438 documents, catalog/state handoff after Exp437 is terminal, and the
   synchronized code map. It must not affect Exp437.
3. **Which tests cover it?** New focused test modules under the Exp438 package,
   plus the existing Exp425 database and Exp436/Exp437 shared-engine regression
   suites as non-regression guards.

The current code map predates commit `06d15834` and is stale in `docs` and
`root`. It must be regenerated before Task 2 modifies product modules.

## Planned File Structure

Create package `experiments/exp438_frequency_complexity_evidence_antigravity/`
with these files:

- `config_antigravity.py`: frozen identity, constants, paths, worker policy.
- `contracts_antigravity.py`: immutable input, calibration, evidence, result,
  backend, and summary records.
- `inputs_antigravity.py`: frozen value-only SQLite/lane input session.
- `aspe_antigravity.py`: attributed source-compatible ASPE implementation.
- `features_antigravity.py`: FFT split, TRAIN-only depth selection, calibration.
- `scoring_antigravity.py`: separate-channel scoring and budget validation.
- `numba_antigravity.py`: optional ASPE acceleration adapter.
- `mlx_antigravity.py`: optional FFT/distance readiness adapter.
- `mps_antigravity.py`: optional FFT/distance readiness adapter.
- `backend_antigravity.py`: frozen cross-backend selection and CPU fallback.
- `checkpoint_antigravity.py`: ordered canonical dataset records and recovery.
- `progress_antigravity.py`: atomic progress, events, PID, and parent log.
- `runtime_antigravity.py`: parent-owned serial/parallel orchestration.
- `reporting_antigravity.py`: frozen review-only publication and hash replay.
- `evaluation_antigravity.py`: lazy post-freeze TEST-label evaluation.
- `cli_antigravity.py`: fixed `readiness/new/resume/inspect/evaluate` interface.

Create matching `tests/test_<module>_antigravity.py` files, plus:

- `scratch/run_exp438_frequency_complexity_evidence_antigravity.py`
- `docs/EXP438_THIRD_PARTY_PROVENANCE_antigravity.md`
- `docs/EXP438_PREREGISTRATION_antigravity_KO.md`
- `docs/EXP438_RUNBOOK_antigravity_KO.md`
- `docs/EXP438_IMPLEMENTATION_REPORT_antigravity.md`

Modify only when their task explicitly requires it:

- `EXPERIMENT_CATALOG.json`
- `RESEARCH_STATE.json` after Exp437 is terminal and the user-direction pause is
  satisfied
- `FAILURE_LEDGER.json` or `CHAMPION_LEDGER.json` only after frozen evaluation
- all three `docs/codemap/` artifacts together

---

### Task 1: Synchronize The Code Map And Freeze Source Provenance

**Files:**
- Create: `docs/EXP438_THIRD_PARTY_PROVENANCE_antigravity.md`
- Regenerate together: `docs/codemap/codemap.html`
- Regenerate together: `docs/codemap/codemap.json`
- Regenerate together: `docs/codemap/codemap.lock`

- [ ] **Step 1: Recompute the stale-module comparison**

Record the current commit, dirty state, tracked paths, and deterministic
top-level fingerprints. Confirm that at least `docs` and `root` differ from the
existing lock. Do not edit only one code-map artifact.

- [ ] **Step 2: Write provenance before implementation**

Pin these facts in the new provenance document:

```text
Brioal/ASPE
commit f17cf8d796c4d25bc1b5ace0e9a7ef7be2944a89
source symbol entropy/aspe.m::aspe
license Apache-2.0

pikipity/Toolbox-for-Adaptive-Fourier-Decomposition
commit 2b11bcc0eacd30f979ef4d6b24715c7fec327b02
license CC BY-NC-ND 4.0
policy read-only research reference; no copy/import/runtime dependency

VUT-HFUT/PhysDiff
commit c4a17679cc3c81d0eea7e5ec07cccada1272bad7
policy excluded namesake physiology project; no detected license; no copied code
```

- [ ] **Step 3: Regenerate all three code-map files from one in-memory model**

Retain no more than 20 primary nodes. Add the Exp438 design/provenance to the
research-control evidence without inventing product-code edges that do not yet
exist. Rebuild HTML's embedded nodes/edges/flows from the exact JSON payload and
recompute lock fingerprints from tracked paths and current bytes.

- [ ] **Step 4: Validate the synchronized map**

Run parsers and a read-only validator that proves node paths exist, evidence
symbols are found, edges/flows reference existing nodes, HTML payload equals
JSON, lock commit/dirty state/fingerprints match, and unsupported relationships
are marked `unknown`.

Run:

```bash
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
```

- [ ] **Step 5: Commit provenance and the synchronized map**

```bash
git add docs/EXP438_THIRD_PARTY_PROVENANCE_antigravity.md \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(exp438): freeze provenance and synchronize code map"
```

---

### Task 2: Freeze Exp438 Configuration And Immutable Contracts

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/config_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/contracts_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_contracts_antigravity.py`

- [ ] **Step 1: Write RED tests for identity and scientific constants**

Freeze `Exp438`, run identity
`exp438-frequency-complexity-evidence-antigravity-20260807`, fixed run parent,
database SHA, 1,117 datasets, minimum length 16, minimum four TRAIN rows,
depth cap 5, delays `(1,2,4)`, `k=3`, `q=0.99`, strict greater-than threshold,
channel budgets 10,000/30,000, seed `43801`, batch sizes
`(16,32,64,128,256)`, and CPU worker policy.

Test that arbitrary path/identity/config changes fail and that every numeric
field is finite, canonical, and immutable.

- [ ] **Step 2: Run tests and confirm missing-module RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_config_antigravity \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_contracts_antigravity -v
```

Expected: `ModuleNotFoundError` for the new modules.

- [ ] **Step 3: Implement the smallest frozen configuration and dataclasses**

Contracts must include immutable records for raw value rows, excluded TRAIN
rows, prepared exact-length groups, decomposition evidence, per-channel
calibration, per-instance evidence, abstention, dataset result, backend batch
evidence, backend decision, and run summary. Arrays are finite, contiguous,
read-only, and shape checked. Payloads reject unknown and label-like fields.

- [ ] **Step 4: Run GREEN tests and configuration serialization checks**

Verify canonical JSON serialization is byte-stable under dictionary insertion
order and does not contain `label`, `labels_blob`, anomaly positions, or metric
fields.

- [ ] **Step 5: Commit the frozen boundary**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/config_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/contracts_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_config_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_contracts_antigravity.py
git commit -m "feat(exp438): freeze configuration and contracts"
```

---

### Task 3: Build The Value-Only Exact-Length Input Boundary

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/inputs_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_inputs_antigravity.py`

**Allowed imports:** generic read-only snapshot/value loaders from
`experiments/exp425_selective_recovery/database.py` and frozen lane loading from
`experiments/exp417_causal_multihorizon_shadow/inputs.py`. Do not import an
Exp436/Exp437 execution profile, runtime, checkpoint, or publication module.

- [ ] **Step 1: Write RED tests for the label firewall and row conservation**

Cover little-endian float32 decoding, exact identity/length preservation,
sorted unique 1,117 population, TRAIN-before-TEST access, zero TEST-label query
count, whole-row nonfinite TRAIN exclusion with blob SHA, row-local nonfinite
TEST abstention, exact-length grouping, fewer-than-four finite TRAIN rows, and
no repair/resampling/tail replacement.

- [ ] **Step 2: Confirm RED failures**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_inputs_antigravity -v
```

- [ ] **Step 3: Implement a parent-owned `Exp438InputSession`**

The session opens one verified read-only descriptor, freezes population and
lanes, exposes normal TRAIN values before TEST values, never exposes label
columns, and closes deterministically. Workers receive values-only immutable
objects and never receive SQLite connections.

- [ ] **Step 4: Run GREEN and predecessor database regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_inputs_antigravity \
  experiments.exp425_selective_recovery.tests.test_database -v
```

- [ ] **Step 5: Commit the input firewall**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/inputs_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_inputs_antigravity.py
git commit -m "feat(exp438): add value-only exact-length inputs"
```

---

### Task 4: Implement Attributed ASPE And The Frozen FFT Split

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/aspe_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/features_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_aspe_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_features_antigravity.py`

- [ ] **Step 1: Write literal ASPE golden-vector RED tests**

Tests independently encode the pinned semantics: whole-vector min-max
normalization; dimension 3; delays 1/2/4; stable ties; per-window sample
standard-deviation divided by mean; weighted ordinal-pattern aggregation; and
base-2 entropy divided by `log2(6)`. Include monotone entropy `0.0`, an
alternating two-pattern literal expected value, equal ordinal counts with
different amplitude weights, and constant/zero-weight exact `0.0` with
`flat_or_zero_weight=true`.

- [ ] **Step 2: Confirm ASPE RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_aspe_antigravity -v
```

- [ ] **Step 3: Implement the attributed float64 ASPE core**

The source header must contain the upstream URL, pinned commit, Apache-2.0
license, symbol, and constant-signal extension. Do not copy MATLAB lines. Reject
all unexpected nonfinite intermediates. Return numeric value plus audit flag;
canonicalize final features to contiguous float32.

- [ ] **Step 4: Write and run FFT/depth-selection RED tests**

Cover exact rFFT cutoff bins, `original == low + high` within `1e-5`, no length
change, finite-only behavior, candidate depths, first-nondecrease rule, lower
median TRAIN depth, and pseudo-test TRAIN exclusion from its own selected depth.

- [ ] **Step 5: Implement the smallest project-owned FFT decomposition**

Implement only the approved cutoff rule. Do not import, port, or name the AFD
toolbox. Produce original/low/high ASPE values and audit flags at delays 1/2/4.

- [ ] **Step 6: Run GREEN tests and source-policy scan**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_aspe_antigravity \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_features_antigravity -v
rg -n "pikipity|VUT-HFUT|cuda" experiments/exp438_frequency_complexity_evidence_antigravity
```

Expected: only provenance/explicit exclusion text if present; no import or
copied runtime code.

- [ ] **Step 7: Commit the feature source**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/aspe_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/features_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_aspe_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_features_antigravity.py
git commit -m "feat(exp438): add attributed frequency complexity features"
```

---

### Task 5: Fit TRAIN-Only Calibrations And Score Three Separate Channels

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/scoring_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_scoring_antigravity.py`

- [ ] **Step 1: Write leakage and scoring RED tests**

Cover scalar robust scaling for each waveform channel, per-column robust scaling
for the nine ASPE values, median of three nearest RMS distances, leave-one-row-
out TRAIN scores, `q=0.99/method='higher'`, strict `score > threshold`, unavailable
channel reasons, exact-length reference selection, TRAIN-row reordering, and
worker-independent ordering.

- [ ] **Step 2: Write RED tests proving no fusion path exists**

Each instance must preserve three named score/threshold/abnormal results. There
must be no combined abnormal boolean, vote count, OR/AND/weighted fusion, channel
selection, or channel-specific alpha. Outside-lane candidates remain keyed by
`(channel,dataset,instance_index)`.

- [ ] **Step 3: Implement CPU scoring only**

Build all final TEST calibrations from normal TRAIN. Repeat depth selection and
calibration without the pseudo-test row for every TRAIN OOF score. Flat flags
remain audit metadata, not distance dimensions. Validate per-channel 10,000 and
total 30,000 budgets before persistence; never truncate top-N.

- [ ] **Step 4: Run GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_scoring_antigravity -v
```

- [ ] **Step 5: Commit canonical CPU scoring**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/scoring_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_scoring_antigravity.py
git commit -m "feat(exp438): add separate train-only score channels"
```

---

### Task 6: Add Truthful Numba, MLX, And MPS Readiness Adapters

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/numba_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/mlx_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/mps_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/backend_antigravity.py`
- Create: matching four `tests/test_*_antigravity.py` files

- [ ] **Step 1: Write RED parity, timing, and fallback tests**

Use deterministic same-input/same-seed fixtures. Require finite outputs,
candidate identity at every frozen threshold, max score error `<=1e-5`, two
untimed warmups, five paired timed calls, transfer/synchronization inside the
timed callable, `>=1.1x` stage speedup, lower median selection, smaller-batch tie,
and lexical backend tie. Missing/unavailable/error/slow backends must record a
reason and select CPU.

- [ ] **Step 2: Implement optional adapters without importing them at module load**

Numba accelerates only the ASPE inner loop. MLX/MPS may accelerate only exact
FFT or dense-distance operations. Keep NumPy CPU authoritative and use lazy
imports so missing optional libraries do not break prediction.

- [ ] **Step 3: Run focused synthetic GREEN tests only**

While Exp437 is active, use injected fake engines/timers and tiny real-engine
smokes only if they are single-process and non-heavy. Do not run official
readiness benchmarks.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_numba_antigravity \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_mlx_antigravity \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_mps_antigravity \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_backend_antigravity -v
```

- [ ] **Step 4: Commit backend gates**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/{numba,mlx,mps,backend}_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_{numba,mlx,mps,backend}_antigravity.py
git commit -m "feat(exp438): add truthful acceleration gates"
```

---

### Task 7: Add Ordered Checkpoints And Self-Written Progress Logs

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/checkpoint_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/progress_antigravity.py`
- Create: matching two test files

- [ ] **Step 1: Write RED persistence tests**

Cover atomic no-clobber writes, canonical ordered dataset records, context and
record-set fingerprints, interrupted commit recovery, corrupt/orphan rejection,
failed-root immutability, exact run identity/root binding, dataset metadata
roundtrip, and cross-Exp437 artifact rejection.

- [ ] **Step 2: Write RED logging tests**

Require atomic `progress_antigravity.json`, durable JSONL events, parent log,
PID, heartbeat, current dataset name, TRAIN/TEST counts and length histograms,
value/byte sizes, stage/backend, completed/total, import/recompute/success/
abstention/error counts, rate, elapsed/ETA, checkpoint, channel counts, and
latest stable error. A terminal snapshot requires `current_dataset=null`.

- [ ] **Step 3: Implement parent-owned persistence**

Reuse algorithms, not runtime identity, from the reviewed same-repository
checkpoint/progress patterns. Workers never write durable files. Bind every
record to Exp438 config, input hashes, code commit, backend decision, sorted
population, and fixed root.

- [ ] **Step 4: Run GREEN and shared persistence regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_progress_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_progress_antigravity -v
```

- [ ] **Step 5: Commit persistence**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/{checkpoint,progress}_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_{checkpoint,progress}_antigravity.py
git commit -m "feat(exp438): add isolated checkpoints and progress"
```

---

### Task 8: Orchestrate Serial And Eleven-Worker Runtime Execution

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/runtime_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_runtime_antigravity.py`

- [ ] **Step 1: Write RED runtime tests**

Cover at most `workers` in-flight futures, official worker count 11 on mocked
12-CPU topology, one numerical thread per worker, deterministic commit order,
accurate current-dataset metadata, parent-only SQLite/durable ownership,
per-dataset exception isolation followed by fail-closed run state, candidate
budget failure before record commit, interrupt/resume equivalence, and adopted
accelerator single-process behavior.

- [ ] **Step 2: Implement `_run` with explicit modes**

`new` starts only from a verified empty Exp438 root. `resume` accepts only an
interrupted Exp438 root. `inspect` is read-only and opens neither DB nor workers.
Checkpoint complete/failure are immutable. Dataset work is committed in sorted
population order even when workers finish out of order.

- [ ] **Step 3: Run GREEN tests and deterministic worker-count comparison**

Compare tiny fixture outputs for 1, 2, and 11 mocked workers byte-for-byte.

- [ ] **Step 4: Commit runtime orchestration**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/runtime_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_runtime_antigravity.py
git commit -m "feat(exp438): add deterministic parallel runtime"
```

---

### Task 9: Freeze Publication And Fixed Five-Command CLI

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/reporting_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/cli_antigravity.py`
- Create: `scratch/run_exp438_frequency_complexity_evidence_antigravity.py`
- Create: matching reporting and CLI test files

- [ ] **Step 1: Write RED publication tests**

Require exactly 1,117 unique completed datasets, zero missing/extra/failed,
inactive PID, terminal null current dataset, budget compliance, zero label
access, checkpoint/event/summary agreement, separate channel outputs, source
provenance, backend decision, manifest, status, and artifact hashes. Publication
must be staged, fsynced, no-clobber, review-only, and replay-verifiable.

- [ ] **Step 2: Write RED CLI/readiness tests**

Parser accepts exactly `readiness/new/resume/inspect/evaluate` and no path,
worker, backend, alpha, threshold, depth, budget, or channel override flags.
Readiness requires a clean stable HEAD, frozen input hashes, absent Exp438 fixed
root, absent/live-check-safe Exp437 PID gate, focused test evidence, and exact
provenance. The wrapper must work outside the repository CWD.

- [ ] **Step 3: Implement reporting and lazy CLI boundaries**

Prediction modes must not import `evaluation_antigravity.py`. `evaluate` imports
it lazily only after publication verification. Readiness owns creation of the
previously absent Exp438 parent and writes one canonical artifact. No command
overwrites an existing root.

- [ ] **Step 4: Run GREEN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_reporting_antigravity \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_cli_antigravity -v
python3 scratch/run_exp438_frequency_complexity_evidence_antigravity.py --help
```

- [ ] **Step 5: Commit publication and entrypoint**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/{reporting,cli}_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_{reporting,cli}_antigravity.py \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py
git commit -m "feat(exp438): add frozen publication and cli"
```

---

### Task 10: Add Lazy Post-Freeze Evaluation

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_evaluation_antigravity.py`

- [ ] **Step 1: Write RED label-boundary tests**

Use spies proving no TEST-label connection/query occurs before prediction,
manifest, artifact, runtime, population, DB, comparator, and provenance hashes
are independently replayed. A single failed gate must keep label access zero.
Evaluation must never change prediction/publication bytes.

- [ ] **Step 2: Write RED metric and decision tests**

For each channel independently compute TP/FP/FN/TN, precision, recall,
specificity, FPR, F1, rescued/missed/regressed datasets, paired micro/macro
deltas, 20,000 dataset-cluster bootstrap samples with seed `43820260807`, TRAIN
OOF Spearman against frozen Exp437 channels when available, and label-free
candidate overlap. Verify direct-improvement and integration-evidence gates are
separate and unavailable correlation cannot pass independence.

- [ ] **Step 3: Implement lazy evaluation**

Load TEST labels through one descriptor-bound read-only connection only after
all freeze gates pass. Produce canonical no-clobber evaluation, dataset metrics,
rescued/regressed rows, confusion matrices, bootstrap, correlation/overlap,
decision, and evaluation hash artifacts. Never select a winning channel inside
Exp438.

- [ ] **Step 4: Run GREEN and evaluation replay tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_evaluation_antigravity -v
```

- [ ] **Step 5: Commit evaluation**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/evaluation_antigravity.py \
  experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_evaluation_antigravity.py
git commit -m "feat(exp438): add post-freeze channel evaluation"
```

---

### Task 11: Freeze Governance, Regenerate The Map, And Verify Implementation

**Files:**
- Create: `docs/EXP438_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP438_RUNBOOK_antigravity_KO.md`
- Create: `docs/EXP438_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Conditionally modify after Exp437 terminal/user-direction pause:
  `RESEARCH_STATE.json`
- Regenerate together: all three `docs/codemap/` files

- [ ] **Step 1: Write preregistration and runbook before readiness**

Copy every approved constant and firewall from the design into the Korean
preregistration. The runbook records exact fixed commands and paths, immutable
failure handling, resume rules, log/PID locations, backend evidence, and the
Exp437-inactive gate. Do not record a backend as selected before readiness.

- [ ] **Step 2: Add Exp438 to the catalog without replacing active Exp437**

Record Exp438 as an isolated implementation track. Do not change
`active_experiment`, Exp437 PID/checkpoint, or Exp437 next action while Exp437 is
active. Change `RESEARCH_STATE.json` only after Exp437 is terminal and the
requested direction-review pause has occurred.

- [ ] **Step 3: Regenerate the code map from the completed module boundary**

Group the low-level Exp438 files into at most two primary nodes: prediction/
runtime and publication/evaluation. Add only source-backed imports, calls,
reads, and writes. Show wrapper -> CLI -> runtime -> inputs/scoring -> checkpoint
and post-freeze CLI -> evaluation -> SQLite labels as separate flows. Rebuild
HTML/JSON/lock together and re-run every code-map validator from Task 1.

- [ ] **Step 4: Run the complete focused Exp438 suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp438_frequency_complexity_evidence_antigravity/tests \
  -p 'test_*_antigravity.py' -v
```

- [ ] **Step 5: Run predecessor non-regression suites**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p 'test_*_antigravity.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
```

- [ ] **Step 6: Verify source policy and repository cleanliness**

```bash
git diff --check
git status --short --branch
git log --oneline --decorate -12
```

Confirm no Exp437 file/runtime path changed, no forbidden AFD/namesake code was
copied, no label loader is imported by prediction modules, all new filenames
contain `antigravity`, and the worktree is clean.

- [ ] **Step 7: Commit governance and final implementation evidence**

```bash
git add docs/EXP438_PREREGISTRATION_antigravity_KO.md \
  docs/EXP438_RUNBOOK_antigravity_KO.md \
  docs/EXP438_IMPLEMENTATION_REPORT_antigravity.md \
  EXPERIMENT_CATALOG.json \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(exp438): freeze implementation handoff"
```

---

### Task 12: Run Readiness And The Official 1,117-Dataset Experiment

This task is intentionally blocked by an active Exp437 process and the user's
requested direction-review pause. Do not repeatedly monitor it.

- [ ] **Step 1: Perform one explicit Exp437 terminal check**

Read Exp437's PID and terminal progress once. Require inactive PID, terminal
state, and `current_dataset=null`. If still active, stop without mutating Exp438
or consuming monitoring turns; report the Exp438 exact next command.

- [ ] **Step 2: Pause for the requested result/direction discussion**

Summarize Exp437 terminal outcome and confirm that starting Exp438 remains the
chosen direction. This is the one planned user exchange; do not ask repeated
implementation approvals.

- [ ] **Step 3: Run exact Exp438 readiness on a clean reviewed HEAD**

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py readiness
```

Readiness records CPU/Numba/MLX/MPS parity, timings, versions, device, selected
batch/backend, fallback reasons, focused tests, source provenance, database and
lane hashes, clean commit, and zero label access.

- [ ] **Step 4: Start the official run without Codex polling**

```bash
PYTHONDONTWRITEBYTECODE=1 nohup python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py new \
  > /Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/launcher_antigravity.log 2>&1 &
```

Report only the PID and these direct-monitoring paths:

```text
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/progress_antigravity.json
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/events_antigravity.jsonl
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/parent_antigravity.log
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/run_antigravity.pid
```

- [ ] **Step 5: Inspect and evaluate only after the user reports completion**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py inspect
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py evaluate
```

- [ ] **Step 6: Record the frozen decision**

If direct improvement passes, update `CHAMPION_LEDGER.json` as review-only. If it
fails, update `FAILURE_LEDGER.json`. If only integration evidence passes, record
that a new experiment number and preregistration are required before any union,
intersection, routing, or fusion with Exp437. Never retune Exp438 from its TEST
results.
