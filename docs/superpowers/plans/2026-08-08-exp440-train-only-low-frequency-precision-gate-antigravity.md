# Exp440 TRAIN-Only Low-Frequency Precision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Exp440 that uses TRAIN-only low-frequency multi-depth agreement and stability to retain at most 100 outside-Hard candidates, then evaluates only after prediction freeze.

**Architecture:** Reuse Exp439's verified pure FFT and database boundaries through explicit adapters, but own all Exp440 configuration, contracts, calibration, budget, checkpoint, runtime, publication, and evaluation artifacts. Compute fold-local OOF thresholds at adjacent depths, reject unstable groups, rank two-depth consensus candidates deterministically, exclude Hard coordinates, and cap globally at 100.

**Tech Stack:** Python 3.14, NumPy float32/float64 and FFT, SQLite read-only snapshot, `unittest`, multiprocessing with 11 dataset workers, optional MLX dense-distance readiness, optional Numba exact-hotspot readiness, canonical JSON, SHA256.

## Global Constraints

- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Exp439 checkpoint, publication, and evaluation remain immutable.
- Prediction never reads TEST labels, anomaly locations, Exp439 evaluation, or past family-level TEST performance.
- TEST labels are opened only by explicit evaluation after the Exp440 prediction SHA is frozen.
- TRAIN nonfinite rows are excluded whole; TEST nonfinite rows abstain; no imputation.
- Outside-Hard candidate budget is a hard maximum of 100.
- CPU workers are `max(1, logical CPUs - 1)`, expected 11; each worker has one numerical thread.
- Adopt MLX/Numba only after exact candidate parity, finite outputs, tolerance `1e-5`, provenance, and end-to-end stage speedup at least `1.1x`.
- Run only one MLX/MPS-heavy process at a time.
- Runtime owns atomic progress JSON, append-only JSONL, text log, PID, checkpoint, heartbeat, throughput, and ETA.
- Failed roots are immutable; only exact interrupted ordered prefixes may resume.
- Code-map HTML, JSON, and lock are regenerated together whenever boundaries or evidence change.

---

### Task 1: Freeze Exp440 governance, identity, and code-map scope

**Files:**
- Create: `docs/EXP440_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP440_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `FAILURE_LEDGER.json`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: approved Exp440 design, Exp439 frozen prediction SHA, Exp404 Hard comparator SHA.
- Produces: exact identity, source hashes, success rule, commands, log paths, and implementation state.

- [ ] **Step 1: Record the frozen literal contract**

The preregistration must contain exactly:

```text
experiment_id = Exp440
run_identity = exp440-train-only-low-frequency-precision-gate-antigravity-20260808
dataset_count = 1117
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
exp439_prediction_sha256 = 500f6956b8640ea329cd47c8cdb504debaeff6ec67c44d0c56b96c1ecd77f3ef
hard_prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
nearest_k = 3
quantile = 0.99, method = higher, comparison = strict greater
stability_ratio_max = 2.0
consensus_depths_min = 2
outside_hard_budget = 100
bootstrap_seed = 44020260808
bootstrap_samples = 20000
```

Run parent:

```text
/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808
```

- [ ] **Step 2: Define exact commands and observability paths**

Expose only `readiness`, `new`, `resume`, `inspect`, and `evaluate`. The runbook lists exact absolute readiness, output, checkpoint, publication, evaluation, progress JSON, event JSONL, parent log, and PID paths.

- [ ] **Step 3: Update research control files**

Append Exp439's completed unfavorable outcome to `FAILURE_LEDGER.json` with frozen prediction/evaluation hashes and no promotion. Add Exp440 to `EXPERIMENT_CATALOG.json` as `design_approved_implementation_planned`. Set `RESEARCH_STATE.json.active_experiment` to `Exp440`, status `in_progress`, PID null, and next action `implement_exp440_test_first`.

- [ ] **Step 4: Validate and commit governance**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git add docs/EXP440_PREREGISTRATION_antigravity_KO.md \
  docs/EXP440_RUNBOOK_antigravity_KO.md EXPERIMENT_CATALOG.json \
  FAILURE_LEDGER.json RESEARCH_STATE.json docs/codemap/codemap.html \
  docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(exp440): freeze low-frequency precision gate"
```

---

### Task 2: Create isolated configuration and immutable contracts

**Files:**
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/config_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/contracts_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_contracts_antigravity.py`

**Interfaces:**
- Produces: `Exp440FrozenConfig`, `DepthCalibrationAntigravity`, `GroupCalibrationAntigravity`, `DepthScoreAntigravity`, `ProvisionalCandidateAntigravity`, `Exp440DatasetResultAntigravity`.

- [ ] **Step 1: Write RED configuration tests**

```python
self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp440")
self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
self.assertEqual(FROZEN_CONFIG.adjacent_offsets, (-1, 0, 1))
self.assertEqual(FROZEN_CONFIG.minimum_consensus_depths, 2)
self.assertEqual(FROZEN_CONFIG.outside_hard_budget, 100)
self.assertEqual(cpu_worker_count_antigravity(12), 11)
self.assertTrue(FROZEN_CONFIG.review_only)
self.assertFalse(FROZEN_CONFIG.operational_alert_eligible)
```

Run and confirm import failure because the package does not exist.

- [ ] **Step 2: Implement the frozen configuration**

Use a frozen dataclass with validation for every literal in Task 1, batch sizes `(16, 32, 64, 128, 256)`, MLX seed `44001`, speedup `1.1`, tolerance `1e-5`, and the fixed run roots. `frozen_config_payload_antigravity()` returns canonical JSON-safe values.

- [ ] **Step 3: Write RED contract invariants**

Tests must reject unordered/duplicate depths, nonfinite scores, negative
thresholds, agreement below two, ranking keys inconsistent with evidence,
candidates inside Hard, and more than 100 retained coordinates. A zero threshold
remains valid and uses the frozen `max(threshold, 1e-12)` ratio denominator.

```python
candidate = ProvisionalCandidateAntigravity(
    dataset="A", instance_index=7, raw_length=32,
    depths=(1, 2), agreement_count=2,
    second_excess_ratio=1.5, second_empirical_rank=0.1,
)
self.assertEqual(candidate.ranking_key, (-2, -1.5, 0.1, "A", 7))
```

- [ ] **Step 4: Implement minimal immutable contracts and run GREEN**

All array inputs are copied into contiguous read-only float32 arrays. Mapping inputs are copied into mapping proxies. Every payload roundtrip is canonical and rejects unknown keys.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_config_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_contracts_antigravity -v
```

- [ ] **Step 5: Commit contracts**

```bash
git add experiments/exp440_train_only_low_frequency_precision_gate_antigravity
git commit -m "feat(exp440): freeze precision-gate contracts"
```

---

### Task 3: Build label-free inputs and source firewall

**Files:**
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/inputs_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/predecessor_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_inputs_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_predecessor_antigravity.py`

**Interfaces:**
- Consumes: Exp439 database snapshot primitives, Exp439 publication manifest, frozen Hard coordinates.
- Produces: `open_exp440_input_session_antigravity(...)`, exact dataset order, grouped finite TRAIN rows, TEST raw rows, Hard exclusion set, and zero label-access counter.

- [ ] **Step 1: Write RED firewall tests**

Assert prediction source validation rejects any path component named `evaluation_antigravity`, any label-bearing field, wrong Exp439 prediction SHA, wrong Hard SHA, wrong DB SHA, population other than 1,117, and noncanonical dataset order.

- [ ] **Step 2: Write RED nonfinite/exact-length tests**

Given finite length-16 siblings plus one NaN TRAIN row, assert the bad row is audited and the finite group remains usable. Given one nonfinite TEST row, assert only that row abstains. Assert no imputation and no cross-length references.

- [ ] **Step 3: Implement narrow adapters**

Reuse only pure read-only database and lane-source functions. Do not import Exp439 evaluation. Bind the Exp439 publication using prediction and manifest hashes, but obtain scientific TRAIN/TEST values from the frozen SQLite snapshot.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_inputs_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_predecessor_antigravity -v
git add experiments/exp440_train_only_low_frequency_precision_gate_antigravity
git commit -m "feat(exp440): enforce label-free source firewall"
```

---

### Task 4: Implement adjacent-depth calibration and TRAIN-only stability

**Files:**
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/features_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/scoring_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_features_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_scoring_antigravity.py`

**Interfaces:**
- Consumes: Exp439 `maximum_depth_antigravity`, `preferred_depth_antigravity`, `fft_split_antigravity`; finite exact-length TRAIN groups.
- Produces: `adjacent_depths_antigravity`, `fit_group_calibration_antigravity`, `score_test_row_antigravity`.

- [ ] **Step 1: Write RED adjacent-depth boundary tests**

```python
self.assertEqual(adjacent_depths_antigravity(1, 1), ())
self.assertEqual(adjacent_depths_antigravity(1, 2), (1, 2))
self.assertEqual(adjacent_depths_antigravity(3, 5), (2, 3, 4))
self.assertEqual(adjacent_depths_antigravity(5, 5), (4, 5))
```

The one-depth case returns empty because Exp440 requires at least two views.

- [ ] **Step 2: Write RED fold-local OOF test**

Use five distinct finite rows and a spy distance function. For every held-out row, assert references contain exactly the other four rows and the OOF threshold equals `np.quantile(scores, 0.99, method="higher")`.

- [ ] **Step 3: Write RED stability tests**

Accept delete-one thresholds `(1.0, 1.5, 2.0, 1.25)` because ratio is `2.0`. Reject `(1.0, 2.01, 1.5, 1.25)`, zero robust scale, nonfinite threshold, and fewer than four OOF scores.

- [ ] **Step 4: Implement minimal multi-depth calibration**

Cache every TRAIN row's preferred depth once and each depth's low-frequency matrix once. At every depth, recompute fold-local robust center/scale and nearest-normal `k=3` scores. Build delete-one threshold replay from the completed OOF vector; never reuse a full-fold transform inside a held-out fold.

- [ ] **Step 5: Write and implement strict consensus scoring**

For each stable depth compute score, threshold, `score > threshold`, excess ratio, and empirical rank. Return a provisional candidate only with at least two exceedances; set the second-largest ratio and second-smallest empirical rank exactly as the design specifies.

- [ ] **Step 6: Run GREEN, Exp439 primitive parity, and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_features_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_scoring_antigravity -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp439_scale_aware_reconstruction_repair_antigravity.tests.test_features_antigravity -v
git add experiments/exp440_train_only_low_frequency_precision_gate_antigravity
git commit -m "feat(exp440): calibrate stable multi-depth consensus"
```

---

### Task 5: Enforce deterministic Hard exclusion and global budget

**Files:**
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/budget_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_budget_antigravity.py`

**Interfaces:**
- Produces: `select_global_candidates_antigravity(candidates, hard_coordinates, budget=100) -> tuple[RetainedCandidateAntigravity, ...]`.

- [ ] **Step 1: Write RED ordering and exclusion tests**

Create 105 shuffled candidates with ties. Assert Hard coordinates are removed before sorting, output is exactly the first 100 ranking keys, ranks are 1 through 100, and repeated shuffled inputs produce byte-identical payloads.

- [ ] **Step 2: Write RED sparse case**

Given seven outside-Hard candidates, assert all seven remain; do not pad or invent candidates.

- [ ] **Step 3: Implement minimal pure selection**

```python
outside = [item for item in candidates if item.coordinate not in hard_coordinates]
ordered = sorted(outside, key=lambda item: item.ranking_key)
return tuple(replace(item, global_rank=index + 1) for index, item in enumerate(ordered[:budget]))
```

Validate unique coordinates and reject budget overrides other than the frozen value at public boundaries.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_budget_antigravity -v
git add experiments/exp440_train_only_low_frequency_precision_gate_antigravity
git commit -m "feat(exp440): cap deterministic outside-hard candidates"
```

---

### Task 6: Benchmark MLX and optional Numba without changing the algorithm

**Files:**
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/backend_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/mlx_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/numba_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_backend_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_mlx_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_numba_antigravity.py`

**Interfaces:**
- Produces: `benchmark_mlx_antigravity`, `benchmark_numba_antigravity`, `select_backend_antigravity`; selected callable must preserve CPU scoring semantics.

- [ ] **Step 1: Write RED finite/parity/speed tests**

For batches `(16, 32, 64, 128, 256)`, reject nonfinite output, any changed candidate index, max absolute error above `1e-5`, speedup below `1.1`, missing synchronization, or absent version/device evidence. Assert CPU fallback is valid and selects 11 workers.

- [ ] **Step 2: Implement MLX dense-distance benchmark**

Benchmark only batched query/reference distance and median-nearest-3. Include host/device transfer and `mx.eval` synchronization inside timed runs. Use seed `44001`, two warmups, five paired timings, and record MLX/metal versions and Apple device.

- [ ] **Step 3: Implement optional Numba benchmark**

If Numba imports, benchmark the exact same dense-distance primitive including compilation warmup outside timed runs. If unavailable or slower than `1.1x`, record truthful CPU fallback. Never require Numba for correctness.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_backend_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_mlx_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_numba_antigravity -v
git add experiments/exp440_train_only_low_frequency_precision_gate_antigravity
git commit -m "feat(exp440): gate optional scoring acceleration"
```

---

### Task 7: Implement ordered checkpoint, progress, and resumable runtime

**Files:**
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/checkpoint_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/progress_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/runtime_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_checkpoint_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_progress_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_runtime_antigravity.py`

**Interfaces:**
- Produces: create-only checkpoint store, atomic progress writer, 11-worker ordered execution, exact interrupted resume, provisional-candidate spool for final global budget.

- [ ] **Step 1: Write RED checkpoint and crash-window tests**

Assert ordered-prefix commits, intent-with-record reconciliation, altered-record rejection, duplicate dataset rejection, context/hash binding, no-clobber creation, and resume only from `interrupted` with dead PID.

- [ ] **Step 2: Write RED progress schema tests**

Require PID, current dataset, TRAIN/TEST counts and lengths, stage, backend, completed/total, import/recompute/success/abstention/error, provisional/retained candidate counts, throughput, elapsed, ETA, checkpoint, heartbeat, and latest error. Assert JSON replacement is atomic and events/text are append-only.

- [ ] **Step 3: Write RED 11-worker ordered scheduling test**

Use delayed thread fixtures whose completion order differs from dataset order. Assert up to 11 tasks remain active, buffered capacity is bounded by `workers**2`, commits remain frozen-order, and serial accelerator execution uses one outer worker.

- [ ] **Step 4: Implement runtime**

Workers receive no labels or live SQLite objects. The parent owns checkpoint and logging. Each dataset result includes all provisional candidates; after all 1,117 ordered records commit, the parent applies the global budget and atomically records the retained-coordinate manifest.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_progress_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_runtime_antigravity -v
git add experiments/exp440_train_only_low_frequency_precision_gate_antigravity
git commit -m "feat(exp440): add ordered observable runtime"
```

---

### Task 8: Add readiness, publication, evaluation, and fixed CLI

**Files:**
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/readiness_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/reporting_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/cli_antigravity.py`
- Create: `scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_readiness_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_reporting_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_evaluation_antigravity.py`
- Create: `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests/test_cli_antigravity.py`

**Interfaces:**
- Produces: five-command fixed CLI, clean-HEAD readiness, label-free prediction publication, lazy post-freeze evaluator, hash-bound artifacts.

- [ ] **Step 1: Write RED readiness and CLI tests**

Assert fixed roots and no scientific/path overrides; readiness requires clean HEAD, exact DB/Exp439/Hard hashes, 1,117 datasets, absent output/checkpoint/evaluation, zero label access, focused tests, and backend decision evidence. CLI help exposes exactly five commands.

- [ ] **Step 2: Implement readiness and wrapper**

Readiness atomically owns the absent run parent. The wrapper resolves repository root before importing Exp440 CLI. `new` requires exact readiness; `resume` accepts only interrupted; `inspect` is read-only; `evaluate` imports label-capable code lazily.

- [ ] **Step 3: Write RED publication tests**

Require 1,117 completed records, zero errors, inactive PID, budget at most 100, unique outside-Hard coordinates, zero prediction label access, complete explanation evidence, deterministic JSONL order, no forbidden label fields, and no-clobber atomic publication.

- [ ] **Step 4: Implement publication artifacts**

Publish prediction, dataset summary, depth evidence, abstention ledger, TRAIN exclusion audit, backend decision, retained-candidate ranking, manifest, status, and artifact SHA256 files. Every new artifact filename contains `antigravity`.

- [ ] **Step 5: Write RED evaluator success/failure tests**

Evaluation validates every frozen source and snapshot before the first label operation. Assert the decision requires candidate count `<=100`, rescued datasets `>=10`, added FP `<=30`, precision `>=0.35`, positive micro/macro deltas, both bootstrap lower bounds `>0`, and no strong dataset with baseline F1 `>=0.8` losing more than `0.2`.

- [ ] **Step 6: Implement lazy evaluation and eight hash-bound artifacts**

Use seed `44020260808` and 20,000 dataset bootstrap samples. Report TP/FP/FN/TN, precision/recall/F1/FPR, per-dataset deltas, rescued/missed/regressed datasets, severe-regression ledger, decision, and hashes. Winner remains false unless every frozen rule passes.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests \
  -p 'test_*_antigravity.py'
git add experiments/exp440_train_only_low_frequency_precision_gate_antigravity \
  scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py
git commit -m "feat(exp440): publish and evaluate frozen precision gate"
```

---

### Task 9: Full verification, code map, and readiness handoff

**Files:**
- Create: `docs/EXP440_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Produces: reviewed clean implementation commit, synchronized map, exact readiness command; does not start the 1,117-dataset run until implementation gates pass.

- [ ] **Step 1: Run focused and inherited regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests \
  -p 'test_*_antigravity.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests \
  -p 'test_*_antigravity.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p 'test_*_antigravity.py'
```

All commands must exit zero. Record exact test counts and durations.

- [ ] **Step 2: Run forbidden-source and naming audits**

```bash
git grep -nE 'evaluation_antigravity|test_label|groundtruth|anomaly_location|family.*test' -- \
  experiments/exp440_train_only_low_frequency_precision_gate_antigravity \
  scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py
find experiments/exp440_train_only_low_frequency_precision_gate_antigravity \
  -type f ! -name '*antigravity*' -print
```

Classify evaluator-only references explicitly. The naming audit must print nothing except generated directories, which are excluded and removed from the audit scope.

- [ ] **Step 3: Run deterministic synthetic end-to-end smoke**

Create a temporary 4-dataset fixture through public APIs. Prove identical CPU outputs across two runs, Hard exclusion before budget, maximum 100 candidates, publication replay, and first label access only after frozen prediction inspection.

- [ ] **Step 4: Run real TRAIN-only smoke**

Use one small and one large TRAIN dataset from the frozen SQLite snapshot. Record dataset names, row counts, lengths, stable depths, backend, elapsed time, candidate parity, excluded rows, and `label_access_count=0`. Do not use TEST labels.

- [ ] **Step 5: Regenerate and validate the code map**

Keep at most 20 nodes by extending the existing frequency-runtime and publication/evaluation nodes. Add exact Exp440 runner/runtime/input/checkpoint/publication/evaluation source symbols and edges. Validate paths, symbols, node/edge/flow references, HTML payload identity, commit/dirty state, and deterministic module fingerprints.

- [ ] **Step 6: Write implementation report and control state**

Set Exp440 to `implementation_complete_readiness_pending`, PID null, prediction not started, TEST labels unopened, and exact next command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py readiness
```

- [ ] **Step 7: Final verification and commit**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git status --short --branch
git add docs/EXP440_IMPLEMENTATION_REPORT_antigravity.md RESEARCH_STATE.json \
  EXPERIMENT_CATALOG.json docs/codemap/codemap.html docs/codemap/codemap.json \
  docs/codemap/codemap.lock
git commit -m "docs(exp440): bind verified readiness handoff"
```

After the clean reviewed commit is established, run readiness once. If readiness passes, start the approved 1,117-dataset label-free run with the exact runbook command and let the process write its own logs without recurring Codex polling.
