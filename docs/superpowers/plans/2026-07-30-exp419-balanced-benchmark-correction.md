# Exp419 Balanced Benchmark Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the infeasible 15-dataset Exp419 timing subset with a frozen,
TRAIN-only 19-dataset balanced subset while preserving every scientific,
parity, memory, measured-speed, approval, and publication gate needed to finish
the official 1,117-dataset run.

**Architecture:** Keep the original quantile selector as the anchor generator,
then add remaining datasets in deterministic `(-cost, dataset)` order until
both 4-worker and 6-worker LPT plans have at least `2.5x` proxy headroom. Bind
an immutable selection proof into `RuntimeBenchmark`, benchmark JSON, the
official verifier, and the exact official artifact bundle. The actual
qualification gate remains measured wall-clock speedup of at least `2.0x`.

**Tech Stack:** Python 3.14, standard-library `dataclasses`, canonical
JSON/SHA-256, `multiprocessing` with `spawn`, deterministic PyTorch CPU,
SQLite, `unittest`.

## Global Constraints

- The official population remains exactly 1,117 datasets from database
  SHA-256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Exp417 preprocessing, architecture, horizons, seeds, scoring, calibration,
  qualification, lane exclusion, workload gates, and candidate identity remain
  unchanged.
- TEST labels, anomaly locations, historical TEST/family performance, and
  human-review outcomes remain unreachable.
- Worker candidates remain exactly `(1, 2, 4, 6)` with start method `spawn`.
- Every worker remains deterministic PyTorch CPU with one intra-op and one
  inter-op thread; MLX and MPS remain excluded.
- Aggregate RSS remains at most `27,487,790,694` bytes.
- Exact one-worker/parallel scientific parity and measured wall-clock speedup
  of at least `2.0x` remain mandatory.
- The old 15-dataset anchors remain included; deterministic additions stop only
  after both 4-worker and 6-worker proxy headroom reach at least `2.5x`.
- No timing result is used to choose, add, remove, or reorder a benchmark
  dataset.
- The terminated diagnostic root
  `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark`
  remains immutable and is never accepted or resumed.
- The corrected benchmark root is new and exact:
  `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced`.
- The official 1,117-dataset run remains forbidden until the corrected
  benchmark qualifies and the user then says exactly
  `Exp419 1,117개 전체 label-free 병렬 실행 승인`.

## File Structure

- `experiments/exp419_deterministic_cpu_parallel/config.py`: freeze the `2.5`
  proxy gate and `(4, 6)` headroom worker counts.
- `experiments/exp419_deterministic_cpu_parallel/contracts.py`: validate and
  hash `BenchmarkSelectionProof`; bind it into `RuntimeBenchmark`.
- `experiments/exp419_deterministic_cpu_parallel/planner.py`: retain original
  anchors and build the deterministic balanced proof.
- `scratch/benchmark_exp419_deterministic_cpu_parallel.py`: execute the proof's
  exact selected population and serialize it once.
- `scratch/run_exp419_deterministic_cpu_parallel.py`: recompute the proof from
  frozen TRAIN metadata and reject any mismatch before opening roots.
- `experiments/exp419_deterministic_cpu_parallel/reporting.py`: round-trip and
  verify the proof in the official runtime benchmark artifact.
- `experiments/exp419_deterministic_cpu_parallel/tests/`: contract, planner,
  CLI, and reporting coverage.
- `docs/EXP419_PREREGISTRATION_KO.md`,
  `docs/EXP419_RUNBOOK_KO.md`, and the original Exp419 design: record the human
  ruling and corrected exact procedure.
- `experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json`: freeze
  the reviewed post-correction runtime sources.

---

### Task 1: Freeze the Balanced-Selection Contract

**Files:**
- Modify: `experiments/exp419_deterministic_cpu_parallel/config.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/contracts.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_config.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_contracts.py`

**Interfaces:**
- Produces:
  - `FROZEN_CONFIG.benchmark_proxy_headroom_gate == 2.5`
  - `FROZEN_CONFIG.benchmark_proxy_headroom_workers == (4, 6)`
  - `BenchmarkSelectionProof`
  - `benchmark_selection_proof_payload(proof) -> dict[str, object]`
  - `benchmark_selection_proof_from_payload(payload) -> BenchmarkSelectionProof`
- Consumes: existing `DatasetCost`, `ShardPlan`, canonical JSON/SHA-256 helpers.

- [ ] **Step 1: Write failing frozen-config tests**

Add:

```python
self.assertEqual(FROZEN_CONFIG.benchmark_proxy_headroom_gate, 2.5)
self.assertEqual(FROZEN_CONFIG.benchmark_proxy_headroom_workers, (4, 6))
with self.assertRaisesRegex(ValueError, "benchmark_proxy_headroom_gate"):
    replace(FROZEN_CONFIG, benchmark_proxy_headroom_gate=2.0).validate()
```

- [ ] **Step 2: Run the config test and verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_config -v
```

Expected: missing config fields.

- [ ] **Step 3: Add the two exact immutable config fields**

```python
benchmark_proxy_headroom_gate: float = 2.5
benchmark_proxy_headroom_workers: tuple[int, ...] = (4, 6)
```

Retain the existing default-instance, exact-type, exact-value validation.

- [ ] **Step 4: Write failing proof-contract tests**

Use this valid fixture:

```python
proof = BenchmarkSelectionProof(
    anchor_datasets=("a", "b"),
    added_datasets=("c", "d"),
    selected_datasets=("a", "b", "c", "d"),
    proxy_headroom_gate=2.5,
    total_cost=100,
    shard_loads={
        2: (50, 50),
        4: (40, 20, 20, 20),
        6: (40, 12, 12, 12, 12, 12),
    },
    proxy_headrooms={2: 2.0, 4: 2.5, 6: 2.5},
)
```

Assert:

- all dataset tuples are lexical, unique, and non-empty;
- anchors and additions are disjoint;
- selected datasets equal the lexical union of anchors and additions;
- gate is exactly `2.5`;
- mapping keys are exactly `(2, 4, 6)`;
- each load tuple length equals its worker key and sums to `total_cost`;
- each headroom equals `total_cost / max(loads)` within `1e-12`;
- 4-worker and 6-worker headroom meet `2.5`;
- 2-worker headroom need not meet `2.5`;
- negative, boolean, non-finite, missing, extra, reordered, and mismatched
  values raise `ValueError`;
- repeated construction produces the same lowercase SHA-256;
- payload round-trip is exact and rejects missing/extra/mutated fields.

- [ ] **Step 5: Run the proof tests and verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_contracts -v
```

Expected: `BenchmarkSelectionProof` import failure.

- [ ] **Step 6: Implement the minimal immutable proof**

Use:

```python
@dataclass(frozen=True)
class BenchmarkSelectionProof:
    anchor_datasets: tuple[str, ...]
    added_datasets: tuple[str, ...]
    selected_datasets: tuple[str, ...]
    proxy_headroom_gate: float
    total_cost: int
    shard_loads: Mapping[int, tuple[int, ...]]
    proxy_headrooms: Mapping[int, float]
    sha256: str = field(init=False)
```

Copy mappings into `MappingProxyType`, reject `bool` as an integer or real,
canonicalize JSON mapping keys as decimal strings, and compute `sha256` over
the payload without its own digest. The serializer adds `sha256`; the parser
reconstructs the record and compares the supplied digest.

- [ ] **Step 7: Run Task 1 tests and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_config \
  experiments.exp419_deterministic_cpu_parallel.tests.test_contracts -v
git diff --check
git add experiments/exp419_deterministic_cpu_parallel/config.py \
  experiments/exp419_deterministic_cpu_parallel/contracts.py \
  experiments/exp419_deterministic_cpu_parallel/tests/test_config.py \
  experiments/exp419_deterministic_cpu_parallel/tests/test_contracts.py
git commit -m "feat(exp419): freeze balanced benchmark proof"
```

---

### Task 2: Select the Deterministic 19-Dataset Balanced Population

**Files:**
- Modify: `experiments/exp419_deterministic_cpu_parallel/planner.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_planner.py`

**Interfaces:**
- Consumes:
  - `select_benchmark_datasets(costs) -> tuple[str, ...]` as the unchanged
    original anchor selector;
  - `build_shard_plan(costs, worker_count) -> ShardPlan`;
  - `BenchmarkSelectionProof`.
- Produces:
  - `select_balanced_benchmark_datasets(
        costs: Sequence[DatasetCost],
    ) -> BenchmarkSelectionProof`.

- [ ] **Step 1: Write failing synthetic selection tests**

Create literal costs where the original anchors are insufficient. Assert the
new function:

1. starts with every exact original anchor;
2. considers only remaining costs sorted by `(-cost, dataset)`;
3. adds exactly one record per iteration;
4. stops at the first prefix for which both 4-worker and 6-worker
   `total_cost / max(shard.total_cost) >= 2.5`;
5. returns lexical anchor, added, and selected names;
6. emits exact 2/4/6 load tuples and headroom values;
7. leaves `select_benchmark_datasets` unchanged as the anchor selector;
8. rejects duplicate datasets, fewer than six costs, non-`DatasetCost`
   elements, or an exhausted population that cannot meet the gate.

The expected values must be hand-derived literals rather than computed by the
production selector inside the assertion.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_planner -v
```

Expected: missing `select_balanced_benchmark_datasets`.

- [ ] **Step 3: Implement the minimal balanced selector**

Use this control flow:

```python
anchors = select_benchmark_datasets(costs)
selected = set(anchors)
remaining = sorted(
    (cost for cost in costs if cost.dataset not in selected),
    key=lambda cost: (-cost.cost, cost.dataset),
)
while True:
    selected_costs = tuple(
        cost for cost in costs if cost.dataset in selected
    )
    plans = {
        count: build_shard_plan(selected_costs, count)
        for count in (2, 4, 6)
    }
    total_cost = sum(cost.cost for cost in selected_costs)
    headrooms = {
        count: total_cost / max(
            assignment.total_cost
            for assignment in plan.assignments
        )
        for count, plan in plans.items()
    }
    if all(
        headrooms[count] >= FROZEN_CONFIG.benchmark_proxy_headroom_gate
        for count in FROZEN_CONFIG.benchmark_proxy_headroom_workers
    ):
        break
    if not remaining:
        raise ValueError("balanced benchmark population cannot meet proxy gate")
    selected.add(remaining.pop(0).dataset)
```

Construct the proof from exact plan loads and lexical names. Do not read
timings, TEST metadata, checkpoints, or prior benchmark files.

- [ ] **Step 4: Run Task 2 tests and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_planner -v
git diff --check
git add experiments/exp419_deterministic_cpu_parallel/planner.py \
  experiments/exp419_deterministic_cpu_parallel/tests/test_planner.py
git commit -m "feat(exp419): select balanced benchmark population"
```

---

### Task 3: Bind the Proof Into Benchmark, Official Verification, and Reporting

**Files:**
- Modify: `experiments/exp419_deterministic_cpu_parallel/contracts.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/coordinator.py`
- Modify: `scratch/benchmark_exp419_deterministic_cpu_parallel.py`
- Modify: `scratch/run_exp419_deterministic_cpu_parallel.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/reporting.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_contracts.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_coordinator.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_reporting.py`

**Interfaces:**
- `RuntimeBenchmark` gains required field
  `selection_proof: BenchmarkSelectionProof`.
- `RuntimeBenchmark.benchmark_datasets` must exactly equal
  `selection_proof.selected_datasets`.
- Benchmark and reporting payloads include
  `"selection_proof": benchmark_selection_proof_payload(...)`.
- The official verifier recomputes
  `select_balanced_benchmark_datasets(expected_costs)` and requires exact proof
  equality before checking qualification.

- [ ] **Step 1: Write failing `RuntimeBenchmark` binding tests**

Update the valid benchmark fixture to pass a proof. Assert construction rejects:

- benchmark datasets different from the proof;
- a proof with mutated anchors, additions, loads, headroom, total, or digest;
- a benchmark digest copied from a different proof.

Assert the benchmark SHA-256 changes when only the valid proof changes.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_contracts -v
```

Expected: `RuntimeBenchmark` does not accept or bind the proof.

- [ ] **Step 3: Add the required proof field to `RuntimeBenchmark`**

Validate the concrete proof type and exact dataset equality. Include the
proof's full canonical payload in the benchmark SHA-256 payload.

- [ ] **Step 4: Write failing benchmark CLI serialization tests**

Assert:

```python
payload = benchmark_payload(benchmark)
self.assertEqual(
    payload["selection_proof"],
    benchmark_selection_proof_payload(benchmark.selection_proof),
)
self.assertEqual(benchmark_from_payload(payload), benchmark)
```

Mutate each proof field and digest separately. Every mutation must make
`benchmark_from_payload` or `verify_benchmark_evidence` raise. A structurally
valid old 15-dataset proof must be rejected against current frozen costs.

- [ ] **Step 5: Update benchmark execution and payload parsing**

In `_run_trials`:

```python
selection_proof = select_balanced_benchmark_datasets(frozen.costs)
selected = selection_proof.selected_datasets
```

Pass `selection_proof` through `select_runtime`. Change
`select_runtime(trials, selection_proof)` so the resulting
`RuntimeBenchmark` always binds the exact proof. Update every call and test.

Serialize and reconstruct the proof only through the contract helper; do not
create a second proof codec.

- [ ] **Step 6: Write failing official verifier tests**

The verifier must reject before `open_roots`:

- old 15-name benchmark evidence;
- a proof recomputed from a cost-mutated dataset;
- correct names with wrong anchors/additions;
- correct names with wrong load order;
- correct proof with a stale Git, context, source-manifest, or benchmark digest;
- arbitrary sibling/temp benchmark roots;
- copied evidence at any path other than the exact direct child;
- trial roots with other descendants, wrong order, or symlink aliases.

It must accept only the exact lexical/canonical corrected root, direct evidence
child, exact ordered direct trial roots, and an exact recomputed proof.

- [ ] **Step 7: Implement exact official proof verification**

After parsing the benchmark and before status selection:

```python
expected_proof = select_balanced_benchmark_datasets(tuple(expected_costs))
if benchmark.selection_proof != expected_proof:
    raise ValueError(
        "benchmark selection proof differs from frozen TRAIN-only balance"
    )
```

Require every `trial.completed_datasets`,
`benchmark.benchmark_datasets`, and `expected_proof.selected_datasets` to be
identical lexical tuples. Freeze the corrected benchmark root, direct evidence
file, and ordered direct trial roots once in shared reviewed production
configuration. Validate exact lexical and canonical equality before creating
or opening benchmark or official roots; reject every arbitrary root,
descendant, copy, reorder, and symlink alias.

- [ ] **Step 8: Write failing reporting round-trip tests**

Assert the official `exp419_parallel_runtime_benchmark.json` contains the exact
proof, its digest, and benchmark digest. Mutate every proof field in an already
rendered ten-file bundle and require `verify_exp419_payloads` to fail.

- [ ] **Step 9: Implement reporting round-trip**

Use the same contract serializer/parser in `_benchmark_payload` and
`_reconstruct_benchmark`. Do not alter the exact ten artifact names.

- [ ] **Step 10: Run Task 3 tests and commit**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_contracts \
  experiments.exp419_deterministic_cpu_parallel.tests.test_coordinator \
  experiments.exp419_deterministic_cpu_parallel.tests.test_cli \
  experiments.exp419_deterministic_cpu_parallel.tests.test_reporting -v
.venv/bin/python -m py_compile \
  experiments/exp419_deterministic_cpu_parallel/*.py \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
git diff --check
git add experiments/exp419_deterministic_cpu_parallel/contracts.py \
  experiments/exp419_deterministic_cpu_parallel/coordinator.py \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py \
  experiments/exp419_deterministic_cpu_parallel/reporting.py \
  experiments/exp419_deterministic_cpu_parallel/tests
git commit -m "feat(exp419): bind balanced benchmark evidence"
```

---

### Task 4: Freeze the Human Ruling, Runbook, and Reviewed Sources

**Files:**
- Modify:
  `docs/superpowers/specs/2026-07-30-exp419-deterministic-cpu-parallel-runtime-design.md`
- Modify: `docs/EXP419_PREREGISTRATION_KO.md`
- Modify: `docs/EXP419_RUNBOOK_KO.md`
- Modify:
  `docs/superpowers/plans/2026-07-30-exp419-deterministic-cpu-parallel-runtime.md`
- Modify:
  `experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json`
- Modify: `experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py`

**Interfaces:**
- Documents declare the 2026-07-30 pre-official human ruling, exact balanced
  algorithm, old invalid root, new root, unchanged gates, and exact official
  approval phrase.
- Source manifest binds the exact same ten runtime source paths with new
  SHA-256 values.

- [ ] **Step 1: Add failing operational-root and manifest tests**

Extend CLI behavior tests so:

- benchmark evidence whose `trial_roots` point at the old diagnostic root or
  its descendants is rejected before official roots open;
- evidence whose `trial_roots` use the corrected disjoint roots passes this
  boundary;
- `propose_source_manifest()` differs before regeneration and matches after it.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m unittest \
  experiments.exp419_deterministic_cpu_parallel.tests.test_cli -v
```

Expected: old selection/root behavior or stale manifest fails.

- [ ] **Step 3: Update design, preregistration, runbook, and original plan**

Record:

- checkpoint-universe fix commits `f54ac4d` and `b706393`;
- approved balanced design commit `4ea4334`;
- original 15-anchor costs and `1.0996059372x` ceiling;
- exact balanced algorithm and 19-dataset result;
- proxy gate `2.5`, unchanged measured gate `2.0`;
- corrected benchmark command and inspection fields;
- invalid-root retention and non-reuse;
- no automatic polling;
- fresh exact official approval after benchmark qualification.

Do not weaken or delete the original scientific, label-isolation, memory,
parity, local-only, or publication constraints.

- [ ] **Step 4: Regenerate the manifest explicitly**

Invoke `propose_source_manifest` with the exact repository root and exact
manifest output path. Immediately verify:

```python
assert propose_source_manifest(REPO_ROOT) == SOURCE_MANIFEST_PATH.read_bytes()
assert verify_source_manifest(REPO_ROOT) == sha256(
    SOURCE_MANIFEST_PATH.read_bytes()
).hexdigest()
```

- [ ] **Step 5: Run all regression and isolation gates**

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp419_deterministic_cpu_parallel/tests -p "test_*.py"
.venv/bin/python -m py_compile \
  experiments/exp419_deterministic_cpu_parallel/*.py \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
.venv/bin/python - <<'PY'
from scratch.benchmark_exp419_deterministic_cpu_parallel import (
    assert_exp419_label_isolation,
)
print(assert_exp419_label_isolation())
PY
git diff --check
```

Expected: zero test failures/errors, syntax and source/label guards pass.

- [ ] **Step 6: Commit**

```bash
git add docs/EXP419_PREREGISTRATION_KO.md \
  docs/EXP419_RUNBOOK_KO.md \
  docs/superpowers/specs/2026-07-30-exp419-deterministic-cpu-parallel-runtime-design.md \
  docs/superpowers/plans/2026-07-30-exp419-deterministic-cpu-parallel-runtime.md \
  experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json \
  experiments/exp419_deterministic_cpu_parallel/tests/test_cli.py
git commit -m "docs(exp419): freeze balanced benchmark correction"
```

---

### Task 5: Run and Audit the Corrected Private Benchmark

**Files:**
- Create local only:
  `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced/`
- Create local only:
  `/Users/mac/Documents/Dataset/exp419_task8_balanced_<HEAD>.log`
- Create local only:
  `/Users/mac/Documents/Dataset/exp419_task8_balanced_<HEAD>.status`

**Interfaces:**
- Consumes: clean reviewed HEAD and the five frozen private inputs.
- Produces:
  `EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json` with four trials and exact
  selection proof.

- [ ] **Step 1: Verify clean immutable inputs**

Run the runbook's exact SHA-256 and SQLite quick-check commands. Require:

- database `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`;
- exact four lane artifact digests already frozen in Exp417/Exp419 config;
- `PRAGMA quick_check` returns `ok`;
- Git is clean and HEAD equals the reviewed manifest-bound commit;
- corrected benchmark root, log, and status paths are new.

- [ ] **Step 2: Verify the exact balanced proof without model execution**

Freeze inputs, recompute the proof, and print:

- 15 anchors;
- 4 additions;
- 19 selected datasets;
- total `888,487,623`;
- exact 2/4/6 loads;
- proof SHA-256;
- context and source-manifest SHA-256.

Abort on any mismatch.

- [ ] **Step 3: Start the corrected benchmark once**

Use the exact runbook command in one tool-managed foreground process. The
wrapper writes its PID, current stage, exit status, and complete log. Do not use
`nohup`, a background shell, an automatic monitor, a guessed timeout, MLX,
MPS, or a competing heavy process.

- [ ] **Step 4: Stop monitoring**

Return the wrapper PID and exact local paths. Do not poll until the user
reports completion or explicitly asks for status.

- [ ] **Step 5: Audit completed benchmark evidence**

After completion, verify:

- wrapper exit status `0`;
- all four trials `(1, 2, 4, 6)` exist exactly once;
- every trial covers the exact 19 datasets;
- one-worker is finite and establishes the scientific reference;
- all parallel trials have exact model, calibration, qualification, lane
  exclusion, candidate, and record-digest parity;
- at least one of 4 or 6 workers has measured speedup `>=2.0`;
- the selected worker is the fastest passing candidate;
- aggregate peak RSS is `<=27,487,790,694`;
- no fallback, missing, duplicate, extra, terminal, or non-finite evidence;
- utilization formulas and every SHA-256 are exact;
- the recomputed balanced proof equals the embedded proof.

If any item fails, do not retune the subset or start the official run.

---

### Task 6: Run and Audit the Official 1,117-Dataset Bundle

**Files:**
- Create local only:
  `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel.checkpoint`
- Create local only:
  `/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_official`
- Create local only: a no-clobber official log, status, stage, and PID set.

**Interfaces:**
- Consumes: qualified corrected benchmark, clean identical Git/input/source
  state, and fresh exact user approval.
- Produces: complete canonical checkpoint and exact ten-file official bundle.

- [ ] **Step 1: Request and record the exact fresh approval**

After Task 5 passes, require the user to say exactly:

`Exp419 1,117개 전체 label-free 병렬 실행 승인`

Do not infer this from earlier blanket approval, design approval, files, or
benchmark success.

- [ ] **Step 2: Revalidate immediately before root creation**

Require:

- clean Git and exact benchmark-bound HEAD;
- current source-manifest digest equals benchmark evidence;
- five frozen input digests and SQLite quick-check match;
- official checkpoint and output roots are new, disjoint, non-symlink paths;
- selected worker count and all proof/trial gates reverify.

- [ ] **Step 3: Start the exact official command**

Run the command from the corrected runbook in a tool-managed foreground
process. Do not pass a worker override. The official CLI must execute:

`clean Git -> label guard -> freeze inputs -> verify benchmark -> open roots
-> run/resume -> revalidate -> render -> verify -> atomic publish`.

- [ ] **Step 4: Stop monitoring**

Return only the wrapper PID and exact log/status/stage paths. Wait for the user
to report completion; do not automatically poll.

- [ ] **Step 5: Audit the checkpoint**

After completion, prove:

- exactly 1,117 expected datasets;
- exactly 1,117 valid records and digests;
- no missing, duplicate, extra, corrupt, terminal, or wrong-shard record;
- exact provenance, context, source, config, plan, runtime, and input digests;
- all worker completions and the canonical merge are complete;
- aggregate RSS and worker-count contract hold.

- [ ] **Step 6: Audit the exact official bundle**

Require exactly:

```text
exp419_shadow_predictions.jsonl
exp419_dataset_qualification.csv
exp419_oof_normal_metrics.csv
exp419_synthetic_safety.csv
exp419_backend_benchmark.json
exp419_parallel_runtime_benchmark.json
exp419_frozen_config.json
exp419_shard_plan.json
EXP419_RESULT_REPORT_KO.md
MANIFEST.json
```

Run `verify_exp419_payloads`, reconstruct every payload, compare every manifest
SHA-256 to disk, require 1,117 unique prediction/qualification datasets, and
confirm no forbidden label/evaluation fields or lane promotion.

- [ ] **Step 7: Final clean-tree and result report**

Run:

```bash
git status --short --branch
git diff --check
git log -1 --oneline
```

Report selected workers, benchmark times/speedup/memory, official elapsed time,
dataset and candidate counts, checkpoint and artifact paths, Git commit,
source/input/context/benchmark/checkpoint/manifest digests, and audit results.
Only after every requirement is proven may the active goal be marked complete.
