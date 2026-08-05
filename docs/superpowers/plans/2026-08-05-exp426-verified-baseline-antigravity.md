# Exp426 Verified Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exp424의 검증된 983개 결과와 TRAIN-only로 다시 확인한 134개 판단 보류를 합쳐, TEST 정답 없이 1,117개 전체 연구 기준 결과를 완성하고 고정한다.

**Architecture:** Exp425의 읽기 전용 출처 검증과 DB snapshot 원칙을 재사용하되 Exp425 MPS runtime은 사용하지 않는다. 부모 프로세스 하나가 DB, checkpoint와 로그를 소유하고, 고정된 12개 CPU worker가 출처 record 해석과 TRAIN 근거 확인을 수행한다. 모든 과학적 record는 데이터셋 이름순으로 저장되어 worker 완료 순서와 무관하게 같은 SHA256을 만든다.

**Tech Stack:** Python 3.14, `concurrent.futures.ProcessPoolExecutor`, SQLite read-only snapshot, NumPy/Numba CPU, canonical JSON, SHA256, `unittest`.

## Global Constraints

- 새 repository 파일 이름에는 모두 소문자 `antigravity`가 포함되어야 한다.
- 새 Exp426 directory는 namespace package이며 새 `__init__.py`를 만들지 않는다.
- worker 수는 사용자의 Exp426 전용 지시에 따라 정확히 12개다. 호출자가 바꿀 수 없다.
- 각 worker의 `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`는 모두 `1`이다.
- MPS와 MLX는 사용하지 않는다. Exp425 retry3 실패와 1.1배 속도 조건 미달을 manifest에 기록한다.
- prediction 전에는 TEST 값, TEST 정답, 실제 이상 위치, TEST 기반 family 성능과 평가 지표를 열지 않는다.
- Exp424 checkpoint와 Exp291/292 lane, DB는 읽기 전용이다.
- Exp424 구성은 정확히 records/importable/terminal = `1117/983/134`다.
- 134개 모두 현재 TRAIN에서 `oof_identity_insufficient`여야 한다. 하나라도 eligible이면 전체 실행을 중단한다.
- 실패한 output/checkpoint root는 삭제하거나 재사용하지 않는다.
- output root는 `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805`다.
- checkpoint root는 `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint`다.
- readiness 파일은 `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_readiness_20260805.json`이다.

---

### Task 1: Frozen Contracts and Crash-Safe Checkpoint

**Files:**
- Create: `experiments/exp426_verified_baseline/contracts_antigravity.py`
- Create: `experiments/exp426_verified_baseline/checkpoint_antigravity.py`
- Create: `experiments/exp426_verified_baseline/tests/test_contracts_antigravity.py`
- Create: `experiments/exp426_verified_baseline/tests/test_checkpoint_antigravity.py`

**Interfaces:**
- Produces: `Exp426Provenance`, `Exp426DatasetRecord`, `Exp426Counts`, `Exp426Progress`, `WorkerEvidence`.
- Produces: `Exp426CheckpointStore.create`, `.open`, `.inspect_read_only`, `.commit`, `.load_record`, `.completed_digests`.
- Consumes: `FrozenRunContext`, canonical JSON and SHA256 helpers.

- [x] **Step 1: Write contract tests first**

Add literal fixtures proving:

```python
record = Exp426DatasetRecord(
    dataset="A_normal_1",
    origin="verified_train_abstention",
    status="abstained",
    candidate_coordinates=(),
    source_record_sha256=None,
    source_result_sha256=None,
    train_identity_sha256="a" * 64,
    eligibility_sha256="b" * 64,
    scientific_payload_sha256="c" * 64,
    reason="oof_identity_insufficient",
)
self.assertEqual(record.to_payload()["candidate_coordinates"], [])
self.assertEqual(record.origin, "verified_train_abstention")
```

Reject unknown keys/status/origin, bool indices, unsorted or duplicate candidates, wrong-dataset candidates, lane overlaps, non-lowercase hashes, imported records without source hashes, abstentions with candidates, and timing/runtime fields inside the scientific payload.

- [x] **Step 2: Run the contract tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp426_verified_baseline.tests.test_contracts_antigravity -v
```

Expected: import failure because `contracts_antigravity.py` does not exist.

- [x] **Step 3: Implement immutable contracts**

Use frozen dataclasses and recursively frozen mappings. `Exp426DatasetRecord.scientific_payload()` must contain exactly:

```python
{
    "schema": "exp426.dataset-record.v1",
    "dataset": dataset,
    "origin": origin,
    "status": status,
    "candidate_coordinates": [[name, index], ...],
    "source_record_sha256": source_record_sha256,
    "source_result_sha256": source_result_sha256,
    "train_identity_sha256": train_identity_sha256,
    "eligibility_sha256": eligibility_sha256,
    "reason": reason,
}
```

`scientific_payload_sha256` is recalculated from that exact payload and is never caller-trusted.

- [x] **Step 4: Write checkpoint RED tests**

Test no-clobber creation, exact provenance/context binding, one-writer lock, sorted universe, atomic intent → dataset → progress → intent cleanup, crash at each boundary, reopen/reconcile once, duplicate commit idempotence only for identical bytes, different-byte rejection, read-only inspect byte/inode/mtime immutability, corrupt/unknown/symlink file rejection, and completed/total conservation.

```python
store.commit(record)
self.assertEqual(store.completed_digests(), {"A_normal_1": record.sha256})
reopened = Exp426CheckpointStore.open(root, provenance, context)
self.assertEqual(reopened.load_record("A_normal_1"), record)
```

- [x] **Step 5: Implement checkpoint store**

The root inventory is exactly:

```text
checkpoint_provenance.json
run_context.json
progress.json
datasets/
intents/
```

Use same-directory temporary files, `fsync`, atomic rename and directory `fsync`. Only the parent process owns a writable store. A worker never receives a store/path/file descriptor.

- [x] **Step 6: Run Task 1 verification**

```bash
python3 -m unittest \
  experiments.exp426_verified_baseline.tests.test_contracts_antigravity \
  experiments.exp426_verified_baseline.tests.test_checkpoint_antigravity -v
python3 -m py_compile \
  experiments/exp426_verified_baseline/contracts_antigravity.py \
  experiments/exp426_verified_baseline/checkpoint_antigravity.py
git diff --check
```

Expected: every command exits 0.

- [x] **Step 7: Commit Task 1**

```bash
git add experiments/exp426_verified_baseline
git commit -m "feat(exp426): add frozen baseline checkpoint"
```

---

### Task 2: Read-Only Source Intake and Twelve-Core TRAIN Verification

**Files:**
- Create: `experiments/exp426_verified_baseline/source_antigravity.py`
- Create: `experiments/exp426_verified_baseline/parallel_antigravity.py`
- Create: `experiments/exp426_verified_baseline/tests/test_source_antigravity.py`
- Create: `experiments/exp426_verified_baseline/tests/test_parallel_antigravity.py`

**Interfaces:**
- Produces: `classify_exp424_record_bytes(dataset, raw) -> WorkerEvidence`.
- Produces: `verify_train_abstention(dataset, train, fold_contract_sha256) -> WorkerEvidence`.
- Produces: `run_ordered_workers(tasks, *, workers=12, prefetch=24) -> Iterator[WorkerEvidence]`.
- Consumes: `experiments.exp425_selective_recovery.predecessor.open_validated_exp424_source`, its descriptor-bound records, `experiments.exp417_causal_multihorizon_shadow.preprocessing.preprocess_length_groups`, `experiments.exp425_selective_recovery.eligibility.apply_group_eligibility`, and Task 1 records.

- [x] **Step 1: Write source-intake RED tests**

Create a minimal real Exp424-format source fixture and assert exact `3/2/1` classification, stable record/result hashes, source mutation rejection before and after worker processing, terminal reason parsing, nonterminal imported candidate preservation, and zero write calls below the source root.

```python
self.assertEqual(report.records, 3)
self.assertEqual(report.importable, 2)
self.assertEqual(report.terminal, 1)
self.assertEqual(report.terminal_reasons, ("oof_identity_insufficient",))
```

The production inspection test must require `1117/983/134` and these existing fixed source hashes:

- root `f7526bf638f4c1051a5970271bdbbccd753986f7d067ec1260817da884c1b757`;
- progress `5e9aa4a4b7240ac89c078bdc1dce4b41bda6b6dbc505dd9ad49782556da7bd84`;
- record set `d74a0def7788db89a1d3e32f46662abf07c44c7cbbce5e02612b510e2fdf4258`.

- [x] **Step 2: Write 12-worker RED tests**

Use real `ProcessPoolExecutor` workers. Assert exactly 12 initializer calls/process identities when at least 24 blocking tasks are queued, environment thread limits all equal `1`, prefetch never exceeds 24, output order is dataset-name order despite reverse completion, worker exceptions retain dataset name, and worker processes never receive paths, DB connections, labels or checkpoint objects.

```python
single = tuple(run_ordered_workers(tasks, workers=1, prefetch=2))
twelve = tuple(run_ordered_workers(tasks, workers=12, prefetch=24))
self.assertEqual([x.scientific_sha256 for x in single],
                 [x.scientific_sha256 for x in twelve])
```

- [x] **Step 3: Run Task 2 RED tests**

```bash
python3 -m unittest \
  experiments.exp426_verified_baseline.tests.test_source_antigravity \
  experiments.exp426_verified_baseline.tests.test_parallel_antigravity -v
```

Expected: imports fail because Task 2 modules do not exist.

- [x] **Step 4: Implement descriptor-bound source intake**

Parent process responsibilities:

1. open the validated Exp424 source authority;
2. read each exact record through its held datasets descriptor;
3. recheck inode and source record digest;
4. pass only `(dataset, immutable_raw_bytes, expected_digest)` to a worker;
5. revalidate the full source immediately before checkpoint publication.

Workers parse canonical bytes, reconstruct `DatasetResult`, and return either an imported compact record or a terminal dataset name/reason. They cannot open any path.

- [x] **Step 5: Implement fixed 12-core executor**

Use spawn-safe top-level functions:

```python
with ProcessPoolExecutor(
    max_workers=12,
    mp_context=multiprocessing.get_context("spawn"),
    initializer=_limit_worker_threads,
) as pool:
    # keep at most 24 futures and emit only next sorted ordinal
```

The production API rejects any worker count other than 12. A test-only private seam may run one worker for the readiness comparison; it is not exposed through CLI or config.

- [x] **Step 6: Implement TRAIN-only abstention verification**

Parent loads `RawTrainDataset` from the retained DB snapshot and submits the object plus fixed metadata. Worker runs preprocessing and `apply_group_eligibility`. It succeeds only when every usable group is `oof_identity_insufficient` or preprocessing-abstained and no group is eligible. It returns full count-bearing eligibility evidence and no candidates. TEST loader symbols are absent from the worker module.

- [x] **Step 7: Run Task 2 verification**

```bash
python3 -m unittest \
  experiments.exp426_verified_baseline.tests.test_source_antigravity \
  experiments.exp426_verified_baseline.tests.test_parallel_antigravity \
  experiments.exp425_selective_recovery.tests.test_predecessor \
  experiments.exp425_selective_recovery.tests.test_eligibility -v
python3 -m py_compile \
  experiments/exp426_verified_baseline/source_antigravity.py \
  experiments/exp426_verified_baseline/parallel_antigravity.py
git diff --check
```

Expected: all pass; production read-only smoke reports `1117/983/134`.

- [ ] **Step 8: Commit Task 2**

```bash
git add experiments/exp426_verified_baseline
git commit -m "feat(exp426): verify baseline on twelve CPU cores"
```

---

### Task 3: Resumable Orchestration and Self-Written Progress Logs

**Files:**
- Create: `experiments/exp426_verified_baseline/runtime_antigravity.py`
- Create: `experiments/exp426_verified_baseline/tests/test_runtime_antigravity.py`

**Interfaces:**
- Produces: `run_exp426_new`, `run_exp426_resume`, `inspect_exp426`.
- Produces: atomic `progress.json`, append-only `events.jsonl`, `parent.log`, and `run.pid`.
- Consumes: Tasks 1–2 and `open_database_snapshot`.

- [ ] **Step 1: Write mixed-run RED test**

Use a 15-dataset fixture: 12 imported and 3 terminal. Assert imported records never load DB/TRAIN, terminal records load TRAIN once, no TEST method is reachable, 12 workers are created, parent alone commits records, and final counts are exactly:

```python
self.assertEqual(result.counts.completed, 15)
self.assertEqual(result.counts.imported, 12)
self.assertEqual(result.counts.verified_train_abstentions, 3)
self.assertEqual(result.counts.failed, 0)
```

- [ ] **Step 2: Write progress/resume RED tests**

Test exact state transitions:

```text
initializing -> validating_source -> importing -> verifying_train ->
finalizing -> complete_review_only
```

For every dataset, require current dataset, TRAIN/TEST sizes and lengths, backend=`cpu`, workers=`12`, completed/total, import/recheck/abstention/error counts, throughput, elapsed, ETA, checkpoint path and latest error. Terminal states clear `current_dataset`.

Inject crashes after worker result, record commit, event append and progress replace. Resume must process each missing record once, never duplicate a terminal event, and never recompute a completed imported/abstention record.

- [ ] **Step 3: Run Task 3 RED tests**

```bash
python3 -m unittest experiments.exp426_verified_baseline.tests.test_runtime_antigravity -v
```

Expected: import failure because `runtime_antigravity.py` does not exist.

- [ ] **Step 4: Implement parent-owned orchestration**

The parent opens one DB snapshot, freezes the context, validates source, dispatches source records to 12 workers, commits 983 imports in sorted order, then loads and dispatches the 134 TRAIN datasets. It never calls a TEST-value or label loader. It verifies the DB boundary before and after each TRAIN load.

Keep two bounded queues of 24 tasks. Worker results may arrive in any order, but a small in-memory reorder map holds them until the next sorted dataset can be committed.

- [ ] **Step 5: Implement durable progress**

Every progress write uses temp + `fsync` + replace. Every JSONL/text append is flushed and `fsync`ed. A 30-second heartbeat thread writes without Codex monitoring and stops before terminal publication. `run.pid` is evidence, not a live lease after exit; live-process rejection uses PID plus run identity.

- [ ] **Step 6: Run Task 3 verification**

```bash
python3 -m unittest \
  experiments.exp426_verified_baseline.tests.test_runtime_antigravity \
  experiments.exp426_verified_baseline.tests.test_checkpoint_antigravity \
  experiments.exp425_selective_recovery.tests.test_database -v
python3 -m py_compile experiments/exp426_verified_baseline/runtime_antigravity.py
git diff --check
```

Expected: all pass and no label-bearing field exists in runtime progress/events.

- [ ] **Step 7: Commit Task 3**

```bash
git add experiments/exp426_verified_baseline
git commit -m "feat(exp426): add resumable baseline runtime"
```

---

### Task 4: Results, Readiness, CLI, and Frozen Documents

**Files:**
- Create: `experiments/exp426_verified_baseline/reporting_antigravity.py`
- Create: `experiments/exp426_verified_baseline/cli_antigravity.py`
- Create: `experiments/exp426_verified_baseline/tests/test_reporting_antigravity.py`
- Create: `experiments/exp426_verified_baseline/tests/test_cli_antigravity.py`
- Create: `scratch/run_exp426_verified_baseline_antigravity.py`
- Create: `docs/EXP426_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP426_RUNBOOK_antigravity_KO.md`

**Interfaces:**
- Produces commands: `new`, `resume`, `inspect`, `readiness`.
- Produces canonical prediction JSONL, dataset summary, import ledger, TRAIN abstention ledger, backend decision, manifest, status and artifact SHA256 map.
- Consumes: complete Task 1–3 checkpoint and runtime evidence.

- [ ] **Step 1: Write reporting RED tests**

Require exact population conservation and governance:

```python
self.assertEqual(manifest["completed_datasets"], 1117)
self.assertEqual(manifest["imported_datasets"], 983)
self.assertEqual(manifest["verified_train_abstentions"], 134)
self.assertEqual(manifest["failed_datasets"], 0)
self.assertTrue(manifest["review_only"])
self.assertFalse(manifest["operational_alert_eligible"])
self.assertFalse(manifest["contains_test_labels"])
self.assertFalse(manifest["test_labels_opened"])
self.assertEqual(manifest["workers"], 12)
```

Prediction rows must be canonical, sorted and unique; abstentions produce none. Reparse every output and independently recalculate checkpoint record hashes, prediction SHA256 and manifest artifact hashes. Recursively reject label/target/ground-truth/confusion/precision/recall/F1/anomaly-location/family-TEST keys except the two literal false governance flags in manifest/status only.

- [ ] **Step 2: Write CLI/readiness RED tests**

Assert:

- `new` requires both frozen roots absent;
- `resume` requires matching incomplete roots;
- `inspect` is byte/inode/mtime read-only;
- `readiness` writes only its explicit file and temporary DB owner link, then removes the link;
- CLI has no worker option and reports fixed workers=12;
- arbitrary roots are rejected before creation;
- label/test-evaluation imports are forbidden;
- complete `resume` returns without DB/source/worker work.

- [ ] **Step 3: Run Task 4 RED tests**

```bash
python3 -m unittest \
  experiments.exp426_verified_baseline.tests.test_reporting_antigravity \
  experiments.exp426_verified_baseline.tests.test_cli_antigravity -v
```

Expected: imports fail because Task 4 modules do not exist.

- [ ] **Step 4: Implement reporting and CLI**

`readiness` checks clean Git, exact DB/lane/source hashes, source `1117/983/134`, output/checkpoint absence, no competing Exp426 process, 1-vs-12 scientific SHA parity, 12-worker process creation, DB one-hash lifecycle, focused test evidence and artifact round trip. It records 1-worker and 12-worker timing but never changes the frozen worker count.

`new` and `resume` invoke only the exact frozen paths. `inspect` uses read-only checkpoint/log readers. Exit codes are 0 complete/ready, 1 validated incomplete/failed, 2 command/contract misuse.

- [ ] **Step 5: Write preregistration and runbook**

Copy the approved design exactly, including source hashes, `983/134`, fixed 12 workers, thread limits, queue 24, TEST firewall, frozen roots, no-clobber rule, progress fields, exact commands, prediction freeze and separate post-hoc evaluation boundary.

- [ ] **Step 6: Run Task 4 verification**

```bash
python3 -m unittest discover -s experiments/exp426_verified_baseline/tests -p 'test_*antigravity.py'
python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
python3 -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs \
  experiments.exp417_causal_multihorizon_shadow.tests.test_contracts \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner \
  experiments.exp417_causal_multihorizon_shadow.tests.test_reporting
python3 -m py_compile experiments/exp426_verified_baseline/*antigravity.py \
  scratch/run_exp426_verified_baseline_antigravity.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 7: Commit Task 4**

```bash
git add experiments/exp426_verified_baseline \
  scratch/run_exp426_verified_baseline_antigravity.py \
  docs/EXP426_PREREGISTRATION_antigravity_KO.md \
  docs/EXP426_RUNBOOK_antigravity_KO.md
git commit -m "docs(exp426): freeze verified baseline run"
```

---

### Task 5: Real Readiness, Full Run, and Prediction Freeze

**Files:**
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `FAILURE_LEDGER.json` only if readiness/run fails
- Modify: `CHAMPION_LEDGER.json` only after a later preregistered evaluation passes its improvement rule

**Interfaces:**
- Consumes the exact Task 4 CLI and frozen external roots.
- Produces a completed Exp426 checkpoint/output root and frozen prediction SHA256, or immutable failure evidence.

- [ ] **Step 1: Run full verification again from clean Git**

```bash
git status --short --branch
python3 -m unittest discover -s experiments/exp426_verified_baseline/tests -p 'test_*antigravity.py'
python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
python3 -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs \
  experiments.exp417_causal_multihorizon_shadow.tests.test_contracts \
  experiments.exp417_causal_multihorizon_shadow.tests.test_runner \
  experiments.exp417_causal_multihorizon_shadow.tests.test_reporting
```

Expected: clean branch and all tests pass.

- [ ] **Step 2: Obtain a fresh Sol/High final review**

Require exact verdict `ship`, `fix-first`, or `rethink`. Any fix invalidates the verdict and requires new verification and a new reviewer. Reviewer remains behaviorally read-only; parent captures before/after HEAD, status and external artifact hashes.

- [ ] **Step 3: Run exact readiness once**

```bash
python3 scratch/run_exp426_verified_baseline_antigravity.py readiness \
  --readiness-output /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_readiness_20260805.json
```

Expected: state `ready`, source `1117/983/134`, workers `12`, 1-vs-12 scientific parity true, TEST values/labels 0, output/checkpoint roots absent.

- [ ] **Step 4: Start the exact 1,117-dataset label-free run**

```bash
python3 scratch/run_exp426_verified_baseline_antigravity.py new \
  --output-root /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805 \
  --checkpoint-root /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint
```

The program writes its own progress and checkpoint. Do not repeatedly poll it with Codex. If interrupted, use only the exact `resume` command recorded in `RESEARCH_STATE.json`.

- [ ] **Step 5: Inspect and freeze outputs**

```bash
python3 scratch/run_exp426_verified_baseline_antigravity.py inspect \
  --output-root /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805 \
  --checkpoint-root /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint
```

Require completed=1117, imported=983, verified_train_abstentions=134, failed=0, current dataset null, TEST values/labels 0, prediction and manifest SHA256 present, checkpoint/output verification clean, and no active PID.

- [ ] **Step 6: Update control files and stop before TEST evaluation**

Record exact commit, prediction SHA256, manifest SHA256, output/checkpoint/log paths and next evaluator command in `RESEARCH_STATE.json` and `EXPERIMENT_CATALOG.json`. Do not open TEST labels in this task. Commit the control-file update and report the frozen result.

---

## Plan Self-Review Checklist

- Every new repository filename includes `antigravity`; no new `__init__.py` exists.
- Every production worker count is exactly 12; no CLI/config selection exists.
- Source imported records remain truthfully marked Exp424/MPS-origin.
- No task turns a TRAIN-insufficient dataset into model scoring.
- No prediction task opens TEST values or labels for the 134 datasets.
- Scientific record hashes exclude runtime timing and worker count.
- Parent alone owns DB, checkpoint, logs and output writes.
- Readiness records speed but cannot adapt worker count or science.
- Prediction freezes before any TEST-label evaluation.
