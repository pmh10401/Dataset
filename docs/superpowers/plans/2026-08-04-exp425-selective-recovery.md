# Exp425 Selective Recovery and Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover Exp424 as a new review-only Exp425 run by importing strictly compatible nonterminal records, converting deterministic group-level OOF identity insufficiency into abstention, and removing proven checkpoint, DB-hash, resume, and progress bottlenecks.

**Architecture:** Build an isolated `experiments.exp425_selective_recovery` package around the frozen Exp417/Exp424 model and scoring code. Exp425 owns group eligibility, compact checkpoint envelopes, predecessor provenance, one-process DB snapshotting, explicit execution modes, and reporting; eligible numerical work continues through the exact Exp424 MPS contracts. Exp424 files are read-only inputs and every imported record retains a verifiable provenance chain.

**Tech Stack:** Python 3.14, NumPy 2.4.6, PyTorch 2.13.0, Apple MPS, SQLite, canonical JSON/SHA256, `unittest`, POSIX descriptor and `fcntl` controls.

## Global Constraints

- Prediction must not read TEST labels, anomaly locations, or historical family TEST performance.
- `mean_combined_f1` is post-hoc human-assisted evidence only.
- Priority-review and Shadow candidates cannot be promoted to automatic alerts.
- Do not use tail replacement or post-TEST alpha/threshold/source/budget reselection.
- Preserve Exp424 artifacts and checkpoint files byte-for-byte as failed review-only evidence.
- Freeze Exp424 model, seed, fold, MPS, batch-size 32, query-chunk-size 256, float precision, percentile, score-cache, threshold, alpha, candidate, and routing contracts.
- Only group eligibility, predecessor import, compact checkpointing, DB snapshotting, execution state, and observability may change in Exp425.
- A new process computes the full 1.6 GB DB SHA256 exactly once; pathname/stat caches are never trusted across processes.
- Run one MPS-heavy process at a time. MLX, altered MPS batching, compilation, and dataset parallelism are deferred to later experiments and must meet `AGENTS.md` parity and at-least-1.1x speed gates.
- Exp425 stays `review_only=true`, `official_prediction=false`, `contains_test_labels=false`, and `test_labels_opened=false`.
- Do not start the 1,117-dataset run without a readiness report and explicit user approval.

## File Structure

- `experiments/exp425_selective_recovery/contracts.py`: immutable Exp425 eligibility, provenance, origin, execution-state, and record-envelope types.
- `experiments/exp425_selective_recovery/eligibility.py`: exact TRAIN-identity fold diagnostics and group filtering.
- `experiments/exp425_selective_recovery/checkpoints.py`: compact descriptor-safe checkpoint store with one shared context.
- `experiments/exp425_selective_recovery/predecessor.py`: read-only Exp424 validation, compatibility attestation, import, and effective-result adaptation.
- `experiments/exp425_selective_recovery/database.py`: exactly-once-per-process DB hash and retained immutable SQLite snapshot.
- `experiments/exp425_selective_recovery/runtime.py`: frozen MPS runtime adapter, native execution traces, selective population orchestration, and atomic progress.
- `experiments/exp425_selective_recovery/reporting.py`: review-only prediction, manifest, status, import ledger, and verification.
- `experiments/exp425_selective_recovery/__init__.py`: narrow public API.
- `experiments/exp425_selective_recovery/tests/`: focused contract, eligibility, checkpoint, import, DB, runtime, CLI, reporting, and benchmark tests.
- `scratch/run_exp425_selective_recovery.py`: `new`, `resume`, and `inspect` CLI.
- `docs/EXP425_PREREGISTRATION_KO.md`: frozen research and governance contract.
- `docs/EXP425_RUNBOOK_KO.md`: safe launch, status, resume, verification, and post-hoc boundary.

---

### Task 1: Deterministic Group Eligibility and Abstention

**Files:**
- Create: `experiments/exp425_selective_recovery/__init__.py`
- Create: `experiments/exp425_selective_recovery/contracts.py`
- Create: `experiments/exp425_selective_recovery/eligibility.py`
- Create: `experiments/exp425_selective_recovery/tests/__init__.py`
- Create: `experiments/exp425_selective_recovery/tests/test_contracts.py`
- Create: `experiments/exp425_selective_recovery/tests/test_eligibility.py`

**Interfaces:**
- Produces: `FoldIdentityCounts`, `GroupEligibilityEvidence`, and `EligibilityAssessment` immutable dataclasses.
- Produces: `evaluate_group_eligibility(dataset: str, group: ProcessedGroup, *, fold_contract_sha256: str) -> GroupEligibilityEvidence`.
- Produces: `apply_group_eligibility(dataset: str, groups: Sequence[ProcessedGroup], *, fold_contract_sha256: str) -> EligibilityAssessment`.
- Consumes: Exp417 `canonical_waveform_sha256`, `ProcessedGroup`, five-fold greedy identity allocation, and hash-modulo-10 early-stop rule.

- [ ] **Step 1: Write failing contract tests**

Add tests that reject unsorted fold records, negative counts, a non-SHA256 fold digest, an eligible group with any early-stop side below five, and an insufficient group whose counts all pass. Assert canonical immutability.

```python
def test_insufficient_evidence_requires_a_failing_split(self):
    evidence = self.make_evidence(
        status="oof_identity_insufficient",
        folds=(FoldIdentityCounts(10, 5, 9, 1),) * 5,
    )
    self.assertEqual(evidence.reason, "oof_identity_insufficient")

def test_eligible_evidence_rejects_small_early_validation(self):
    with self.assertRaisesRegex(ValueError, "eligible split"):
        self.make_evidence(
            status="eligible",
            folds=(FoldIdentityCounts(10, 5, 9, 1),) * 5,
        )
```

- [ ] **Step 2: Run contract tests and confirm the expected import failure**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_contracts -v`  
Expected: `ModuleNotFoundError: No module named 'experiments.exp425_selective_recovery'`.

- [ ] **Step 3: Implement immutable eligibility contracts**

Use exact status values `eligible` and `oof_identity_insufficient`. `reason` is `None` for eligible evidence and the stable status string for insufficient evidence.

```python
@dataclass(frozen=True)
class FoldIdentityCounts:
    oof_fit: int
    oof_validation: int
    early_stop_fit: int
    early_stop_validation: int

@dataclass(frozen=True)
class GroupEligibilityEvidence:
    dataset: str
    length: int
    processed_status: str
    train_identity_sha256: str
    fold_contract_sha256: str
    unique_identity_count: int
    folds: tuple[FoldIdentityCounts, ...]
    final_fit: int | None
    final_validation: int | None
    status: str
    reason: str | None

@dataclass(frozen=True)
class EligibilityAssessment:
    groups: tuple[ProcessedGroup, ...]
    evidence: Mapping[int, GroupEligibilityEvidence]
```

Validate five folds, sorted unique group lengths, exact 64-character lowercase digests, nonnegative counts, and status/count consistency in `__post_init__`. Require final-fit/final-validation counts only when `processed_status == "candidate_eligible"`; require both to be `None` for `diagnostic_only`. Already preprocessing-abstained groups retain their original status and use a distinct `preprocessing_abstained` evidence status without OOF/final-fit applicability.

- [ ] **Step 4: Run contract tests and confirm they pass**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_contracts -v`  
Expected: all tests pass.

- [ ] **Step 5: Write failing eligibility tests**

Cover these fixtures:

1. an eligible group whose Exp425 fold index tuples exactly equal `identity_grouped_folds`;
2. a group with fewer than five early-validation identities becomes `oof_identity_insufficient` without calling Exp417 model fitting;
3. mixed eligible and insufficient exact-length groups retain the eligible group byte-for-byte and replace only the insufficient group with `status="insufficient_train_abstain"` and no channels;
4. ordering is stable under input permutation;
5. duplicate waveforms are one identity, matching the semiconductor repeated-normal interpretation;
6. `diagnostic_only` groups require all five OOF early-stop splits but do not require or record a final-fit split;
7. `candidate_eligible` groups require both all OOF splits and the one final-fit early-stop split.

```python
assessment = apply_group_eligibility(
    "Synthetic_normal_1",
    (eligible_group, insufficient_group),
    fold_contract_sha256="a" * 64,
)
self.assertEqual(
    tuple(assessment.evidence),
    (eligible_group.group.length, insufficient_group.group.length),
)
self.assertEqual(assessment.groups[0], eligible_group)
self.assertEqual(assessment.groups[1].status, "insufficient_train_abstain")
self.assertEqual(dict(assessment.groups[1].channels), {})
```

- [ ] **Step 6: Implement exact preflight fold diagnostics and filtering**

Build waveform identity groups with `canonical_waveform_sha256`. Assign sorted identities greedily to the currently smallest fold, breaking ties by fold index. For each validation fold, derive fit identities, select early validation with `int(identity, 16) % 10 == 0`, and derive early fit by complement. For `candidate_eligible` only, apply the same hash split once across all identities for final-fit early stopping. Mark a usable group eligible only when all five OOF folds pass and, when applicable, the final-fit split also passes. Compute `train_identity_sha256` from sorted identity plus sorted member-index records and include the preregistered fold-contract digest.

```python
status = (
    "eligible"
    if all(
        fold.early_stop_fit >= 5 and fold.early_stop_validation >= 5
        for fold in fold_counts
    )
    and (
        group.status == "diagnostic_only"
        or (
            final_fit is not None
            and final_validation is not None
            and final_fit >= 5
            and final_validation >= 5
        )
    )
    else "oof_identity_insufficient"
)
filtered = (
    group
    if status == "eligible"
    else replace(group, status="insufficient_train_abstain", channels=None)
)
```

Do not catch or classify the Exp417 exception text. For eligible groups, call `identity_grouped_folds` in the parity test and require exact index equality.

- [ ] **Step 7: Run eligibility and frozen preprocessing regressions**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_eligibility experiments.exp417_causal_multihorizon_shadow.tests.test_preprocessing -v`  
Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add experiments/exp425_selective_recovery
git commit -m "feat(exp425): classify OOF identity insufficiency"
```

---

### Task 2: Compact Provenance-Bound Checkpoint Store

**Files:**
- Modify: `experiments/exp425_selective_recovery/contracts.py`
- Create: `experiments/exp425_selective_recovery/checkpoints.py`
- Create: `experiments/exp425_selective_recovery/tests/test_checkpoints.py`

**Interfaces:**
- Produces: `Exp425RunProvenance`, `SourceProvenGroupEvidence`, `NativeRecordEnvelope`, `ImportedRecordEnvelope`, and `Exp425Record`.
- Produces: `Exp425CheckpointStore.create(root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext) -> Exp425CheckpointStore`.
- Produces: `Exp425CheckpointStore.open(root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext) -> Exp425CheckpointStore`.
- Produces: `write_native(record: NativeRecordEnvelope) -> str`, `write_imported(record: ImportedRecordEnvelope) -> str`, `completed_record_digests() -> Mapping[str, str]`, `load_record(dataset: str) -> Exp425Record`, and `iter_records() -> Iterator[Exp425Record]`.
- Consumes later: Task 3 predecessor envelopes and Task 5 native execution records.

`Exp425RunProvenance` contains `run_identity`, `git_commit`, `database_sha256`, four lane artifact digests, frozen config/fold/model/backend digests, model and perturbation seeds, runtime versions, expected datasets, predecessor compatibility digest, and shared run-context digest. `NativeRecordEnvelope` contains dataset, native effective result, native group eligibility evidence, model/calibration/backend audit hashes, elapsed time, reason, and `origin="exp425_native"`. `ImportedRecordEnvelope` contains dataset, adapted effective result, source-proven group evidence, source root/progress/record/result digests, compatibility attestation, elapsed time, reason, and `origin="exp424_import"`. `Exp425Record` is the validated union of these two envelopes.

- [ ] **Step 1: Write failing root/context and compact-record tests**

Assert that:

- root creation writes exactly `checkpoint_provenance.json`, `run_context.json`, `progress.json`, `datasets/`, and `transactions/`;
- `run_context.json` canonical SHA256 equals every dataset record's `run_context_sha256`;
- a dataset record never contains keys `context`, `lanes`, `datasets`, or `test_identity_manifest` below the envelope root;
- round-trip reconstruction injects the exact shared context;
- `new` creation rejects an existing path and open rejects an absent path;
- unknown files, symlinks, digest mismatch, duplicate dataset, path replacement, and concurrent writers fail closed;
- simulated process death after intent fsync, record fsync, progress fsync, and intent removal has one deterministic recovery result at each boundary.

```python
raw = (root / "datasets" / record_filename("Tiny_normal_1")).read_bytes()
self.assertLess(len(raw), len(canonical_json_bytes(source_exp424_record)) // 5)
payload = json.loads(raw)
self.assertEqual(payload["run_context_sha256"], store.run_context_sha256)
self.assertNotIn(b'"test_identity_manifest"', raw)
```

- [ ] **Step 2: Run checkpoint tests and confirm the expected import failure**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_checkpoints -v`  
Expected: `ModuleNotFoundError` for `experiments.exp425_selective_recovery.checkpoints`.

- [ ] **Step 3: Implement schemas and canonical payload validation**

Use schema `exp425.checkpoint.v1`. Root provenance includes the Exp425 Git commit, DB and lane hashes, frozen Exp424 numerical-contract digests, runtime versions, expected datasets, predecessor compatibility digest, and run identity. Store the full `FrozenRunContext` once in `run_context.json` and its digest in root provenance.

```python
record_body = {
    "schema_version": "exp425.dataset.v1",
    "dataset": record.dataset,
    "origin": record.origin,
    "run_context_sha256": self.run_context_sha256,
    "evidence": record.evidence_payload(),
    "elapsed_seconds": record.elapsed_seconds,
}
record_body["payload_sha256"] = sha256(
    canonical_json_bytes(record_body)
).hexdigest()
```

Reject duplicate JSON keys, extra keys, noncanonical bytes, nonfinite numbers, unknown statuses, and any embedded TEST-label or evaluation key.

- [ ] **Step 4: Implement descriptor-safe creation, writes, progress, and lazy reads**

Use `O_NOFOLLOW`, directory descriptors, `fcntl.flock`, exclusive temp creation, `fsync`, and atomic no-replace publication. `progress.json` contains sorted `record_digests`, completed/expected counts, current dataset, and a payload digest. Validate the root and progress once per `open`; `load_record` validates and decodes one requested dataset only. Never deserialize all completed records in the constructor.

```python
def iter_records(self) -> Iterator[Exp425Record]:
    for dataset in self.expected_datasets:
        if dataset in self._record_digests:
            yield self.load_record(dataset)
```

Every write uses a durable intent file in `transactions/`: publish and fsync the canonical intent containing dataset and intended record digest; publish and fsync the dataset record; replace and fsync progress; remove and fsync the intent. Open/resume reconciles under an exclusive root lock: intent without record is removed; matching intent plus durable record absent from progress completes progress; matching intent plus record already in progress removes the intent. A record absent from progress and without a valid intent is corruption and must not be adopted. Quarantine a corrupted store beside the original only when the exact root inode remains owned and the quarantine destination does not exist.

- [ ] **Step 5: Run checkpoint tests and the Exp417 checkpoint regression suite**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_checkpoints experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints -v`  
Expected: all tests pass.

- [ ] **Step 6: Add and run compact-size/resume benchmark tests**

Generate 25 synthetic records with a representative shared context. Compare old-style repeated-context bytes against Exp425 root plus record bytes and compare two validated open/iteration cycles after one warm-up. Require at least 80% byte reduction and at least 2x validation speed; use the same records and local temporary filesystem.

```python
self.assertLess(exp425_bytes, repeated_context_bytes * 0.20)
self.assertLess(exp425_resume_seconds, old_resume_seconds / 2.0)
self.assertEqual(exp425_candidate_coordinates, old_candidate_coordinates)
```

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_checkpoints.CheckpointBenchmarkTests -v`  
Expected: all benchmark assertions pass on the development Mac.

- [ ] **Step 7: Commit Task 2**

```bash
git add experiments/exp425_selective_recovery/contracts.py \
  experiments/exp425_selective_recovery/checkpoints.py \
  experiments/exp425_selective_recovery/tests/test_checkpoints.py
git commit -m "perf(exp425): compact provenance-bound checkpoints"
```

---

### Task 3: Strict Exp424 Predecessor Import

**Files:**
- Create: `experiments/exp425_selective_recovery/predecessor.py`
- Create: `experiments/exp425_selective_recovery/tests/test_predecessor.py`
- Modify: `experiments/exp425_selective_recovery/__init__.py`

**Interfaces:**
- Produces: `Exp424CompatibilityContract`, `Exp424ValidatedSnapshot`, and `Exp424ImportReport`.
- Produces context manager `open_validated_exp424_source(root: Path, contract: Exp424CompatibilityContract) -> Iterator[Exp424ValidatedSnapshot]`.
- Produces: `import_exp424_nonterminal(source: Exp424ValidatedSnapshot, target: Exp425CheckpointStore, target_context: FrozenRunContext) -> Exp424ImportReport`.
- Produces: `adapt_imported_result(source: DatasetResult, target_context: FrozenRunContext) -> DatasetResult`.
- Consumes: Task 2 `write_imported`, the exact Exp424 checkpoint schema version 3, and frozen compatibility digests recorded in the Exp424 root.

`Exp424CompatibilityContract` includes the exact source commit, entrypoint SHA256/blob, DB and four lane digests, config/fold/model/backend digests, runtime versions, seeds, MPS device and device guard, batch/chunk values, score formula, score-cache component tuple and contract digest, percentile method, score dtype, canonical thresholds payload, alpha tuple, candidate-formula digest, routing-contract digest, device-transfer policy, and synchronization-policy digest. `Exp424ValidatedSnapshot` retains open descriptor-bound root/datasets handles, the held shared lock, root/progress identities and digests, per-record identities/digests, plus a lazy sorted iterator of validated source records. `Exp424ImportReport` contains sorted imported, rejected-terminal, and rejected-incompatible dataset tuples and the source/attestation digests.

- [ ] **Step 1: Write failing compatibility and mutation tests**

Create a minimal valid Exp424-format source fixture and assert acceptance only when all of these match: exact source commit `4a398374d880f859f285a271ad63de4a5e55b028`, entrypoint SHA256 `98c909f6100dd2814e3fe102c3f95054a3d6bb42f85657d19b17780ae4934fe1`, entrypoint Git blob `da5374b0bf38803aa506dfda664d76395d7f28f9`, root/progress digest, record digest, universe, DB/lane/config/fold/model/seed/runtime/backend/MPS device and device guard/batch/chunk/score-cache components and digest/device-transfer/synchronization/percentile/precision/threshold/alpha/candidate/routing contracts, candidate coordinates, and nonterminal evidence. Add an independent mismatch rejection test for every explicit compatibility field, a terminal result, a symlink, an unknown file, and source mutation between validation, staging, final revalidation, and promotion.

```python
with open_validated_exp424_source(source_root, contract) as snapshot:
    report = import_exp424_nonterminal(snapshot, target, target_context)
self.assertEqual(report.imported, ("Eligible_normal_1",))
self.assertEqual(report.rejected_terminal, ("Small_normal_1",))
self.assertEqual(target.load_record("Eligible_normal_1").origin, "exp424_import")
```

- [ ] **Step 2: Run predecessor tests and confirm the expected import failure**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_predecessor -v`  
Expected: `ModuleNotFoundError` for `experiments.exp425_selective_recovery.predecessor`.

- [ ] **Step 3: Implement the compatibility contract and attestation digest**

The compatibility payload explicitly names every frozen field; it does not rely on “same Git commit” because Exp425 has a new commit. Bind instead to exact source-code contract digests and Exp424 actual runtime/MPS settings.

```python
compatibility_payload = {
    "schema": "exp425.exp424-compatibility.v1",
    "source_experiment": "Exp424-MPS-SCORE-CACHE",
    "source_git_commit": "4a398374d880f859f285a271ad63de4a5e55b028",
    "entrypoint_sha256": "98c909f6100dd2814e3fe102c3f95054a3d6bb42f85657d19b17780ae4934fe1",
    "entrypoint_git_blob": "da5374b0bf38803aa506dfda664d76395d7f28f9",
    "database_sha256": contract.database_sha256,
    "lane_hashes": dict(contract.lane_hashes),
    "config_sha256": contract.config_sha256,
    "fold_contract_sha256": contract.fold_contract_sha256,
    "model_contract_sha256": contract.model_contract_sha256,
    "model_seeds": list(contract.model_seeds),
    "perturbation_seeds": list(contract.perturbation_seeds),
    "backend_contract_sha256": contract.backend_contract_sha256,
    "device": "mps",
    "batch_size": 32,
    "query_chunk_size": 256,
    "score_formula": "Exp417.score_horizon.v1",
    "score_cache_components": [
        "oof_score_calibration_cache",
        "reusable_batch_residual_predictor",
    ],
    "score_cache_contract_sha256": contract.score_cache_contract_sha256,
    "percentile_method": "linear",
    "score_dtype": "float64",
    "mps_device_guard": "mps_and_mps0_equivalent",
    "device_transfer_policy": "numpy_copy_to_mps_then_detach_cpu_numpy",
    "synchronization_contract_sha256": contract.synchronization_contract_sha256,
    "thresholds": contract.thresholds_payload,
    "alpha_values": contract.alpha_values,
    "candidate_formula_sha256": contract.candidate_formula_sha256,
    "routing_contract_sha256": contract.routing_contract_sha256,
}
```

- [ ] **Step 4: Implement one-pass descriptor-bound source validation**

Expose validation as a context manager that retains the shared lock plus open root/datasets descriptors through import. Capture identities, validate exact inventories, read root and progress once, then stream records in sorted dataset order. For each file, verify canonical JSON and progress digest before reconstructing the `DatasetResult`. Retain per-record descriptor identity, stable metadata, and digest; do not construct a 1,117-result mapping.

Write converted compact envelopes only into an uncommitted target import-staging directory. Before publishing any staged envelope into the target dataset/progress inventory, re-read and verify source root, progress, inventories, and every source record digest through the retained descriptors. Promote only after this final revalidation. A crashed import batch remains staged and must repeat full source revalidation before promotion; mutation or cleanup failure aborts and quarantines only the owned target staging batch.

- [ ] **Step 5: Implement imported-result adaptation and envelopes**

Require identical DB digest, lane snapshot, dataset universe, TEST identity manifest, test size, frozen TRAIN contract, decisions, candidates, and status. Rebuild only the context-bearing `DatasetDecisionEvidence` and `DatasetResult` with `target_context`; preserve the source result digest in an `ImportedRecordEnvelope` with `origin="exp424_import"` and a compatibility-attestation digest. For each exact-length source group, create immutable `SourceProvenGroupEvidence` containing its source qualification status, whether OOF/final model and calibration evidence exists, source record/result digests, and compatibility attestation. Set split counts to unavailable rather than deriving or inventing them; the successfully validated nonterminal source execution is the evidence.

```python
adapted_evidence = replace(source.decision_evidence, context=target_context)
adapted = replace(
    source,
    decision_evidence=adapted_evidence,
    lanes=target_context.lanes,
    context=target_context,
)
if coordinates(adapted) != coordinates(source):
    raise ValueError("Exp424 imported candidate coordinates changed")
```

Terminal Exp424 records are counted as `rejected_terminal` and never adapted. A canonically valid nonterminal record that fails only target-compatibility gates is counted as `rejected_incompatible` and enters the native queue with its reason. Structural schema corruption, digest/inventory disagreement, or source mutation fails the whole import rather than silently recomputing from untrusted evidence.

Imported datasets perform no DB or model work. The source-proven group evidence is derived exclusively from the fully validated checkpoint record and is visibly distinct from native count-bearing eligibility evidence.

- [ ] **Step 6: Run predecessor, checkpoint, and reporting reconstruction tests**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_predecessor experiments.exp425_selective_recovery.tests.test_checkpoints experiments.exp417_causal_multihorizon_shadow.tests.test_reporting -v`  
Expected: all tests pass.

- [ ] **Step 7: Run a read-only real Exp424 inventory smoke**

Run a new inspection command against `/Users/mac/Documents/Dataset/exp424_mps_score_cache_20260802-153831.checkpoint` that validates root/progress/record accounting without writing a target. Require exactly 1,117 records, 983 importable nonterminal records, 134 rejected terminal records, one terminal reason code derived from structured source evidence, and zero source mutations.

Expected summary fields:

```json
{"records":1117,"importable":983,"terminal":134,"source_mutated":false}
```

- [ ] **Step 8: Commit Task 3**

```bash
git add experiments/exp425_selective_recovery/predecessor.py \
  experiments/exp425_selective_recovery/tests/test_predecessor.py \
  experiments/exp425_selective_recovery/__init__.py
git commit -m "feat(exp425): import compatible Exp424 evidence"
```

---

### Task 4: One-Hash Database Snapshot

**Files:**
- Create: `experiments/exp425_selective_recovery/database.py`
- Create: `experiments/exp425_selective_recovery/tests/test_database.py`
- Modify: `experiments/exp425_selective_recovery/__init__.py`

**Interfaces:**
- Produces: `DatabaseSnapshotAudit` and `Exp425DatabaseSnapshot`.
- Produces context manager `open_database_snapshot(database: Path, *, expected_sha256: str) -> Iterator[Exp425DatabaseSnapshot]`.
- Produces methods `freeze_context(lanes: FrozenLaneSnapshot, *, verified_code_commit: str) -> FrozenRunContext`, `load_train(dataset: str, context: FrozenRunContext) -> RawTrainDataset`, `load_test(dataset: str, contract: FrozenTrainContract, context: FrozenRunContext) -> RawTestDataset`, and `verify_boundary(name: str) -> None`.
- Consumes: Exp417 label-free SQL contract and `load_raw_train_values_from_connection` / `load_raw_test_values_from_connection` decoders.

- [ ] **Step 1: Write failing one-hash and mutation tests**

Patch the descriptor hashing function with a counter. Freeze context, load two TRAIN datasets and two TEST datasets, verify three boundaries, and close. Assert hash count exactly one. Add fail-closed tests for replacement, truncation, byte mutation, mtime/ctime change, symlink input, pathname disappearance, pre-existing or newly created `-wal`, `-shm`, and rollback-journal sidecars, and a journal-enabled writer. Add a macOS test proving the owned hard-link creation does not make the snapshot reject its own accepted metadata.

```python
with open_database_snapshot(db, expected_sha256=expected) as snapshot:
    context = snapshot.freeze_context(lanes, verified_code_commit="a" * 40)
    snapshot.load_train(context.datasets[0], context)
    snapshot.load_train(context.datasets[1], context)
    snapshot.verify_boundary("after_train")
self.assertEqual(snapshot.audit.full_hash_count, 1)
```

- [ ] **Step 2: Run DB tests and confirm the expected import failure**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v`  
Expected: `ModuleNotFoundError` for `experiments.exp425_selective_recovery.database`.

- [ ] **Step 3: Implement retained descriptor, full hash, and immutable SQLite view**

Open with `O_RDONLY | O_CLOEXEC | O_NOFOLLOW`, require a linked regular file, capture pre-link identity metadata, bind the parent directory identity, inventory and reject the DB basename's `-wal`, `-shm`, and journal sidecars, hash through the descriptor once, and compare with the preregistered digest. On macOS, create a private hard link to the exact inode, verify it against the pre-link identity, then capture the operational accepted metadata after the expected ctime/link-count change. Open SQLite with `mode=ro&immutable=1`, set `PRAGMA query_only=ON`, and retain descriptor, parent descriptor, owned link, and connection.

```python
audit = DatabaseSnapshotAudit(
    sha256=actual_sha256,
    device=before.st_dev,
    inode=before.st_ino,
    size=before.st_size,
    mtime_ns=before.st_mtime_ns,
    ctime_ns=before.st_ctime_ns,
    full_hash_count=1,
    boundaries=(),
)
```

At every boundary, compare against the post-link operational metadata and verify the published pathname and owned link still name the same inode and all sidecars remain absent. Before cleanup, run one final operational check; after removing only the owned link, allow only the expected link-count/ctime transition and recheck identity, size, mtime, pathname, parent identity, and sidecar absence.

- [ ] **Step 4: Implement context freezing and loaders on the retained connection**

Execute the exact Exp417 population and TEST identity SQL through the retained connection. Validate sorted unique dataset and instance identities, exact frozen lane coverage, raw float32 byte lengths, and no label columns. Construct `FrozenRunContext` with the verified digest and current Exp425 clean commit. Load values only through Exp417 `_from_connection` functions and verify the boundary before and after each dataset transaction without rehashing.

- [ ] **Step 5: Run DB tests plus Exp417 input and SQL-label regressions**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -v`  
Expected: all tests pass and the Exp417 SQL guard still rejects label access.

- [ ] **Step 6: Benchmark repeated-hash elimination**

Use a temporary 256 MB file and 20 synthetic dataset loads. Compare one Exp425 snapshot hash against 20 Exp417-style verified opens. Require `full_hash_count == 1`, identical decoded rows, and lower wall time. Record the timing but do not introduce a timing-dependent runtime branch.

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_database.DatabaseBenchmarkTests -v`  
Expected: benchmark passes.

- [ ] **Step 7: Commit Task 4**

```bash
git add experiments/exp425_selective_recovery/database.py \
  experiments/exp425_selective_recovery/tests/test_database.py \
  experiments/exp425_selective_recovery/__init__.py
git commit -m "perf(exp425): retain one verified database snapshot"
```

---

### Task 5: Selective MPS Runtime, Explicit Resume, and Progress

**Files:**
- Create: `experiments/exp425_selective_recovery/runtime.py`
- Create: `experiments/exp425_selective_recovery/tests/test_runtime.py`
- Create: `experiments/exp425_selective_recovery/tests/test_cli.py`
- Create: `scratch/run_exp425_selective_recovery.py`
- Modify: `experiments/exp425_selective_recovery/__init__.py`

**Interfaces:**
- Produces: `ExecutionCommand = Literal["new", "resume", "inspect"]`, `RuntimeProgress`, and `Exp425RunResult`.
- Produces: `AtomicProgressWriter` and `Exp425RuntimeSession`.
- Produces: `build_exp425_runtime(snapshot: Exp425DatabaseSnapshot, audit: BackendAudit, progress: AtomicProgressWriter) -> Exp425RuntimeSession`.
- Produces: `run_selective_population(context: FrozenRunContext, store: Exp425CheckpointStore, runtime: Exp425RuntimeSession) -> Exp425RunResult`.
- Produces CLI exit codes: 0 validated `complete_review_only` or matching read-only inspect; 1 validated failed/incomplete run; 2 CLI/contract misuse.
- Consumes: Tasks 1–4, Exp417 `run_dataset`, Exp424 MPS training/scoring functions, and Exp417 global workload rule.

- [ ] **Step 1: Write failing runtime tests for mixed import/native execution**

Seed a store with one imported result, leave one eligible dataset pending, and leave one OOF-insufficient dataset pending. Assert the imported dataset never calls DB/model work, eligible work uses `torch.device("mps")`, insufficient work never calls model or TEST loading, all three end in sorted coverage, and only unexpected terminal evidence fails the run.

```python
result = run_selective_population(context, store, runtime)
self.assertEqual(result.counts.imported, 1)
self.assertEqual(result.counts.native, 2)
self.assertEqual(result.counts.oof_identity_insufficient, 1)
self.assertEqual(model_calls, ["Eligible_normal_1"])
self.assertEqual(test_load_calls, ["Eligible_normal_1"])
```

- [ ] **Step 2: Write failing state/progress and CLI mode tests**

Test exact transitions `initializing -> importing -> running -> finalizing -> complete_review_only`, atomic failure transition, stale-running resume, current dataset clearing, per-dataset flushed JSON log lines, PID/run identity, and counters. Test `new` rejects existing paths, `resume` rejects absent/mismatched paths, `inspect` performs no writes, hashes, benchmarks, model/DB computation, or reconciliation, and a matching complete run returns idempotently. Readiness is not an alias for inspect and is added as a separate subcommand in Task 6.

```python
self.assertEqual(progress["state"], "complete_review_only")
self.assertIsNone(progress["current_dataset"])
self.assertEqual(progress["completed_datasets"], 3)
self.assertEqual(progress["remaining_datasets"], 0)
self.assertFalse(progress["test_labels_opened"])
```

- [ ] **Step 3: Run runtime and CLI tests and confirm expected import failures**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime experiments.exp425_selective_recovery.tests.test_cli -v`  
Expected: missing runtime/entrypoint failures.

- [ ] **Step 4: Implement the frozen MPS runtime session**

Build a `RunnerRuntime` whose loaders use the retained snapshot, whose preprocessing calls Exp417 preprocessing followed by Task 1 eligibility, and whose training/qualification/scoring functions reproduce Exp424 exactly on remaining eligible groups. Capture TRAIN, processed groups, model evidence, contract, scoring evidence, eligibility evidence, and elapsed time in a per-dataset trace. Use `run_dataset` without the Exp417 store, convert the trace to a Task 2 native record, then write it transactionally.

```python
assessment = apply_group_eligibility(
    train.dataset,
    preprocess_length_groups(train),
    fold_contract_sha256=provenance.fold_contract_sha256,
)
trace.eligibility = assessment.evidence
return assessment.groups
```

Require MPS availability and forbid CPU fallback. Keep batch size 32, query chunk size 256, cached float64 score calibration, seeds, ordering, and synchronization identical to Exp424.

- [ ] **Step 5: Implement selective population orchestration**

Load only compact record digests at startup. Iterate the context's sorted universe; lazily load completed imported/native records and natively compute only missing datasets. Validate every effective result against the target context. Apply the unchanged global workload rule after all records are reconstructed. Treat `oof_identity_insufficient` as abstention; treat `DatasetTerminalEvidence`, a failed result, corruption, backend fallback, or incomplete coverage as failed.

- [ ] **Step 6: Implement atomic progress, parent logs, and explicit CLI modes**

Write progress through temp-file `fsync` plus atomic replace. Flush one canonical JSON line per terminal dataset with ordinal, total, dataset, origin, stage, status, reason, elapsed/cumulative time, counters, PID, and run identity. `new` creates both roots, `resume` opens both exact roots and reconciles stale state, and `inspect` reads already-published status/checkpoint metadata without mutation or expensive gate computation. Do not use launchd restart semantics.

```python
terminal_line = {
    "event": "dataset_terminal",
    "ordinal": ordinal,
    "total": len(context.datasets),
    "dataset": dataset,
    "origin": record.origin,
    "status": effective.status,
    "reason": record.reason,
    "elapsed_seconds": record.elapsed_seconds,
    "completed_datasets": ordinal,
    "remaining_datasets": len(context.datasets) - ordinal,
    "pid": os.getpid(),
    "run_identity": store.run_identity,
}
print(json.dumps(terminal_line, sort_keys=True), flush=True)
```

- [ ] **Step 7: Run runtime, CLI, runner, model, scoring, and qualification tests**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_runtime experiments.exp425_selective_recovery.tests.test_cli experiments.exp417_causal_multihorizon_shadow.tests.test_runner experiments.exp417_causal_multihorizon_shadow.tests.test_model experiments.exp417_causal_multihorizon_shadow.tests.test_scoring experiments.exp417_causal_multihorizon_shadow.tests.test_qualification -v`  
Expected: all tests pass.

- [ ] **Step 8: Run TRAIN-only MPS parity smoke**

On one preregistered eligible dataset, compare Exp424 and Exp425 with the same seed and input. Require finite residuals/scores, identical candidate coordinates at every frozen alpha, score deltas within the Exp424 tolerance, identical result-evidence digest after context normalization, and no TEST-label query. Record transfer and synchronization wall time but do not reselect settings.

- [ ] **Step 9: Commit Task 5**

```bash
git add experiments/exp425_selective_recovery/runtime.py \
  experiments/exp425_selective_recovery/tests/test_runtime.py \
  experiments/exp425_selective_recovery/tests/test_cli.py \
  experiments/exp425_selective_recovery/__init__.py \
  scratch/run_exp425_selective_recovery.py
git commit -m "feat(exp425): run selective recovery with explicit resume"
```

---

### Task 6: Review-Only Reporting, Documentation, and Readiness Gate

**Files:**
- Create: `experiments/exp425_selective_recovery/reporting.py`
- Create: `experiments/exp425_selective_recovery/tests/test_reporting.py`
- Create: `docs/EXP425_PREREGISTRATION_KO.md`
- Create: `docs/EXP425_RUNBOOK_KO.md`
- Modify: `scratch/run_exp425_selective_recovery.py`
- Modify: `experiments/exp425_selective_recovery/__init__.py`

**Interfaces:**
- Produces: `render_exp425_artifacts(run: Exp425RunResult, store: Exp425CheckpointStore, db_audit: DatabaseSnapshotAudit) -> Mapping[str, bytes]`.
- Produces: `verify_exp425_artifacts(payloads: Mapping[str, bytes], store: Exp425CheckpointStore) -> None`.
- Produces files: prediction JSONL, dataset summary, group eligibility evidence, predecessor import ledger, backend audit, manifest, status, runtime progress, and parent log.
- Consumes: complete Task 1–5 evidence and no TEST labels.

- [ ] **Step 1: Write failing governance and artifact round-trip tests**

Require sorted canonical prediction coordinates, exact hashes and row counts, imported/native/recomputed/abstained/failed conservation, compatibility and source digests, DB audit with `full_hash_count=1`, MPS/runtime versions, checkpoint benchmark, and explicit false governance flags. Recursively reject keys containing label, target, ground truth, confusion, precision, recall, F1, anomaly location, or family TEST performance.

```python
self.assertTrue(manifest["review_only"])
self.assertFalse(manifest["official_prediction"])
self.assertFalse(manifest["publication_eligible"])
self.assertFalse(manifest["contains_test_labels"])
self.assertFalse(manifest["test_labels_opened"])
self.assertEqual(manifest["db_full_hash_count"], 1)
self.assertEqual(
    manifest["imported_datasets"] + manifest["native_datasets"],
    manifest["completed_datasets"],
)
```

- [ ] **Step 2: Run reporting tests and confirm the expected import failure**

Run: `python3 -m unittest experiments.exp425_selective_recovery.tests.test_reporting -v`  
Expected: `ModuleNotFoundError` for `experiments.exp425_selective_recovery.reporting`.

- [ ] **Step 3: Implement rendering and independent verification**

Render canonical payloads from validated record envelopes. The import ledger records source root/progress/record/result digests and compatibility attestation. The eligibility artifact records every exact-length group and stable reason, using count-bearing native evidence for Exp425 work and visibly distinct source-proven evidence with unavailable split counts for imported Exp424 work. Verify by reparsing every artifact, reconstructing effective results, checking checkpoint digests, recalculating prediction SHA256, checking population conservation, and rejecting forbidden content. A failed summary has no prediction rows; a complete review-only summary may contain rows but remains nonpublishable operationally.

- [ ] **Step 4: Write preregistration and runbook documents**

Copy the exact frozen boundaries, 983/134 source diagnosis, import gates, group-level abstention rule, one-hash DB contract, checkpoint targets, `new|resume|inspect` semantics, one-shot launch rule, log fields, artifact verification commands, and post-hoc separation from the approved design. Include commands with the actual DB and Exp424 checkpoint arguments but no command that starts the full run without `--mode new` and a user-approved output path.

- [ ] **Step 5: Add a separate readiness subcommand and fail-closed launch gate**

Add `readiness` as a separate CLI subcommand, not an `inspect` alias. It accepts an explicit `--readiness-output` path and may write that artifact plus, on macOS only, one hidden random owned DB hard link required by the exact snapshot smoke. The link must name the accepted inode, is recorded in readiness evidence, and must be removed and verified absent before readiness can pass; cleanup failure returns blocked and reports the path. Before full execution, require clean Git, MPS availability, exact input hashes, no other MPS-heavy process, real Exp424 inventory `1117/983/134`, compact checkpoint benchmark targets, DB one-hash smoke, focused regression success, and artifact round-trip success. Do not create or mutate an Exp425 run output/checkpoint root and do not launch if any gate is false.

```python
if not all(gates.values()):
    write_readiness(state="blocked", gates=gates)
    return 1
write_readiness(state="ready", gates=gates)
return 0
```

- [ ] **Step 6: Run focused Exp425 tests and full Exp417 regression**

Run:

```bash
python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
python3 -m unittest discover -s experiments/exp417_causal_multihorizon_shadow/tests -p 'test_*.py'
python3 -m py_compile experiments/exp425_selective_recovery/*.py \
  scratch/run_exp425_selective_recovery.py
git diff --check
```

Expected: every test and compile command exits 0; `git diff --check` prints nothing.

- [ ] **Step 7: Run real readiness verification separately from inspect**

First run the read-only `inspect` subcommand against an existing synthetic completed fixture and prove its filesystem tree is unchanged. Then run `scratch/run_exp425_selective_recovery.py readiness --readiness-output /Users/mac/Documents/Dataset/exp425_readiness_20260804.json` against the frozen DB/lane inputs and Exp424 checkpoint root. Require:

- Exp424 records/importable/terminal = `1117/983/134`;
- terminal reason = OOF early-stop identity insufficiency only;
- DB SHA256 equals the preregistered value and full-hash count = 1;
- no TEST labels opened;
- checkpoint byte/resume targets pass;
- MPS available and no competing MPS-heavy process;
- no file below the Exp424 output or checkpoint root changed.

- [ ] **Step 8: Commit Task 6 and stop before the population run**

```bash
git add experiments/exp425_selective_recovery/reporting.py \
  experiments/exp425_selective_recovery/tests/test_reporting.py \
  experiments/exp425_selective_recovery/__init__.py \
  scratch/run_exp425_selective_recovery.py \
  docs/EXP425_PREREGISTRATION_KO.md docs/EXP425_RUNBOOK_KO.md
git commit -m "docs(exp425): freeze selective recovery run gates"
```

Report readiness, predicted native work count, checkpoint import count, PID/log paths that would be used, and the exact full-run command. Wait for explicit user approval before starting the long MPS population run.

---

## Final Verification and Review

- [ ] Confirm `git status --short --branch` is clean on `codex/exp425-selective-recovery`.
- [ ] Confirm the full Exp425 and Exp417 regression suites exit 0.
- [ ] Confirm real read-only source inspection reports exactly `1117/983/134` and no Exp424 mutation.
- [ ] Confirm MPS parity, checkpoint size/resume, DB one-hash, artifact round-trip, label guard, and readiness gates all pass.
- [ ] Confirm the 1,117-dataset run has not been started.
- [ ] Request a fresh Sol/High final review with exact verdict `ship`, `fix-first`, or `rethink`.
- [ ] If verdict is `ship`, present the readiness report and request explicit full-run approval.
