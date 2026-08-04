# Exp425 Selective Recovery and Runtime Hardening Design

**Status:** Approved architecture, implementation not started  
**Date:** 2026-08-04  
**Experiment:** Exp425  
**Predecessor:** Exp424 MPS Score Cache  
**Publication class:** review-only; never an automatic alert source

## 1. Purpose

Exp425 recovers the completed but fail-closed Exp424 population run without
rewriting Exp424 evidence. It also removes the proven resume, checkpoint, input
verification, launch, and progress-reporting bottlenecks that do not change the
frozen score or routing algorithm.

The research objective remains operational trust, not maximum mean F1. Candidate
generation uses normal TRAIN data only, suppresses false alarms, and keeps all
Priority-review and Shadow outputs outside automatic alert promotion.

## 2. Observed Exp424 Failure

The preserved Exp424 run has the following state:

- all 1,117 per-dataset checkpoint records exist;
- 983 records contain nonterminal decision evidence;
- 134 records contain the same terminal OOF error:
  `early stopping requires at least five identities on both sides`;
- the population summary failed closed because any terminal dataset evidence
  fails the entire run;
- the final prediction file therefore has zero rows and the empty-file SHA256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- the checkpoint tree is approximately 4.2 GB because each dataset record
  repeats the large frozen run context;
- the resume launcher repeatedly restarted a command that unconditionally
  required a nonexistent output directory, producing `FileExistsError` instead
  of resuming;
- the final runtime progress file can remain stale even after a failed summary.

Exp424 remains immutable, failed, review-only evidence. No Exp424 file is
modified, relabeled, deleted, or republished by Exp425.

## 3. Frozen Boundaries

Exp425 freezes the following values to the actual Exp424 contracts:

- dataset universe and deterministic dataset order;
- normal TRAIN-only fitting and qualification;
- TEST values used only after the TRAIN contract is frozen;
- TEST labels, anomaly locations, and family TEST performance unopened during
  prediction;
- model architecture, model and perturbation seeds, horizons, five-fold identity
  allocation, early-stop identity hash rule, final-fit rule, score calibration,
  thresholds, alpha values, candidate coordinates, and routing rules;
- MPS backend, batch size, query chunk size, float precision, percentile method,
  score cache, and synchronization behavior used by Exp424;
- no tail replacement and no post-TEST retuning;
- `mean_combined_f1` remains post-hoc human-assisted evidence only.

The only semantic change is a preregistered TRAIN-eligibility outcome for an
exact-length group that cannot construct the frozen OOF/early-stop splits. The
other changes are storage, verification, resume, and observability controls.

## 4. OOF Identity-Insufficiency Contract

### 4.1 Decision level

Identity sufficiency is evaluated per exact-length TRAIN group because folds,
models, and score calibration are defined at that level. It is not a dataset-wide
exception.

Before model fitting, Exp425 deterministically constructs the same five folds and
the same hash-modulo early-stop split that Exp424 would construct. For every fold
it records:

- unique TRAIN identity count;
- OOF fit and validation identity counts;
- early-stop fit and validation identity counts;
- the frozen fold-contract digest.

Candidate-eligible groups also record the final-fit and final-validation identity
counts used by final-model early stopping. Diagnostic-only groups never fit final
models, so their final-fit counts are explicitly non-applicable rather than being
used as an additional eligibility gate.

If any required split has fewer than five identities on either side, that group
receives `oof_identity_insufficient`. No model is trained for that group, no TEST
values are scored for it, and it contributes no candidate.

This outcome is an expected deterministic abstention, not a runtime failure. A
dataset is `abstained` for this reason only when it has no remaining eligible
group. If at least one group remains eligible, the existing Exp424 qualification,
scoring, and dataset-status rules operate unchanged on those eligible groups.

Unexpected exceptions, nonfinite backend output, provenance mismatch, corrupted
input, or incomplete accounting remain terminal failures and fail the run closed.

### 4.2 Structured evidence

Exp425 adds immutable group eligibility evidence with at least:

- dataset and exact length;
- eligibility status and stable reason code;
- identity and split counts;
- fold-contract digest;
- TRAIN identity digest;
- evidence payload digest.

Native evidence distinguishes candidate-eligible, diagnostic-only, and already
preprocessing-abstained groups. Final-fit counts are required only for the
candidate-eligible case. Imported records use a separate
`source_proven_group_evidence` type containing source group status, qualification,
model/calibration presence, source result/record digests, and compatibility
attestation. Exp424 did not persist split counts, so imported evidence never
invents them and remains visibly distinct from native Exp425 evidence.

Human-readable exception text is diagnostic only and never the decision key.

## 5. Selective Predecessor Import

Exp425 may import only the 983 nonterminal Exp424 records. This is an explicit
predecessor import, not ordinary same-experiment resume.

Every imported record must pass all of these gates:

1. the source root is the exact preserved Exp424 checkpoint store;
2. source root provenance, progress digest, dataset inventory, record filename,
   record digest, and embedded payload digest agree;
3. the exact source Git commit
   `4a398374d880f859f285a271ad63de4a5e55b028`, Exp424 entrypoint SHA256
   `98c909f6100dd2814e3fe102c3f95054a3d6bb42f85657d19b17780ae4934fe1`,
   entrypoint Git blob `da5374b0bf38803aa506dfda664d76395d7f28f9`, database,
   lane snapshot, dataset universe, TRAIN/TEST identity manifests, frozen
   configuration, folds, models, seeds, runtime versions, backend, MPS settings,
   batch size, query chunk size, score cache, percentile/precision, thresholds,
   alpha values, candidate formula, routing, MPS device guard, device-transfer,
   and synchronization contracts match the preregistered Exp425 compatibility
   declaration;
4. the source record contains complete decision, model, calibration, and backend
   evidence and contains no terminal evidence;
5. candidate coordinates and result evidence reconstruct canonically;
6. the source is read through descriptor-bound, no-follow validation and cannot
   change during import.

Source root/dataset descriptors and the shared source lock remain open from
validation through import staging. Imported envelopes are first written to a
target staging area and are not added to the committed dataset/progress inventory.
Before promotion, Exp425 revalidates source root/progress/inventory plus every
source record identity, metadata, and digest through the retained descriptors.
Only that final accepted snapshot may be promoted through the checkpoint intent
protocol. A crash or mutation before promotion leaves no committed import; a
crash during promotion leaves source-attestation-aware import intents that the
generic checkpoint opener must not reconcile. Only the predecessor importer may
resume them, after reacquiring source descriptors/shared lock and repeating the
entire source revalidation against the immutable batch attestation.

`open_validated_exp424_source` is the only capability mint. It returns a
non-serializable opaque source object whose private construction token, live
descriptor identities, held lock, validation nonce, and source-attestation digest
are checked by every import staging and promotion operation. The public
checkpoint API exposes no caller-constructible import authorization. Closing the
source context invalidates the capability. A crash/resume must open and fully
validate the source again to mint a new capability; copied, forged, closed,
wrong-source, and wrong-batch capabilities are rejected.

The score-cache and synchronization contracts are not caller-supplied free-form
digests. Exp425 embeds their exact canonical JSON payloads and requires these
fixed SHA256 values:

- Exp424 score-cache contract:
  `82d6f3b431b2d57f45f2aae33ee4f16729709b0090f8c4984bbdf5557baff257`;
- Exp424 MPS synchronization contract:
  `e1e928ddde0a4793a63cb95a6a643888251e4ddabba9b2bbedd12859df95133a`.

The score-cache payload fixes the two optimization components, Exp417 score
formula, float64 OOF residuals, linear percentile, batch 32, chunk 256, one
predictor materialization per seed/scoring stage, and exact candidate-coordinate
parity. The synchronization payload fixes MPS/device guard, contiguous float32
NumPy copy to MPS, per-prediction/horizon `detach().cpu().numpy()` output transfer,
implicit MPS-to-CPU synchronization, zero explicit `torch.mps.synchronize()` calls,
and float64 residual output. Tests recalculate the canonical digests and reject
every one-field mutation.

An imported Exp425 envelope records:

- `origin = exp424_import`;
- source experiment and root-provenance digest;
- source record and result-evidence digests;
- Exp425 compatibility-attestation digest;
- source-proven group evidence without fabricated split counts;
- the shared Exp425 run-context digest.

Imported evidence is never represented as natively computed Exp425 evidence.
A structurally valid record that is merely incompatible with the target contract
is not partially trusted and enters the native Exp425 work queue with a recorded
reason. Source corruption, digest/inventory disagreement, or mutation fails the
entire import closed instead of being hidden by recomputation. The 134 terminal
Exp424 records are never imported and enter the native queue.

## 6. Compact Checkpoint Schema

Exp425 uses a new checkpoint schema and does not alter Exp417/Exp424 checkpoint
files.

### 6.1 Root objects

The checkpoint root stores once:

- run provenance and schema version;
- the complete immutable frozen run context;
- its canonical SHA256 digest;
- expected dataset universe;
- predecessor compatibility declaration;
- execution mode and run identity;
- empty `transactions/` and `import_batches/` roots for native intents and
  source-revalidation-gated predecessor staging.

### 6.2 Dataset records

Each dataset file stores only:

- dataset-local eligibility, training, calibration, decision, and backend
  evidence, or an imported predecessor envelope;
- origin (`exp425_native` or `exp424_import`);
- reference to the shared run-context digest;
- elapsed timing and terminal diagnostics;
- canonical payload digest.

The full run context is injected only while reconstructing an in-memory result.
It is not serialized inside every dataset record.

### 6.3 Resume index

`progress.json` is an atomic, digest-bound inventory of dataset record digests and
counts. Resume validation reads the root and progress once, verifies directory
inventory, validates each record digest once, and decodes dataset evidence lazily.
Unknown files, duplicate datasets, mismatched digests, missing records, and
pathname/inode changes fail closed.

Dataset commits use a durable `transactions/` intent journal. The writer fsyncs
an intent containing the dataset and record digest, publishes and fsyncs the
dataset record, atomically publishes and fsyncs progress, then removes and fsyncs
the intent. On open/resume, an exclusive descriptor-bound reconciliation may
finish progress for an intent whose matching record is durable, remove an intent
whose record was never published, or remove a leftover intent already reflected
in progress. A record absent from both progress and a valid intent is corruption,
not a recoverable orphan. Crash tests cover termination after every boundary.

Native and imported transactions are distinct. Generic open/resume may reconcile
only `origin=exp425_native` intents. An imported intent must include import batch
identity and final source-attestation digest; generic open reports
`import_revalidation_required` without changing progress. The predecessor importer
must reacquire and fully revalidate the Exp424 source before it may resume any
import promotion. Staging and promotion additionally require the newly minted,
live opaque capability matching that exact source and batch.

Targets for the checkpoint-only optimization are:

- at least 80% reduction in total checkpoint bytes on a representative sample;
- at least 2x faster validation/resume on the same completed sample;
- exact reconstructed candidate-coordinate and result-evidence parity.

Failure to meet a speed target does not permit weaker validation.

## 7. Database Snapshot Verification

At each new or resumed process, Exp425:

1. opens the DB with no-follow, read-only controls and captures pre-link descriptor
   identity, size, and high-resolution timestamps;
2. inventories and rejects existing SQLite `-wal`, `-shm`, and journal sidecars;
3. computes the full SHA256 exactly once through the accepted descriptor;
4. creates and verifies the owned macOS hard link, then captures operational
   accepted metadata after the expected link-induced ctime/link-count change;
5. opens one query-only immutable SQLite view bound to that accepted identity;
6. retains the identity descriptor and sidecar-parent identity for the process
   lifetime;
7. verifies descriptor/path identity, operational metadata, and sidecar absence
   at transaction and finalization boundaries.

A new process, including a resume process, performs a new full hash. A pathname
or stat cache is never trusted across processes. Before owned-link cleanup, a
final operational-metadata check must pass. Cleanup then permits only the expected
owned link-count/ctime change and rechecks identity, size, mtime, pathname, and
sidecar absence. Any other mismatch closes the run without publication.

The manifest records the full-hash count, accepted identity metadata, digest,
verification boundaries, and any failure reason.

## 8. Execution and Resume State Machine

Exp425 exposes explicit modes:

- `new`: output and checkpoint paths must not exist;
- `resume`: both paths must contain an initialized Exp425 run with exactly the
  requested run identity and provenance;
- `inspect`: read-only validation and status reporting, with no computation.
- `readiness`: performs preregistered hashes, benchmarks, and test/smoke gates and
  may write only the explicitly requested readiness artifact; it never creates
  or mutates an Exp425 checkpoint/output run.

On macOS, readiness DB verification has one narrowly scoped temporary exception:
it may create exactly one hidden, random, no-follow hard link to the already
accepted DB inode in the DB directory, use it for the immutable SQLite view, and
remove it before returning. The link path, inode checks, creation/removal results,
and any cleanup failure are recorded in readiness evidence. No readiness success
is possible while the owned link remains. Read-only `inspect` never uses this
exception.

A matching completed review-only run returns idempotently after validation.
Existing unrelated, partial, or mismatched paths fail closed. The launcher is
one-shot and does not automatically restart a failed or completed process.

The atomic runtime state machine is:

`initializing -> importing -> running -> finalizing -> complete_review_only`

Any unexpected exception transitions atomically to `failed`. A stale `running`
state is reconciled only by explicit `resume` after checkpoint validation.

The command returns zero only after all 1,117 datasets are accounted for, all
records validate, expected abstentions are accepted, and review-only artifacts
are finalized. Unexpected terminal evidence or incomplete accounting returns
nonzero.

## 9. Progress and Logs

The run writes atomic structured progress and a line-oriented parent log. Every
terminal dataset line is flushed immediately and includes:

- ordinal and total;
- dataset;
- origin (`import`, `native`, or `recompute_after_import_reject`);
- current and terminal stage;
- dataset status and stable reason code;
- elapsed and cumulative times;
- completed, imported, recomputed, abstained, failed, and remaining counts;
- current process ID and run identity.

`runtime_progress.json` uses `current_dataset = null` only between datasets and in
terminal process states. Finalization always writes `complete_review_only` or
`failed`, including the exit reason. No TEST labels or derived TEST metrics appear
in progress files.

## 10. MLX, MPS, and Parallelism Assessment

Exp425 is not a new score source and does not change Exp424 numerical execution.
It retains the single MPS-heavy process and the actual Exp424 MPS path. Its new
hot paths are JSON/checkpoint validation, filesystem integrity, and SQLite input
control; MLX does not accelerate these operations without changing their
contracts.

Exp425 therefore does not adopt a new MLX path. The manifest records this frozen
fallback reason. MLX kernels, altered MPS batching, compilation, different
chunk/batch sizes, parallel dataset workers, and concurrent CPU/GPU dataset
execution require later experiment numbers. They must satisfy `AGENTS.md` finite
output, candidate-coordinate parity, numerical tolerance, and at least 1.1x
end-to-end speed gates before adoption. Only one MLX/MPS-heavy process runs at a
time unless a separate deterministic benchmark proves otherwise.

## 11. Outputs and Publication Gates

Exp425 produces:

- compact checkpoint root, progress, and per-dataset records;
- predecessor-import ledger and compatibility attestation;
- group eligibility and abstention evidence;
- runtime progress and parent log;
- prediction JSONL, dataset summary, backend audit, manifest, and status;
- hashes for every published artifact and predecessor root.

The manifest must state:

- `review_only = true`;
- `official_prediction = false`;
- `contains_test_labels = false`;
- `test_labels_opened = false`;
- imported/native/recomputed/abstained/failed counts;
- Exp424 source digests and Exp425 compatibility digest;
- MPS/runtime versions and frozen numerical contract;
- DB snapshot and hash accounting;
- checkpoint size/resume benchmark and optimization decision.

Priority-review and Shadow candidates cannot become automatic alerts. Confusion
matrices or F1 metrics may be computed only by a separate post-hoc evaluator after
the prediction manifest and prediction SHA256 are frozen.

## 12. Verification Strategy

Implementation follows test-driven development and must cover:

- exact group-level eligibility boundary tests, including fewer than five
  identities on either split, candidate-only final-fit applicability,
  diagnostic-only OOF parity, mixed eligible/ineligible groups, and all-group
  abstention;
- regression parity for eligible Exp424 groups and candidate coordinates;
- all predecessor-import gates plus mutation, symlink, inventory, digest, and
  provenance rejection tests;
- compact checkpoint canonical round trip, concurrent-writer exclusion, atomic
  intent-journal crash recovery at every commit boundary, lazy resume, and
  corruption quarantine tests;
- DB replacement/truncation/timestamp/inode mutation, hard-link ordering,
  WAL/SHM/journal sidecar tests, and exactly-one-hash accounting per process;
- `new`, `resume`, completed-idempotent, mismatched-path, stale-state, exit-code,
  terminal-progress, read-only inspect, and separate readiness tests;
- no-label-open and review-only publication tests;
- representative checkpoint byte-size and resume-time benchmarks;
- full existing Exp417/Exp424 regression tests, compile checks, and
  `git diff --check`.

A small TRAIN-only smoke must prove that imported and natively recomputed eligible
records yield identical candidate coordinates and result-evidence digests. The
1,117-dataset run is not started until readiness verification is reported and the
user explicitly approves the long run.

## 13. Implementation Boundary

Exp425 is implemented in a new experiment package and entrypoint. Frozen Exp424
artifacts and checkpoint files are read-only inputs. Frozen Exp417/Exp424 behavior
must not be silently edited; reusable code may be exposed only through behavior-
preserving refactors with existing regression parity.

The likely implementation units are:

1. Exp425 contracts and deterministic group eligibility;
2. compact checkpoint store and predecessor importer;
3. descriptor-bound database snapshot/runtime adapter;
4. explicit CLI modes, one-shot execution state, progress, and logging;
5. reporting, manifests, documentation, and readiness verification.

Implementation is complete only after parent verification and a fresh Sol/High
review returns `ship`.
