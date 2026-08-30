# Exp471 Exp467 Evaluation Recovery Design

Date: 2026-08-24  
Status: approved architecture, implementation not started  
Source experiment: Exp467  
Derived experiment: Exp471

## 1. Purpose

Exp467 completed all 4,468 official coordinates with zero failures. Its first
official evaluation then failed only when the final evaluation digest attempted
to serialize scikit-learn's initial ROC threshold, which is positive infinity,
through strict finite JSON.

Exp471 shall produce the paper evaluation and report without rerunning the 4,468
source experiments. It shall preserve the original Exp467 authority, make the
expensive label-free score preparation resumable, and normalize non-finite ROC
threshold metadata only at the derived artifact boundary.

## 2. Fixed Source Authority

The following source identity is immutable:

- source commit: `1e616731a88da9bc0268823e1d2cef2a53c1054f`;
- source context SHA-256:
  `31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af`;
- source root:
  `/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_official_antigravity_20260823`;
- source population: A/B/C/D = 1,117 each, 4,468 total;
- execution origins: 4,234 imported and 234 native;
- source failures: zero;
- official Python executable:
  `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`.

Exp471 must not modify any Exp467 module, source checkpoint, source manifest,
progress log, database, or result root. Before reading score payloads, Exp471
reconstructs the Exp467 code closure from the fixed Git commit, recreates the
source context, and requires the exact source-context digest above. It then
verifies the migration manifest, all 4,468 source envelopes, the four condition
manifests, inventory identity, catalog identity, database identity, payload
hashes, and execution-origin counts.

The existing Exp451 historical-Git-byte reconstruction pattern is the reference
design. Worktree bytes may be used only after their hashes match the fixed Git
bytes. Any mismatch stops before score preparation or label access.

## 3. Rejected Alternatives

### 3.1 Relax Exp467 context verification

Rejected. Accepting old artifacts under changed Exp467 bytes would weaken the
paper authority and make unrelated future changes appear compatible.

### 3.2 Runtime monkeypatch or unrecorded wrapper

Rejected. It would execute behavior that is absent from the recorded source
closure and would not be independently reproducible.

### 3.3 Derived Exp471 authority

Selected. Exp467 remains frozen source evidence. Exp471 records a separate code
closure, cache manifest, evaluation digest, report manifest, and an explicit
binding back to the verified Exp467 source authority.

## 4. Data Flow

```text
fixed Exp467 Git bytes + official source root
                  |
                  v
        verify-source (no labels)
                  |
                  v
   verified Exp467 source authority
                  |
                  v
 prepare resumable raw/micro score shards (no labels)
                  |
                  v
      complete score-cache manifest
                  |
                  v
 open labels once -> frozen Exp450 metric/curve functions
                  |
                  v
 durable Exp471 evaluation artifact
                  |
                  v
         atomic paper report
```

The public `report` command orchestrates every stage. Completed verified stages
are reused. Operators do not run a separate expensive `evaluate` command.

## 5. Components and Ownership

The implementation creates a new namespace and does not edit Exp467:

- Exp471 root:
  `/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_antigravity_20260824`;
- score cache: `<Exp471 root>/score_cache_antigravity`;
- durable evaluation: `<Exp471 root>/evaluation_antigravity`;
- final report: `<Exp471 root>/report_antigravity`.

The root and its three children use create-only publication. A command refuses a
symlink, a path outside the frozen root, or a foreign existing destination.

- `experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py`
  freezes source and target identities, roots, method order, worker policy, and
  expected hashes.
- `contracts_antigravity.py` defines immutable source authority, score shard,
  cache manifest, evaluation artifact, and report authority records.
- `source_authority_antigravity.py` reconstructs fixed Git bytes and verifies
  the complete Exp467 source evidence without labels.
- `score_cache_antigravity.py` builds, validates, resumes, and publishes
  label-free score shards and the terminal cache manifest.
- `evaluation_antigravity.py` opens labels only after cache completion and calls
  the unchanged Exp450 metric and curve definitions.
- `reporting_antigravity.py` produces canonical JSON/CSV/figures and publishes
  the report atomically.
- `cli_antigravity.py` exposes `verify-source`, `prepare`, `status`, and `report`.
- `scratch/run_exp471_exp467_evaluation_recovery_antigravity.py` is the closed
  operator entry point.
- `docs/codemap/codemap_exp471_antigravity.html`, `.json`, and `.lock` are the
  synchronized Exp471 code-map triplet.

All new files include `antigravity` in their filename.

## 6. Durable Score Cache

The expensive label-free boundary is persisted before any TEST label access.
Each method/dataset shard contains:

- Exp471 cache schema and cache-context digest;
- verified Exp467 source-context and inventory digests;
- dataset identity and exact test-row count;
- method identity in the fixed ten-method order;
- raw per-dataset score vector used by per-dataset and macro metrics;
- micro score vector used by pooled metrics and curves;
- source/baseline manifest hashes;
- canonical payload SHA-256 and envelope SHA-256.

For A/B/C/D, the raw and micro vectors are the unchanged Exp467 score vectors.
For public baselines, the raw vector and the Numba-calibrated micro vector are
stored separately so the frozen Exp450 metric call receives exactly the same two
score maps as before.

Shards use create-only atomic publication and exact-byte reuse. A partial run
verifies existing shards and schedules only missing or previously failed shards.
A terminal cache manifest is published only after all ten methods cover all
1,117 datasets with finite vectors, exact row counts, and the fixed ordering.
Changed or malformed existing bytes fail closed; they are never overwritten.

Progress is appended to `progress_antigravity.jsonl` with durable completed,
failed, remaining, elapsed time, and ETA fields. The latest event never claims a
completion that is not already durable.

## 7. Evaluation and JSON Normalization

The evaluation stage requires a complete, reopened cache manifest before loading
TEST labels. It then passes the cached raw and micro score maps to the unchanged
Exp450 metric and curve functions.

The in-memory frozen evaluation remains unchanged, including every initial ROC
threshold equal to positive infinity. A recursive artifact-view canonicalizer
maps only non-finite floating-point values to JSON `null` immediately before
strict canonical JSON hashing and publication. It does not remove the threshold
field, stringify infinity, use `allow_nan=True`, or substitute an arbitrary
finite number.

AP, AUROC, paired statistics, ROC false-positive/true-positive coordinates, and
PR recall/precision coordinates must be identical before and after artifact-view
normalization. The durable evaluation artifact binds:

- Exp467 source-context and inventory digests;
- Exp471 code-closure digest;
- complete score-cache manifest digest;
- fixed method and dataset order;
- metric tables and curve tables;
- normalized canonical evaluation digest;
- interpreter and dependency versions.

Once this artifact exists and reopens successfully, report rendering never
repeats score preparation or metric calculation.

## 8. Parallelism and Acceleration

Score shards are independent and use the project CPU-worker policy: detected
logical CPU count minus one, minimum one. The current expected record is 12
logical CPUs and 11 workers. Each worker uses one inner numerical thread.

The exact baseline upper-tail calibration remains the existing Numba float64,
`fastmath=False`, `parallel=False`, `nogil=True` algorithm. Exp471 may schedule
independent shards concurrently but must not change score ordering, calibration,
metric definitions, or thresholds.

PyTorch, MLX, and MPS are recorded explicitly. This derived evaluation workload
does not rerun tensor feature generation. The order-statistic calibration and
canonical JSON stages are not promoted to MLX/MPS because no exact validated
expression provides the required source semantics. The canonical fallback is the
Numba/CPU path. No MPS-heavy process is launched by Exp471.

Before 11-worker preparation becomes canonical, a public/synthetic smoke compares
one-worker and 11-worker shard bytes and measures wall time. Parallel preparation
is accepted only when bytes are identical and wall time improves; otherwise the
same resumable shard design runs with one worker.

## 9. Failure Handling and Recovery

- Source commit, Git byte, context, catalog, database, manifest, checkpoint, or
  origin mismatch: stop before cache access.
- Existing cache shard mismatch: fail closed and name the exact path and field;
  never overwrite it.
- Worker failure: append one unique failure event and continue unrelated first
  attempts; retry failures only after the first wave completes.
- Interrupted preparation: reopen and verify completed shards, then schedule only
  missing work.
- Label access before complete cache authority: reject.
- Evaluation artifact mismatch: reject before reporting.
- Report interruption before atomic rename: retry from the durable evaluation
  artifact without recalculation.
- Existing foreign or markerless report destination: never reclaim or delete it.

## 10. CLI and Operator Workflow

The official interpreter is required for every command:

```sh
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py verify-source

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py prepare

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py status

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py report
```

`report` also invokes incomplete prerequisite stages and resumes them. The
separate commands exist for observation and cheap preflight, not to duplicate
successful work.

## 11. Verification

Tests must prove all of the following:

1. Fixed Git bytes reconstruct the exact Exp467 source context and changed bytes,
   catalog, backend, database, or commit are rejected.
2. An economical fixture represents all 4,468 coordinates and both execution
   origins without exposing private source data or TEST labels.
3. Source verification completes before any score payload or label callback.
4. Cache shards are finite, ordered, create-only, tamper-evident, and resumable.
5. A worker interruption resumes missing shards without recomputing verified
   shards; retry wave ordering is deterministic.
6. One-worker and accepted parallel preparation produce identical shard bytes.
7. A synthetic real ROC curve contains initial positive-infinity thresholds, the
   raw evaluation remains unchanged, and the artifact view contains JSON null.
8. AP/AUROC, ROC coordinates, and PR coordinates are exactly unchanged.
9. Reporting from a complete evaluation artifact calls neither score preparation
   nor metric calculation.
10. Report publication is atomic and safely retryable.
11. The complete Exp471 test suite, code-map check, Python compilation, and
    `git diff --check` pass.

No official TEST labels or private raw data are provided to external agents.

## 12. Pull Request Handling

PR #25 must not be merged into Exp467 because it changes the source experiment's
runtime closure. It remains open only as evidence of the reproduced defect and
tested normalization rule while Exp471 is implemented. The Exp471 PR supersedes
PR #25, contains no Exp467 file modification, and references the source context,
the failed evaluation log, the cache/evaluation manifest digests, and final test
evidence. Closing or retargeting PR #25 is a separate explicit repository action.

## 13. Acceptance Criteria

Exp471 is complete only when:

- the original Exp467 root remains byte-identical;
- source verification reports 4,468 successes and zero failures under the exact
  fixed source context;
- the label-free cache is complete and restartable;
- the durable evaluation artifact and final report both verify;
- continuous-score AP/AUROC, ROC, and PR artifacts exist for all ten methods;
- no duplicate 4,468-coordinate experiment run occurs;
- all provenance and code-map artifacts are committed and the working tree is
  clean.
