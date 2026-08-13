# Paper r4 Operational Launch Antigravity Design

## Goal

Make the 1,117-dataset paper proposal run finish safely even when individual
datasets fail or the process is interrupted. A completed dataset must never be
recomputed or lost, failures must be collected for one later analysis, and no
global Task6/Task7/freeze result may exist before all Task4+Task5 results pass
typed verification.

This work prepares the launch surface only. It does not create the official r4
root and does not start the official experiment.

## Chosen architecture

Add one operations-only launcher outside the scientific paper package. It pins
the reviewed Git commit, capable Python interpreter, database, official r4
root, CPU worker policy, and code closure. It owns process locking, PID
metadata, structured progress, status, stop, and explicit resume commands.

Keep scientific scoring, queueing, create-only success publication, append-only
failure attempts, and exact-coverage global gates in the existing paper
package. Make only the minimal public CLI correction required to print its
canonical result and return a nonzero exit status for incomplete or failed
runs.

## Frozen operational paths

Current executable authority:

- Repository: `/Users/mac/Documents/Time Series/.worktrees/grok-paper-r4-operational-antigravity`
- Interpreter: `/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python`
- Database: `/Users/mac/Documents/Dataset/univariate_ts.db`
- Immutable prior root: `/Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813`
- Future official root: `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813`
- Official operations namespace: `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity`
- Prelaunch smoke root: `/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813`
- Prelaunch smoke-ops root: `/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813_ops_antigravity`

The exp443 MiniRocket interpreter is not the current paper r4 authority.
Official r4 and its ops namespace remain unauthorized and must stay absent.
The v2 smoke and smoke-ops roots are the current executable smoke authority
and must also stay absent until a later authorized smoke.

Immutable failed v1 evidence, present and never current or reusable:

- Failed smoke root: `/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813`
- Failed smoke-ops root: `/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813_ops_antigravity`

The launcher rejects alternate values rather than accepting configurable
official paths. The reviewed commit is supplied explicitly and must be an exact
lowercase 40-hex clean HEAD. The interpreter path, executable identity, Python
version, package versions, database SHA-256, code-closure hashes, CPU facts, and
backend decision are recorded in canonical authority evidence.

## Commands and lifecycle

The launcher exposes these operations:

- `check`: read-only authority validation. It must not create r4, smoke, or ops
  roots.
- `smoke`: run the bounded Task4+Task5 operational smoke in the separate smoke
  root and publish canonical evidence.
- `launch`: after a separate manual user approval, create the r4 and ops roots,
  acquire the exclusive worker lock, and start one detached worker.
- `status`: validate authority and artifacts, then report stage, completed,
  failed, pending, percentage, elapsed time, rate, ETA, PID metadata, and lock
  state without treating PID existence as liveness authority.
- `stop`: send SIGINT first, wait a bounded interval, then SIGTERM if required.
  Already published success records remain reusable.
- `resume`: after batch root-cause review and another explicit approval, start
  only the failed and never-attempted queue under the identical authority.
- `_worker`: internal command holding the exclusive `flock`; it runs the paper
  CLI, relays SIGINT/SIGTERM, writes timestamped JSONL events, and exits with the
  child result.

Only `flock` is the single-process authority. PID is operator metadata. A stale
PID cannot block launch when the lock is free, and a live unrelated PID cannot
authorize or stop a worker.

## Prelaunch smoke

Do not repeat the already-passed 19-dataset/76-comparison MiniRocket readiness
gate. Exercise the real Task4+Task5 collection and persistence path on this
frozen, sorted 23-dataset set:

- All 19 nonfinite-repair affected datasets frozen by
  `EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY`.
- `MelbournePedestrian_normal_10`: official TRAIN-or-TEST union TEST-only
  member with 3 nonfinite TEST rows / 3 scalars and finite TRAIN.
- `Phoneme_normal_39`: one normal TRAIN row.
- `ChlorineConcentration_normal_3`: more than 1,000 normal TRAIN rows and the
  historical capped-reference boundary.
- `PigAirwayPressure_normal_1`: four normal TRAIN rows and series length 2,000.

The smoke passes only when the first run either finishes or reports a verified
incomplete wave, a second same-authority invocation reuses all published
successes, final typed artifacts verify, and TEST-label access remains zero.
The smoke root is never reused as official evidence.

## Failure and resume semantics

Each completed Task4+Task5 result is parent-published as a create-only canonical
record only after typed round-trip and hash verification. Each failure is an
append-only attempt record. Remaining futures are not cancelled. At the end of
the wave, the pipeline returns `proposal_score_incomplete`, lists failed and
pending datasets, and exits nonzero. There is no automatic retry.

Grok performs independent failure analysis and alternatives. Codex makes the
final decision, integrates changes, and verifies results. The user explicitly
approves resume. Same-context transient failures may reuse verified successes.
Any code, scientific configuration, database, catalog, environment, backend,
or closure change requires a fresh root; checkpoints are never migrated by
assumption.

## Completion boundary

Task6, Task7, proposal shards, freeze, and journal proposal completion remain
unreachable until exactly 1,117 Task4+Task5 successes reconstruct in catalog
order and all success namespaces verify. Final completion additionally requires
the proposal manifest and journal semantic verifiers to pass.

The pipeline CLI prints canonical JSON for `run`, `status`, and `verify`.
`completed` exits 0; `proposal_score_incomplete`, invalid authority, semantic
verification failure, and terminal child failure exit nonzero.

## Acceleration and worker policy

The launcher does not introduce a new score source or accelerated algorithm.
It verifies the existing acceleration manifest. PyTorch, MLX, and MPS facts and
the CPU fallback reason remain recorded. CPU is canonical unless the existing
same-seed finite-output, identical-candidate, score-tolerance, and at-least
1.1x end-to-end MLX gate passes.

For a new official run, dataset workers equal
`max(1, detected_logical_cpu_count - 1)` and numerical inner threads equal one.
Both values are non-overridable and recorded in authority/provenance.

## Safety and review

Grok implementation occurs only in a dedicated Grok worktree and has no write
access to the official experiment roots. All new filenames include
`antigravity`. Tests are written and observed failing before production edits.
The Exp448 and paper codemap triplets are regenerated together whenever their
mapped boundaries change.

After implementation, Grok performs a fresh read-only self-review and waits for
its final verdict. Codex independently inspects the diff and reruns focused and
relevant regression tests. A fresh Sol Advisor final review must also return a
ship verdict. Only then may Codex present evidence and request the final manual
approval that authorizes creation of the official r4 root and process.
