# Exp437 Recovery And Exp438 Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the frozen Exp437 checkpoint, prove the replacement CPU process is live, and start the official 1,117-dataset Exp438 run concurrently without changing either experiment's scientific rules.

**Architecture:** Add one focused Exp438 predecessor-state module that turns Exp437 checkpoint, progress, PID, and publication files into a strictly validated snapshot. Readiness accepts either a terminal inactive snapshot or a fresh live zero-error CPU snapshot, while CLI replay permits only monotonic progress or a valid terminal transition. A separate operational launcher imports the exact reviewed Exp437 worktree and calls its existing public resume/publication APIs without modifying Exp437 product code.

**Tech Stack:** Python 3 standard library, existing Exp436/Exp437 checkpoint and progress contracts, Exp438 readiness/CLI, `unittest`, canonical JSON and SHA-256, POSIX `nohup`.

## Global Constraints

- Exp437 source worktree is `/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery` at exact commit `cc0f507f290dc7f613763d3653498f8f54cad88a` and must remain clean.
- Exp437 and Exp438 scientific features, scores, thresholds, candidate budgets, frozen lanes, and label firewalls must not change.
- Prediction may not access TEST labels; TEST labels remain post-freeze evaluation only.
- Official CPU worker count is `max(1, logical_cpu_count - 1)`, which is 11 on this Mac.
- Only a CPU Exp437 process may run concurrently with Exp438; MLX/MPS predecessor execution is rejected.
- Exp438 still records its in-situ Numba/MLX/MPS parity and speed assessment and uses CPU unless an exact candidate passes parity and the 1.1x speed gate.
- Both processes own atomic progress, PID, event, parent-log, and checkpoint evidence. Codex performs one post-launch liveness check and does not poll repeatedly.
- Every newly created file name contains `antigravity`.
- The three code-map files are regenerated together before product changes and whenever the module/data-flow map changes.

---

### Task 1: Refresh The Code Map And Freeze This Plan

**Files:**
- Create: `docs/superpowers/plans/2026-08-07-exp437-recovery-exp438-concurrency-antigravity.md`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: current Git HEAD, tracked paths, and current file bytes.
- Produces: a committed executable plan and synchronized code map.

- [ ] **Step 1: Compare the repository and lock**

Run:
```bash
git rev-parse HEAD
git status --short --branch
python3 - <<'PY'
import json, subprocess
from pathlib import Path
lock = json.loads(Path("docs/codemap/codemap.lock").read_text())
print(lock["current_commit"])
print(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print(lock.get("changed_modules_since_prior_lock", []))
PY
```

Expected: the lock points to the preceding design commit and `docs` is stale.

- [ ] **Step 2: Regenerate and validate all three code-map artifacts together**

Retain at most 20 primary nodes; add this plan/design under research-control evidence; calculate deterministic fingerprints from sorted tracked paths and current bytes while excluding `docs/codemap`. Validate JSON parsing, source paths and symbols, node/edge/flow references, HTML embedded-data equality, current commit/dirty state, fingerprints, and unknown-relation marking.

- [ ] **Step 3: Commit the plan and map**

```bash
git add docs/superpowers/plans/2026-08-07-exp437-recovery-exp438-concurrency-antigravity.md docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git diff --cached --check
git commit -m "docs(exp438): plan concurrent predecessor recovery"
```

---

### Task 2: Validate Exp437 As A Rich Predecessor Snapshot

**Files:**
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/predecessor_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_predecessor_antigravity.py`
- Modify: `experiments/exp438_frequency_complexity_evidence_antigravity/readiness_antigravity.py`
- Modify: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_readiness_antigravity.py`

**Interfaces:**
- Consumes: `EXP437_PROFILE`, `Exp436CheckpointStore.open_classified_read_only`, `inspect_progress`, injected `now_utc: Callable[[], datetime]` and `process_is_live: Callable[[int], bool]`.
- Produces: `read_exp437_snapshot_antigravity(run_parent, *, now_utc, process_is_live) -> dict[str, object]`, `classify_exp437_snapshot_antigravity(snapshot) -> str`, and `CONCURRENT_HEARTBEAT_MAX_AGE_SECONDS_ANTIGRAVITY = 300.0`.

- [ ] **Step 1: Write failing snapshot tests**

Construct canonical temporary Exp437 checkpoint/progress evidence with shared Exp436 helpers. Assert a live snapshot contains exact identity/root, `state="running"`, PID, OS liveness, `backend="cpu"`, progress/checkpoint counts, total 1,117, error zero, heartbeat age, and publication/evaluation absence. Assert tampered hashes, broken event/checkpoint prefixes, wrong identity, and malformed timestamps fail.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_predecessor_antigravity -v
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement strict snapshot reading**

Use:
```python
CONCURRENT_HEARTBEAT_MAX_AGE_SECONDS_ANTIGRAVITY = 300.0
EXP437_CONCURRENCY_AUTHORIZED_ANTIGRAVITY = True

def _process_is_live_antigravity(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
```

Open the checkpoint with `profile=EXP437_PROFILE` and call `inspect_progress(..., process_is_live=None)` so canonical hashes and event/checkpoint prefixes are verified without rejecting a live owner. Parse a timezone-aware UTC heartbeat; reject negative/non-finite age. Check `output_antigravity/publication_antigravity` and `evaluation_antigravity` existence without opening TEST labels.

- [ ] **Step 4: Write failing classification/readiness tests**

Accept:
- `"terminal"` only for complete/checkpoint_complete with inactive PID and complete counters.
- `"concurrent"` only for running, live CPU PID, heartbeat age at most 300 seconds, zero errors, `0 <= completed < 1117`, checkpoint count not behind progress, no publication/evaluation, and authorization exactly true.

Reject stale/dead PID, MLX/MPS, failed/interrupted, nonzero error, invalid counters, active publication/evaluation, wrong identity/root, or disabled authorization. Readiness must persist `exp437_execution`, `exp437_execution_mode`, and `concurrent_with_exp437`.

- [ ] **Step 5: Run readiness tests and verify RED**

```bash
python3 -m unittest experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_predecessor_antigravity experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_readiness_antigravity -v
```

- [ ] **Step 6: Integrate the classifier and verify GREEN**

Keep `ReadinessDependenciesAntigravity.exp437_terminal` as the injectable callable for compatibility, but treat its return as the rich snapshot. Keep `schema="exp438.readiness.v1"` and existing science/config/hash fields unchanged.

- [ ] **Step 7: Commit**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/predecessor_antigravity.py experiments/exp438_frequency_complexity_evidence_antigravity/readiness_antigravity.py experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_predecessor_antigravity.py experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_readiness_antigravity.py
git diff --cached --check
git commit -m "feat(exp438): validate live CPU predecessor"
```

---

### Task 3: Permit Only Monotonic Exp437 Progress During CLI Replay

**Files:**
- Modify: `experiments/exp438_frequency_complexity_evidence_antigravity/predecessor_antigravity.py`
- Modify: `experiments/exp438_frequency_complexity_evidence_antigravity/cli_antigravity.py`
- Modify: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_predecessor_antigravity.py`
- Modify: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_cli_antigravity.py`

**Interfaces:**
- Consumes: readiness-bound `exp437_execution` and a current snapshot.
- Produces: `revalidate_exp437_snapshot_antigravity(bound, current) -> str`.

- [ ] **Step 1: Write failing transition tests**

Accept running CPU forward progress with the same live PID, a different verified live recovery PID, running-to-terminal completion, and stable terminal state. Reject counter regression, dead/stale running PID, failed/interrupted state, CPU-to-MLX/MPS change, identity/root/total change, nonzero error, active publication/evaluation, or incomplete terminal counters.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_predecessor_antigravity -v
```

Expected: the transition validator is missing.

- [ ] **Step 3: Implement the transition validator**

Require identity/root/backend/total equality and monotonic progress/checkpoint counts. A running current snapshot must pass concurrent classification; its PID may change only when the replacement PID is live. A terminal current snapshot requires inactive PID, both counters at 1,117, zero errors, and no evaluation; publication may exist after completion.

- [ ] **Step 4: Write CLI replay tests**

Patch the snapshot reader. Assert forward progress and terminal completion pass. Patch the Exp438 input factory to raise if invoked and prove every invalid transition fails before inputs/workers. Keep commit, paths, hashes, dataset order, frozen config, backend decision, focused tests, and three zero-label counters mandatory.

- [ ] **Step 5: Run RED, implement CLI integration, and run GREEN**

```bash
python3 -m unittest experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_cli_antigravity -v
```

Lazily read/revalidate the current snapshot in `_read_readiness_antigravity` and remove only the old terminal-only gate.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp438_frequency_complexity_evidence_antigravity/predecessor_antigravity.py experiments/exp438_frequency_complexity_evidence_antigravity/cli_antigravity.py experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_predecessor_antigravity.py experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_cli_antigravity.py
git diff --cached --check
git commit -m "feat(exp438): revalidate concurrent predecessor"
```

---

### Task 4: Add A Reproducible Exp437 Recovery Launcher

**Files:**
- Create: `operational/recover_exp437_checkpoint_antigravity.py`
- Create: `experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_recover_exp437_checkpoint_antigravity.py`

**Interfaces:**
- Consumes: exact worktree/commit, `EXP437_PROFILE`, `resume_exp436`, shared CLI `_input_factory` and `_publish_after_complete`, and read-only checkpoint/progress inspection.
- Produces: `preflight_exp437_recovery_antigravity() -> dict[str, object]` and `recover_exp437_checkpoint_antigravity() -> dict[str, object]`; stdout is canonical JSON.

- [ ] **Step 1: Write failing launcher tests**

Load the operational module by file path with injected git/process/runtime dependencies. Reject dirty/wrong source HEAD, live old PID, wrong identity/backend/readiness binding, failed checkpoint classification, event/checkpoint mismatch, publication/evaluation, or nonzero label access. Valid incident evidence is old PID 81739, progress/event 938, checkpoint 939, error zero, and recoverable reconciliation; preflight must not change fixture hashes.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_recover_exp437_checkpoint_antigravity -v
```

- [ ] **Step 3: Implement preflight and recovery**

Prepend only the exact clean Exp437 worktree to `sys.path`, verify its HEAD before imports, and never edit checkpoint/progress manually. Invoke:
```python
summary = resume_exp436(
    readiness_path=EXP437_PROFILE.readiness,
    run_root=EXP437_PROFILE.output,
    checkpoint_root=EXP437_PROFILE.checkpoint,
    reviewed_commit=REVIEWED_EXP437_COMMIT_ANTIGRAVITY,
    input_session_factory=_input_factory(REVIEWED_EXP437_COMMIT_ANTIGRAVITY, EXP437_PROFILE),
    profile=EXP437_PROFILE,
)
result = (_publish_after_complete(summary, EXP437_PROFILE)
          if summary.status == "checkpoint_complete"
          else summary.to_payload())
```

Record preflight SHA-256 for readiness, progress, PID, events, and checkpoint manifest. Do not import evaluation or TEST-label code.

- [ ] **Step 4: Run launcher and shared recovery tests**

```bash
python3 -m unittest experiments.exp438_frequency_complexity_evidence_antigravity.tests.test_recover_exp437_checkpoint_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_progress_antigravity -v
```

- [ ] **Step 5: Commit**

```bash
git add operational/recover_exp437_checkpoint_antigravity.py experiments/exp438_frequency_complexity_evidence_antigravity/tests/test_recover_exp437_checkpoint_antigravity.py
git diff --cached --check
git commit -m "feat(exp437): add checkpoint recovery launcher"
```

---

### Task 5: Freeze Documentation And Run Regression Gates

**Files:**
- Modify: `docs/EXP438_PREREGISTRATION_antigravity_KO.md`
- Modify: `docs/EXP438_RUNBOOK_antigravity_KO.md`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: implemented contracts and test evidence.
- Produces: operational-only concurrency amendment, synchronized research records, and current code map.

- [ ] **Step 1: Document fixed operational rules**

Record user authorization, 300-second heartbeat, CPU-only/zero-error/monotonic replay, 11 workers, in-situ acceleration evidence, unchanged science, zero prediction-label access, and one-check/no-poll monitoring.

- [ ] **Step 2: Update state/catalog without claiming results**

Record Exp437 stale-PID recovery at 938 progress events/939 checkpoint records and Exp438 implementation/readiness state. Do not write performance results.

- [ ] **Step 3: Run full regression**

```bash
python3 -m unittest discover -s experiments/exp438_frequency_complexity_evidence_antigravity/tests -p 'test_*antigravity.py'
python3 -m unittest discover -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests -p 'test_*antigravity.py'
python3 -m unittest discover -s experiments/exp425_selective_recovery_antigravity/tests -p 'test_*antigravity.py'
```

Expected: zero failures/errors/skips.

- [ ] **Step 4: Verify science identity and label firewall**

Compare frozen config payloads and run targeted label-boundary tests. Only predecessor operational readiness evidence may differ; channels, quantile, strict comparison, budgets, batch sizes, seeds, worker policy, and review-only status remain unchanged.

- [ ] **Step 5: Regenerate and validate all code-map files together**

Map the predecessor and recovery files under existing primary nodes; update tests/flows/evidence; recalculate fingerprints; validate HTML/JSON identity and lock state.

- [ ] **Step 6: Commit**

```bash
git add docs/EXP438_PREREGISTRATION_antigravity_KO.md docs/EXP438_RUNBOOK_antigravity_KO.md RESEARCH_STATE.json EXPERIMENT_CATALOG.json docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git diff --cached --check
git commit -m "docs(exp438): freeze concurrent execution evidence"
```

---

### Task 6: Recover Exp437 And Prove It Is Alive

**Files:**
- Runtime log: `/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/recovery_launcher_antigravity.log`
- Runtime PID: `/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/recovery_launcher_antigravity.pid`
- Existing Exp437 process-owned progress/event/PID/checkpoint files.

**Interfaces:**
- Consumes: committed recovery launcher and frozen checkpoint.
- Produces: one new live Exp437 PID or a valid just-completed publication.

- [ ] **Step 1: Run read-only preflight**

```bash
python3 operational/recover_exp437_checkpoint_antigravity.py --preflight
```

Expected: exact clean source, old PID absent, 938 progress/events, 939 checkpoint records, zero errors, no publication/evaluation, recoverable reconciliation.

- [ ] **Step 2: Launch once**

```bash
nohup python3 operational/recover_exp437_checkpoint_antigravity.py > /Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/recovery_launcher_antigravity.log 2>&1 &
```

Write the shell PID to the run-parent `recovery_launcher_antigravity.pid`; the runtime replaces canonical `output_antigravity/run_antigravity.pid` through its owner API. Launcher files must remain outside `output_antigravity`, whose strict inventory allows only process-owned runtime files.

- [ ] **Step 3: Perform one liveness check**

Confirm new canonical PID, OS liveness, active CPU state, fresh heartbeat, error zero, and non-regressing counts. Accept terminal inactive plus publication if it finishes during launch. Do not poll again.

---

### Task 7: Start The Official Exp438 1,117-Dataset Run

**Files:**
- Runtime root: `/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807`
- Process-owned readiness, progress, events, PID, log, and checkpoint artifacts.

**Interfaces:**
- Consumes: verified Exp437 snapshot, clean committed Exp438 worktree, frozen sources, tests, and backend assessment.
- Produces: one official Exp438 process for exactly 1,117 datasets with zero prediction-label access.

- [ ] **Step 1: Confirm absent root and clean HEAD**

```bash
test ! -e /Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807
git status --short --branch
git rev-parse HEAD
```

- [ ] **Step 2: Create readiness once**

```bash
python3 scratch/run_exp438_frequency_complexity_evidence_antigravity.py readiness
```

Expected: 1,117 sorted datasets, current hashes, zero TEST value/label access, verified Exp437 execution, concurrency mode, 11 workers, and truthful backend assessment.

- [ ] **Step 3: Launch official run once**

```bash
nohup python3 scratch/run_exp438_frequency_complexity_evidence_antigravity.py new > /Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/launcher_antigravity.log 2>&1 &
```

Write the shell PID to the run-parent `launcher_antigravity.pid`; canonical process PID remains `output_antigravity/run_antigravity.pid`. Keeping launcher files outside `output_antigravity` also allows shell redirection before the runtime creates that directory.

- [ ] **Step 4: Perform one liveness check and hand off**

Confirm canonical live PID, fresh heartbeat, total 1,117, readiness-selected backend, error zero, worker activity, and zero label access. Report both PIDs, counts, current datasets, elapsed/ETA, and direct progress/event/log/PID/checkpoint paths. Do not start recurring monitoring.
