# Paper r4 Operational Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, observable, explicitly approved operations surface that can complete the 1,117-dataset paper proposal run without losing verified per-dataset results.

**Architecture:** Keep all scientific scoring and checkpoint semantics in the existing paper package. Add one fixed-authority operations launcher outside that package, make the public pipeline CLI expose truthful JSON/exit codes, and verify a bounded real Task4+Task5 smoke before any official launch authorization.

**Tech Stack:** Python 3.14, argparse, canonical JSON, `fcntl.flock`, subprocess/signal handling, unittest, SQLite read-only catalog, existing paper artifact verifiers.

## Global Constraints

- Do not create `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813` or start any official experiment.
- Grok writes only in its dedicated worktree; it must not mutate r3, r4, smoke, or operations roots during unit/regression implementation.
- All newly created filenames include `antigravity`.
- Resolve active Exp448 and regenerate the complete Exp448 and paper codemap triplets together before/with mapped boundary changes.
- Use strict RED → GREEN → REFACTOR. Record the exact failing reason before production code.
- Dataset workers are `max(1, logical_cpu_count - 1)` and inner numerical threads are one; neither is operator-overridable.
- No TEST-label module import is allowed before proposal freeze/evaluation.
- No automatic retry. Preserve verified successes, append failures, finish the wave, exit nonzero, then wait for analysis and explicit resume approval.
- Do not change scientific algorithms, thresholds, candidate definitions, source identities, or Task4–Task7 routing.

---

### Task 1: Freeze the CLI result and exit-code contract

**Files:**
- Modify: `paper/equipment_anomaly_detection/pipeline_paper.py`
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_artifacts_runner_paper.py`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.lock`

**Interfaces:**
- Consumes: `run_pipeline(config) -> dict[str, object]` and `verify_completed(config) -> dict[str, object]`.
- Produces: `exit_code_for_result_paper(result: Mapping[str, object]) -> int` and a `main()` that always prints canonical JSON for all three public commands.

- [ ] **Step 1: Add behavior-first CLI tests**

  Add subprocess or redirected-stdio tests proving literal JSON output and exit status: `completed` returns 0; `proposal_score_incomplete` returns 2; malformed/terminal errors propagate as nonzero. The test must call the real `pipeline_paper.main` boundary and must not assert source text.

- [ ] **Step 2: Run the focused test and capture RED**

  Run:
  `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -v`

  Expected RED: `run` emits no JSON and/or incomplete returns 0.

- [ ] **Step 3: Implement the minimum truthful CLI contract**

  Serialize with `json.dumps(value, sort_keys=True, separators=(",", ":"))`, print for every command, and return 0 only for an accepted completed state. Return 2 for `proposal_score_incomplete`; do not catch semantic exceptions as success.

- [ ] **Step 4: Run focused GREEN**

  Repeat the focused command and confirm zero failures.

### Task 2: Add the fixed-authority r4 operations launcher

**Files:**
- Create: `scratch/manage_paper_r4_proposal_antigravity.py`
- Create: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py`

**Interfaces:**
- Produces: `check`, `smoke`, `launch`, `status`, `stop`, `resume`, `_worker` argparse commands.
- Produces: canonical `paper-r4-launch-authority-antigravity.v1`, `paper-r4-ops-state-antigravity.v1`, and timestamped `paper-r4-progress-event-antigravity.v1` JSONL records.
- Consumes: fixed paths and `paper.equipment_anomaly_detection.pipeline_paper` CLI only; no label-capable module.

- [ ] **Step 1: Write authority/check RED tests**

  Test real temporary Git/DB/path fixtures for exact lowercase reviewed commit, clean HEAD, fixed interpreter/DB/root identities, closure hashes, r3 digest, r4 absence, logical CPU minus one workers, inner thread one, backend facts, symlink rejection, and no filesystem writes from `check`.

- [ ] **Step 2: Observe RED for the absent launcher**

  Run:
  `python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_operations_antigravity -v`

  Expected RED: import/file absence only.

- [ ] **Step 3: Implement authority/check minimally**

  Use immutable constants for all official paths. Hash the launcher and scoring closure. Reject unknown fields and alternate paths. Capture Python/package/platform/CPU/PyTorch/MLX/MPS facts without importing TEST labels.

- [ ] **Step 4: Add launch/liveness RED tests**

  Prove exclusive `flock` authority, metadata-only PID behavior, no duplicate worker, no root creation before launch, create-only authority, canonical environment threads, exact child command, signal relay, bounded stop escalation, and stale PID recovery.

- [ ] **Step 5: Implement launch, worker, stop, and resume minimally**

  The internal worker holds the lock for its lifetime, writes its PID only after lock acquisition, relays SIGINT/SIGTERM, streams stdout/stderr to the human log and JSONL events, and exits with the child code. `resume` requires identical authority and an incomplete verified root; it never changes closure/config.

- [ ] **Step 6: Add status RED tests**

  Exercise empty, running, incomplete, tampered, and completed fixtures through real artifact verifiers. Assert completed/failed/pending, rate/ETA semantics, lock state, and non-authoritative PID.

- [ ] **Step 7: Implement status minimally and run GREEN**

  Do not scan arbitrary files or infer success from logs. Use journal, typed success/attempt namespaces, and existing semantic verifiers.

### Task 3: Add the bounded Task4+Task5 smoke gate

**Files:**
- Modify: `scratch/manage_paper_r4_proposal_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py`

**Interfaces:**
- Consumes: exact 19-dataset readiness authority plus three exact boundary dataset IDs.
- Produces: canonical create-only smoke authority/result; official `launch` requires the verified smoke result hash but does not reuse smoke science artifacts.

- [ ] **Step 1: Write smoke selection and resume-equivalence RED tests**

  Hand-check the exact sorted 23-name tuple. Use a small real typed fixture to prove first-wave partial success, append-only failure, no sibling cancellation, second-invocation reuse, byte-identical final projection versus uninterrupted execution, zero label access, and no official-root writes.

- [ ] **Step 2: Observe RED**

  Run the operations test module and confirm failures arise from absent smoke behavior.

- [ ] **Step 3: Implement the minimum smoke command**

  Invoke the existing pipeline in `smoke` mode with the exact 23 names, fixed capable interpreter, fixed DB, separate smoke root, and exact authority. Refuse an existing mismatched smoke root. Publish a result only after journal and proposal manifest verification.

- [ ] **Step 4: Run focused GREEN**

  Run the operations module and the existing Task4/Task5 recovery tests. Do not run the real 23-dataset smoke during implementation.

### Task 4: Synchronize maps and operator documentation

**Files:**
- Modify together: `docs/codemap/codemap_exp448_antigravity.html`
- Modify together: `docs/codemap/codemap_exp448_antigravity.json`
- Modify together: `docs/codemap/codemap_exp448_antigravity.lock`
- Modify: `docs/EXP448_RUNBOOK_antigravity_KO.md`
- Modify: `docs/EXP448_IMPLEMENTATION_REPORT_antigravity.md`

**Interfaces:**
- Produces: a map that answers callers, effects, and tests for the operations launcher and updated CLI.
- Produces: exact human commands for `check`, `smoke`, `status`, `stop`, `resume`, and the final post-review `launch` approval boundary.

- [ ] **Step 1: Regenerate both complete codemap triplets**

  Ensure HTML/JSON equivalence and refresh every lock hash changed by Tasks 1–3. Do not update only one member of either triplet.

- [ ] **Step 2: Update runbook and report**

  State explicitly that implementation verification does not authorize r4 creation. Include exact log/status commands and the batch-analysis-before-resume rule.

- [ ] **Step 3: Run codemap/lineage validation**

  Run the repository validators that cover Exp448 triplet fingerprints, paper triplet equivalence, and lineage closure. Confirm zero failures.

### Task 5: Grok review, Codex verification, and launch hold

**Files:**
- Review all Task 1–4 diffs; do not create official artifacts.

**Interfaces:**
- Produces: Grok final verdict, Codex verification evidence, and fresh Sol Advisor final verdict.
- Does not produce: an r4 root, process, journal, or official result.

- [ ] **Step 1: Run focused and relevant regressions**

  At minimum run the operations module, CLI/artifact runner module, hard-route recovery tests, MiniRocket boundary tests, config/lineage tests, both codemap validators, `py_compile`, and `git diff --check`.

- [ ] **Step 2: Ask Grok for a fresh read-only review and wait**

  Require a final `SHIP` or `HOLD` verdict covering authority binding, root confinement, lock/PID semantics, signal handling, exit codes, checkpoint reuse, label boundary, and no official-root mutation. Resolve every `HOLD` item with a new RED→GREEN cycle and ask again.

- [ ] **Step 3: Codex independently verifies**

  Inspect the complete diff, rerun the commands, and verify the official r4 root remains absent and r3 digest remains unchanged.

- [ ] **Step 4: Obtain fresh Sol Advisor final review**

  Require a ship verdict on the final commit and evidence. Resolve findings before proceeding.

- [ ] **Step 5: Stop at manual approval**

  Present check/smoke readiness evidence and exact future launch command. Wait for a new explicit user approval before any smoke root or official r4 root/process creation.
