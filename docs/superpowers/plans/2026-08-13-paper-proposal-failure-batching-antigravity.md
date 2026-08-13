# Paper Proposal Failure Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the 1,117-dataset paper proposal run preserve every successful Task4+5 result, quarantine dataset-local failures, and retry only the exact failed/unattempted queue before global Task6/7 and final freeze.

**Architecture:** Restore the frozen Exp249 linear-edge/nearest nonfinite repair only in the MiniRocket input path. Add context-bound create-only Task4+5 work records and an append-only attempt ledger under a registered incomplete-stage namespace. Each wave processes every pending dataset without cancelling siblings; Task6/7 and journal completion remain fail-closed until a sorted, exactly complete population is reconstructed through typed validators.

**Tech Stack:** Python 3.14, NumPy, aeon MiniRocket, canonical JSON/SHA-256, `ProcessPoolExecutor`, `unittest`.

## Global Constraints

- Do not read TEST labels or evaluation artifacts in preflight, scoring, checkpoint, retry, or freeze.
- Preserve the frozen Task4, Task5, Task6, and Task7 mathematics; only restore the proven Exp249 MiniRocket preprocessing omission.
- Successful work records are create-only and context-bound; failed attempts are append-only operational evidence.
- Do not mark `proposal_score` complete, run Task6/7, or create final proposal shards until all catalog datasets have verified Task4+5 success records.
- Same-context retries may reuse successes. Any code, config, catalog, DB, backend, environment, or context change requires a fresh root; never migrate r3 work into r4.
- Preserve `/Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813` unchanged as failed evidence.
- New repository filenames must include `antigravity`; existing paper module names remain unchanged.
- CPU dataset workers remain the verified logical-CPU-minus-one policy; inner numerical threads remain one.

---

### Task 1: Restore the frozen Exp249 MiniRocket nonfinite preprocessing

**Files:**
- Modify: `paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py`
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_minirocket_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/source_provenance_paper.json`

**Interfaces:**
- Produce: `repair_linear_edge_nearest_paper(values: np.ndarray) -> np.ndarray`.
- `score_minirocket_confirmations(values)` repairs every TRAIN and TEST row before alignment and z-normalization.
- Empty or all-nonfinite rows fail with a typed, dataset-local exception; other hard sources are unchanged.

- [ ] Add RED tests proving the current paper path fails on the exact internal/edge NaN patterns and differs from Exp443 Policy A.
- [ ] Run the focused tests and capture the failure.
- [ ] Copy the Exp249/Exp443 control formula: interpolate internal nonfinite positions on the original grid and extend edges with the nearest finite value, preserving `<f4`.
- [ ] Add an exact parity test against `repair_linear_edge_nearest_antigravity` and a capable-interpreter `DodgerLoopDay_normal_1` score smoke.
- [ ] Run the MiniRocket and provenance suites; update only the helper provenance hash required by the changed paper module.
- [ ] Commit the preprocessing restoration.

### Task 2: Add typed Task4+5 work records

**Files:**
- Modify: `paper/equipment_anomaly_detection/artifacts_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_artifacts_runner_paper.py`

**Interfaces:**
- Produce create-only per-dataset work records under `stage_artifacts_paper/proposal_work/task4_task5_success/`.
- Each record binds schema, context SHA, catalog SHA, raw TRAIN/TEST identities, backend record, scientific configuration/closure SHA, dataset, complete typed `HardSourceBundle`, complete typed `HardRouteResult`, and payload SHA.
- Produce a strict reader that reconstructs the same typed objects, including TRAIN OOF arrays, and round-trips canonical bytes.

- [ ] Add RED round-trip tests covering eligible/abstained sources, every route stage, tampered arrays, wrong context/backend/raw identity, unknown/extra files, and symlink/path attacks.
- [ ] Run the focused tests and capture failures.
- [ ] Implement canonical serialization/deserialization using existing validators and derived frozen lineage/threshold identities; do not use pickle.
- [ ] Write success records only in the parent process after a worker returns and after an encode/decode semantic equality check.
- [ ] Prove a reread result gives byte-identical downstream final shard bytes to the in-memory result.
- [ ] Run focused tests and commit.

### Task 3: Continue after failures and retry only the exact queue

**Files:**
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `paper/equipment_anomaly_detection/runner_paper.py`
- Modify: `paper/equipment_anomaly_detection/pipeline_paper.py`
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_artifacts_runner_paper.py`

**Interfaces:**
- `_collect_task4_task5` accepts root/context/catalog and reuses verified successes.
- Worker exceptions append one canonical attempt record under `proposal_work/task4_task5_attempts/`; remaining futures continue.
- Queue identity is exactly sorted `(catalog datasets - verified successes)` and includes failed plus never-attempted coordinates.
- `run` returns `proposal_score_incomplete` with completed/failed/pending counts and paths; a later `run` retries only the queue.

- [ ] Add RED tests where one of several datasets fails while later datasets succeed and are persisted.
- [ ] Add RED tests that a second invocation does not call successful coordinates and retries only failed/unattempted coordinates.
- [ ] Implement parent-only success publication, append-only attempts, no sibling cancellation, and exact end-of-wave queue reporting.
- [ ] Block Task6/7, final shards, proposal manifest, proposal journal record, and `completed` status while the queue is nonempty.
- [ ] Ensure `verify_journal` accepts only the registered, strictly verified incomplete work namespace and rejects pollution/tampering.
- [ ] Run focused tests and commit.

### Task 4: Prove resume equivalence and population gates

**Files:**
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_artifacts_runner_paper.py`
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_minirocket_paper.py`

**Interfaces:**
- Interrupted/resumed and uninterrupted paths must create byte-identical final shards and manifest.
- Global Task6 and Task7 execute exactly once and only after exact catalog coverage.

- [ ] Add an interruption fixture that persists a strict prefix, fails selected coordinates, then resumes.
- [ ] Compare all final shard bytes, shard hashes, manifest bytes, policy coordinates, continuous scores, and stage order against an uninterrupted run.
- [ ] Add a 19-affected-dataset capable parity gate using the frozen Exp443 Policy A/Exp249 evidence; require finite outputs, score tolerance `1e-6`, and identical candidates at every frozen alpha.
- [ ] Add full-population preflight inventory assertions: 19 affected datasets, 164 affected normal TRAIN rows, 983 nonfinite scalars, zero all-missing rows.
- [ ] Run focused, inherited Task4–9, lineage, and capable MiniRocket suites; commit.

### Task 5: Regenerate governance and prepare r4 without launching it

**Files:**
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.html`, `.json`, `.lock`
- Modify together: `docs/codemap/codemap_exp448_antigravity.html`, `.json`, `.lock`
- Modify: paper lineage/provenance locks required by changed scientific helper bytes
- Modify: relevant paper runbook/implementation report

**Interfaces:**
- Codemaps must answer callers, effects, and tests for preprocessing, MiniRocket, work records, proposal scoring, runner, and CLI.
- Documentation names a new r4 root but does not create or launch it.

- [ ] Regenerate both complete codemap triplets and validate all fingerprints, edge references, flow references, and embedded HTML JSON identity.
- [ ] Validate lineage/source provenance without importing legacy side-effect modules.
- [ ] Verify r3 tree/hash is unchanged and record it as failed evidence.
- [ ] Verify the proposed r4 root is absent and no proposal process is running.
- [ ] Run `git diff --check`, full relevant tests, and a fresh Sol Advisor final review before any r4 launch.
- [ ] Commit the final governance/docs update.

## Self-review

- Spec coverage: known nonfinite failure, fail-later batching, exact retry queue, strict completion gate, same-context reuse, r3 preservation, parity, and byte-identical resume are each assigned above.
- Placeholder scan: no implementation step delegates unspecified error handling or tests.
- Type consistency: Task 2 work records reconstruct `HardSourceBundle` and `HardRouteResult`, which Task 3 supplies unchanged to the existing population-global Task6/7 composition.
