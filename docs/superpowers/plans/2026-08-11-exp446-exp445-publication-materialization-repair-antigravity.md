# Exp446 Exp445 Publication Materialization Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Native implementation tasks use `sol_advisor_terra_implementer`; commitment-boundary and final review use a fresh `sol_advisor_sol_reviewer`.

**Goal:** Exp445의 과학 계산과 30개 record를 바꾸지 않고 frozen control row의 중첩 tuple을 canonical JSON list로 변환하는 출판 결함을 고친 뒤, 새 Exp446 root에서 CPU/11로 30개를 다시 계산하여 동일 record SHA를 확인하고 label-free publication과 inspect를 완성한다.

**Architecture:** Exp444의 공유 과학 모듈과 Exp445의 검증된 lifecycle을 재사용하되, shared reporting의 JSON encoding boundary만 최소 수정한다. Exp446는 닫힌 singleton profile, 불변 Exp445 실패-root verifier, single-use runtime/publication, 세 명령 CLI로 구성한다. Exp445 root는 항상 읽기 전용이며 Exp446는 publication-only 복사나 resume 없이 30개를 처음부터 계산한다.

**Tech Stack:** Python 3, `unittest`, SQLite values-only source boundary, canonical JSON/JSONL, descriptor-relative `os.open`/`dir_fd` verification, CPU multiprocessing 11 workers, Git, self-contained HTML/JSON code map. PyTorch·MPS·MLX·Numba는 이 engineering-only replay에서 재선택하지 않는다.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-08-11-exp446-exp445-publication-materialization-repair-antigravity-design.md`
- Design commit: `ac88f553979cd73a7c210edec9107d25d4365434`
- Design SHA256: `3fce139171efbfc0a85fffeb44ea201116053efbe76e5a190e7198a3bdf3d281`
- Exp446 identity: `Exp446` / schema `exp446` / run identity `exp446-exp445-publication-materialization-repair-antigravity-20260811`.
- Official root: `/Users/mac/Documents/Dataset/exp446_exp445_publication_materialization_repair_antigravity_20260811`.
- Official backend: CPU, `outer_workers=max(1, 12-1)=11`, `inner_threads=1`.
- Exp445 failed root and Exp444 root are immutable, descriptor-pinned predecessors. Never write, rename, unlink, publish, evaluate, resume, or retry them.
- Prediction/lifecycle code must not import, open, query, or serialize TEST labels. Exp446 has no `evaluate` command.
- Every new filename contains `antigravity`.
- Before each product-code task, compare the current repository with `docs/codemap/codemap_exp446_antigravity.lock`, then use `codemap_exp446_antigravity.json` to identify callers, effects, and tests. If stale, regenerate the HTML/JSON/lock triplet together before editing.
- All long-running state uses atomic JSON plus text/event logs and checkpoint resume mechanics internally, but the public Exp446 CLI exposes no resume command.
- Focused tests run at each task. Run the combined Exp444/445/446 regression only once at the clean implementation boundary, not after every review observation.
- Official `readiness`, `new`, and `inspect` commands are forbidden until a fresh Sol High review returns ship on the exact clean reviewed commit.

---

## Task 1: Freeze Exp446 governance and generate the initial versioned code map

**Files:**

- Create: `docs/EXP446_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP446_RUNBOOK_antigravity_KO.md`
- Create: `docs/EXP446_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `FAILURE_LEDGER.json`
- Modify: `CHAMPION_LEDGER.json`
- Create together: `docs/codemap/codemap_exp446_antigravity.html`
- Create together: `docs/codemap/codemap_exp446_antigravity.json`
- Create together: `docs/codemap/codemap_exp446_antigravity.lock`
- Test/validate: JSON parsers and code-map validator commands recorded in the implementation report

### Step 1: Record the immutable Exp445 failure

Write the exact official facts from the approved design, including 30/30, error 0, label access 0, CPU/11, aggregate `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`, publication/evaluation absence, five terminal SHA values, and failure text `Exp444 original-control replay is invalid`.

The preregistration must freeze:

```text
only change = recursive JSON materialization at the encoding boundary
record parity = 30 index-wise wrapper SHA equal to Exp445 and Exp444
aggregate parity = af24dd48...abab034
commands = readiness, new, inspect
labels = 0
backend = cpu / outer 11 / inner 1
```

### Step 2: Register a non-promoted engineering experiment

Add Exp446 as `preregistered_implementation_pending`. Keep prediction, publication, evaluation, promotion, and champion hashes null. Record Exp445 as an immutable failed-publication predecessor, not a failed scientific result.

### Step 3: Create the initial Exp446 code map

Use the Exp445 map as the structural baseline, but do not add nonexistent product edges. Add the approved Exp446 design/governance node and mark product relationships unknown until code exists. Keep at most 20 primary nodes.

Validate:

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp446_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp446_antigravity.lock >/dev/null
git diff --check
```

Also validate that HTML embedded JSON is byte-identical to `codemap_exp446_antigravity.json`, all evidence paths/symbols exist, every edge/flow reference resolves, and module fingerprints match.

### Step 4: Commit governance and map together

```bash
git add docs/EXP446_PREREGISTRATION_antigravity_KO.md docs/EXP446_RUNBOOK_antigravity_KO.md docs/EXP446_IMPLEMENTATION_REPORT_antigravity.md EXPERIMENT_CATALOG.json RESEARCH_STATE.json FAILURE_LEDGER.json CHAMPION_LEDGER.json docs/codemap/codemap_exp446_antigravity.html docs/codemap/codemap_exp446_antigravity.json docs/codemap/codemap_exp446_antigravity.lock
git commit -m "docs(exp446): freeze publication materialization repair"
```

---

## Task 2: Reproduce the real publication boundary failure and apply the one-point fix

**Files:**

- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_publication_source_authority_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/reporting_antigravity.py`
- Modify together: `docs/codemap/codemap_exp446_antigravity.html`
- Modify together: `docs/codemap/codemap_exp446_antigravity.json`
- Modify together: `docs/codemap/codemap_exp446_antigravity.lock`

### Step 1: Confirm map impact before editing

Record in the task report:

1. Caller: Exp444/Exp445 publication flows call `_payloads_antigravity`.
2. Effects: control replay validation and canonical publication leaf bytes.
3. Tests: shared source-authority/reporting tests and Exp445 publication tests.

### Step 2: Write the real-boundary RED test

Extend the existing source-authority regression so it executes:

```python
values = reporting._capture_values_only_publish_authority_antigravity()
summaries = reporting._parse_dataset_summaries_antigravity(summary_bytes)
authority = reporting._compose_publish_source_authority_antigravity(summaries, values)
payloads, selection = reporting._payloads_antigravity(
    terminal_attempt,
    authority,
    terminal_snapshot,
    profile=EXP445_PROFILE_ANTIGRAVITY,
    results=frozen_results,
)
```

The fixture must use the real frozen values-only handoff and a real terminal, Exp445-compatible 30-record result set. Do not mock `_payloads_antigravity`, `_parse_control_replays_antigravity`, or `_json_payload_value_antigravity`.

Required RED:

```text
ValueError: Exp444 original-control replay is invalid
```

Run only the new test first and save the exact RED output in the task report.

### Step 3: Implement the minimal materialization boundary

Replace the outer-only conversion with one recursively materialized value:

```python
control_rows = _json_payload_value_antigravity(authority.control_rows)
_parse_control_replays_antigravity(control_rows, source)
```

Use the same `control_rows` object in `CONTROL_REPLAY_FILE_ANTIGRAVITY` canonical encoding. Do not change the parser to accept tuple and do not change source, score, rank, lane, threshold, mask, view, or selection logic.

### Step 4: Prove GREEN and byte semantics

Assertions must show:

- 30 control rows parse successfully.
- `depth_scores` and `replayed_depths` are JSON lists.
- coordinate, rank, agreeing depths, empirical ranks, and excess ratios are unchanged.
- the frozen handoff remains deeply immutable.
- shared public wrapper signatures remain unchanged.

Run:

```bash
python3 -m unittest -v experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_publication_source_authority_antigravity
python3 -m unittest -v experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_reporting_antigravity experiments.exp445_exp444_publication_source_authority_repair_antigravity.tests.test_publication_antigravity
```

### Step 5: Regenerate the three Exp446 map files and commit

Update the reporting evidence symbol and test edge, validate the triplet, then commit only this bounded fix and its map:

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/reporting_antigravity.py experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_publication_source_authority_antigravity.py docs/codemap/codemap_exp446_antigravity.html docs/codemap/codemap_exp446_antigravity.json docs/codemap/codemap_exp446_antigravity.lock
git commit -m "fix(exp446): materialize frozen control rows"
```

Request one fresh Sol High commitment-boundary review of this exact clean commit. Fix only Critical/Important findings that directly affect the encoding boundary or immutable artifact contract.

---

## Task 3: Add the closed Exp446 shared profile and runtime contracts

**Files:**

- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/profiles_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/contracts_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/readiness_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/runtime_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/reporting_antigravity.py`
- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/tests/test_profile_contracts_antigravity.py`
- Modify together: `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

### Step 1: Write closed-profile RED tests

Tests must require an exact singleton:

```python
EXP446_PROFILE_ANTIGRAVITY = LifecycleProfileAntigravity(
    experiment_id="Exp446",
    schema="exp446",
    run_identity="exp446-exp445-publication-materialization-repair-antigravity-20260811",
    run_root=Path("/Users/mac/Documents/Dataset/exp446_exp445_publication_materialization_repair_antigravity_20260811"),
    reviewed_design_path=...,
    preregistration_path=...,
    predecessor_required=True,
)
```

RED cases: caller-minted equal object, altered root/schema/worker/backend, Exp445 context reused as Exp446, label-capable field, and arbitrary profile dispatch.

### Step 2: Extend shared internal profile acceptance only

Allow exactly three reviewed singleton identities. Keep Exp444 and Exp445 public wrappers byte/signature compatible. Extend internal checkpoint context serialization with Exp446 schema/document authority without changing legacy payloads.

### Step 3: Bind readiness and runtime invariants

Exp446 readiness must check, before root allocation:

- reviewed commit equals current clean HEAD;
- design/prereg/environment/scientific hashes;
- 12 logical CPUs, CPU backend, outer 11, inner 1;
- DB inode/link authority and label access 0;
- Exp445 immutable predecessor verifier success;
- Exp446 root absent and not a symlink.

Runtime accepts only `mode="new"`, the fixed root, and `attempt_0000_antigravity`. It rejects caller-supplied root/attempt/backend/worker/source overrides.

### Step 4: Run focused contracts

```bash
python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_profile_contracts_antigravity
python3 -m unittest -v experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_profiles_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity
```

Regenerate the map triplet and commit:

```bash
git commit -m "feat(exp446): add closed lifecycle profile"
```

---

## Task 4: Verify the immutable failed Exp445 predecessor

**Files:**

- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/profile_antigravity.py`
- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/predecessor_antigravity.py`
- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/tests/test_predecessor_antigravity.py`
- Modify together: `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

### Step 1: Freeze exact predecessor constants

Include the official Exp445 root, attempt segments, all five terminal/failure SHA values, 30 record-wrapper SHA values, aggregate SHA, expected state/progress/PID semantics, publication/evaluation absence, and source DB identity requirements.

### Step 2: Write descriptor-adversarial RED tests

Cover exact inventory, no-follow root/attempt/records/leaves, single-link source DB, pre/post inode+mode+nlink+size+mtime+ctime identity, terminal complete/inactive state, 30 ordered records, aggregate SHA, publication absence, evaluation absence, and exact immutable failure marker.

Mutation cases must include symlink, hardlink, rename, same-size in-place overwrite, extra leaf, missing/reordered record, altered failure text/SHA, live PID, and publication/evaluation appearance. Verification must never write.

### Step 3: Implement values-only immutable authority

Return a frozen authority containing the 30 expected record wrapper SHA values and aggregate. Do not return writable paths or open descriptors.

### Step 4: Verify twice and commit

```bash
python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_predecessor_antigravity
python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_predecessor_antigravity
```

Then perform two read-only real predecessor calls, confirm identical returned authority, DB nlink 1, and Exp446 root absence. Regenerate the map and commit:

```bash
git commit -m "feat(exp446): verify immutable exp445 failure"
```

---

## Task 5: Implement Exp446 single-use recomputation and parity gates

**Files:**

- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/lifecycle_antigravity.py`
- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/tests/test_lifecycle_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/runtime_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/checkpoint_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/progress_antigravity.py`
- Modify together: `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

### Step 1: Write lifecycle RED tests

The new lifecycle facade exposes only:

```python
run_readiness_antigravity(reviewed_commit: str)
run_new_antigravity(reviewed_commit: str)
publish_prediction_antigravity()
inspect_prediction_antigravity()
```

Test an in-memory/temp-root run with 30 ordered records. Before every success checkpoint write, compare the canonical wrapper SHA against both Exp445 and Exp444 authority at the same index. Before terminal complete, compare the ordered aggregate.

RED cases: index 0/15/29 mismatch, aggregate mismatch, predecessor changes after readiness, root exists/symlink, wrong commit, worker not 11, label counter nonzero, and any publication attempt before terminal parity.

### Step 2: Generalize the existing Exp445 parity-gated runtime narrowly

Use an internal predecessor-authority protocol selected only by the closed profile. Exp445 keeps its existing verifier and behavior; Exp446 uses the failed-Exp445 verifier. Do not duplicate scientific scoring modules.

The sequence is fixed:

```text
fresh predecessor verification
→ allocate new root once
→ ordered CPU/11 dispatch
→ index SHA gate before success checkpoint
→ aggregate gate before terminal complete
→ fresh predecessor verification
→ publication eligibility
```

### Step 3: Preserve failure truth

Any post-allocation mismatch writes the exact next error record, inactive progress/PID, immutable failure evidence, and no publication. No same-root retry path is exposed.

### Step 4: Verify runtime behavior

```bash
python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_lifecycle_antigravity
python3 -m unittest -v experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_progress_antigravity experiments.exp445_exp444_publication_source_authority_repair_antigravity.tests.test_lifecycle_antigravity
```

Regenerate the map and commit:

```bash
git commit -m "feat(exp446): add parity-gated recomputation"
```

---

## Task 6: Add Exp446 publication, source-free inspection, and three-command CLI

**Files:**

- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/reporting_antigravity.py`
- Modify: `experiments/exp446_exp445_publication_materialization_repair_antigravity/lifecycle_antigravity.py`
- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/cli_antigravity.py`
- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/tests/test_publication_antigravity.py`
- Create: `experiments/exp446_exp445_publication_materialization_repair_antigravity/tests/test_cli_antigravity.py`
- Create: `scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py`
- Modify together: `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

### Step 1: Write publication and CLI RED tests

The parser accepts exactly:

```text
readiness --reviewed-commit <40 lowercase hex>
new --reviewed-commit <40 lowercase hex>
inspect
```

Reject `resume`, `retry`, `evaluate`, root/path/backend/worker/source overrides, and extra arguments. An inspect subprocess must prove that SQLite label/evaluation modules are not imported.

### Step 2: Add closed Exp446 reporting entrypoints

Implement private profile-aware publication/inspection while preserving Exp444/445 wrappers. Required ordering:

```text
terminal checkpoint snapshot
→ fresh Exp445 failed-root verification
→ 30-wrapper and aggregate parity
→ fresh descriptor-pinned provenance and values-only source capture
→ materialized control validation
→ create-only staging
→ atomic rename
→ frozen-leaf replay audit
```

Capture/staging/post-rename failures must leave immutable failure evidence; a failed post-rename audit quarantines the canonical publication path. Source-free inspect opens only frozen publication leaves.

### Step 3: Add an outside-CWD-safe launcher

Set all seven numerical thread environment variables to `1` before importing NumPy/scientific modules. The launcher delegates only to the fixed Exp446 CLI.

### Step 4: Run focused tests

```bash
python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_publication_antigravity experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_cli_antigravity
python3 -m unittest -v experiments.exp445_exp444_publication_source_authority_repair_antigravity.tests.test_publication_antigravity experiments.exp445_exp444_publication_source_authority_repair_antigravity.tests.test_cli_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_reporting_antigravity
```

Regenerate the map and commit:

```bash
git commit -m "feat(exp446): add closed publication lifecycle"
```

---

## Task 7: Final implementation verification and clean-candidate review

**Files:**

- Modify: `docs/EXP446_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `docs/EXP446_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

### Step 1: Run deterministic sequential/11-worker parity

Use the same 30 ordered candidates, input bytes, seed, scientific config, and environment lock. Assert every record wrapper byte/SHA and aggregate SHA are identical. Record CPU model, 12 logical CPUs, worker 11, inner 1, elapsed timings, and label access 0.

### Step 2: Run one combined regression boundary

Run each discovery once on the final functional tree and capture exact counts/exit codes:

```bash
python3 -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p "test_*_antigravity.py"
python3 -m unittest discover -s experiments/exp445_exp444_publication_source_authority_repair_antigravity/tests -p "test_*_antigravity.py"
python3 -m unittest discover -s experiments/exp446_exp445_publication_materialization_repair_antigravity/tests -p "test_*_antigravity.py"
```

Also run `compileall`, JSON parse, `git diff --check`, DB inode/nlink check, Exp445 frozen-authority verification, Exp446 root-absence check, and complete code-map validation.

### Step 3: Commit the clean implementation candidate

Update status to `implementation_verified_awaiting_official_readiness`; keep official execution/publication hashes null and promotion false.

```bash
git commit -m "docs(exp446): record verified implementation candidate"
```

Regenerate the map from that clean functional/docs commit and make one map-only commit if the lock self-reference convention requires it.

### Step 4: Fresh Sol High final review

Review the exact clean candidate commit for:

- the approved design and preregistration hashes;
- one-point materialization fix;
- no parser weakening;
- Exp445 immutable predecessor and Exp446 root absence;
- exact CPU11/label0 routing;
- per-record and aggregate parity gates;
- no evaluator/import/command;
- source-free inspect and failure preservation;
- code-map truth.

Only a `ship` verdict permits Task 8. Fix Critical/Important findings once, rerun only affected focused tests, then rerun the combined regression once if product code changed.

---

## Task 8: Run official Exp446 once, inspect twice, record truth, and stop

**Files:**

- Modify after execution: `docs/EXP446_IMPLEMENTATION_REPORT_antigravity.md`
- Modify after execution: `docs/EXP446_RUNBOOK_antigravity_KO.md`
- Modify after execution: `EXPERIMENT_CATALOG.json`
- Modify after execution: `RESEARCH_STATE.json`
- Modify after execution: `FAILURE_LEDGER.json`
- Modify after execution: `CHAMPION_LEDGER.json`
- Modify together after truth update: `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

### Step 1: Read-only preflight

Confirm exact reviewed clean HEAD, frozen document hashes, DB nlink 1, immutable Exp445 authority, no active related PID, and absent Exp446 root. If any check fails, stop without creating the root and record the exact blocker.

### Step 2: Run official readiness exactly once

```bash
python3 scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py readiness --reviewed-commit <EXACT_CLEAN_REVIEWED_HEAD>
```

Require CPU/11, inner1, labels0, predecessor valid, root absent.

### Step 3: Run official new exactly once

```bash
python3 scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py new --reviewed-commit <EXACT_CLEAN_REVIEWED_HEAD>
```

The process writes its own PID, current candidate, sizes, stage/backend, completed/30, success/error, speed, elapsed, ETA, checkpoint, and latest error to atomic JSON and text logs. Do not token-poll it; report the PID and log paths, then wait for the user to say it finished if it remains long-running.

Success requires 30/30, error0, labels0, 30 index-wise SHA equality, aggregate `af24dd48...abab034`, and create-only publication.

### Step 4: Inspect twice without sources or labels

```bash
python3 scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py inspect
python3 scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py inspect
```

Require byte-identical manifest/publication/prediction SHA results and confirm no evaluation artifact or label access.

### Step 5: Record final truth and stop for discussion

Update governance files with exact execution commit, PID, elapsed time, terminal SHA, 30 record SHA/aggregate, publication/prediction SHA, two inspect results, label count, and any failure. Do not evaluate or promote. Regenerate the code-map triplet and commit:

```bash
git commit -m "docs(exp446): record official publication repair result"
```

Stop and report the result to the user. Any improvement/research direction after this engineering repair requires a new experiment number and preregistration.

---

## Plan self-review checklist

- [ ] Every approved design requirement maps to a task and test.
- [ ] No unresolved placeholder, placeholder hash, or undefined alternate route remains before official readiness.
- [ ] Exp445 failed root is read-only and Exp446 gets a new single-use root.
- [ ] The parser remains strict; recursive materialization occurs only at JSON encoding.
- [ ] Exp446 recomputes 30 records on CPU/11 and compares every SHA before publication.
- [ ] Exp446 CLI has exactly three commands and no evaluation import.
- [ ] Long-run logs are process-written; Codex does not repeatedly poll.
- [ ] Full regression and fresh final review occur once at the clean boundary.
- [ ] Versioned Exp446 code-map HTML/JSON/lock are always regenerated together.
- [ ] Official execution ends after two label-free inspections and truth recording.
