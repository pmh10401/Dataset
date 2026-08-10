# Exp445 Exp444 Publication Source Authority Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute Exp444's frozen 30 label-free records under a new Exp445 identity, prove byte-for-byte record parity with the immutable failed predecessor, and publish/evaluate only after repairing the descriptor-pinned dataset-summary authority boundary.

**Architecture:** Keep every Exp444 scientific module and public wrapper compatible. Add one closed lifecycle-profile layer inside the Exp444 package, then implement Exp445 as a thin package containing only its fixed profile, immutable predecessor verifier, lifecycle facade, and four-command CLI. The shared publication path receives the already descriptor-pinned Exp440 summary bytes from provenance; it never derives an Exp442 sibling path or queries the full database. Exp445 creates a fresh root, recomputes all 30 records with CPU `max(1, logical CPU count - 1)` = 11 on this Mac, compares every ordered record SHA before publication, and permits TEST labels only after a frozen publication is independently inspected twice.

**Tech Stack:** Python 3.14.6, NumPy 2.4.6, existing Exp444 isolated environment and lock SHA256 `8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94`, `unittest`, immutable/read-only descriptor pinning, `ProcessPoolExecutor` with 11 outer workers and one inner numerical thread, canonical JSON/JSONL, SHA256, create-only artifact publication, and versioned self-contained code maps.

## Global Constraints

- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Approved design path is `docs/superpowers/specs/2026-08-10-exp445-exp444-publication-source-authority-repair-antigravity-design.md`. Compute and freeze its final SHA only after the acceleration wording and this plan are accepted.
- Experiment identity is `Exp445`; schema is `exp445`; run identity is `exp445-exp444-publication-source-authority-repair-antigravity-20260810`.
- Fixed run root is `/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810`; it must be absent before official readiness.
- Exp444 root `/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810` is immutable. Never create `attempt_0001_antigravity`, a publication, an evaluation, or a failure marker there.
- The Exp444 predecessor must remain complete/inactive with 30/30, zero errors, zero labels, no publication/evaluation, database link count one, and the exact terminal hashes declared by the design.
- Ordered predecessor record-prefix payload SHA256 is `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`; all 30 individual record SHA values are binding.
- Inherited scientific configuration SHA256 remains `8ba0a9524600bef7479df0626834bfeef4b3410f64d3f2d1cbd3885c784eb11a`. Identity/root/document authority belongs to the lifecycle profile and must not alter scientific result bytes.
- Exp445 is an engineering-only repair, not a new score source. Do not change candidates, TRAIN values, TEST values, features, masks, seeds, views, thresholds, scores, lane rules, ordering, or evaluation formulas.
- Official workers are `max(1, os.cpu_count() - 1)`, expected 11 from 12 logical CPUs; inner numerical threads are one. Worker count is not a CLI override.
- Do not reopen backend selection. Revalidate the existing CPU/11 decision and sequential/11-worker byte parity. PyTorch/PyTorch-MPS are unavailable and inapplicable under the frozen lock; MLX exact-float64 failed; Numba measured `0.9968x`, below 1.1x.
- Prediction and publication must never import/open/query TEST labels, actual anomaly positions, family TEST performance, or prior TEST metrics. Evaluation imports the label-capable module only after two independent publication inspections.
- `readiness` and `new` accept only `--reviewed-commit`; `inspect` and `evaluate` accept no scientific/path/runtime overrides. Exp445 has no `resume` command.
- Readiness and `new` failures before Exp445 root allocation are mutation-free and leave the fixed root absent. After allocation, record, failure, publication, and evaluation artifacts are create-only; same-run checkpoint state, progress, and PID retain the inherited atomic-replacement protocol. Any post-allocation failure preserves the new Exp445 failed root and forbids same-root retry.
- Exp445 stays review-only, descriptive-only, not operationally promotable, and not a champion candidate.
- Before each boundary-changing task, inspect `docs/codemap/codemap_exp445_antigravity.json` for callers, effects, and tests. Regenerate the HTML/JSON/lock triplet together in the same commit as each boundary change.
- Do not run official lifecycle commands or access TEST labels until a fresh Sol High review passes on a clean reviewed commit.

## File Structure

- Modify `experiments/exp444_train_only_stability_precision_gate_antigravity/config_antigravity.py`, `contracts_antigravity.py`, `provenance_antigravity.py`, `readiness_antigravity.py`, `runtime_antigravity.py`, `reporting_antigravity.py`, `evaluation_antigravity.py`, and `cli_antigravity.py` only to accept a closed profile internally while preserving Exp444 public wrappers.
- Create `experiments/exp444_train_only_stability_precision_gate_antigravity/profiles_antigravity.py` for exactly two fixed lifecycle profiles and no arbitrary profile constructor exposed to the CLI.
- Create `experiments/exp445_exp444_publication_source_authority_repair_antigravity/profile_antigravity.py`, `predecessor_antigravity.py`, `lifecycle_antigravity.py`, and `cli_antigravity.py` as the thin Exp445 package.
- Create focused `test_*_antigravity.py` files under both experiment test directories; extend existing Exp444 tests only when verifying wrapper compatibility.
- Create `scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py` as the only operator launcher.
- Create `docs/EXP445_PREREGISTRATION_antigravity_KO.md`, `docs/EXP445_RUNBOOK_antigravity_KO.md`, and `docs/EXP445_IMPLEMENTATION_REPORT_antigravity.md`.
- Update `EXPERIMENT_CATALOG.json`, `RESEARCH_STATE.json`, `FAILURE_LEDGER.json` only on an actual failure, and `CHAMPION_LEDGER.json` with non-promotion history only.
- Regenerate `docs/codemap/codemap_exp445_antigravity.html`, `.json`, and `.lock` together.

---

### Task 1: Freeze Exp445 governance and operator contract

**Files:**
- Create: `docs/EXP445_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP445_RUNBOOK_antigravity_KO.md`
- Create: `docs/EXP445_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify together: the Exp445 code-map triplet

**Interfaces:** The documents freeze the four commands, label firewall, immutable predecessor, exact record-parity gate, CPU/11 topology, failure preservation, and descriptive-only governance.

- [ ] **Step 1: Verify the approved design, clean tree, and current map**

```bash
git status --short --branch
shasum -a 256 docs/superpowers/specs/2026-08-10-exp445-exp444-publication-source-authority-repair-antigravity-design.md
python3 -m json.tool docs/codemap/codemap_exp445_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp445_antigravity.lock >/dev/null
```

Expected: only the explicitly approved planning/document changes are present; map files parse.

- [ ] **Step 2: Write preregistration and runbook before product code**

Include the exact identity/root/hashes, all five predecessor conditions, `max(1, logical CPU - 1)=11`, the inherited backend authorities, the four commands, precise log/checkpoint/publication/evaluation paths, and the literal governance block from the design. State that `new` is single-use and no resume exists.

- [ ] **Step 3: Register design-only state without promotion**

Add Exp445 as `design_approved_implementation_planned`, keep prediction/evaluation hashes null, keep `CHAMPION_LEDGER.pending_candidate` empty, and set `RESEARCH_STATE` to implementation planning. Do not change the verified champion.

- [ ] **Step 4: Regenerate map, validate, and commit**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp445_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp445_antigravity.lock >/dev/null
git diff --check
git add docs/EXP445_PREREGISTRATION_antigravity_KO.md docs/EXP445_RUNBOOK_antigravity_KO.md docs/EXP445_IMPLEMENTATION_REPORT_antigravity.md EXPERIMENT_CATALOG.json RESEARCH_STATE.json CHAMPION_LEDGER.json docs/codemap
git commit -m "docs(exp445): freeze publication repair governance"
```

---

### Task 2: Add closed lifecycle profiles without changing Exp444 wrappers

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/profiles_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_profiles_antigravity.py`
- Modify: `config_antigravity.py`, `contracts_antigravity.py`
- Modify: existing Exp444 config/contracts/CLI tests as needed
- Modify together: the Exp445 code-map triplet

**Interfaces:**

```python
@dataclass(frozen=True)
class LifecycleProfileAntigravity:
    experiment_id: str
    schema: str
    run_identity: str
    run_root: Path
    reviewed_design_path: Path
    preregistration_path: Path
    predecessor_required: bool

EXP444_PROFILE_ANTIGRAVITY: LifecycleProfileAntigravity
EXP445_PROFILE_ANTIGRAVITY: LifecycleProfileAntigravity
```

- [ ] **Step 1: RED — prove identity/root separation and wrapper compatibility**

Tests must reject arbitrary profile/root/schema values, prove exactly two closed profiles, prove each fixed root differs, and snapshot existing Exp444 public call signatures.

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_profiles_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_config_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_cli_antigravity
```

Expected RED: profile module is absent.

- [ ] **Step 2: GREEN — implement only identity/document/root indirection**

Keep `FROZEN_CONFIG` and `scientific_config_sha256_antigravity()` byte-identical. Add profile-aware internal validators; Exp444 public constructors/wrappers continue selecting only `EXP444_PROFILE_ANTIGRAVITY`.

- [ ] **Step 3: Verify frozen science and commit with map**

Assert scientific SHA remains `8ba0a9524600bef7479df0626834bfeef4b3410f64d3f2d1cbd3885c784eb11a`, run the four test modules, regenerate map triplet, `git diff --check`, and commit `refactor(exp445): add closed lifecycle profiles`.

---

### Task 3: Repair publication summary authority by composition

**Files:**
- Modify: `provenance_antigravity.py`, `reporting_antigravity.py`, `runtime_antigravity.py`
- Create: `tests/test_publication_source_authority_antigravity.py`
- Modify: `tests/test_provenance_antigravity.py`, `test_reporting_antigravity.py`, `test_runtime_antigravity.py`
- Modify together: the Exp445 code-map triplet

**Interfaces:**

```python
_parse_dataset_summaries_antigravity: Callable[
    [bytes], tuple[dict[str, object], ...]
]
_capture_values_only_publish_authority_antigravity: Callable[
    [], _ValuesOnlyPublishAuthorityAntigravity
]
_compose_publish_source_authority_antigravity: Callable[
    [tuple[dict[str, object], ...], _ValuesOnlyPublishAuthorityAntigravity],
    _PublishSourceAuthorityAntigravity,
]
```

- [ ] **Step 1: RED — reproduce the real composed failure**

Write a test that calls real provenance capture, pure summary parsing, values-only source replay, and final source-authority composition without mocking any boundary. Add spies that fail on Exp442 summary path access, SQLite full query, fallback, or synthesis. Add malformed/noncanonical/wrong-order/wrong-count/label-field summary-byte cases. Prove that the values-only session is closed before composition begins.

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_publication_source_authority_antigravity
```

Expected RED: current no-argument helper derives the absent Exp442 sibling.

- [ ] **Step 2: GREEN — pass descriptor-pinned Exp440 bytes**

Use `Exp444ProvenanceAntigravity.dataset_summary_bytes` captured from the retained Exp440 directory descriptor. The pure parser reads only those bytes and validates 1,117 sorted unique label-free rows. A separate existing-values-only replay captures candidates, source hashes, and control rows, proves label access zero, closes its session, and returns an immutable handoff. The compose helper combines only the parsed rows and closed handoff; neither parser nor composer opens paths or SQLite. Lifecycle passes these private values; public APIs expose no source override.

- [ ] **Step 3: Regression and commit**

Run provenance/reporting/runtime tests plus the new composed test. Verify `rg -n "_exp442_dataset_summaries_antigravity"` returns no production definition/call. Regenerate map triplet and commit `fix(exp445): compose frozen summary authority`.

---

### Task 4: Authenticate the immutable Exp444 predecessor

**Files:**
- Create: `experiments/exp445_exp444_publication_source_authority_repair_antigravity/profile_antigravity.py`
- Create: `predecessor_antigravity.py`
- Create: `tests/test_predecessor_antigravity.py`
- Modify together: the Exp445 code-map triplet

**Interfaces:**

```python
@dataclass(frozen=True)
class Exp444PredecessorAuthorityAntigravity:
    reviewed_commit: str
    attempt_root: Path
    record_sha256_prefix: tuple[str, ...]
    record_prefix_payload_sha256: str

verify_exp444_predecessor_antigravity: Callable[
    [], Exp444PredecessorAuthorityAntigravity
]
```

- [ ] **Step 1: RED — enumerate every rejection boundary**

Use temporary fixtures to reject wrong terminal hashes, non-regular/hardlinked/replaced leaves, wrong context/config/workers, running PID, non-complete state, any error, non-30 prefix, record mismatch, existing Exp444 publication/evaluation, wrong database link count, and any attempted write.

- [ ] **Step 2: GREEN — descriptor-pin and validate read-only**

Read the exact `attempt_0000_antigravity` inventory with descriptor-relative operations, compare all design-declared file SHA values, validate record wrapper SHA at every index, compute the aggregate prefix payload SHA, and return an immutable authority. Never call Exp444 publication or resume paths.

- [ ] **Step 3: Test and commit**

Run predecessor tests twice, verify fixture bytes are unchanged, regenerate map triplet, and commit `feat(exp445): verify immutable predecessor authority`.

---

### Task 5: Make readiness and runtime profile-aware while preserving record bytes

**Files:**
- Modify: Exp444 `readiness_antigravity.py`, `runtime_antigravity.py`, `checkpoint_antigravity.py`, `progress_antigravity.py`
- Create: Exp445 `lifecycle_antigravity.py`, `tests/test_readiness_antigravity.py`, `tests/test_lifecycle_antigravity.py`
- Modify: Exp444 readiness/runtime/checkpoint/progress tests
- Modify together: the Exp445 code-map triplet

**Interfaces:**

```python
readiness_for_profile_antigravity: Callable[
    [LifecycleProfileAntigravity, str], ReadinessEvidenceAntigravity
]
run_new_for_profile_antigravity: Callable[
    [LifecycleProfileAntigravity, str], RuntimeOutcomeAntigravity
]
```

- [ ] **Step 1: RED — bind Exp445 context and forbid overrides**

Test fixed Exp445 identity/root/docs, root absence, clean reviewed HEAD, 12 detected CPUs/11 workers/inner-1, inherited environment/science/backend hashes, zero labels, and verified predecessor. Reject explicit root, attempt, worker, backend, resume, or mismatched commit.

- [ ] **Step 2: GREEN — thread the closed profile internally**

Keep existing `run_new_antigravity()` and readiness wrappers selecting Exp444. Exp445 lifecycle selects only `EXP445_PROFILE_ANTIGRAVITY`. Checkpoint context records Exp445 identity separately while candidate record serialization stays exactly `{"sequence": sequence, "result": result.to_payload()}`.

- [ ] **Step 3: Add deterministic parity gate before terminal completion**

Before each candidate record is written, canonicalize `{"sequence": sequence, "result": result.to_payload()}` and compare its SHA with the same predecessor index. Before the 30th result commit, revalidate the immutable predecessor and aggregate prefix SHA `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`. On mismatch, write the exact next-sequence immutable error record, close progress failed, and never create terminal-complete state or publication staging. Only 30 successful pre-write comparisons may cause checkpoint/progress completion; no completed file is rewritten.

- [ ] **Step 4: Run focused and inherited tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp445_exp444_publication_source_authority_repair_antigravity.tests.test_readiness_antigravity \
  experiments.exp445_exp444_publication_source_authority_repair_antigravity.tests.test_lifecycle_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_progress_antigravity
```

Regenerate map and commit `feat(exp445): add parity-gated new lifecycle`.

---

### Task 6: Publish only after source and predecessor parity gates

**Files:**
- Modify: Exp444 `reporting_antigravity.py`
- Modify: Exp445 `lifecycle_antigravity.py`
- Create: Exp445 `tests/test_publication_antigravity.py`
- Modify: Exp444 reporting tests
- Modify together: the Exp445 code-map triplet

- [ ] **Step 1: RED — prove publication ordering**

Call spies must require: complete checkpoint → predecessor revalidation → 30-record parity → descriptor-pinned Exp440 summary revalidation → create-only staging → atomic publication. Fail each boundary and assert no complete publication exists.

- [ ] **Step 2: GREEN — add profile-aware schema/identity internals**

Make manifest/schema identity profile-aware without changing prediction rows, control rows, candidate ordering, lane conservation, hash DAG, or Exp444 wrapper behavior. Inspection reads only frozen publication leaves and never reopens live sources.

- [ ] **Step 3: Preserve failures and no-clobber semantics**

Tests cover pre-staging failure, partial staging, post-rename failure audit, complete publication replay, and same-root retry rejection. Hash Exp444/Exp442 fixtures before and after.

- [ ] **Step 4: Test, map, commit**

Run both reporting suites, regenerate map, and commit `feat(exp445): gate create-only publication`.

---

### Task 7: Expose exactly four Exp445 commands and lazy evaluation

**Files:**
- Create: Exp445 `cli_antigravity.py`, `tests/test_cli_antigravity.py`, `tests/test_evaluation_antigravity.py`
- Create: `scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py`
- Modify: Exp444 `evaluation_antigravity.py`, `cli_antigravity.py`
- Modify: Exp444 CLI/evaluation tests
- Modify together: the Exp445 code-map triplet

- [ ] **Step 1: RED — freeze the operator surface**

Parser tests accept only `readiness --reviewed-commit`, `new --reviewed-commit`, `inspect`, and `evaluate`. Reject resume and every path/profile/source/seed/mask/threshold/worker/backend/evaluation override. Verify the launcher sets all inner-thread variables before NumPy import.

- [ ] **Step 2: GREEN — implement thin CLI dispatch**

`readiness` and `new` validate clean HEAD equals the supplied 40-hex commit. `inspect` targets only the fixed Exp445 publication. `evaluate` performs two independent read-only inspections before lazily importing the profile-aware evaluator.

- [ ] **Step 3: Protect the label boundary**

Subprocess/AST/call-order tests prove readiness/new/inspect do not import `sqlite3` through evaluation, labels remain zero, evaluate opens TEST labels only after both inspections, and evaluation writes create-only Exp445 artifacts. Re-entry of complete evaluation is source- and label-free replay.

- [ ] **Step 4: Test, map, commit**

Run both Exp444 and Exp445 CLI/evaluation suites, regenerate map, and commit `feat(exp445): expose closed recovery commands`.

---

### Task 8: Run complete regression and fresh reviews before official execution

**Files:**
- Modify: `docs/EXP445_IMPLEMENTATION_REPORT_antigravity.md`, `docs/EXP445_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`, `RESEARCH_STATE.json`, `CHAMPION_LEDGER.json`
- Modify together: the Exp445 code-map triplet

- [ ] **Step 1: Run all Exp444 and Exp445 tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover \
  -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*_antigravity.py'
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover \
  -s experiments/exp445_exp444_publication_source_authority_repair_antigravity/tests -p 'test_*_antigravity.py'
```

Expected: all pass; no official root is created; no TEST label is opened.

- [ ] **Step 2: Run read-only smokes only**

Run profile construction, predecessor verification against immutable files, real provenance+source-authority composition, CLI help/rejection, and a temporary-root 30-record sequential/11-worker parity smoke. Do not invoke official readiness/new/evaluate.

- [ ] **Step 3: Validate map and tree**

Verify every node/edge evidence path and symbol, all node paths, edge/flow references, HTML embedded JSON byte equality, deterministic module fingerprints, current commit/dirty convention, and zero unmarked unknown relations. Run `git diff --check` and full JSON parsing.

- [ ] **Step 4: Obtain commitment-boundary review, fix, and commit the candidate**

Use a fresh Sol High commitment-boundary review of the working diff. Important findings require fix-first, focused regression, full regression, map regeneration, and re-review. When that review passes, record exact test counts/hashes and status `implementation_verified_awaiting_clean_head_final_review`, keep promotion fields false, and commit `chore(exp445): verify publication repair implementation`.

- [ ] **Step 5: Review the exact clean candidate commit**

Verify `git status --porcelain` is empty and capture the exact 40-hex HEAD. Use a different fresh Sol High final reviewer on that exact clean commit. Any finding requires a new fix commit, full regression/map validation, and another fresh clean-HEAD review. Only a passing final review may change state to `implementation_verified_awaiting_official_readiness` and permit Task 9 readiness/new; if recording that status creates a new commit, review that exact clean commit as well.

---

### Task 9: Run one official label-free Exp445 computation and freeze publication

**Files:**
- Runtime artifacts only under the fixed Exp445 Dataset root during commands
- Modify after completion: implementation report, runbook, catalog, state, failure ledger only if needed, champion non-promotion history
- Modify together: the Exp445 code-map triplet for final recorded state

- [ ] **Step 1: Recheck immutable authorities and root absence**

```bash
git status --porcelain
REVIEWED_COMMIT_ANTIGRAVITY="$(git rev-parse HEAD)"
test ! -e /Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810
shasum -a 256 docs/EXP445_PREREGISTRATION_antigravity_KO.md docs/EXP445_RUNBOOK_antigravity_KO.md
```

Expected: clean tree, absent root, all Exp444 predecessor hashes unchanged, database link count one.

- [ ] **Step 2: Run official readiness once**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py \
  readiness --reviewed-commit "$REVIEWED_COMMIT_ANTIGRAVITY"
```

Expected: CPU/11/inner-1, 30 source candidates, 1,117 summary rows, exact predecessor authority, zero labels, root still absent.

- [ ] **Step 3: Run official new once**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py \
  new --reviewed-commit "$REVIEWED_COMMIT_ANTIGRAVITY"
```

The process writes PID, current dataset/candidate/view, dimensions, stage/backend, counters, throughput, elapsed time, ETA, checkpoint, and latest error to atomic JSON plus append-only JSONL/text logs. Report paths once; do not repeatedly poll. Expected terminal state: 30/30, error 0, labels 0, exact ordered predecessor record parity, create-only publication complete.

- [ ] **Step 4: Inspect frozen publication twice without labels**

Run the fixed `inspect` command twice and compare returned publication/prediction hashes. Both inspections must be read-only, source-free, label-free, and identical.

- [ ] **Step 5: Stop on any failure**

If readiness or `new` pre-allocation validation fails, leave the Exp445 root absent and create no failure artifact. If a gate fails after root allocation, preserve the Exp445 root, write exact immutable failure/hash/next-command evidence there, do not retry the root, do not evaluate, and do not alter Exp444/Exp442.

---

### Task 10: Evaluate descriptively, record exact result, and pause

**Files:**
- Modify: `docs/EXP445_IMPLEMENTATION_REPORT_antigravity.md`, `docs/EXP445_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`, `RESEARCH_STATE.json`, `CHAMPION_LEDGER.json`, and `FAILURE_LEDGER.json` only if applicable
- Modify together: the Exp445 code-map triplet

- [ ] **Step 1: Run the sole post-freeze evaluator**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py evaluate
```

Expected: evaluator independently repeats the two-inspection gate, then performs only the fixed 1,117-dataset TEST-label evaluation and produces create-only descriptive artifacts.

- [ ] **Step 2: Inspect evaluation and record exact hashes**

Record stable/unstable counts, confusion matrices, micro/macro F1, paired bootstrap intervals, rescued/regressed datasets, source/prediction/publication/evaluation hashes, predecessor/new record parity, backend evidence, PID/checkpoint/log paths, and all errors. Do not declare a scientific winner or operational promotion.

- [ ] **Step 3: Recheck predecessor immutability**

Recompute every declared Exp444 terminal hash, verify no Exp444 publication/evaluation/attempt_0001 exists, confirm Exp442 source hashes, and confirm database link count one.

- [ ] **Step 4: Final validation and commit**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp445_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp445_antigravity.lock >/dev/null
git diff --check
```

Set the project to paused discussion state, clear PID/active command, keep `pending_candidate` empty, regenerate the map triplet, obtain a final Sol High result review, and commit `docs(exp445): record publication repair result`. Do not start Exp446 automatically.

## Plan Self-Review Checklist

- [ ] Every approved design section maps to at least one task and executable gate.
- [ ] Every new filename contains `antigravity`; no `__init__.py`, TODO, TBD, placeholder hash, or future result claim exists.
- [ ] Exp444 public wrappers and scientific SHA stay unchanged; profile differences are limited to identity/schema/root/documents/predecessor authority.
- [ ] The real provenance→publication composition test would have caught the Exp444 failure.
- [ ] All 30 individual record SHA values and their aggregate payload SHA are checked before publication.
- [ ] CPU worker policy is max-minus-one = 11 on this Mac; backend selection is not reopened.
- [ ] Prediction and publication remain label-free; evaluator import occurs only after two inspections.
- [ ] Exp444/Exp442 are immutable; the Exp445 root is single-use and its immutable artifacts are create-only while active runtime surfaces use only the inherited atomic-replacement protocol.
- [ ] Every module-boundary change updates all three versioned code-map files in the same commit.
- [ ] Official commands remain blocked until a clean reviewed commit and fresh Sol High approval.
