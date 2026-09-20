# Exp447 Exp446 Post-Freeze Performance Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the immutable Exp446 prediction on all 1,117 datasets after freeze, publish confusion matrices, dataset transitions, regressions, and paired bootstrap intervals without changing prediction bytes or promoting a winner.

**Architecture:** A new evaluation-only Exp447 package authenticates the immutable Exp446 publication and Exp404 Hard comparator twice before lazily opening one read-only TEST-label session. It closes the label boundary, revalidates every frozen authority, computes deterministic descriptive metrics, and writes a separate create-only Exp447 artifact tree. Exp446 source, CLI, root, publication, and prediction remain unchanged.

**Tech Stack:** Python 3.14.6, NumPy 2.4.6, SQLite immutable read-only descriptors, canonical JSON/JSONL, SHA256, `unittest`, NumPy PCG64 paired bootstrap, and Darwin inode/link-count validation.

## Global Constraints

- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Experiment identity is `Exp447`; schema is `exp447`; run identity is `exp447-exp446-postfreeze-performance-evaluation-antigravity-20260811`.
- Run root is `/Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811` and must be absent before official readiness.
- Approved design path is `docs/superpowers/specs/2026-08-11-exp447-exp446-postfreeze-performance-evaluation-antigravity-design.md` with SHA256 `043009941619335b658af99911b1463f66ea73e93d600fe1cfb884391afea868`.
- Exp446 prediction SHA is `602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100`; manifest file SHA is `bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11`; artifact-map file SHA is `d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d`.
- Exp446 population is exactly 1,117 datasets, 30 source coordinates, 26 stable coordinates, 4 unstable coordinates, and prediction label-access count 0.
- Exp404 Hard prediction SHA is `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`.
- Database SHA is `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`; expected inode is `1048138`; official preflight requires link count 1.
- TEST labels become reachable only after two byte/root-identical Exp446 inspections and Exp404 validation. Exactly 1,117 ordered queries and 195,900 `(instance_index,label)` rows are allowed.
- Frozen baseline confusion in `[TP,FP,FN,TN]` order is `[1915,303,2003,191679]`.
- Bootstrap uses NumPy `Generator(PCG64(44720260811))`, exactly 20,000 paired dataset draws, int64 indices, `endpoint=False`, and linear 2.5/97.5 percentiles.
- Evaluation is always `descriptive_complete_no_promotion`; there is no success threshold, scientific winner, same-experiment reselection, champion update, or operational-alert promotion.
- Exp446 root, CLI, publication, prediction, manifest, and artifacts are read-only. Do not add an evaluate command to Exp446.
- Exp447 exposes exactly `readiness --reviewed-commit`, `evaluate --reviewed-commit`, and `inspect`; no root/hash/source/lane/threshold/budget overrides, `new`, `resume`, or `retry`.
- The isolated interpreter is `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python`; do not alter its package set.
- Exp447 is an evaluation-only experiment, not a score source; no MLX/MPS benchmark or training is performed.
- Compare `docs/codemap/codemap_exp446_antigravity.lock` with the current repository before the first source edit. Use `docs/codemap/codemap_exp446_antigravity.json` to identify callers, effects, and tests. Generate `docs/codemap/codemap_exp447_antigravity.html`, `.json`, and `.lock` together whenever the new boundary changes.
- Native implementation uses `sol_advisor_terra_implementer`; design, commitment-boundary review, and final review use a fresh `sol_advisor_sol_reviewer`.
- The private owner-only dashboard is `https://timeseries-research-observatory.pmh10401.chatgpt.site`; its separate repository is `/Users/mac/.codex/visualizations/2026/08/10/019fedc4-df12-73b2-9812-a5f8f387ac78/research-observatory-antigravity-site`. Do not touch it until immutable Exp447 evaluation plus label-free inspect have succeeded and that repository is clean.
- The local/deterministic evaluator imports no network/dashboard code. A separate exporter may emit only sanitized aggregate data after truth closeout: never raw TEST labels, instance/anomaly indices, dataset names, local paths, hashes, logs, checkpoints, DB names, secrets, or per-dataset rows.
- Live status is a separate non-scientific sync and uses only the existing exact 12-field `PublicExperimentStatus` allowlist: experiment_id, display_name, status, stage, completed, total, success, errors, backend, throughput, eta_seconds, updated_at. Exp447 total is 1117.
- The completed dashboard payload is `exp447.public-evaluation.v1`: experiment_id, generated_at, dataset_count, decision, four aggregate comparison rows (hard baseline, hard+source30, hard+stable26, hard+unstable4; each TP/FP/FN/TN/precision/recall/F1/FPR), aggregate diagnostic counts (rescued, added-false-alarm, regressed, strong-regressed), and three paired-bootstrap aggregate delta/CI rows. It contains no dataset lists.
- Dashboard work is a post-evaluation presentation boundary, not a scientific or promotion gate. Dashboard failure never changes immutable scientific artifacts and never permits evaluation retry or reevaluation.

## File Structure

- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/config_antigravity.py`: fixed identity, paths, hashes, populations, bootstrap values, and governance.
- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/contracts_antigravity.py`: immutable publication identity, confusion/metric, bootstrap, evaluation, and inspection results.
- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/artifacts_antigravity.py`: canonical parsing and create-only/no-follow wrappers.
- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/sources_antigravity.py`: Exp446 publication, Exp404 Hard, DB identity, and lazy TEST-label session authority.
- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/bootstrap_antigravity.py`: pure project F1 and three deterministic paired bootstrap comparisons.
- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/evaluation_antigravity.py`: double-inspection label boundary, metrics, transitions, audit, create-only publication, and replay.
- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/readiness_antigravity.py`: clean reviewed HEAD, document/hash, source, DB, root-absence, and import-firewall gates.
- `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/cli_antigravity.py`: closed three-command dispatcher with lazy evaluator import.
- `scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py`: outside-CWD-safe launcher that sets seven numerical thread variables to 1 before imports.
- Matching `test_*_antigravity.py` files under `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/` cover each boundary.
- `docs/EXP447_PREREGISTRATION_antigravity_KO.md`, `docs/EXP447_RUNBOOK_antigravity_KO.md`, and `docs/EXP447_IMPLEMENTATION_REPORT_antigravity.md` preserve the frozen governance and execution truth.

---

### Task 1: Freeze Exp447 governance and versioned code map

**Files:**
- Create: `docs/EXP447_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP447_RUNBOOK_antigravity_KO.md`
- Create: `docs/EXP447_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify: `FAILURE_LEDGER.json`
- Create together: `docs/codemap/codemap_exp447_antigravity.html`
- Create together: `docs/codemap/codemap_exp447_antigravity.json`
- Create together: `docs/codemap/codemap_exp447_antigravity.lock`

**Interfaces:**
- Consumes: approved design bytes/SHA and immutable Exp446 truth.
- Produces: frozen preregistration, exact operator runbook, planned/non-promoted ledgers, and design-only Exp447 map evidence.

- [ ] **Step 1: Verify the approved design and current map convention**

```bash
test "$(shasum -a 256 docs/superpowers/specs/2026-08-11-exp447-exp446-postfreeze-performance-evaluation-antigravity-design.md | awk '{print $1}')" = "043009941619335b658af99911b1463f66ea73e93d600fe1cfb884391afea868"
python3 -m json.tool docs/codemap/codemap_exp446_antigravity.lock >/dev/null
git status --short --branch
git rev-parse HEAD
```

Expected: matching design SHA, parseable map lock, and a clean tree.

- [ ] **Step 2: Write the preregistration and runbook**

Record every Global Constraint verbatim. The runbook fixes these future commands and marks them unavailable until a reviewed implementation exists:

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py evaluate --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py inspect
```

The report starts with `implementation_status=not_started`, null evaluation hashes, and no official label access.

- [ ] **Step 3: Register planned truth without changing the champion**

Add one Exp447 catalog/state/history record with status `design_approved_implementation_planned`, null evaluation/champion/promotion hashes, `test_labels_opened=false`, and all non-promotion flags. Keep `pending_candidate` and current champion unchanged. Add no failure entry unless a real Exp447 attempt exists.

- [ ] **Step 4: Generate the complete Exp447 code-map triplet**

Copy the current topology only as an evidence-backed starting point. Add the new design/preregistration/runbook/plan to `research_control`, keep at most 20 primary nodes, and state that no Exp447 source edge exists before implementation. Recompute all nine module fingerprints from tracked paths and current contents, excluding `docs/codemap`.

- [ ] **Step 5: Validate and commit Task 1**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp447_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp447_antigravity.lock >/dev/null
git diff --check
git commit -m "docs(exp447): freeze postfreeze evaluation contract"
```

Expected: one bounded governance/map commit and no product source.

---

### Task 2: Implement frozen configuration, contracts, artifacts, and bootstrap

**Files:**
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/config_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/contracts_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/bootstrap_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_contracts_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_artifacts_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_bootstrap_antigravity.py`
- Modify together: `docs/codemap/codemap_exp447_antigravity.html`
- Modify together: `docs/codemap/codemap_exp447_antigravity.json`
- Modify together: `docs/codemap/codemap_exp447_antigravity.lock`

**Interfaces:**
- Produces: `FROZEN_CONFIG_ANTIGRAVITY`, `PublicationIdentityAntigravity`, `ConfusionAntigravity`, `MetricSummaryAntigravity`, `BootstrapComparisonAntigravity`, `canonical_json_bytes_antigravity`, `write_create_only_antigravity`, `project_f1_antigravity`, and `paired_bootstrap_antigravity`.

- [ ] **Step 1: Write RED tests for exact config and immutable types**

```python
def test_frozen_identity_and_governance_antigravity():
    assert FROZEN_CONFIG_ANTIGRAVITY.experiment_id == "Exp447"
    assert FROZEN_CONFIG_ANTIGRAVITY.dataset_count == 1117
    assert FROZEN_CONFIG_ANTIGRAVITY.label_rows == 195900
    assert (FROZEN_CONFIG_ANTIGRAVITY.source_count, FROZEN_CONFIG_ANTIGRAVITY.stable_count, FROZEN_CONFIG_ANTIGRAVITY.unstable_count) == (30, 26, 4)
    assert FROZEN_CONFIG_ANTIGRAVITY.scientific_success_criterion is None
    assert FROZEN_CONFIG_ANTIGRAVITY.promotion_allowed is False

def test_confusion_rejects_bool_and_computes_project_metrics_antigravity():
    confusion = ConfusionAntigravity(tp=1, fp=1, fn=1, tn=7)
    assert confusion.f1 == 0.5
    with self.assertRaises(ValueError):
        ConfusionAntigravity(tp=True, fp=0, fn=0, tn=0)
```

Also test exact mapping keys, finite floats, deep immutability, duplicate dataset rejection, source/lane conservation, and recursive rejection of label-like keys from pre-label objects.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_config_antigravity \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_contracts_antigravity \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_artifacts_antigravity \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_bootstrap_antigravity
```

Expected: import errors because the Exp447 modules are absent.

- [ ] **Step 3: Implement minimal exact configuration and contracts**

Use frozen dataclasses with exact-type validation. `ConfusionAntigravity` exposes `precision`, `recall`, `f1`, and `false_positive_rate`; each zero denominator returns `0.0`. `PublicationIdentityAntigravity` binds root device/inode, every Exp446 leaf SHA, manifest/prediction/artifact SHA, dataset order, source/stable/unstable order, and label-access count 0.

- [ ] **Step 4: Implement canonical artifacts and deterministic bootstrap**

`paired_bootstrap_antigravity(baseline_rows, comparator_rows, comparator_name)` validates identical dataset order, constructs one `(20000,1117)` int64 PCG64 draw matrix, computes micro and macro comparator-minus-baseline F1 arrays, records draw SHA, and uses linear percentiles. The caller reuses the same draw bytes for all three comparisons.

- [ ] **Step 5: Run GREEN twice and commit**

```bash
python3 -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
python3 -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
git diff --check
git commit -m "feat(exp447): freeze evaluation primitives"
```

---

### Task 3: Implement immutable source authority and readiness

**Files:**
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/sources_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/readiness_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_sources_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_readiness_antigravity.py`
- Modify together: `docs/codemap/codemap_exp447_antigravity.html`
- Modify together: `docs/codemap/codemap_exp447_antigravity.json`
- Modify together: `docs/codemap/codemap_exp447_antigravity.lock`

**Interfaces:**
- Consumes: Task 2 config/contracts/artifacts and the immutable Exp446/Exp404 roots.
- Produces: `capture_exp446_publication_antigravity()`, `capture_hard_predictions_antigravity(identity)`, `open_test_labels_antigravity(identity)`, and `readiness_antigravity(reviewed_commit)`.

- [ ] **Step 1: Write RED source and readiness tests**

Test exact Exp446 leaf inventory/hash, manifest/artifact/prediction SHA, dataset order, 30/26/4 partition, zero label access, symlink/extra/missing/replaced leaves, root and nested-directory inode replacement, Exp404 byte/hash tamper, DB SHA/inode/nlink, root absence, clean/dirty/wrong reviewed commit, exact design/prereg SHA, and that readiness imports neither `sqlite3` nor the label-session function.

```python
def test_label_session_is_unreachable_before_two_equal_captures_antigravity():
    first = capture_exp446_publication_antigravity()
    with self.assertRaises(ValueError):
        open_test_labels_antigravity(first, independently_confirmed=None)
```

- [ ] **Step 2: Run RED and capture the missing interface**

```bash
python3 -m unittest \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_sources_antigravity \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_readiness_antigravity -v
```

Expected: missing module/interface failures.

- [ ] **Step 3: Implement descriptor-pinned Exp446 and Exp404 capture**

Retain root/publication directory descriptors through exact inventory reads, reject symlinks with no-follow opens, verify every byte SHA and semantic population, then revalidate descriptor and namespace identities. Return only deeply immutable bytes-derived values; do not return SQLite connections or mutable publication objects.

- [ ] **Step 4: Implement the closure-private TEST-label session**

Import `sqlite3` only inside `open_test_labels_antigravity`. Open the verified DB through the project read-only descriptor boundary, install an authorizer that permits only the parameterized query for `(dataset, split='TEST')`, return a slots-only facade with `labels_for_dataset_antigravity(dataset)` and `close_antigravity()`, enforce one query per dataset in frozen order, and revalidate DB identity/link count on close.

- [ ] **Step 5: Implement mutation-free readiness**

Validate clean reviewed HEAD, fixed documents and hashes, source/DB identity, Python/NumPy versions, no label import/access, and absent/non-symlink Exp447 root without creating any directory.

- [ ] **Step 6: Run GREEN twice and commit**

```bash
python3 -m unittest \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_sources_antigravity \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_readiness_antigravity
python3 -m unittest \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_sources_antigravity \
  experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_readiness_antigravity
git diff --check
git commit -m "feat(exp447): pin postfreeze source authority"
```

---

### Task 4: Implement the single-use evaluator and immutable replay

**Files:**
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap_exp447_antigravity.html`
- Modify together: `docs/codemap/codemap_exp447_antigravity.json`
- Modify together: `docs/codemap/codemap_exp447_antigravity.lock`

**Interfaces:**
- Consumes: Task 2 pure metrics/bootstrap and Task 3 source authority.
- Produces: `evaluate_antigravity(reviewed_commit) -> FrozenEvaluationAntigravity` and `inspect_evaluation_antigravity() -> FrozenEvaluationAntigravity`.

- [ ] **Step 1: Write RED ordering and conservation tests**

Use an event-recording dependency fixture. Assert this prefix before label open:

```python
assert events[:5] == [
    "capture_exp446", "capture_exp404", "derive_hard",
    "capture_exp446", "compare_exp446_identity",
]
assert events[5] == "open_labels"
```

Test 1,117/195,900 production counts, contiguous indices, exact integer labels, source/stable/unstable transition conservation, baseline `[1915,303,2003,191679]`, source-30 historical confusion agreement, confusion sums, rescued/false-alarm/regressed/strong-regression definitions, and all three bootstrap comparisons.

- [ ] **Step 2: Write RED failure-preservation tests**

Cover drift at second inspection, label close, post-close inspection, post-compute inspection, pre-commit inspection, partial destination write, audit write failure, existing complete replay, existing partial root, and frozen semantic leaf tamper with recomputed artifact-map hashes.

- [ ] **Step 3: Run RED**

```bash
python3 -m unittest experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_evaluation_antigravity -v
```

Expected: missing evaluator interfaces.

- [ ] **Step 4: Implement evaluation ordering and metrics**

Compute these four predictions for every dataset: Hard, Hard union source 30, Hard union stable 26, and Hard union unstable 4. Store dataset confusion rows and exact candidate transitions. Compute aggregate precision/recall/F1/FPR, macro F1, diagnostics, and three paired bootstrap comparisons only after the label session has closed.

- [ ] **Step 5: Implement create-only publication and replay**

Write exactly the eleven design-listed leaves. `artifact_sha256_antigravity.json` is self-excluding. Replay pins the exact inventory, validates all schemas/types/orders/hashes, reconstructs every derived leaf from dataset rows and transitions, and rechecks current Exp446 identity without importing SQLite.

- [ ] **Step 6: Implement truthful failure audit**

Before label open, fail without creating Exp447 root. From label-open onward, create one unique sibling `evaluation_audit_<uuid>_antigravity.json` containing phase, frozen identity, query attempts/completions, row count, session state, destination state, and original error. Never overwrite complete or partial evidence.

- [ ] **Step 7: Run GREEN twice and commit**

```bash
python3 -m unittest experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_evaluation_antigravity
python3 -m unittest experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_evaluation_antigravity
git diff --check
git commit -m "feat(exp447): evaluate immutable Exp446 prediction"
```

---

### Task 5: Add the closed CLI and outside-CWD launcher

**Files:**
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/cli_antigravity.py`
- Create: `experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests/test_cli_antigravity.py`
- Create: `scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap_exp447_antigravity.html`
- Modify together: `docs/codemap/codemap_exp447_antigravity.json`
- Modify together: `docs/codemap/codemap_exp447_antigravity.lock`

**Interfaces:**
- Consumes: Task 3 readiness and Task 4 evaluate/inspect.
- Produces: `build_parser_antigravity()`, `main_antigravity(argv)`, and a runnable launcher.

- [ ] **Step 1: Write RED CLI tests**

Assert exactly three commands, `allow_abbrev=False`, required full 40-hex reviewed commit for readiness/evaluate, rejection of new/resume/retry and all override flags, lazy `sqlite3`/evaluation imports for help/readiness/inspect, JSON-safe dataclass output without `asdict`, and outside-CWD launcher behavior.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_cli_antigravity -v
```

- [ ] **Step 3: Implement the minimal closed dispatcher**

`readiness` imports readiness only. `evaluate` validates the clean reviewed commit again and then lazily imports the evaluator. `inspect` imports only replay and must leave `sqlite3` and the label-session module unloaded. Recursively serialize dataclasses through `fields()` and `getattr()` rather than `dataclasses.asdict`.

- [ ] **Step 4: Implement the launcher**

Before package import set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, and `NUMBA_NUM_THREADS` to `1`; insert the repository root based on `__file__`; then call `main_antigravity()`.

- [ ] **Step 5: Run GREEN and commit**

```bash
python3 -m unittest experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.tests.test_cli_antigravity
python3 scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py --help
git diff --check
git commit -m "feat(exp447): add closed evaluation CLI"
```

---

### Task 6: Verify the clean candidate and pass fresh Sol review

**Files:**
- Modify: `docs/EXP447_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify together: `docs/codemap/codemap_exp447_antigravity.html`
- Modify together: `docs/codemap/codemap_exp447_antigravity.json`
- Modify together: `docs/codemap/codemap_exp447_antigravity.lock`

**Interfaces:**
- Consumes: Tasks 1-5 implementation commits.
- Produces: a clean exact reviewed candidate that alone may be used for official readiness/evaluate.

- [ ] **Step 1: Run focused Exp447 tests twice**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
```

- [ ] **Step 2: Run affected predecessor regressions once**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*.py'
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp445_exp444_publication_source_authority_repair_antigravity/tests -p 'test_*.py'
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp446_exp445_publication_materialization_repair_antigravity/tests -p 'test_*.py'
```

Capture exact test counts, durations, and exit codes. Do not launch duplicate runs when output collection is slow.

- [ ] **Step 3: Run read-only authority checks**

```bash
shasum -a 256 /Users/mac/Documents/Dataset/univariate_ts.db
/usr/bin/stat -f 'db_inode=%i db_nlink=%l size=%z' /Users/mac/Documents/Dataset/univariate_ts.db
test ! -e /Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811
git status --short --branch
git rev-parse HEAD
```

- [ ] **Step 4: Update implementation truth and regenerate the map**

Record exact tests and hashes, status `implementation_verified_awaiting_official_readiness`, label access 0, null evaluation hash, no promotion, and the reviewed-candidate field. Regenerate/validate the three map files together from the clean functional commit.

- [ ] **Step 5: Request a fresh Sol High final review**

The reviewer checks exact clean HEAD, design/prereg/hash binding, TEST-label firewall, source/DB descriptor authority, metric/bootstrap definitions, create-only failure semantics, CLI closure, regressions, map truth, root absence, and no promotion. Any Critical or Important finding returns to a bounded RED/GREEN fix and a fresh review.

- [ ] **Step 6: Commit verification truth only after `ship`**

```bash
git diff --check
git commit -m "docs(exp447): verify clean evaluation candidate"
```

---

### Task 7: Run the official 1,117-dataset evaluation and report results

**Files:**
- Modify after success: `docs/EXP447_IMPLEMENTATION_REPORT_antigravity.md`
- Modify after success: `docs/EXP447_RUNBOOK_antigravity_KO.md`
- Modify after success: `EXPERIMENT_CATALOG.json`
- Modify after success: `RESEARCH_STATE.json`
- Modify after success: `CHAMPION_LEDGER.json`
- Modify after actual failure only: `FAILURE_LEDGER.json`
- Modify together after truth change: `docs/codemap/codemap_exp447_antigravity.html`
- Modify together after truth change: `docs/codemap/codemap_exp447_antigravity.json`
- Modify together after truth change: `docs/codemap/codemap_exp447_antigravity.lock`

**Interfaces:**
- Consumes: fresh Sol `ship` candidate from Task 6 and absent official Exp447 root.
- Produces: one immutable Exp447 evaluation, one label-free replay, verified metrics, and a discussion-ready Korean analysis.

- [ ] **Step 1: Run official readiness once**

```bash
PY=/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python
REVIEWED_COMMIT=$(git rev-parse HEAD)
"$PY" scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py readiness --reviewed-commit "$REVIEWED_COMMIT"
```

Expected: root remains absent, all source/DB/document/environment checks pass, label access remains 0.

- [ ] **Step 2: Run official evaluate once**

```bash
"$PY" scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py evaluate --reviewed-commit "$REVIEWED_COMMIT"
```

Expected: 1,117 queries, 195,900 rows, closed session, immutable eleven-leaf Exp447 evaluation, no Exp446 mutation, and no promotion.

- [ ] **Step 3: Run source-free label-free inspect once**

```bash
"$PY" scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py inspect
```

Expected: replayed hashes and metrics match the evaluate result without opening SQLite or labels.

- [ ] **Step 4: Validate the artifact and recompute headline metrics independently**

Parse all eleven files, verify the self-excluding artifact map, dataset/transition order, confusion conservation, exact baseline, source-30 historical agreement, macro denominators, three bootstrap draw SHA/intervals, and reconstructed diagnostics. Confirm Exp446 leaf tree bytes and DB SHA/inode/nlink are unchanged.

- [ ] **Step 5: Produce the easy Korean result analysis**

Report:

```text
Hard baseline confusion and F1
Hard + source 30 confusion and F1
Hard + stable 26 confusion and F1
Hard + unstable 4 confusion and F1
new TP / added FP
rescued datasets
false-alarm datasets
regressed and strong-regressed datasets
micro/macro paired bootstrap deltas and 95% intervals
```

Include a confusion-matrix chart, stable-versus-source comparison chart, and dataset transition table. Explain TP/FP/FN/TN in plain Korean. State verified findings, uncertainty, caveats, and that no automatic promotion occurred.

- [ ] **Step 6: Close truth and stop for discussion**

Record exact evaluation/prediction/manifest/artifact hashes, label query accounting, result metrics, artifact paths, inspect result, and `descriptive_complete_no_promotion` in the ledgers/report. Keep champion/promotion false. Regenerate and validate the Exp447 map triplet, commit truth then map, request a final read-only Sol verification, and stop for user discussion.

---

### Task 8: Present immutable aggregate results in the private research observatory

**Files:**
- Modify only after Task 7 truth closeout and clean external site repository: the existing private site repository and its existing presentation/test surfaces.
- Export: sanitized `exp447.public-evaluation.v1` aggregate payload and the Exp447 code-map version; no raw or per-dataset scientific input is exported.

**Interfaces:**
- Consumes: immutable, independently verified Exp447 evaluation plus label-free inspect result; separate exact-allowlist live status.
- Produces: an owner-only Exp447 observatory card/detail presentation and a saved private deployment version; no scientific artifact or ledger change.

- [ ] **Step 1: Hold until the scientific and site gates are both closed**

Require completed immutable Exp447 evaluation, successful label-free inspect, independently verified truth closeout, and a clean
`/Users/mac/.codex/visualizations/2026/08/10/019fedc4-df12-73b2-9812-a5f8f387ac78/research-observatory-antigravity-site`.
If the site repository is dirty, do not edit, build, commit, deploy, or contact it. Dashboard work never opens TEST labels,
changes Exp447 evaluation bytes, or authorizes evaluation retry/reevaluation.

- [ ] **Step 2: Export only the fixed public allowlists**

Keep the evaluator local/deterministic and free of network/dashboard imports. Implement the exporter as a separate post-closeout
boundary. Status sync is separate from the evaluator and contains exactly the 12 allowed `PublicExperimentStatus` fields
(`experiment_id`, `display_name`, `status`, `stage`, `completed`, `total`, `success`, `errors`, `backend`, `throughput`,
`eta_seconds`, `updated_at`) with `total=1117`. The completed `exp447.public-evaluation.v1` payload contains only its identity,
generation time, aggregate decision, four aggregate comparison rows, aggregate diagnostic counts, and three aggregate
paired-bootstrap delta/CI rows. Reject all raw TEST labels, indices, names, paths, hashes, logs, checkpoints, DB names,
secrets, per-dataset rows, and dataset lists.

- [ ] **Step 3: TDD the private observatory presentation**

Use the Sites building workflow in the clean existing site repository. Add RED/GREEN parser and UI tests for exact payload
allowlists, rejection of forbidden fields, status-field closure, Exp447 card/detail rendering, aggregate comparison chart,
confusion presentation, and paired-bootstrap chart. Preserve private owner-only access and do not add presentation data to
scientific decision or promotion logic.

- [ ] **Step 4: Validate, commit, save, and deploy only the private presentation**

Use the Sites hosting workflow after site tests, lint, deployment build, and requested browser verification pass. Export the
Exp447 codemap version with the presentation. Commit the exact validated external site repository, save a version, and deploy
that saved version to `https://timeseries-research-observatory.pmh10401.chatgpt.site` with owner-only access preserved.
If any dashboard/parser/UI/build/deploy step fails, preserve immutable Exp447 scientific artifacts unchanged, do not retry
evaluation, and report the separate presentation failure.

## Plan Self-Review

- Design coverage: every identity, frozen hash, label-order gate, comparator, metric, bootstrap, audit, CLI, map, official command, non-promotion requirement, and private aggregate-only dashboard handoff has an owning task.
- Scope: one evaluation-only subsystem; Exp446 product code and artifacts remain read-only.
- Type consistency: Task 2 types and functions are the exact interfaces consumed by Tasks 3-5.
- The only intentionally deferred boundary is Task 8: it is blocked until immutable scientific truth exists and the separately owned site repository is clean.
- Official TEST labels are inaccessible until Task 7 and only after a fresh reviewed clean candidate.
