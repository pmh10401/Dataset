# Exp444 TRAIN-only Stability Precision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute only the frozen Exp442 30 candidates under deterministic normal-TRAIN omission views, publish stable and unstable review lanes without TEST labels, and report the post-freeze result without a scientific success criterion or promotion.

**Architecture:** A frozen-source layer verifies Exp404/440/442 bytes and reads only normal TRAIN plus the 30 bound TEST value rows. A pure stability layer reproduces the original Exp440 candidate first, creates up to 16 deterministic omission views, and classifies every source coordinate into exactly one review-only lane. An ordered 11-worker runtime writes its own checkpoint and progress evidence; create-only publication freezes all predictions before a separately imported evaluator may read TEST labels.

**Tech Stack:** Python 3.14.6, NumPy 2.4.6, MLX 0.32.0, Numba 0.66.0, SQLite immutable read-only descriptors, `unittest`, `ProcessPoolExecutor` with 11 outer workers and one numerical thread per worker, canonical JSON/JSONL, SHA256, Darwin inode checks, and create-only no-clobber publication.

## Global Constraints

- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Experiment identity is `Exp444`; run identity is `exp444-train-only-stability-precision-gate-antigravity-20260810`.
- Approved design path is `docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md` with exact SHA256 `1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8`.
- Run root is `/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810` and must be absent before official readiness.
- Isolated interpreter root is `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314`.
- Prediction source coordinates are exactly the frozen Exp442 30 rows, at most one coordinate per dataset. Exp444 never adds, replaces, relocates, or pads a candidate.
- Input hashes are database `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`, Exp404 Hard `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`, Exp442 prediction `b79bad6012ad6bc07050ade9968e815796a2cc871b5c85d63f0bd7dbdf6f81f4`, Exp442 ranking `d8d858cd74553aa64f6f0a5424064872b59f682fed1c8b2ace63fffba6ad20cc`, Exp442 selection `06004f18dbd9b0df3fa857bb8b1a9fd7eafd4d927f3736fb8cfb25db56b72408`, and selected-coordinate identity `db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`.
- Normal TRAIN grouping, whole-row nonfinite exclusion, FFT low-frequency features, robust center/scale, nearest `k=3`, OOF quantile `0.99` with method `higher`, strict `score > threshold`, and two-depth consensus are unchanged from Exp440.
- Mask seed is `44420260810`; bootstrap seed is `44420260811`; bootstrap samples are 20,000.
- For exact-length group size `n`, omit `max(1, floor(0.10*n))`, generate at most 16 unique masks, and require `ceil(0.875*view_count)` surviving views. This is the candidate classification rule, not an experiment success criterion.
- A view survives only when the unchanged Exp440 rule produces a candidate and at least two original Exp442 agreeing depths still exceed their view thresholds.
- Prediction cannot import or read TEST `label`, `labels_blob`, actual anomaly positions, past TEST metrics, family TEST performance, TEST missingness as a gate, candidate index position, or tail rules.
- Existing Exp440/442/443 roots and artifacts are immutable. Exp444 uses a fresh namespace, root, checkpoint, publication, and evaluation.
- Exp444 has no experiment pass/fail score, no scientific winner, no champion update, and no operational alert promotion. Evaluator status is always `descriptive_complete_no_promotion` after a coherent evaluation.
- Official CPU workers are `max(1, os.cpu_count() - 1)`, expected 11. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, and `NUMBA_NUM_THREADS` are all `1` before importing NumPy.
- MLX or Numba is eligible only with identical masks, candidate/view survival, stable/unstable lane coordinates, finite outputs, maximum absolute scalar difference at most `1e-5`, and end-to-end stage speed at least 1.1x CPU including transfer and synchronization.
- Run only one MLX/MPS-heavy process. Never combine outer multiprocessing with concurrent MLX jobs.
- Long execution writes atomic progress JSON, PID JSON, append-only event JSONL and text log, current dataset/candidate/view dimensions, counters, throughput, elapsed time, ETA, checkpoint, and latest error itself. Codex does not poll repeatedly.
- Compare `docs/codemap/codemap_exp444_antigravity.lock` before every code change. Query `docs/codemap/codemap_exp444_antigravity.json` for callers, effects, and tests before editing a mapped module. Regenerate `docs/codemap/codemap_exp444_antigravity.html`, `docs/codemap/codemap_exp444_antigravity.json`, and `docs/codemap/codemap_exp444_antigravity.lock` together when a boundary or major flow changes.

## File Structure

- `experiments/exp444_train_only_stability_precision_gate_antigravity/config_antigravity.py`: frozen identity, paths, hashes, view rules, workers, backend limits, and governance.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/contracts_antigravity.py`: immutable source candidate, mask, view score, stability result, checkpoint, backend, publication, and evaluation objects.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/artifacts_antigravity.py`: thin Exp444 wrappers around verified canonical/create-only artifact primitives.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/environment_antigravity.py`: interpreter, packages, thread controls, and environment-lock verification.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/sources_antigravity.py`: strict Exp404/440/442 replay, selected-coordinate validation, descriptor-bound values-only SQLite access, and source snapshot.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/masks_antigravity.py`: deterministic unique omission masks and survival-count arithmetic.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/stability_antigravity.py`: original Exp442 control replay, exact Exp440 per-view refit, and stable/unstable classification.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/backend_antigravity.py`: CPU reference, optional MLX dense distance, optional Numba dense distance, parity, timing, and fallback decision.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/checkpoint_antigravity.py`: ordered 30-candidate create-only records, context/state hashes, error/interruption, and prefix resume.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/progress_antigravity.py`: parent-owned atomic progress/PID and append-only event/text logs.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/runtime_antigravity.py`: 11-worker ordered scheduling, sequential reference route, signal handling, resume, and finalization.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/reporting_antigravity.py`: create-only prediction publication, lane conservation, manifest/hash DAG, and read-only inspection.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/bootstrap_antigravity.py`: project F1 and paired-dataset bootstrap used only by evaluation.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/evaluation_antigravity.py`: lazy post-freeze label snapshot, baseline/Exp442/stable metrics, transitions, diagnostics, and descriptive publication.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/readiness_antigravity.py`: clean reviewed HEAD, frozen source, environment, backend, root absence, control replay, and zero-label proof.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/cli_antigravity.py`: fixed `readiness/new/resume/inspect/evaluate` interface with lazy evaluator import.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/requirements_antigravity.txt`: direct package pins.
- `experiments/exp444_train_only_stability_precision_gate_antigravity/requirements_lock_antigravity.txt`: full sorted resolved environment lock.
- `scripts/create_exp444_environment_antigravity.sh`: idempotent isolated-environment creator that refuses mismatched existing environments.
- `scratch/run_exp444_train_only_stability_precision_gate_antigravity.py`: outside-CWD-safe entrypoint that sets seven thread controls before importing the CLI.
- `scratch/verify_exp444_worker_parity_antigravity.py`: one-worker/11-worker read-only parity entrypoint.
- `docs/EXP444_PREREGISTRATION_antigravity_KO.md`: frozen label boundary and descriptive experiment contract without a scientific success gate.
- `docs/EXP444_RUNBOOK_antigravity_KO.md`: exact commands and self-written progress/checkpoint/log paths.
- `docs/EXP444_IMPLEMENTATION_REPORT_antigravity.md`: implementation, review, smoke, official hashes, results, and non-promotion record.
- Matching `test_*_antigravity.py` files under `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/` cover each boundary.

---

### Task 1: Freeze Exp444 governance without a scientific success gate

**Files:**
- Create: `docs/EXP444_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP444_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify together: `docs/codemap/codemap_exp444_antigravity.html`
- Modify together: `docs/codemap/codemap_exp444_antigravity.json`
- Modify together: `docs/codemap/codemap_exp444_antigravity.lock`

**Interfaces:**
- Consumes: approved design SHA256 `1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8` and the Global Constraints.
- Produces: frozen preregistration, operator commands, research-state entry, and design-only code-map evidence used by every later task.

- [ ] **Step 1: Verify approved design bytes and current code map**

```bash
test "$(shasum -a 256 docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md | awk '{print $1}')" = "1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8"
python3 -m json.tool docs/codemap/codemap_exp444_antigravity.lock >/dev/null
git status --short --branch
```

Expected: matching SHA, parseable lock, and a clean Exp444 branch.

- [ ] **Step 2: Write the preregistration with descriptive-only governance**

Include all frozen input hashes, the exact mask equations, the 0.875 candidate rule, label firewall, backend gate, publication order, and this literal block:

```text
scientific_success_criterion = none
evaluation_status = descriptive_complete_no_promotion
adaptive_followup = true
independent_validation = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
```

Do not include a minimum TP, maximum FP, F1 threshold, confidence-interval threshold, winner branch, or promotion branch.

- [ ] **Step 3: Write the pre-implementation runbook**

Document these fixed future commands and mark them unavailable until a clean reviewed implementation exists:

```bash
bash scripts/create_exp444_environment_antigravity.sh
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py new --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py resume --reviewed-commit REVIEWED_COMMIT --attempt attempt_0000_antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py inspect
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py evaluate
```

List the exact attempt, checkpoint, progress, events, text, PID, publication, and evaluation paths. State that the program writes logs and Codex does not poll.

- [ ] **Step 4: Register Exp444 as planned and non-promoted**

Add one catalog entry with status `design_approved_implementation_planned`, `source_candidate_count=30`, null prediction/evaluation hashes, the eight governance fields, and exact document/root paths. Set `RESEARCH_STATE.active_experiment="Exp444"`, `active_phase="implementation_planning_complete"`, and `test_labels_opened=false`. Add a champion-history record with status `design_only_not_promoted`; do not populate `pending_candidate`.

- [ ] **Step 5: Regenerate the code-map triplet as design-only evidence**

Keep the existing 20 primary nodes. Add the plan/preregistration/runbook to `research_control` evidence and explicitly state that no Exp444 source edge exists yet.

- [ ] **Step 6: Validate and commit**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp444_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp444_antigravity.lock >/dev/null
git diff --check
git add docs/EXP444_PREREGISTRATION_antigravity_KO.md docs/EXP444_RUNBOOK_antigravity_KO.md \
  EXPERIMENT_CATALOG.json RESEARCH_STATE.json CHAMPION_LEDGER.json \
  docs/codemap/codemap_exp444_antigravity.html docs/codemap/codemap_exp444_antigravity.json docs/codemap/codemap_exp444_antigravity.lock
git commit -m "docs(exp444): freeze TRAIN stability contract"
```

---

### Task 2: Freeze environment, configuration, immutable contracts, and artifact primitives

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/requirements_antigravity.txt`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/requirements_lock_antigravity.txt`
- Create: `scripts/create_exp444_environment_antigravity.sh`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/config_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/contracts_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/environment_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_contracts_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_artifacts_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_environment_antigravity.py`
- Modify together: `docs/codemap/codemap_exp444_antigravity.html`
- Modify together: `docs/codemap/codemap_exp444_antigravity.json`
- Modify together: `docs/codemap/codemap_exp444_antigravity.lock`

**Interfaces:**
- Produces: `FROZEN_CONFIG`, `frozen_config_payload_antigravity()`, `scientific_config_sha256_antigravity()`, `verify_environment_antigravity() -> EnvironmentEvidenceAntigravity`, canonical/create-only artifact wrappers, and immutable value objects consumed by Tasks 3-12.

- [ ] **Step 1: Write RED config and contract tests**

```python
def test_frozen_config_has_descriptive_governance_antigravity():
    assert FROZEN_CONFIG.source_candidate_count == 30
    assert FROZEN_CONFIG.maximum_stability_views == 16
    assert FROZEN_CONFIG.stability_fraction == 0.875
    assert FROZEN_CONFIG.scientific_success_criterion is None
    assert FROZEN_CONFIG.evaluation_status == "descriptive_complete_no_promotion"
    assert FROZEN_CONFIG.cpu_workers == 11

def test_result_conserves_two_review_lanes_antigravity():
    result = StabilitySelectionAntigravity(source=SOURCE, stable=(A,), unstable=(B,))
    assert result.source_count == result.stable_count + result.unstable_count == 2
```

Also test exact payload keys, integer-vs-boolean rejection, finite scalar enforcement, immutable copied arrays, duplicate coordinate rejection, recursive label/evaluation key rejection, and round trips for `SourceCandidateAntigravity`, `OmissionMaskAntigravity`, `DepthViewScoreAntigravity`, `StabilityViewAntigravity`, `CandidateStabilityResultAntigravity`, `StabilitySelectionAntigravity`, `BackendDecisionAntigravity`, checkpoint context/state, and evaluation summary.

- [ ] **Step 2: Run RED tests**

```bash
python3 -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_config_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_artifacts_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_environment_antigravity
```

Expected: import failure because the Exp444 modules do not exist.

- [ ] **Step 3: Create the isolated environment contract**

`requirements_antigravity.txt` contains:

```text
mlx==0.32.0
numba==0.66.0
numpy==2.4.6
```

The shell script requires `/usr/local/bin/python3` version 3.14.6, creates the fixed mode-0700 venv, installs the complete tracked lock with `--no-deps`, runs `pip check`, and refuses to mutate a mismatched existing venv. A one-time `--bootstrap-lock-antigravity` mode is permitted only while the lock file is empty; after the resolved lock is committed the default path must require byte-identical sorted `pip freeze --all`.

- [ ] **Step 4: Implement frozen config and immutable contracts**

Define these exact public types and functions:

```python
@dataclass(frozen=True)
class SourceCandidateAntigravity:
    dataset: str
    instance_index: int
    raw_length: int
    original_depths: tuple[int, ...]
    original_rank: int

@dataclass(frozen=True)
class OmissionMaskAntigravity:
    view_index: int
    excluded_train_indices: tuple[int, ...]
    identity_sha256: str

@dataclass(frozen=True)
class CandidateStabilityResultAntigravity:
    candidate: SourceCandidateAntigravity
    views: tuple[StabilityViewAntigravity, ...]
    survival_count: int
    required_survival_count: int
    lane: str  # exactly stable_review or unstable_review

def frozen_config_payload_antigravity() -> dict[str, object]: ...
def scientific_config_sha256_antigravity() -> str: ...
```

Use wrappers from `experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity`; do not copy the full artifact implementation.

- [ ] **Step 5: Verify environment and GREEN tests**

```bash
bash scripts/create_exp444_environment_antigravity.sh --bootstrap-lock-antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover \
  -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests \
  -p 'test_*_antigravity.py'
```

Expected: all Task 2 tests pass; record the exact lock SHA in config before committing.

- [ ] **Step 6: Map the new package boundary and commit**

Regenerate all three code-map files. Add one Exp444 node only after source files exist, with evidence symbols from config/contracts and tests. Validate evidence paths/symbols, HTML JSON identity, node/edge/flow references, fingerprints, and zero unknown relationships.

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity \
  scripts/create_exp444_environment_antigravity.sh docs/codemap
git commit -m "feat(exp444): freeze stability contracts and environment"
```

---

### Task 3: Verify frozen Exp442 sources and load only bound values

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/sources_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_sources_antigravity.py`
- Modify: `experiments/exp444_train_only_stability_precision_gate_antigravity/contracts_antigravity.py`

**Interfaces:**
- Consumes: frozen config/artifact wrappers and Exp442 publication/attempt files.
- Produces: `verify_frozen_sources_antigravity() -> FrozenSourceSnapshotAntigravity`, `load_candidate_values_antigravity(snapshot) -> tuple[CandidateInputAntigravity, ...]`, and a parent-owned descriptor session with zero TEST-label access.

- [ ] **Step 1: Write RED source and SQL-firewall tests**

```python
def test_source_snapshot_has_exact_thirty_coordinates_antigravity():
    snapshot = verify_frozen_sources_antigravity(FIXTURE)
    assert len(snapshot.candidates) == 30
    assert len({x.dataset for x in snapshot.candidates}) == 30
    assert snapshot.selected_coordinate_identity_sha256 == EXPECTED_IDENTITY

def test_prediction_sql_rejects_test_labels_antigravity():
    with self.assertRaisesRegex(ValueError, "label-capable SQL"):
        validate_prediction_sql_antigravity(
            "SELECT instance_index, label FROM instances WHERE split='TEST'"
        )
```

Cover path replacement, symlink/FIFO, truncated JSONL, duplicate candidates, rank not 1-30, altered raw length/depths, wrong artifact map, selection mismatch, DB inode/path replacement, `labels_blob`, unrestricted `SELECT *`, unknown tables/columns, and non-contiguous TEST identity.

- [ ] **Step 2: Run the focused RED test**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity
```

Expected: missing `verify_frozen_sources_antigravity` or missing module.

- [ ] **Step 3: Implement strict source replay and values-only loader**

Use exact SQL constants:

```python
NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY = """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index"""

BOUND_TEST_VALUE_SQL_ANTIGRAVITY = """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST' AND i.instance_index = ?"""
```

The loader decodes `<f4`, excludes an entire nonfinite TRAIN row exactly as Exp440, groups remaining rows by raw length, requires the candidate group and TEST length to match `raw_length`, and returns copied read-only arrays. The authorizer exposes no raw connection/cursor attribute and denies TEST label/labels_blob before execution. Close removes only its own pin, verifies source DB link count and inode, and preserves any failure root.

- [ ] **Step 4: Run source tests and real read-only smoke**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_sources_antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python - <<'PY'
from experiments.exp444_train_only_stability_precision_gate_antigravity.sources_antigravity import verify_frozen_sources_antigravity
s = verify_frozen_sources_antigravity()
assert len(s.candidates) == 30 and s.label_access_count == 0
print("EXP444_SOURCE_OK", len(s.candidates))
PY
```

Expected: tests pass and `EXP444_SOURCE_OK 30`.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/contracts_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/sources_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_sources_antigravity.py
git commit -m "feat(exp444): bind frozen Exp442 candidate sources"
```

---

### Task 4: Generate deterministic masks and reproduce the original 30 candidates

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/masks_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/stability_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_masks_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_stability_antigravity.py`

**Interfaces:**
- `generate_omission_masks_antigravity(dataset: str, raw_length: int, train_indices: tuple[int, ...]) -> tuple[OmissionMaskAntigravity, ...]`
- `replay_original_candidate_antigravity(candidate_input: CandidateInputAntigravity, distance=score_channel_cpu_antigravity) -> OriginalReplayAntigravity`
- `score_stability_candidate_antigravity(candidate_input: CandidateInputAntigravity, distance=score_channel_cpu_antigravity) -> CandidateStabilityResultAntigravity`

- [ ] **Step 1: Write RED deterministic-mask tests**

```python
def test_n_five_produces_five_unique_single_omission_masks_antigravity():
    masks = generate_omission_masks_antigravity("A", 32, (0, 1, 2, 3, 4))
    assert len(masks) == 5
    assert len({m.excluded_train_indices for m in masks}) == 5
    assert all(len(m.excluded_train_indices) == 1 for m in masks)
    assert required_survival_count_antigravity(len(masks)) == 5

def test_large_group_caps_at_sixteen_masks_antigravity():
    masks = generate_omission_masks_antigravity("A", 32, tuple(range(160)))
    assert len(masks) == 16
    assert all(len(m.excluded_train_indices) == 16 for m in masks)
    assert required_survival_count_antigravity(16) == 14
```

Also assert canonical SHA identity, order independence after validated contiguous identities, same seed reproducibility, dataset/raw-length separation, no duplicate masks, retain at least four TRAIN rows, and exact `ceil(0.875*view_count)` arithmetic.

- [ ] **Step 2: Write RED original-control and view tests**

```python
def test_original_replay_requires_exact_exp442_candidate_antigravity():
    replay = replay_original_candidate_antigravity(FIXTURE_INPUT)
    assert replay.coordinate == FIXTURE_INPUT.candidate.coordinate
    assert replay.original_depths == FIXTURE_INPUT.candidate.original_depths

def test_view_requires_two_original_depths_antigravity():
    result = classify_view_antigravity(
        original_depths=(2, 3), view_scores=(depth(2, True), depth(4, True))
    )
    assert result is False
```

Cover a candidate that survives 14/16, one that survives 13/16, missing original depth, unavailable refit, nonfinite score, altered candidate coordinate, and exact stable+unstable conservation.

- [ ] **Step 3: Run RED tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_masks_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_stability_antigravity
```

Expected: missing functions.

- [ ] **Step 4: Implement the pure mask and stability engine**

Generate masks by canonical JSON SHA ordering with a monotonically increasing view counter; deduplicate excluded-index tuples until reaching `min(16, comb(n, omit_count))`. For each view:

```python
retained = train_matrix[[i not in excluded for i in train_indices]]
fitted = fit_group_calibration_antigravity(retained, distance=distance)
scores, candidate = score_test_row_antigravity(
    source.dataset,
    source.instance_index,
    test_values,
    fitted,
    distance=distance,
)
survived = candidate is not None and len(
    set(source.original_depths).intersection(
        score.depth for score in scores if score.exceeds
    )
) >= 2
```

Call the unchanged public Exp440 calibration/scoring functions; do not fork their equations. Copy all returned evidence into Exp444 immutable contracts before returning from a worker.

- [ ] **Step 5: Run GREEN tests and 30-candidate control smoke**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_masks_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_stability_antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python - <<'PY'
from experiments.exp444_train_only_stability_precision_gate_antigravity.sources_antigravity import load_candidate_values_antigravity, verify_frozen_sources_antigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.stability_antigravity import replay_original_candidate_antigravity
inputs = load_candidate_values_antigravity(verify_frozen_sources_antigravity())
replays = tuple(replay_original_candidate_antigravity(item) for item in inputs)
assert len(replays) == 30 and all(item.matches_source for item in replays)
print("EXP444_CONTROL_OK", len(replays))
PY
```

Expected: tests pass and `EXP444_CONTROL_OK 30` with label access zero.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/masks_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/stability_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_masks_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_stability_antigravity.py
git commit -m "feat(exp444): classify deterministic TRAIN stability views"
```

---

### Task 5: Benchmark CPU, MLX, and Numba without changing candidate semantics

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/backend_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_backend_antigravity.py`

**Interfaces:**
- `score_distance_cpu_antigravity(query, references, k) -> float`
- `score_distance_mlx_antigravity(query, references, k) -> float`
- `score_distance_numba_antigravity(query, references, k) -> float`
- `benchmark_backends_antigravity(inputs: tuple[CandidateInputAntigravity, ...]) -> BackendDecisionAntigravity`

- [ ] **Step 1: Write RED backend contract tests**

```python
def test_backend_requires_identity_and_speed_gate_antigravity():
    decision = choose_backend_antigravity(
        cpu=run("cpu", seconds=2.0, lanes=(("A", 1, "stable_review"),)),
        mlx=run("mlx", seconds=1.7, lanes=(("A", 1, "stable_review"),), max_diff=1e-6),
    )
    assert decision.selected_backend == "mlx"

def test_fast_backend_with_lane_difference_falls_back_antigravity():
    decision = choose_backend_antigravity(
        cpu=run("cpu", seconds=2.0, lanes=(("A", 1, "stable_review"),)),
        mlx=run("mlx", seconds=1.0, lanes=(("A", 1, "unstable_review"),), max_diff=0.0),
    )
    assert decision.selected_backend == "cpu"
```

Test finite outputs, `k=3`, scalar tolerance `1e-5`, synchronization included, speed ratio exactly 1.1 accepted, 1.099 rejected, unavailable imports, and no simultaneous MLX with process pool.

- [ ] **Step 2: Run RED test**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_backend_antigravity
```

Expected: missing backend module.

- [ ] **Step 3: Implement exact distance backends and end-to-end benchmark**

CPU delegates to `score_channel_cpu_antigravity`. MLX computes float64-equivalent RMS distances if supported by the installed device, uses `mx.eval()` before stopping time, selects `k=3`, and returns the median. Numba uses an explicit compiled loop with the same float64 accumulation and median-of-three rule. The benchmark runs whole candidate-view scoring, not a synthetic primitive only, on fixed candidates covering group sizes 5, median, and 160.

- [ ] **Step 4: Run GREEN tests and record local decision**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_backend_antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python - <<'PY'
from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import benchmark_backends_antigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.sources_antigravity import load_candidate_values_antigravity, verify_frozen_sources_antigravity
items = load_candidate_values_antigravity(verify_frozen_sources_antigravity())
print(benchmark_backends_antigravity(items).to_payload())
PY
```

Expected: one selected backend plus versions, device, timing, parity, speed, and fallback reasons; no labels accessed.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/backend_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_backend_antigravity.py
git commit -m "feat(exp444): gate MLX and Numba stability backends"
```

---

### Task 6: Add ordered checkpoint and self-written progress evidence

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/checkpoint_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/progress_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_checkpoint_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_progress_antigravity.py`

**Interfaces:**
- `Exp444CheckpointStoreAntigravity.create(root, context, process) -> Exp444CheckpointStoreAntigravity`
- `Exp444CheckpointStoreAntigravity.open_interrupted(root, context, replacement_process) -> Exp444CheckpointStoreAntigravity`
- `commit_antigravity(sequence: int, result: CandidateStabilityResultAntigravity) -> None`
- `ProgressWriterAntigravity.create(...)`, `start_candidate_antigravity(...)`, `start_view_antigravity(...)`, `complete_candidate_antigravity(...)`, `close_antigravity(status)`.

- [ ] **Step 1: Write RED checkpoint and progress tests**

Test create-only context, exact 0-29 commit order, duplicate/out-of-order rejection, result-file hash in state, one exact-next orphan reconciliation, corrupted orphan rejection, terminal error immutability, distinct replacement PID requirement, live PID resume rejection, stale PID resume, interruption counter restoration, and complete checkpoint finalization-only reopen.

```python
def test_progress_records_current_dimensions_and_eta_antigravity():
    writer.start_candidate_antigravity("A", 7, train_count=20, test_count=50, raw_length=144)
    writer.start_view_antigravity(view_index=3, view_count=16, excluded_count=2)
    state = json.loads(progress.read_text())
    assert state["current_dataset"] == "A"
    assert state["current_candidate_index"] == 7
    assert state["current_view"] == 3
    assert state["raw_length"] == 144
```

- [ ] **Step 2: Run RED tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_progress_antigravity
```

Expected: missing modules.

- [ ] **Step 3: Implement atomic ordered persistence**

Each record is canonical JSON in `records_antigravity/{sequence:04d}_record_antigravity.json`, created exclusively and fsynced before atomic state replacement. Progress contains every user-required field and sets `current_dataset/current_candidate_index/current_view` to null and `eta_seconds` to null when inactive terminal. Events and Korean text logs are append-only; only the parent writes them.

- [ ] **Step 4: Run GREEN tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_progress_antigravity
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/checkpoint_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/progress_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_checkpoint_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_progress_antigravity.py
git commit -m "feat(exp444): persist ordered stability progress"
```

---

### Task 7: Execute 30 candidates with deterministic 11-worker ordering and resume

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/runtime_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_runtime_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_worker_parity_antigravity.py`
- Create: `scratch/verify_exp444_worker_parity_antigravity.py`

**Interfaces:**
- `run_new_antigravity(reviewed_commit: str) -> RuntimeOutcomeAntigravity`
- `run_resume_antigravity(reviewed_commit: str, attempt_name: str) -> RuntimeOutcomeAntigravity`
- `ordered_parallel_map_antigravity(inputs, workers=11, distance_backend="cpu") -> tuple[CandidateStabilityResultAntigravity, ...]`
- `sequential_reference_antigravity(inputs) -> tuple[CandidateStabilityResultAntigravity, ...]`.

- [ ] **Step 1: Write RED runtime tests**

Test exact case-sensitive `(dataset, instance_index)` order, at most `workers**2` pending futures, out-of-order completion with ordered commit, worker exception attribution, 11-worker count, inner thread environment, parent-only writer, SIGINT interruption, exact-prefix resume, source/config/backend context mismatch, complete finalization re-entry, and no evaluator import.

```python
def test_parallel_and_sequential_bytes_match_antigravity():
    sequential = sequential_reference_antigravity(FIXTURE_INPUTS)
    parallel = ordered_parallel_map_antigravity(FIXTURE_INPUTS, workers=3)
    assert canonical_json_bytes_antigravity(parallel) == canonical_json_bytes_antigravity(sequential)
```

- [ ] **Step 2: Run RED tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_worker_parity_antigravity
```

Expected: missing runtime functions.

- [ ] **Step 3: Implement bounded spawn-worker orchestration**

The parent opens/verifies source once, materializes immutable candidate inputs, closes the prediction DB session before worker scheduling, and passes only copied arrays/contracts to spawn workers. Use `multiprocessing.get_context("spawn")`, `ProcessPoolExecutor(max_workers=11, mp_context=context)`, and a buffer keyed by sequence. The MLX backend, if selected, uses a single parent process and no outer process pool.

- [ ] **Step 4: Run GREEN tests and real parity command**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_worker_parity_antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/verify_exp444_worker_parity_antigravity.py
```

Expected: exact 30-result semantic and canonical-byte parity, 11 production workers, finite evidence, and zero label access.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/runtime_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_runtime_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_worker_parity_antigravity.py \
  scratch/verify_exp444_worker_parity_antigravity.py
git commit -m "feat(exp444): run ordered candidate stability replay"
```

---

### Task 8: Freeze stable and unstable review lanes and inspect them read-only

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/reporting_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_reporting_antigravity.py`

**Interfaces:**
- `publish_prediction_antigravity(attempt_root: Path) -> FrozenPublicationAntigravity`
- `inspect_publication_antigravity(publication_root: Path) -> FrozenPublicationAntigravity`.

- [ ] **Step 1: Write RED publication tests**

Assert complete inactive 30-record checkpoint, original-control parity, stable+unstable=30, coordinate disjointness, exact source union, 1,117 dataset summaries, canonical order, no labels/evaluation keys, create-only destination, self-excluding artifact map, tamper detection, and idempotent finalization only when every canonical byte already matches.

```python
def test_publication_conserves_source_coordinates_antigravity():
    frozen = publish_prediction_antigravity(ATTEMPT)
    assert frozen.stable_count + frozen.unstable_count == 30
    assert frozen.stable_coordinates.isdisjoint(frozen.unstable_coordinates)
    assert frozen.stable_coordinates | frozen.unstable_coordinates == frozen.source_coordinates
    assert frozen.label_access_count == 0
```

- [ ] **Step 2: Run RED test**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_reporting_antigravity
```

Expected: missing reporting module.

- [ ] **Step 3: Implement the fixed publication DAG**

Create these canonical leaves before manifest/status/map:

```text
dataset_summary_antigravity.jsonl
source_candidates_antigravity.jsonl
candidate_stability_antigravity.jsonl
view_evidence_antigravity.jsonl
stable_review_prediction_antigravity.jsonl
unstable_review_prediction_antigravity.jsonl
control_replay_antigravity.json
source_audit_antigravity.json
backend_decision_antigravity.json
runtime_antigravity.json
```

Then create `manifest_antigravity.json`, `status_antigravity.json`, and `artifact_sha256_antigravity.json`. `inspect` reparses every contract and recomputes every leaf, manifest, status, and map hash without SQLite, workers, checkpoint mutation, or evaluator import.

- [ ] **Step 4: Run GREEN tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_reporting_antigravity
```

Expected: all publication/replay tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/reporting_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_reporting_antigravity.py
git commit -m "feat(exp444): freeze descriptive stability lanes"
```

---

### Task 9: Evaluate only after prediction freeze and make no winner decision

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/bootstrap_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_bootstrap_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_evaluation_antigravity.py`

**Interfaces:**
- `paired_bootstrap_antigravity(baseline_rows, comparator_rows) -> PairedIntervalsAntigravity`
- `evaluate_frozen_publication_antigravity() -> FrozenEvaluationAntigravity`
- `inspect_evaluation_antigravity() -> FrozenEvaluationAntigravity`.

- [ ] **Step 1: Write RED metric/bootstrap tests**

Test project confusion order `[TP, FP, FN, TN]`, zero-denominator metric = 0.0, simple 1,117-dataset macro mean, NumPy `Generator(PCG64(44420260811))`, 20,000 `int64` paired samples, `endpoint=False`, linear 2.5/97.5 percentiles, and exact replay.

- [ ] **Step 2: Write RED label-boundary and descriptive-status tests**

```python
def test_evaluator_reads_labels_only_after_double_inspection_antigravity():
    result = evaluate_frozen_publication_antigravity(dependencies=SPY)
    assert SPY.events[:2] == ["inspect_publication", "inspect_publication"]
    assert SPY.events[2] == "open_labels"
    assert result.status == "descriptive_complete_no_promotion"
    assert result.scientific_winner_selected is False
```

Cover publication tamper before labels, exactly 1,117 ordered label queries from the Task8-published, Exp440-bound dataset summary (not an Exp442 dataset-summary authority), 195,900 labels, baseline confusion 1,915/303/2,003/191,679, Exp442-origin 30-coordinate reproduction cross-attested by Task8, stable subset relation, retained/excluded TP/FP conservation, rescued/regressed datasets, strong regression table, failure after label access with truthful audit sibling, and existing evaluation replay without reopening labels.

- [ ] **Step 3: Run RED tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_bootstrap_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_evaluation_antigravity
```

Expected: missing evaluator modules.

- [ ] **Step 4: Implement pure evaluation and create-only publication**

Publish exactly these evaluation leaves:

```text
evaluation_antigravity.json
dataset_metrics_antigravity.jsonl
candidate_transitions_antigravity.jsonl
rescued_datasets_antigravity.jsonl
regressed_datasets_antigravity.jsonl
strong_regressions_antigravity.jsonl
bootstrap_antigravity.json
confusion_antigravity.json
decision_antigravity.json
artifact_sha256_antigravity.json
```

`decision_antigravity.json` contains no branchable success value. It records only `status="descriptive_complete_no_promotion"`, `scientific_success_criterion=null`, false promotion flags, and immutable source/prediction/evaluation hashes.

- [ ] **Step 5: Run GREEN tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_bootstrap_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_evaluation_antigravity
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/bootstrap_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/evaluation_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_bootstrap_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_evaluation_antigravity.py
git commit -m "feat(exp444): evaluate stability lanes after freeze"
```

---

### Task 10: Gate readiness and expose a fixed five-command operator boundary

**Files:**
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/readiness_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/cli_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_readiness_antigravity.py`
- Create: `experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_cli_antigravity.py`
- Create: `scratch/run_exp444_train_only_stability_precision_gate_antigravity.py`
- Modify: `docs/EXP444_RUNBOOK_antigravity_KO.md`

**Interfaces:**
- `run_readiness_antigravity(reviewed_commit: str) -> ReadinessEvidenceAntigravity`
- `main_antigravity(argv: tuple[str, ...] | None = None) -> int` with only `readiness`, `new`, `resume`, `inspect`, `evaluate`.

- [ ] **Step 1: Write RED readiness and CLI tests**

Test clean HEAD, exact reviewed commit, approved design/prereg hashes, environment lock, seven pre-NumPy thread controls, absent fixed root, source hashes, 30-coordinate control replay, backend decision, one-worker/11-worker parity, zero label access, and source revalidation immediately before root creation. Reject root/backend/worker/seed/view/fraction/source/threshold/budget path flags.

```python
def test_wrapper_sets_threads_before_first_numpy_import_antigravity():
    completed = subprocess.run([PYTHON, WRAPPER, "thread-probe-antigravity"], capture_output=True, text=True)
    assert json.loads(completed.stdout) == {name: "1" for name in THREAD_ENV_NAMES}
```

Test `inspect` never mutates checkpoint/progress/PID/log/publication bytes, `evaluate` imports the evaluator only after publication inspection, resume accepts exactly one safe `attempt_####_antigravity`, and wrapper exceptions emit one JSON object without corrupting stdout.

- [ ] **Step 2: Run RED tests**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_cli_antigravity
```

Expected: missing readiness/CLI/wrapper.

- [ ] **Step 3: Implement readiness, CLI, and pre-import wrapper**

The wrapper imports only `os`, `sys`, `json`, and `pathlib`, assigns all seven thread variables to `1`, inserts repo root, then imports `cli_antigravity`. The CLI parser exposes no scientific or path override. `new` performs readiness-equivalent source checks inside the same process before root allocation; `resume` revalidates frozen context; `inspect` is read-only; `evaluate` lazy-imports the label-capable module.

- [ ] **Step 4: Run GREEN tests and help audit**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity \
  experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_cli_antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp444_train_only_stability_precision_gate_antigravity.py --help
```

Expected: tests pass and help exposes exactly five commands with only reviewed-commit/attempt lifecycle arguments.

- [ ] **Step 5: Update runbook and commit**

Replace pre-implementation warnings with exact reviewed commands while retaining the no-poll policy and the prohibition on execution before final review.

```bash
git add experiments/exp444_train_only_stability_precision_gate_antigravity/readiness_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/cli_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_readiness_antigravity.py \
  experiments/exp444_train_only_stability_precision_gate_antigravity/tests/test_cli_antigravity.py \
  scratch/run_exp444_train_only_stability_precision_gate_antigravity.py \
  docs/EXP444_RUNBOOK_antigravity_KO.md
git commit -m "feat(exp444): expose fixed stability lifecycle"
```

---

### Task 11: Run full regression, complete the report, sync the map, and obtain final review

**Files:**
- Create: `docs/EXP444_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap_exp444_antigravity.html`
- Modify together: `docs/codemap/codemap_exp444_antigravity.json`
- Modify together: `docs/codemap/codemap_exp444_antigravity.lock`

**Interfaces:**
- Consumes: all Task 1-10 commits and real smoke evidence.
- Produces: a clean reviewed implementation commit eligible for official readiness.

- [ ] **Step 1: Run Exp444 and inherited regressions**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover \
  -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests \
  -p 'test_*_antigravity.py'
python3 -m unittest discover -s experiments/exp440_train_only_low_frequency_precision_gate_antigravity/tests -p 'test_*_antigravity.py'
python3 -m unittest discover -s experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests -p 'test_*_antigravity.py'
python3 -m unittest discover -s experiments/exp442_terminal_eta_publication_repair_antigravity/tests -p 'test_*_antigravity.py'
```

Expected: zero failures and zero errors in every suite.

- [ ] **Step 2: Run real label-free source, control, backend, and worker parity smokes**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/verify_exp444_worker_parity_antigravity.py
```

Record exact source hashes, 30 control matches, view counts, backend timings, lane byte parity, worker count, elapsed time, and label-access count zero. This smoke uses a private temporary root and must not create the official run root.

- [ ] **Step 3: Write the implementation report and update state**

Document exact commits, environment lock, test counts, parity hashes, backend choice, failures fixed during implementation, and the next command `fresh_sol_high_read_only_review_current_clean_head`. Set catalog/state status to `implementation_verified_awaiting_fresh_review`; keep prediction/evaluation hashes null and `test_labels_opened=false`.

- [ ] **Step 4: Regenerate and validate the complete code map**

Map the Exp444 node's verified dependencies on Exp440/442 sources, SQLite values, artifact utilities, runtime, publication, evaluator, and matching tests. Every edge must have a real source symbol. Validate 20-node cap, 3-5 flows, allowed edge types, evidence symbols, HTML embedded JSON equality, lock fingerprints, and unknown relationships.

- [ ] **Step 5: Commit the verified implementation record**

```bash
git diff --check
git add docs/EXP444_IMPLEMENTATION_REPORT_antigravity.md EXPERIMENT_CATALOG.json RESEARCH_STATE.json docs/codemap
git commit -m "docs(exp444): record verified stability implementation"
```

- [ ] **Step 6: Request fresh read-only final review**

Review the exact clean HEAD for Critical/Important/Minor findings, label-boundary violations, source substitution, candidate conservation, mask determinism, backend parity, lifecycle mutation, publication replay, evaluation promotion, and code-map evidence. Do not run official readiness until the verdict is `ship` with no unresolved Critical or Important finding. Fix findings through new RED/GREEN commits and repeat the review on a new clean HEAD.

---

### Task 12: Run official Exp444, freeze predictions, evaluate descriptively, and pause

**Files:**
- Modify: `docs/EXP444_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `docs/EXP444_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `FAILURE_LEDGER.json` when any unfavorable/engineering failure occurs
- Modify: `CHAMPION_LEDGER.json`
- Modify together: `docs/codemap/codemap_exp444_antigravity.html`
- Modify together: `docs/codemap/codemap_exp444_antigravity.json`
- Modify together: `docs/codemap/codemap_exp444_antigravity.lock`

**Interfaces:**
- Produces: immutable official readiness, 30-coordinate stable/unstable prediction publication, post-freeze descriptive evaluation, hashes, ledgers, and a paused research state.

- [ ] **Step 1: Run official readiness on the exact reviewed commit**

```bash
REVIEWED_COMMIT_ANTIGRAVITY="$(git rev-parse HEAD)"
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp444_train_only_stability_precision_gate_antigravity.py \
  readiness --reviewed-commit "$REVIEWED_COMMIT_ANTIGRAVITY"
```

Expected: 1,117 dataset authority, 30 source candidates, 30 original controls, exact source hashes, chosen backend evidence, 11-worker parity, zero label access, and official root still created only by the reviewed lifecycle.

- [ ] **Step 2: Launch one official `new` process and expose logs**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp444_train_only_stability_precision_gate_antigravity.py \
  new --reviewed-commit "$REVIEWED_COMMIT_ANTIGRAVITY"
```

If readiness estimates more than 60 seconds, run the same command in one persistent process, report its PID and the exact `progress_antigravity.json` and `text_antigravity.log` paths, and stop Codex polling. The program must complete/recover from its own checkpoint.

- [ ] **Step 3: Inspect frozen prediction without labels**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp444_train_only_stability_precision_gate_antigravity.py inspect
```

Expected: stable+unstable=30, label access zero, complete publication hash replay, and no file mutation.

- [ ] **Step 4: Run the sole post-freeze evaluator**

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python \
  scratch/run_exp444_train_only_stability_precision_gate_antigravity.py evaluate
```

Expected: exactly 1,117 TEST-label queries after prediction freeze, coherent baseline/Exp442/stable metrics, status `descriptive_complete_no_promotion`, and no winner or promotion.

- [ ] **Step 5: Record exact results without judging success**

Write stable/unstable counts, TP/FP transitions, confusion, micro/macro values, paired intervals, rescued/regressed datasets, source/prediction/evaluation hashes, PID/checkpoint/log paths, backend evidence, and any engineering error into the report and ledgers. Keep `CHAMPION_LEDGER.pending_candidate` empty and mark Exp444 `official_evaluated_descriptive_no_promotion`.

- [ ] **Step 6: Sync code map, validate, commit, and pause**

```bash
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp444_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp444_antigravity.lock >/dev/null
git diff --check
git add docs/EXP444_IMPLEMENTATION_REPORT_antigravity.md docs/EXP444_RUNBOOK_antigravity_KO.md \
  EXPERIMENT_CATALOG.json RESEARCH_STATE.json FAILURE_LEDGER.json CHAMPION_LEDGER.json docs/codemap
git commit -m "docs(exp444): record descriptive stability result"
```

Set `RESEARCH_STATE.status="paused"` and next action `DISCUSS_EXP444_RESULT_BEFORE_NEXT_EXPERIMENT`. Do not start Exp445 automatically.
