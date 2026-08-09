# Exp443 Missing Normal TRAIN Policy Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the frozen Exp249 MiniROCKET contribution to the current 2,218-coordinate Hard lane, change only how non-finite normal TRAIN rows are handled, and compare the three preregistered policies on all 1,117 datasets after three label-free prediction freezes.

**Architecture:** A strict source layer pins the database and every inherited artifact, a raw-input layer reads only normal TRAIN values and label-free TEST values, and a pure preprocessing boundary applies one of three normal-TRAIN policies while leaving TEST preprocessing fixed. The unchanged Exp249 MiniROCKET scorer runs in an isolated pinned environment, a control gate must exactly reproduce the 19 affected datasets and the final Hard lane before alternatives can run, and an ordered 57-unit checkpoint drives an 11-worker CPU runtime. Prediction publication is create-only and label-free; a separately imported evaluator opens TEST labels only after all three predictions replay successfully.

**Tech Stack:** Python 3.14.6, NumPy 2.3.5, aeon 1.5.0 official MiniRocket, scikit-learn 1.8.0, SQLite immutable read-only snapshots, `unittest`, `ProcessPoolExecutor` with 11 outer workers and one numerical thread per worker, canonical JSON/JSONL, SHA256, Darwin inode pinning and no-clobber publication. MLX, MPS, and PyTorch are recorded as inapplicable for the official third-party CPU transformer.

## Global Constraints

- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Experiment identity is `Exp443`; run identity is `exp443-missing-normal-train-policy-comparison-antigravity-20260809`.
- The final reviewed design is `docs/superpowers/specs/2026-08-09-exp443-missing-normal-train-policy-comparison-antigravity-design.md` with exact content SHA256 `b52142a0a48f9c557a56180ee96dc1a7a8e46760a15a547aa92f8b5999808f`.
- Run root is `/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809`.
- Isolated interpreter root is `/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314`.
- Prediction may use only raw normal TRAIN values, label-free TEST values and identity, the frozen Exp249 prediction/cache, the frozen Exp291 lanes, the frozen Exp414 missing audit, the frozen Exp404 Hard comparator, and reviewed Exp443 code/config.
- Prediction must not import an evaluator or a TEST-label loader, select a TEST `label`, read `labels_blob`, read actual anomaly positions, read Exp416/Exp442 evaluation, or use historical family TEST performance.
- TEST values always use Exp249's linear interpolation with nearest finite edge extension. Only normal TRAIN handling changes.
- Target length is computed from the original normal TRAIN lengths before any row is repaired or excluded, then fixed across all three policies.
- Policy order is exactly `linear_edge_nearest_control`, `exclude_nonfinite_row`, `leave_one_out_peer_median`; work order is policy-major followed by case-sensitive sorted dataset name.
- Missing inventory is exactly 1,117 datasets, 19 affected datasets, 4 affected families, 164 affected normal TRAIN rows, 983 non-finite scalar values, and zero all-missing rows. Any difference fails before scoring.
- Policy C uses only other normal TRAIN rows of the same original length. It takes the ordinary NumPy median of finite peer values at each missing coordinate and excludes the whole row if any missing coordinate has no finite peer. It never falls back to linear interpolation.
- The unsupported Policy C rows are expected to be exactly two rows: `MelbournePedestrian_normal_8` instance 1 and `MelbournePedestrian_normal_9` instance 242. Their four unsupported coordinates remain audit evidence, not hard-coded repair inputs.
- The MiniROCKET implementation is the existing `experiments.exp249_250_official_mini_multirocket.scoring.score_configuration`; do not fork, rewrite, vectorize, or translate the transformer.
- Fixed model values are kernels 1,024/4,096, actual dimensions 1,008/4,032, seeds 20260721/20261721, maximum dilations 32, five TRAIN folds, KNN 3 local gap, maximum reference 1,000, query chunk 128, per-instance z-normalization, and alpha 0.005 seed intersection.
- Existing `LocalGapModel` centers by the median and scales by the 10th-to-90th percentile range through `RobustScaler(quantile_range=(10, 90))`; do not substitute a 25th-to-75th percentile IQR.
- Frozen routing is `Exp236 Hard OR (Exp236 Standard AND (Mini1024 OR Mini4096)) MINUS frozen exact-normal`.
- Control counts are exactly 2,116 inherited Hard + 137 MiniROCKET promotions = 2,253 pre-exact Hard - 35 exact-normal = 2,218 current Hard.
- Before Policies B/C, Policy A must have finite scores, exact candidate parity at both kernel settings, maximum scalar absolute difference at most `1e-6`, and exact final-coordinate parity with Exp291 and Exp404. Failure status is `control_reproduction_failed`; no B/C result or evaluator may run.
- Official CPU workers are exactly `max(1, os.cpu_count() - 1)`, expected 11 on this 12-logical-CPU Mac. Inner aeon/scikit-learn/BLAS/Numba threads are 1.
- MLX decision is `inapplicable_official_cpu_transformer`; no MLX/MPS-heavy process runs and no GPU rewrite is permitted.
- A real one-worker and 11-worker replay must produce identical semantic score payloads, candidates, routing, and canonical prediction bytes before publication.
- Checkpoint work contains 57 recomputed units. Source import is separately counted as 1,098 unchanged datasets per policy; it is never reported as recomputation.
- Completed checkpoint records are create-only and hash-bound. An interruption can resume only the verified ordered prefix under an inactive PID. An error record is terminal and can never be replaced by success.
- Prediction publication contains three 1,117-row predictions and no labels or evaluation metrics. All three must be frozen and inspected before evaluation imports the label-capable module.
- Baseline confusion must reproduce TP 1,915 / FP 303 / FN 2,003 / TN 191,679 and micro F1 `0.6241851368970013` before policy comparison.
- Dataset precision, recall, F1, and false-positive rate use `0.0` when their denominator is zero. Macro metrics are the simple mean over all 1,117 datasets.
- `rescued` means baseline TP is zero and policy TP is at least one. `regressed` means policy dataset F1 is strictly below baseline dataset F1.
- Bootstrap is NumPy 2.3.5 `Generator(PCG64(44320260809))`, 20,000 paired dataset samples, `int64`, `endpoint=False`, and linear 2.5/97.5 percentiles.
- Each policy is classified independently. Never choose one policy from its TEST result inside Exp443, even when multiple policies pass.
- Exp443 remains adaptive, retrospective, review-only, not independently validated, and not operational-alert eligible. Exp404/Exp416 remains the verified reference.
- Do not start Exp444. After the Exp443 result is recorded, stop for direction discussion.
- Compare `docs/codemap/codemap.lock` before every code-changing task. Before editing a mapped module, use `codemap.json` to identify callers, effects, and tests. Regenerate `codemap.html`, `codemap.json`, and `codemap.lock` together whenever a boundary, dependency, or major flow changes.

## Frozen Input Hashes

```python
FROZEN_INPUT_SHA256_ANTIGRAVITY = {
    "database": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
    "exp249_prediction": "fc72d2dfbaf6a1653ec804dbe349397214992387083b22bdab0e8c26a66876c2",
    "exp249_prediction_manifest": "25ade7c3f8362559a29c15e43fead6070a7fac6cab648f03f62bcb20382b8c8f",
    "exp249_artifact_manifest": "42fee6f9a26b2c6fd19747f83887a431bf5edad144e2af3464033256a0108da9",
    "exp249_cache_manifest": "44c0ecc80ff7cbae383d49d2b8b0cefed442c49b8a12a7bd0cf8569ce121c971",
    "exp249_audit": "144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee",
    "exp291_routing": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
    "exp291_manifest": "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e",
    "exp414_repair_audit": "548f493a1608521bc81e4b742cc061fab7298f0d03c69f23845e37528b26ab91",
    "exp414_manifest": "99556922dc8f5178feaf03bc61eca4bfae628ae8f6b6092c2f6159da8c5dd3cc",
    "exp404_hard": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
    "exp404_manifest": "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8",
}
```

## File Structure

- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/config_antigravity.py`: frozen identities, paths, policies, model values, counts, hashes, workers, metric rules, and JSON-safe config.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/contracts_antigravity.py`: immutable raw-series, policy-audit, score, source, checkpoint, prediction, publication, and evaluation value objects.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/artifacts_antigravity.py`: Exp443 wrappers around verified descriptor reads, canonical serialization, create-only writes, fsync, and Darwin no-clobber rename.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/environment_antigravity.py`: exact interpreter/package/thread/backend verification and package-lock binding.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/database_antigravity.py`: parent-owned inode/hash pin, worker-safe immutable SQLite connections, label-free SQL tracing, and final source revalidation.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/sources_antigravity.py`: strict Exp249/Exp291/Exp414/Exp404 parsing, cache inventory verification, frozen lane reconstruction, and source snapshot.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/preprocessing_antigravity.py`: original target length, three normal-TRAIN policies, fixed TEST repair, alignment, z-normalization, and audit generation.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/minirocket_antigravity.py`: lazy adapter to the unchanged Exp249 scorer and fixed seed-intersection calibration.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/routing_antigravity.py`: pure per-policy promotion and final-Hard reconstruction for all 1,117 datasets.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/control_antigravity.py`: real Policy A archive-score, candidate, routing, and worker-topology parity gate.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/checkpoint_antigravity.py`: ordered 57-unit create-only records, atomic state, context hashes, terminal error semantics, and resume verification.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/progress_antigravity.py`: process-owned atomic progress JSON plus append-only event/text logs and PID heartbeat.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/readiness_antigravity.py`: clean reviewed HEAD, frozen sources, environment, backend, absent-output, and label-firewall proof.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/runtime_antigravity.py`: new/resume orchestration, 11-worker scheduling, ordered buffering, control-before-alternative enforcement, signals, and terminal evidence.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/reporting_antigravity.py`: three create-only prediction publications, complete replay inspection, and artifact SHA map.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/bootstrap_antigravity.py`: the only Exp443 project-metric and paired-dataset bootstrap implementation.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/evaluation_antigravity.py`: post-freeze TEST-label snapshot, confusion, transitions, strong regression, family diagnostics, gates, and evaluation publication.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/cli_antigravity.py`: fixed readiness/new/resume/inspect/evaluate interface with lazy evaluator import.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_antigravity.txt`: direct scientific dependency pins.
- `experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_lock_antigravity.txt`: complete sorted resolved environment lock produced once and reviewed before config binding.
- `scripts/create_exp443_environment_antigravity.sh`: idempotent isolated-environment creator that refuses a mismatched existing environment.
- `scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py`: outside-CWD-safe entrypoint.
- `scratch/verify_exp443_worker_parity_antigravity.py`: read-only one-worker/11-worker semantic parity command.
- `docs/EXP443_PREREGISTRATION_antigravity_KO.md`: frozen Korean preregistration.
- `docs/EXP443_RUNBOOK_antigravity_KO.md`: exact commands, PID/checkpoint/log paths, failure rules, and no-poll monitoring guidance.
- `docs/EXP443_IMPLEMENTATION_REPORT_antigravity.md`: reviewed implementation evidence, tests, environment, smoke, and official result hashes.
- Matching `test_*_antigravity.py` files under the Exp443 `tests` directory cover each boundary.

---

### Task 1: Freeze governance documents and register Exp443 without promotion

**Files:**
- Create: `docs/EXP443_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP443_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: the approved design at SHA256 `b52142a0a48f9c557a56180ee96dc1a7a8e46760a15a547aa92f8b5999808f` and the frozen input table above.
- Produces: the exact scientific contract, commands, artifact paths, research state, and non-promotion status consumed by later code.

- [ ] **Step 1: Verify the corrected approved design bytes**

```bash
test "$(shasum -a 256 docs/superpowers/specs/2026-08-09-exp443-missing-normal-train-policy-comparison-antigravity-design.md | awk '{print $1}')" = "b52142a0a48f9c557a56180ee96dc1a7a8e46760a15a547aa92f8b5999808f"
rg -n "10th-to-90th|분모가 0|rescued" docs/superpowers/specs/2026-08-09-exp443-missing-normal-train-policy-comparison-antigravity-design.md
```

Expected: one matching SHA and all three corrected contract phrases.

- [ ] **Step 2: Write the Korean preregistration**

Include every Global Constraint, the twelve input hashes, three policy definitions, the control fail-closed rule, exact success gates, and this governance block:

```text
official_run_purpose = controlled_missing_normal_train_policy_comparison
adaptive_followup = true
independent_validation = false
external_generalization_supported = false
confirmatory_statistical_claim = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
test_informed_reselection_allowed = false
```

- [ ] **Step 3: Write the runbook before implementation**

Document these eventual commands without claiming they work yet:

```bash
bash scripts/create_exp443_environment_antigravity.sh
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py new --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py resume --reviewed-commit REVIEWED_COMMIT --attempt attempt_0000_antigravity
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py inspect
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py evaluate
```

Mark `REVIEWED_COMMIT` as a runtime argument, never a placeholder stored in the final preregistration. List `progress_antigravity.json`, `events_antigravity.jsonl`, `text_antigravity.log`, `pid_antigravity.json`, `checkpoint_antigravity`, publication, and evaluation paths.

- [ ] **Step 4: Register a planned review-only experiment**

Add exactly one Exp443 catalog entry with status `design_approved_implementation_planned`, the approved design/plan/preregistration/runbook paths, run root, 1,117 datasets, three policies, 57 recompute units, null prediction/evaluation hashes, and the nine governance fields. Set `RESEARCH_STATE.active_experiment` to `Exp443`, `active_phase` to `implementation_planning_complete`, and add one `exp443_missing_train_policy_comparison` object with exact next command `bash scripts/create_exp443_environment_antigravity.sh`. Add a `CHAMPION_LEDGER.history` record that explicitly leaves Exp404/Exp416 as the verified reference and does not put Exp443 in `pending_candidate`.

- [ ] **Step 5: Regenerate the map triplet without inventing code nodes**

Only extend `research_control` evidence with the approved design, plan, preregistration, and runbook. Keep source relationships unknown until implementation exists.

- [ ] **Step 6: Validate and commit**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git add docs/EXP443_PREREGISTRATION_antigravity_KO.md \
  docs/EXP443_RUNBOOK_antigravity_KO.md EXPERIMENT_CATALOG.json \
  RESEARCH_STATE.json CHAMPION_LEDGER.json \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(exp443): freeze missing TRAIN comparison contract"
```

---

### Task 2: Build the isolated environment, frozen config, contracts, and artifact primitives

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_antigravity.txt`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_lock_antigravity.txt`
- Create: `scripts/create_exp443_environment_antigravity.sh`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/config_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/contracts_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/environment_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_contracts_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_artifacts_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_environment_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `frozen_config_payload_antigravity() -> dict[str, object]`
- `scientific_config_sha256_antigravity() -> str`
- `verify_environment_antigravity() -> EnvironmentEvidenceAntigravity`
- Artifact wrappers expose canonical JSON/JSONL, pinned regular reads, create-only writes, fsync, and exclusive rename without replace-style publication.

- [ ] **Step 1: Write RED config and contract tests**

Assert exact policy order, paths, hashes, counts, kernels, seeds, alpha, scaling range `(10, 90)`, workers 11, bootstrap rules, and governance. Mutation of any frozen dataclass field must fail `validate_antigravity()`. Add round-trip and wrong-type tests for raw series, policy audits, score arrays, prediction rows, checkpoint context/state, and evaluation evidence. Prediction contracts must reject label/evaluation keys recursively.

- [ ] **Step 2: Create direct requirements and the environment script**

`requirements_antigravity.txt` contains only:

```text
aeon==1.5.0
numpy==2.3.5
scikit-learn==1.8.0
```

The script must require Python 3.14.6, create the fixed venv with mode 0700, install `pip==26.1.2`, install the direct pins, run `pip check`, and refuse to overwrite a nonmatching existing venv. Its one development-only `--bootstrap-lock-antigravity` mode is allowed only while the tracked lock is absent: it resolves the direct pins and generates sorted `pip freeze --all` bytes. After the lock is committed, the default mode installs every exact lock entry with `--no-deps`, runs `pip check`, and requires byte-identical sorted `pip freeze --all`; it never regenerates the lock. Do not activate or mutate the project `.venv`.

- [ ] **Step 3: Resolve and review the complete lock**

```bash
bash scripts/create_exp443_environment_antigravity.sh --bootstrap-lock-antigravity
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python - <<'PY'
import aeon, numpy, sklearn, platform
assert platform.python_version() == "3.14.6"
assert (aeon.__version__, numpy.__version__, sklearn.__version__) == ("1.5.0", "2.3.5", "1.8.0")
print("EXP443_ENVIRONMENT_OK")
PY
```

Bind the exact generated lock SHA256 in `config_antigravity.py`; do not leave a placeholder hash.

- [ ] **Step 4: Implement minimal frozen types and artifact wrappers**

Reuse the already-tested generic primitives from `experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity`; wrap them only to enforce Exp443 filenames, mode 0600/0700, canonical payloads, and no-clobber behavior. Do not copy the full implementation.

Core immutable types include:

```python
@dataclass(frozen=True)
class RawSeriesAntigravity:
    dataset: str
    split: str
    instance_index: int
    raw_blob_sha256: str
    values: np.ndarray
    missing_positions: tuple[int, ...]

@dataclass(frozen=True)
class PolicyDatasetResultAntigravity:
    policy: str
    dataset: str
    target_length: int
    train_total: int
    train_missing: int
    train_used: int
    train_excluded: int
    test_rows: int
    kernel_scores: tuple[KernelSeedScoresAntigravity, ...]
    candidates_by_kernel: Mapping[int, tuple[int, ...]]
    audits: tuple[RunPolicyAuditAntigravity, ...]
    status: str
```

Arrays are copied, read-only, exact dtype-checked, finite where required, and serialized as deterministic lists plus explicit dtype/shape for checkpoint records.

- [ ] **Step 5: Implement exact environment evidence**

Record Python, platform, machine, logical CPU count, aeon/NumPy/scikit-learn/Numba/SciPy/pandas/threadpoolctl versions, complete lock SHA, BLAS thread pools, and acceleration decision. Fail if direct versions, lock SHA, logical CPUs, worker count, or inner-thread contract differ. Record PyTorch/MLX/MPS versions when installed but classify them as `inapplicable_official_cpu_transformer`.

- [ ] **Step 6: Run GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_config_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_contracts_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_artifacts_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_environment_antigravity -v
git diff --check
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  scripts/create_exp443_environment_antigravity.sh \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): freeze environment and contracts"
```

---

### Task 3: Pin and verify raw database plus all inherited sources

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/database_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/sources_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_database_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_sources_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `pin_database_snapshot_antigravity(database, attempt_root) -> PinnedDatabaseAntigravity`
- `load_raw_dataset_antigravity(pinned_database, dataset) -> RawDatasetAntigravity`
- `scan_missing_inventory_antigravity(pinned_database, population) -> MissingInventoryAntigravity`
- `verify_frozen_sources_antigravity() -> FrozenSourceSnapshotAntigravity`
- `FrozenSourceSnapshotAntigravity` contains 1,117 sorted datasets, 19 affected datasets, row counts/target lengths, Exp236 lanes, Exp249 candidate maps and cache inventory, exact-normal coordinates, Exp291/Exp404 Hard coordinates, and zero label accesses.

- [ ] **Step 1: Write RED inode, SQL, and source-hash tests**

Cover symlink/FIFO rejection, source replacement during read, wrong DB hash, wrong cache hash, duplicate CSV/JSON keys, missing NPZ/metadata, extra cache entry, duplicate dataset, malformed coordinate, and out-of-range candidate. A worker connection must be tied to the parent-created inode pin. Source and pin revalidation after query must detect mutation.

- [ ] **Step 2: Freeze the prediction SQL allowlist**

Only these data queries are allowed:

```sql
SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index
```

```sql
SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
```

Allow `PRAGMA quick_check` and exact population/identity queries. Reject `labels_blob`, a TEST predicate combined with selected `label`, evaluation tables, and any unregistered statement. Tests inspect SQL constants and traced statements.

- [ ] **Step 3: Implement a parent-owned SQLite inode pin**

Open the original DB with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`, require a regular linked file, hash the descriptor once, create a private same-inode hard link named `database_pin_antigravity.db` in the attempt, and keep the descriptor open. Workers open the pin with `mode=ro&immutable=1`, verify inode metadata before and after, and return their SQL trace summary. On close/resume, rehash and verify both the source pathname and pin still name the approved bytes. Remove only the owned hard link after terminal evidence is durable.

- [ ] **Step 4: Parse and validate every inherited artifact**

Require:

- Exp249 prediction: 17,872 rows, 1,117 datasets, exact 4 variants × 4 alphas per dataset, label/evaluation flags false.
- Exp249 cache manifest: 1,117 rows, unique cache/metadata paths, exact config hash, and every declared NPZ plus metadata SHA.
- Exp249 audit: exact versions/config and label flag false.
- Exp291: 1,117 rows and lane counts 2,218 Hard / 35 exact-normal / 806 Standard / 31 Priority.
- Exp414 audit: exact 19/164/983/0 inventory and affected-dataset identity SHA `ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9`.
- Exp404: 2,218 unique Hard coordinates and exact coordinate equality with Exp291 `automatic_hard_indices`.

Read NPZ files from already-pinned bytes with `np.load(io.BytesIO(payload), allow_pickle=False)`; do not trust paths after hashing.

- [ ] **Step 5: Reproduce the frozen routing algebra without scoring**

From Exp249's existing `seed_intersection`, alpha 0.005 rows and its embedded Exp236 lanes, prove all 1,117 rows satisfy 2,116 + 137 - 35 = 2,218 and exactly equal Exp291/Exp404. This source-only proof precedes any raw DB scoring.

- [ ] **Step 6: Verify the real missing inventory**

Use the pinned DB to scan original `<f4` normal TRAIN blobs. Require blob length divisible by four, contiguous unique instance indices, finite TEST identity coverage, 19/164/983/0 counts, 4 families, and exact agreement with Exp414 row blob hashes and missing counts. Do not use Exp414 repaired values as model input.

- [ ] **Step 7: Run GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_database_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_sources_antigravity -v
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): verify frozen raw sources"
```

---

### Task 4: Implement the three normal-TRAIN policies and fixed TEST preprocessing

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/preprocessing_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_preprocessing_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `original_target_length_antigravity(train_rows) -> int`
- `repair_linear_edge_nearest_antigravity(values) -> np.ndarray`
- `apply_train_policy_antigravity(raw, policy, target_length) -> PreparedTrainAntigravity`
- `prepare_dataset_antigravity(raw, policy) -> PreparedDatasetAntigravity`
- Prepared output is finite float32, aligned to the original target length, then row-wise z-normalized exactly like Exp249.

- [ ] **Step 1: Write RED linear-repair tests**

```python
source = np.asarray([np.nan, 1.0, np.nan, 5.0, np.inf], dtype="<f4")
self.assertTrue(np.array_equal(
    repair_linear_edge_nearest_antigravity(source),
    np.asarray([1.0, 1.0, 3.0, 5.0, 5.0], dtype=np.float32),
))
```

Cover `-Inf`, one finite value, all finite, empty input, and all-missing fail-closed. Compare output byte-for-byte with `experiments.exp226_235_hard_alert_quality.preprocessing.sanitize_series` on representative fixtures.

- [ ] **Step 2: Write RED exclusion tests**

Assert any NaN/+Inf/-Inf excludes the complete normal TRAIN row and records dataset, index, original length, exact missing positions, missing-position hash, raw blob hash, and `excluded_nonfinite_row`. Finite rows retain original order and values. Fewer than five used rows yields a policy abstention record, never a zero-filled row.

- [ ] **Step 3: Write RED peer-median tests**

Use unequal-length groups and prove only same-length peers contribute. Prove self-exclusion, ordinary even-count median, missing peers ignored per coordinate, no mask feature reaches the model, and one unsupported coordinate excludes the entire row. Assert the two real unsupported rows and four coordinates are discovered from data, not from branching on dataset name.

- [ ] **Step 4: Freeze cross-policy invariants**

For a mixed fixture and all 19 real datasets require:

- target length is calculated once before policy application;
- every originally finite TRAIN row has identical prepared bytes in A/B/C;
- TEST prepared bytes are identical in A/B/C;
- original TRAIN order is retained among used rows;
- Policy B leaves at least 15 rows per real affected dataset;
- Policy C repairs 162 affected rows and excludes 2;
- no prepared value is non-finite.

- [ ] **Step 5: Implement exact alignment and z-normalization**

Use the same formulas as Exp249's existing foundation:

```python
target = int(np.clip(np.median(original_lengths), 8, 2048))
aligned = np.interp(
    np.linspace(0.0, 1.0, target, dtype=np.float64),
    np.linspace(0.0, 1.0, len(row), dtype=np.float64),
    repaired,
).astype(np.float32)
center = aligned.mean(dtype=np.float32)
scale = aligned.std(dtype=np.float32)
normalized = ((aligned - center) / (scale if scale > 1e-8 else 1.0)).astype(np.float32)
```

Mirror the existing early return exactly: when `len(repaired) == target`, use the float32 repaired row directly and do not call `np.interp`. Test parity against `foundation.load_bundle` for Policy A on all 19 affected datasets.

- [ ] **Step 6: Run GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_preprocessing_antigravity -v
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): apply frozen missing TRAIN policies"
```

---

### Task 5: Adapt the unchanged MiniROCKET scorer and pure routing

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/minirocket_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/routing_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_minirocket_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_routing_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `score_prepared_dataset_antigravity(prepared) -> PolicyDatasetResultAntigravity`
- `seed_intersection_candidates_antigravity(kernel_scores, alpha=0.005) -> tuple[int, ...]`
- `route_policy_antigravity(snapshot, result_by_dataset, policy) -> PolicyPredictionAntigravity`
- `canonical_prediction_bytes_antigravity(prediction) -> bytes`

- [ ] **Step 1: Write RED adapter-boundary tests**

Patch `score_configuration` and assert exactly four calls per dataset: two kernels × two seeds, each with fixed config, expected dimension, max reference 1,000, chunk 128, inner `n_jobs=1`, and the policy-prepared TRAIN plus common TEST matrix. Reject any altered seed/kernel/alpha, non-finite score, wrong score length, or wrong feature dimension.

- [ ] **Step 2: Implement a lazy exact scorer adapter**

Import `RocketModelConfig`, `score_configuration`, and `expected_feature_dimension` inside the scoring function so config/source verification can run without importing aeon. Instantiate the exact MiniROCKET config and call the existing scorer. Apply `empirical_candidates` separately to each seed and intersect their indices for each kernel at alpha 0.005.

- [ ] **Step 3: Write RED routing fixtures**

For each dataset require:

```python
promotion = standard & (mini1024 | mini4096)
pre_exact = inherited_hard | promotion
final_hard = pre_exact - frozen_exact_normal
```

Test disjointness, range checks, duplicate rejection, exact removal even when a new policy again proposes it, and unchanged priority/review evidence. Candidate coordinates outside Standard may be audited but cannot enter Hard.

- [ ] **Step 4: Implement all-population routing**

For the 1,098 unaffected datasets import both verified Exp249 candidate sets. For the 19 affected datasets require one recomputed result. Produce exactly one canonical row per dataset with policy, n_test, Mini1024/Mini4096 candidates, inherited Hard/Standard, promotions, exact-normal removals, final Hard, import/recompute provenance, and both label/evaluation flags false.

- [ ] **Step 5: Prove control routing counts**

Using the inherited Exp249 results as a temporary control fixture, assert 1,117 rows, 137 promotions, 2,253 pre-exact coordinates, 35 removals, and 2,218 final coordinates exactly equal Exp291 and Exp404.

- [ ] **Step 6: Run GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_minirocket_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_routing_antigravity -v
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): reproduce MiniROCKET routing"
```

---

### Task 6: Implement the fail-closed real control gate

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/control_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_control_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `compare_policy_a_to_archive_antigravity(snapshot, results) -> ControlGateResultAntigravity`
- `require_control_gate_antigravity(result) -> None`
- Control evidence records per dataset/kernel/seed array shapes, max absolute difference, candidate equality, routing identity, environment identity, and final pass/fail.

- [ ] **Step 1: Write RED finite/shape/tolerance tests**

Use tiny synthetic score arrays to prove equality passes, exactly `1e-6` passes, greater than `1e-6` fails, shape changes fail, NaN/Inf fail, and candidate mismatch fails even when scalar tolerance passes.

- [ ] **Step 2: Compare every corresponding original scalar**

For each affected dataset, load the verified Exp249 NPZ arrays:

```text
kernels_1024__train_by_seed
kernels_1024__test_by_seed
kernels_4096__train_by_seed
kernels_4096__test_by_seed
```

Compare to Policy A in seed order, require exact shapes, compute finite maximum absolute error in float64, and separately reproduce both seed-intersection candidate sets.

- [ ] **Step 3: Bind candidate and final-Hard parity**

Require candidate equality with both archive-derived empirical candidates and the matching frozen Exp249 prediction rows. Route A with the 1,098 imported datasets and require exact canonical coordinate identity with both Exp291 and Exp404, not only the total 2,218 count.

- [ ] **Step 4: Enforce fail-closed sequencing**

`require_control_gate_antigravity` must be called before the runtime can construct a Policy B/C work unit. A failed gate returns terminal evidence `control_reproduction_failed`, keeps labels unopened, and makes publication/evaluation APIs reject the attempt.

- [ ] **Step 5: Add an opt-in real control test marker**

The normal unit suite uses fixtures. A real test runs only when `EXP443_REAL_CONTROL_ANTIGRAVITY=1`, recomputes all 19 Policy A datasets in the pinned environment, and asserts all control fields. It must not be silently skipped in readiness or the official `new` route.

- [ ] **Step 6: Run fixture GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_control_antigravity -v
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): gate exact control reproduction"
```

---

### Task 7: Add ordered checkpointing and process-owned progress logs

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/checkpoint_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/progress_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_checkpoint_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_progress_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `Exp443CheckpointStoreAntigravity.create_antigravity(root: Path, units: tuple[WorkUnitAntigravity, ...], binding: CheckpointBindingAntigravity, process: ProcessIdentityAntigravity) -> Exp443CheckpointStoreAntigravity`
- `Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(root: Path, binding: CheckpointBindingAntigravity, pid_probe: Callable[[ProcessIdentityAntigravity], bool]) -> Exp443CheckpointStoreAntigravity`
- `commit_result_antigravity(sequence: int, result: PolicyDatasetResultAntigravity) -> str` and `commit_error_antigravity(sequence: int, unit: WorkUnitAntigravity, error: BaseException) -> str`
- `ProgressWriterAntigravity.begin_units_antigravity(active: tuple[WorkUnitAntigravity, ...]) -> None`, `accept_unit_antigravity(result: PolicyDatasetResultAntigravity) -> None`, `heartbeat_antigravity(active: tuple[WorkUnitAntigravity, ...]) -> None`, and `terminal_antigravity(stage: str, latest_error: str | None) -> None`.

- [ ] **Step 1: Write RED ordered-prefix tests**

Create a three-policy/two-dataset fixture. Reject unsorted or duplicate work units, out-of-order commits, wrong policy/dataset, altered config/environment/source/commit hash, missing record, changed record bytes, state count mismatch, and any record following an error. Verify a completed record can never be replaced.

- [ ] **Step 2: Freeze checkpoint layout**

```text
checkpoint_antigravity/
  context_antigravity.json
  state_antigravity.json
  records_antigravity/
    0000_record_antigravity.json
    0001_record_antigravity.json through 0055_record_antigravity.json
    0056_record_antigravity.json
```

Context contains exact ordered units plus reviewed commit, source snapshot SHA, scientific config SHA, environment lock SHA, DB SHA, and run identity. Each canonical record contains deterministic score lists, candidates, audits, timing, SQL trace digest, and status. State is the only atomically replaced file and stores the ordered record SHA prefix plus terminal state.

- [ ] **Step 3: Write RED PID/resume tests**

Use a fake PID probe. Active PID blocks resume; missing/inactive PID permits resume only with a valid prefix. Changed process start identity, stale run identity, terminal error, completed attempt, or malformed heartbeat blocks resume. SIGTERM interruption writes inactive PID and `interrupted_resumable` without inventing a result.

- [ ] **Step 4: Implement required atomic progress schema**

`progress_antigravity.json` contains exact keys for PID/active, current policy/dataset, active datasets, TRAIN total/missing/used/excluded, TEST rows, target length, stage/backend, completed/57, imported/recomputed/success/abstention/error, throughput, elapsed, ETA, checkpoint, heartbeat, and latest error. The parent atomically replaces JSON and appends canonical events plus easy Korean text lines. Workers never write shared logs.

- [ ] **Step 5: Keep heartbeats alive without Codex polling**

The parent waits for worker results with a short timeout and updates its own heartbeat/progress even when no unit completes. `current_dataset` is the lowest-sequence active unit and `active_datasets` lists all submitted units. Timing scalars must be finite; terminal ETA is JSON null.

- [ ] **Step 6: Run GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_progress_antigravity -v
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): checkpoint policy comparison"
```

---

### Task 8: Implement readiness, 11-worker runtime, resume, and worker parity

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/readiness_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/runtime_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_readiness_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_runtime_antigravity.py`
- Create: `scratch/verify_exp443_worker_parity_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_worker_parity_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `run_readiness_antigravity(reviewed_commit) -> ReadinessBindingAntigravity`
- `run_new_antigravity(reviewed_commit) -> CompletedAttemptAntigravity`
- `run_resume_antigravity(reviewed_commit, attempt_name) -> CompletedAttemptAntigravity`
- `run_unit_antigravity(task) -> PolicyDatasetResultAntigravity`
- `verify_worker_parity_antigravity(snapshot) -> WorkerParityAntigravity`

- [ ] **Step 1: Write RED readiness tests**

Reject dirty worktree, wrong HEAD, wrong design/prereg/config/lock SHA, missing source, wrong environment, wrong logical CPU count, output-root collision, symlinked parent, label-firewall violation, and applicable-MLX claims. Readiness must prove source routing algebra and missing inventory but must not compute TEST labels or publish predictions.

- [ ] **Step 2: Add structural prediction-firewall tests**

Walk production AST/imports and SQL constants. Permit only the explicit Exp249 scorer/calibration, Exp441 artifact primitives, Exp417 descriptor-safe connection primitive, and Exp410 family parser. Reject evaluator imports, `load_test_labels*`, Exp416 audit metrics, Exp442 evaluation, TEST label SQL, `labels_blob`, dynamic imports outside the lazy CLI evaluator branch, and fields matching label/evaluation names in prediction payloads.

- [ ] **Step 3: Write RED scheduling tests**

Use fake delayed workers to prove at most 11 tasks are active, inner thread environment is one, completion order does not change checkpoint order, Policy B/C submission cannot happen before A control pass, and a worker exception commits one terminal error without submitting further work. Test SIGTERM and KeyboardInterrupt leave a resumable prefix only when no error record exists.

- [ ] **Step 4: Implement production scheduling**

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `NUMBA_NUM_THREADS` to `1` before importing numerical libraries in workers. Use a spawn `ProcessPoolExecutor(max_workers=11)`. Buffer out-of-order completions by sequence and commit only the next contiguous prefix. Revalidate DB/source/config/environment before scheduling, after A, before B/C, and before terminal evidence.

- [ ] **Step 5: Implement real topology parity**

Run all 19 Policy A units sequentially and with 11 workers from the same raw snapshot and seed. Strip timing/PID fields, then require exact canonical semantic result bytes, exact candidates, exact control evidence, and exact routed prediction bytes. Record sequential/parallel elapsed time and speed ratio; parity is mandatory even if speedup is below 1.1 because the CPU topology is fixed by user policy.

- [ ] **Step 6: Run fixture GREEN and a one-dataset real smoke**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_readiness_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_runtime_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_worker_parity_antigravity -v
EXP443_REAL_SMOKE_ANTIGRAVITY=1 PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_minirocket_antigravity -v
```

The real smoke uses one affected dataset, all fixed kernels/seeds, no TEST labels, and records measured time for ETA planning.

- [ ] **Step 7: Map and commit**

```bash
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  scratch/verify_exp443_worker_parity_antigravity.py \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): run deterministic CPU comparison"
```

---

### Task 9: Publish and inspect three label-free predictions

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/reporting_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_reporting_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `publish_predictions_antigravity(attempt) -> FrozenPublicationAntigravity`
- `inspect_publication_antigravity() -> FrozenPublicationAntigravity`
- Publication uses private same-parent staging, create-only files, fsync, no-clobber rename, retained source/checkpoint descriptors, and complete replay.

- [ ] **Step 1: Write RED publication-boundary tests**

Reject incomplete/failed/control-failed attempts, 56 or 58 records, missing policy, wrong dataset, label/evaluation fields, non-finite values, changed imported candidates, 1,098-coordinate drift, wrong Hard range, duplicate coordinate, altered terminal authority, output collision, symlink/FIFO, partial staging, and artifact-map mismatch.

- [ ] **Step 2: Freeze the exact prediction inventory**

```text
publication_antigravity/
  prediction_linear_edge_nearest_control_antigravity.jsonl
  prediction_exclude_nonfinite_row_antigravity.jsonl
  prediction_leave_one_out_peer_median_antigravity.jsonl
  policy_score_audit_antigravity.jsonl
  train_policy_audit_antigravity.jsonl
  control_gate_antigravity.json
  worker_parity_antigravity.json
  source_audit_antigravity.json
  backend_decision_antigravity.json
  manifest_antigravity.json
  status_antigravity.json
  artifact_sha256_antigravity.json
```

Each prediction is exactly 1,117 case-sensitive sorted rows. The manifest contains three separate prediction SHA256 values and asserts labels/evaluation absent. The artifact map binds every other file and does not hash itself.

- [ ] **Step 3: Implement complete read-only replay**

Inspection pins the exact regular-file inventory, parses canonical bytes, reverifies all inherited sources and checkpoint records, reroutes all policies, reconstructs all publication bytes, and requires byte identity. It opens no SQLite label query and reports label access zero.

- [ ] **Step 4: Test create-only failure preservation**

Inject failures before file write, after one file, before rename, and after rename verification. Preserve failed staging as a uniquely named `_failed_*_antigravity` sibling with failure evidence. Never replace an existing publication.

- [ ] **Step 5: Run GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_reporting_antigravity -v
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): freeze three policy predictions"
```

---

### Task 10: Implement the post-freeze evaluator and paired bootstrap

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/bootstrap_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_bootstrap_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- `project_metrics_antigravity(confusion) -> MetricSetAntigravity`
- `paired_dataset_bootstrap_antigravity(rows) -> BootstrapResultAntigravity`
- `load_frozen_test_labels_antigravity(publication) -> FrozenLabelsAntigravity`
- `compute_policy_evaluation_antigravity(publication, labels) -> EvaluationResultAntigravity`
- `evaluate_publication_antigravity() -> FrozenEvaluationAntigravity`

- [ ] **Step 1: Write RED metric and bootstrap tests**

Test exact confusion, zero-denominator behavior, macro mean over every dataset, rescued/regressed definitions, strong baseline threshold `>=0.8`, added/removed TP/FP, and family grouping. Use a tiny injectable bootstrap population to test deterministic PCG64 indices, paired resampling, int64/endpoint false, and linear quantiles. Production must reject NumPy other than 2.3.5, population other than 1,117, or samples other than 20,000.

- [ ] **Step 2: Freeze the post-publication label boundary**

Evaluation calls `inspect_publication_antigravity()` twice before importing/opening labels. Then it uses exactly one pinned DB snapshot and one ordered query per dataset:

```sql
SELECT i.instance_index, i.label
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
```

It requires contiguous identity and `0`/`1` labels, closes the connection, rehashes/replays publication and source again, and records pre-freeze label access zero. No TRAIN or prediction setting can be changed afterward.

- [ ] **Step 3: Reproduce the baseline before comparing policies**

Build the Exp404/Exp291 baseline coordinate set from the frozen publication authority and require exact confusion `(1915, 303, 2003, 191679)` plus F1 `0.6241851368970013`. Require Policy A coordinate and confusion identity with baseline. Any mismatch fails evaluation publication.

- [ ] **Step 4: Compute each policy independently**

For A/B/C produce 1,117 dataset rows with baseline/policy confusion, precision, recall, F1, FPR, delta, added/removed TP/FP, rescued/regressed/strong flags, affected flag, and family. Aggregate whole-population and 19-dataset diagnostics. Verify no final-Hard change outside the 19 affected datasets.

- [ ] **Step 5: Apply the complete success gate without choosing a winner**

For each policy independently require micro delta > 0, macro F1 delta > 0, both CI lower bounds > 0, FP <= 303, rescued >= 1, zero strong regressions, exact 1,098-dataset invariance, zero pre-freeze labels, zero errors, and finite outputs. Report each passing policy as `retrospective_improvement_signal`; otherwise `no_retrospective_improvement_signal`. A may only reproduce baseline and is not expected to pass.

- [ ] **Step 6: Freeze evaluation inventory**

```text
evaluation_antigravity/
  evaluation_antigravity.json
  dataset_metrics_antigravity.jsonl
  coordinate_transitions_antigravity.jsonl
  rescued_datasets_antigravity.jsonl
  regressed_datasets_antigravity.jsonl
  family_diagnostics_antigravity.jsonl
  confusion_antigravity.json
  bootstrap_antigravity.json
  decision_antigravity.json
  artifact_sha256_antigravity.json
```

Use create-only staging and full replay. Decision names every policy status and explicitly sets `scientific_winner_selected=false`, `review_only=true`, and `operational_alert_eligible=false`.

- [ ] **Step 7: Run GREEN, map, and commit**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_bootstrap_antigravity \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_evaluation_antigravity -v
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): evaluate frozen TRAIN policies"
```

---

### Task 11: Add the lazy CLI, wrapper, implementation report, and full verification

**Files:**
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/cli_antigravity.py`
- Create: `scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py`
- Create: `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_cli_antigravity.py`
- Create: `docs/EXP443_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `docs/EXP443_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Fixed commands: `readiness`, `new`, `resume`, `inspect`, `evaluate`.
- `evaluate` is the only branch that lazily imports `evaluation_antigravity`.
- Wrapper inserts the repo root derived from its own path and rejects execution with the wrong interpreter/environment.

- [ ] **Step 1: Write RED CLI/import tests**

Assert exact commands/options, JSON-safe status output, nonzero exit on failure, no worker override, no policy/alpha/kernel/seed override, and required reviewed commit for readiness/new/resume. Patch imports and prove readiness/new/resume/inspect never import the evaluator; only evaluate does so after publication inspection.

- [ ] **Step 2: Implement the minimal CLI and wrapper**

Keep scientific settings out of CLI flags. `resume` accepts only a basename matching `attempt_[0-9]{4}_antigravity`; absolute paths and traversal fail. The wrapper works outside the repo CWD and uses no product `.venv` fallback.

- [ ] **Step 3: Run complete regression and static checks**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest discover \
  -s experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests \
  -p 'test_*_antigravity.py' -v
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest discover \
  -s experiments/exp249_250_official_mini_multirocket/tests -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m compileall -q \
  experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py \
  scratch/verify_exp443_worker_parity_antigravity.py
git diff --check
```

- [ ] **Step 4: Run the real source/preprocessing smoke and full Policy A control**

```bash
EXP443_PYTHON=/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python
EXP443_REAL_CONTROL_ANTIGRAVITY=1 PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" -m unittest \
  experiments.exp443_missing_normal_train_policy_comparison_antigravity.tests.test_control_antigravity -v
PYTHONDONTWRITEBYTECODE=1 "$EXP443_PYTHON" scratch/verify_exp443_worker_parity_antigravity.py
```

This is long-running but mandatory. Record elapsed times, speed ratio, all max score differences, candidate parity, 2,218-coordinate parity, and label access zero.

- [ ] **Step 5: Write the implementation report and update planned state**

Record exact implementation commit candidate, test counts, environment lock/hash, MLX decision, real control evidence, worker parity, known limitations, and exact official next command. Set catalog/state to `implementation_verified_awaiting_fresh_review`; do not claim an official prediction exists.

- [ ] **Step 6: Regenerate and validate the final implementation map**

Map no more than 20 primary nodes by grouping Exp443 source/input/preprocessing/scoring/runtime/publication/evaluation files. Add evidenced flows:

1. readiness/source pin;
2. raw normal TRAIN policy processing;
3. MiniROCKET score and control gate;
4. checkpointed 11-worker prediction publication;
5. post-freeze evaluation.

Validate every evidence symbol/path and exact HTML/JSON payload parity.

- [ ] **Step 7: Commit the complete reviewed implementation candidate**

```bash
git add experiments/exp443_missing_normal_train_policy_comparison_antigravity \
  scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py \
  scratch/verify_exp443_worker_parity_antigravity.py \
  docs/EXP443_IMPLEMENTATION_REPORT_antigravity.md docs/EXP443_RUNBOOK_antigravity_KO.md \
  EXPERIMENT_CATALOG.json RESEARCH_STATE.json \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp443): complete missing TRAIN comparison"
```

- [ ] **Step 8: Perform a fresh read-only review**

Review the final diff against the approved design and preregistration. Require zero critical/important findings, rerun any affected tests after fixes in new commits, regenerate the map with each boundary fix, and bind the final clean reviewed HEAD in the implementation report/state before official readiness.

---

### Task 12: Execute the official 57-unit label-free run and stop automatic monitoring

**Files:**
- No product-code modification.
- Runtime writes only under `/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809`.

- [ ] **Step 1: Verify a clean reviewed checkout and absent official root**

```bash
git status --short --branch
git rev-parse HEAD
test ! -e /Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809
```

Expected: clean branch, one reviewed commit, absent run root.

- [ ] **Step 2: Run readiness**

```bash
EXP443_REVIEWED_COMMIT_ANTIGRAVITY="$(git rev-parse HEAD)"
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python \
  scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py \
  readiness --reviewed-commit "$EXP443_REVIEWED_COMMIT_ANTIGRAVITY"
```

Verify readiness reports exact sources, environment, CPU 11, MLX inapplicable, label access zero, and no publication/evaluation.

- [ ] **Step 3: Launch one official background process**

Use the runbook's exact `nohup` command so the process, not Codex, owns progress updates. Record the returned PID once. Do not start another MiniROCKET/MLX/MPS-heavy process.

The user-visible paths are:

```text
/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809/attempts_antigravity/attempt_0000_antigravity/progress_antigravity.json
/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809/attempts_antigravity/attempt_0000_antigravity/events_antigravity.jsonl
/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809/attempts_antigravity/attempt_0000_antigravity/text_antigravity.log
/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809/attempts_antigravity/attempt_0000_antigravity/pid_antigravity.json
/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809/attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity
```

- [ ] **Step 4: Do not consume tokens by polling**

Return the PID/log/checkpoint paths and ETA written by the process. Wait for the user to say the process completed or failed. If interrupted, use the exact `resume` command only after PID inactivity and checkpoint inspection; never delete the attempt.

- [ ] **Step 5: Inspect the completed prediction publication**

After the user reports completion:

```bash
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python \
  scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py inspect
```

Require 57 successful recomputes, zero errors/non-finite values, control pass, worker parity, 3 × 1,117 prediction rows, 1,098 unaffected imports per policy, and three frozen SHA256 values.

---

### Task 13: Evaluate once, record the result, and pause for direction discussion

**Files:**
- Modify after result: `docs/EXP443_IMPLEMENTATION_REPORT_antigravity.md`
- Modify after result: `EXPERIMENT_CATALOG.json`
- Modify after result: `RESEARCH_STATE.json`
- Modify after result: `FAILURE_LEDGER.json` only if no alternative policy passes all gates or the official attempt fails.
- Modify after result: `CHAMPION_LEDGER.json` to record review-only evidence without changing the verified reference.
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

- [ ] **Step 1: Open TEST labels exactly once after the three freezes**

```bash
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python \
  scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py evaluate
```

No same-experiment rerun, threshold change, policy change, or winner selection follows this command.

- [ ] **Step 2: Replay evaluation independently**

Run `evaluate` again in inspect-only behavior: it must detect and replay the existing create-only evaluation rather than query labels or overwrite files. Recompute confusion/bootstrap from pinned published bytes in a test-only temporary destination and require exact canonical payload identity.

- [ ] **Step 3: Classify and record all three policies**

Report in easy Korean:

- what changed in normal TRAIN rows;
- whether Policy A reproduced the current model;
- full confusion and F1 for A/B/C;
- added/removed TP/FP;
- rescued and regressed datasets;
- strong-dataset preservation;
- micro/macro paired CIs;
- 19-dataset diagnostic versus all 1,117;
- each independent gate status;
- why no operational promotion or within-experiment winner is allowed.

- [ ] **Step 4: Update research control files truthfully**

If B or C passes every gate, record each passing arm as `retrospective_improvement_signal_review_only` and leave Exp404/Exp416 as the verified reference. If neither passes, append one Exp443 failure-ledger entry with exact prediction/evaluation hashes and failed gates. Set `RESEARCH_STATE` to `paused_after_exp443_for_direction_discussion`, clear active PID, and set no Exp444 next command.

- [ ] **Step 5: Regenerate map, validate, and commit result evidence**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git add docs/EXP443_IMPLEMENTATION_REPORT_antigravity.md \
  EXPERIMENT_CATALOG.json RESEARCH_STATE.json FAILURE_LEDGER.json CHAMPION_LEDGER.json \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(exp443): record official policy comparison"
```

- [ ] **Step 6: Stop and discuss the next direction**

Do not preregister or implement Exp444 automatically. Present the Exp443 evidence and practical choices to the user in simple terms.

---

## Final Acceptance Checklist

- [ ] All new filenames contain `antigravity`.
- [ ] Design, preregistration, config, environment lock, reviewed commit, source snapshot, checkpoint, prediction, and evaluation hashes form one replayable chain.
- [ ] Policy A exactly reproduces original score arrays within `1e-6`, both candidate sets exactly, and the full 2,218-coordinate Hard baseline exactly.
- [ ] Policies B/C cannot begin before the control pass.
- [ ] Normal TRAIN is the only changing input; TEST prepared bytes are identical across policies.
- [ ] Missing inventory is exactly 19/164/983/0; Policy C repairs 162 and excludes 2.
- [ ] Official MiniROCKET is unchanged and runs CPU-only with 11 outer/1 inner workers.
- [ ] One-worker and 11-worker semantic prediction bytes match.
- [ ] Atomic progress includes PID, current policy/dataset, sizes, stage/backend, counts, rate, elapsed, ETA, checkpoint, heartbeat, and latest error.
- [ ] Resume accepts only a valid inactive ordered prefix and never changes an error to success.
- [ ] Three label-free predictions freeze and inspect before any TEST-label access.
- [ ] Baseline confusion and F1 reproduce exactly before comparison.
- [ ] Evaluation uses the exact paired bootstrap and all preregistered success gates.
- [ ] No policy is selected from Exp443 TEST results and no operational promotion occurs.
- [ ] Code map triplet is current, internally consistent, and evidence-backed.
- [ ] Exp443 result is committed and the project pauses without starting Exp444.
