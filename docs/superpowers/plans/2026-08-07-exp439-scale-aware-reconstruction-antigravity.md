# Exp439 Scale-Aware Reconstruction Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an isolated Exp439 that preserves Exp438 scientific rules, replaces the false fixed FFT reconstruction guard with a one-float32-ULP scale-aware guard, imports the 89 valid Exp438 prefix records, and names future failing datasets.

**Architecture:** Fork the reviewed Exp438 package under a new Exp439 identity and new immutable roots. Keep feature definitions, scoring, thresholds, frozen lanes, candidate rules, publication, and label boundaries unchanged. Add one numeric guard helper, one exact-prefix importer, and parent-side dataset error attribution.

**Tech Stack:** Python 3.14, NumPy FFT and float32/float64 arrays, SQLite read-only snapshots, `unittest`, 11 multiprocessing workers, optional Numba/MLX/MPS readiness gates, canonical JSON and SHA256.

## Global Constraints

- Every new filename contains `antigravity`.
- The failed Exp438 root remains immutable.
- Exp439 has a new identity, run root, readiness, checkpoint, publication, and evaluation root.
- Prediction never reads TEST labels, anomaly locations, or past family-level TEST performance.
- TEST labels are available only after prediction publication through `evaluate`.
- A TRAIN row with NaN or infinity is excluded whole; no imputation is allowed.
- Official CPU workers are `max logical CPUs - 1`, which is 11 on this Mac.
- MLX, MPS, or Numba is selected only after parity, finite output, tolerance, provenance, and at least 1.1x speedup pass.
- Code-map HTML, JSON, and lock are regenerated together.
- Authored edits use `apply_patch`; deterministic mechanical fork and generated map serialization may use scripts.

---

### Task 1: Create an isolated worktree and freeze Exp439 governance

**Files:**
- Create worktree: `.worktrees/exp439-scale-aware-reconstruction-antigravity`
- Create: `docs/EXP439_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP439_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: approved Exp439 design and failed Exp438 evidence.
- Produces: branch `codex/exp439-scale-aware-reconstruction-antigravity`, frozen identity, exact commands, and synchronized map.

- [ ] **Step 1: Create the worktree**

Use `superpowers:using-git-worktrees`, then run:

```bash
git worktree add .worktrees/exp439-scale-aware-reconstruction-antigravity \
  -b codex/exp439-scale-aware-reconstruction-antigravity
```

- [ ] **Step 2: Compare and synchronize the code map before edits**

```bash
git rev-parse HEAD
git status --short --branch
python3 - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('docs/codemap/codemap.lock').read_text())['current_commit'])
PY
```

- [ ] **Step 3: Freeze literal identity and policy**

The preregistration and runbook record:

```text
experiment_id = Exp439
run_identity = exp439-scale-aware-reconstruction-repair-antigravity-20260807
run_parent = /Users/mac/Documents/Dataset/exp439_scale_aware_reconstruction_repair_antigravity_20260807
predecessor_commit = d1da286e9c74c64abe76e0c255c86c30141813bd
population = 1117
workers = 11
tolerance = max(1e-5, one float32 ULP at maximum component scale)
```

Expose only `readiness`, `new`, `resume`, `inspect`, and `evaluate`.

- [ ] **Step 4: Record state and catalog**

Set Exp438 to `official_failed_immutable_89_of_1117` and Exp439 to
`design_approved_implementation_planned`. Record the exact exception,
`Chinatown_normal_1`, the failed dispatch window, and zero nonfinite rows.

- [ ] **Step 5: Validate and commit**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git add docs/EXP439_PREREGISTRATION_antigravity_KO.md \
  docs/EXP439_RUNBOOK_antigravity_KO.md EXPERIMENT_CATALOG.json \
  RESEARCH_STATE.json docs/codemap/codemap.html docs/codemap/codemap.json \
  docs/codemap/codemap.lock
git commit -m "docs(exp439): freeze reconstruction repair"
```

---

### Task 2: Fork Exp438 under an isolated Exp439 identity

**Files:**
- Create: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/*_antigravity.py`
- Create: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests/test_*_antigravity.py`
- Create: `scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py`

**Interfaces:**
- Consumes: reviewed Exp438 package at commit `d1da286e`.
- Produces: importable Exp439 package with distinct frozen configuration and artifact roots.

- [ ] **Step 1: Mechanically fork policy-compliant files**

Copy only `*_antigravity.py` and `test_*_antigravity.py`. Do not create
`__init__.py`. Replace Exp438/exp438/EXP438 tokens with Exp439 equivalents and
replace `frequency_complexity_evidence` with
`scale_aware_reconstruction_repair` in copied files only.

- [ ] **Step 2: Write and run RED identity tests**

```python
self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp439")
self.assertEqual(
    FROZEN_CONFIG.run_identity,
    "exp439-scale-aware-reconstruction-repair-antigravity-20260807",
)
self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
self.assertEqual(cpu_worker_count_antigravity(12), 11)
```

Run the copied config test and confirm it fails until the new identity is fixed.

- [ ] **Step 3: Freeze config and runner**

Set smoke seed `43901`, bootstrap seed `43920260807`, and root:

```python
RUN_PARENT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp439_scale_aware_reconstruction_repair_antigravity_20260807"
)
```

The runner imports only Exp439 `main_antigravity`.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests \
  -p 'test_config_antigravity.py'
git diff --check
git add experiments/exp439_scale_aware_reconstruction_repair_antigravity \
  scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py
git commit -m "feat(exp439): isolate reconstruction repair runtime"
```

---

### Task 3: Implement the one-ULP guard test-first

**Files:**
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/features_antigravity.py`
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests/test_features_antigravity.py`

**Interfaces:**
- Produces: `reconstruction_tolerance_antigravity(original, low, high) -> float` and unchanged FFT component bytes.

- [ ] **Step 1: Add the real Chinatown RED fixture**

```python
values = np.asarray([
    554.0, 267.0, 172.0, 109.0, 106.0, 77.0, 23.0, 43.0,
    69.0, 159.0, 365.0, 706.0, 1378.0, 1471.0, 1347.0,
    1300.0, 1082.0, 1200.0, 1350.0, 1211.0, 936.0, 813.0,
    713.0, 401.0,
], dtype=np.float32)
```

Assert little-endian float32 SHA256
`33bd5923753b1a3b6e4903ad9e8405d0c761a08f432b7a76c2177921034702a7`
and assert `fft_split_antigravity(values, 1)` succeeds.

- [ ] **Step 2: Verify RED**

Run the single test and confirm the old fixed guard raises
`decomposition reconstruction tolerance failed`.

- [ ] **Step 3: Add a corruption rejection test**

Alter one stored component by four ULPs at component peak and assert the helper
rejects it. Keep the existing NaN and infinity tests.

- [ ] **Step 4: Implement the minimal helper**

```python
def reconstruction_tolerance_antigravity(original, low, high) -> float:
    peak = max(
        1.0,
        float(np.max(np.abs(original))),
        float(np.max(np.abs(low))),
        float(np.max(np.abs(high))),
    )
    tolerance = max(1e-5, float(np.spacing(np.float32(peak))))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("reconstruction tolerance is invalid")
    return tolerance
```

Use it only in the existing guard. Do not change FFT bins, casts, returned
components, depth selection, features, scores, or thresholds.

- [ ] **Step 5: Verify GREEN and parity, then commit**

Prove old-pass inputs return byte-identical Exp438/Exp439 `low` and `high`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp439_scale_aware_reconstruction_repair_antigravity.tests.test_features_antigravity -v
git add experiments/exp439_scale_aware_reconstruction_repair_antigravity/features_antigravity.py \
  experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests/test_features_antigravity.py
git commit -m "fix(exp439): scale reconstruction guard by one ulp"
```

---

### Task 4: Import the exact 89-record Exp438 prefix

**Files:**
- Create: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/predecessor_import_antigravity.py`
- Create: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests/test_predecessor_import_antigravity.py`
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/checkpoint_antigravity.py`
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/runtime_antigravity.py`
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/readiness_antigravity.py`

**Interfaces:**
- Produces: `validate_exp438_prefix_antigravity(root) -> Exp438PrefixImportAntigravity` and all-or-zero import.

- [ ] **Step 1: Write RED prefix tests**

Accept only status failed, 89 records, last dataset `BeetleFly_normal_1`, exact
database/source/context hashes, zero label access, frozen ordered prefix, and
matching record SHA256 values. Reject one altered byte, missing/reordered record,
wrong commit, nonzero label access, or a 90th record.

- [ ] **Step 2: Verify RED and implement read-only validation**

The returned object holds immutable source bytes, parsed result objects,
metadata, SHA256 values, source paths, and predecessor context hash. It never
opens TEST labels or mutates Exp438.

- [ ] **Step 3: Add all-or-zero create-time import**

Before workers start, commit all 89 results in order and write
`import_manifest_antigravity.json`. Set `import_count=89`,
`recompute_count=0`, and emit `prefix_imported`. If validation fails, create no
Exp439 root and never partially import.

- [ ] **Step 4: Verify scheduling continues at position 89**

Assert the first recomputed dataset is `BeetleFly_normal_2`.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp439_scale_aware_reconstruction_repair_antigravity
git commit -m "feat(exp439): import verified exp438 prefix"
```

---

### Task 5: Name worker failures in terminal evidence

**Files:**
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/runtime_antigravity.py`
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/progress_antigravity.py`
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests/test_runtime_antigravity.py`
- Modify: `experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests/test_progress_antigravity.py`

**Interfaces:**
- Produces: `DatasetExecutionErrorAntigravity` and structured `latest_error.dataset`.

- [ ] **Step 1: Write and run RED failure-attribution test**

Make dataset `B` raise `ValueError("synthetic reconstruction failure")` and
assert terminal evidence contains dataset `B`, original type, and message.

- [ ] **Step 2: Implement parent-side attribution**

```python
try:
    result = future.result()
except BaseException as error:
    raise DatasetExecutionErrorAntigravity(dataset, error) from error
```

Write structured fields directly; do not parse traceback text.

- [ ] **Step 3: Verify and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp439_scale_aware_reconstruction_repair_antigravity.tests.test_runtime_antigravity \
  experiments.exp439_scale_aware_reconstruction_repair_antigravity.tests.test_progress_antigravity -v
git add experiments/exp439_scale_aware_reconstruction_repair_antigravity
git commit -m "fix(exp439): name failing datasets in progress"
```

---

### Task 6: Complete regression, code-map, and readiness validation

**Files:**
- Create: `docs/EXP439_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

- [ ] **Step 1: Run all relevant test suites**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests -p 'test_*_antigravity.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/exp438_frequency_complexity_evidence_antigravity/tests -p 'test_*_antigravity.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests -p 'test_*_antigravity.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
```

- [ ] **Step 2: Run the real read-only Chinatown smoke**

Load only its TRAIN values, assert zero exclusions, score without TEST labels,
confirm the old exception is gone, and verify DB link count returns to one.

- [ ] **Step 3: Regenerate and validate the code map**

Retain at most 20 nodes. Validate all paths, symbols, edge/flow references,
embedded HTML payload, module fingerprints, and unknown relationships.

- [ ] **Step 4: Write implementation report and freeze reviewed commit**

Record RED/GREEN evidence, tests, Chinatown SHA and ULP, import proof, zero label
access, backend policy, and exact readiness command. Set state to
`implementation_complete_waiting_readiness`.

- [ ] **Step 5: Commit and run one-shot readiness**

```bash
git diff --check
git add docs/EXP439_IMPLEMENTATION_REPORT_antigravity.md EXPERIMENT_CATALOG.json \
  RESEARCH_STATE.json docs/codemap/codemap.html docs/codemap/codemap.json \
  docs/codemap/codemap.lock
git commit -m "docs(exp439): freeze implementation evidence"
test -z "$(git status --porcelain)"
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py readiness
```

Readiness must report 1,117 datasets, valid 89-record import, zero prediction
label access, backend parity, and a clean reviewed commit.

---

### Task 7: Launch and hand off the official Exp439 run

**Files created by runtime:**
- `/Users/mac/Documents/Dataset/exp439_scale_aware_reconstruction_repair_antigravity_20260807/output_antigravity/*`
- `/Users/mac/Documents/Dataset/exp439_scale_aware_reconstruction_repair_antigravity_20260807/checkpoint_antigravity/*`

- [ ] **Step 1: Start in a persistent session**

```bash
echo $$ > /Users/mac/Documents/Dataset/exp439_scale_aware_reconstruction_repair_antigravity_20260807/launcher_antigravity.pid
exec env PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py new
```

- [ ] **Step 2: Perform one startup handoff check**

Verify live PID, backend, 11 workers, `import_count=89`, current dataset and
metadata, error zero, checkpoint, DB link count two, throughput, and ETA.

- [ ] **Step 3: Stop recurring Codex monitoring**

Report the PID and paths. The process owns heartbeat, logs, ETA, and checkpoint.
After completion, inspect and publish before explicit post-freeze evaluation.
