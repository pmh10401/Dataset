# Deferred Deep Public-Baseline Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare, without starting training, independent resumable launch/status/verify tooling for the frozen 50-epoch Deep SVDD 1D and DROCC 1D public baselines.

**Architecture:** Add one operational wrapper around the unchanged `experiments.paper_public_baselines` CLI. The wrapper pins an immutable launch authority, runs each deep method in its own detached one-method root, monitors create-only shard/meta pairs, and exposes read-only status and frozen-root verification; it never changes the model, scorer, runner, artifact schema, or label boundary.

**Tech Stack:** Python 3 standard library (`argparse`, `fcntl`, `hashlib`, `json`, `os`, `pathlib`, `subprocess`, `time`), existing public-baseline canonical artifact helpers, PyTorch runtime inspection, `unittest`.

## Global Constraints

- Do not start Deep SVDD or DROCC training, do not create either future official run root, and do not run a deep smoke while proposal r3 is live.
- Do not modify `paper/equipment_anomaly_detection/**`, `paper/equipment_anomaly_detection/codemap_paper.*`, `/Users/mac/Documents/Dataset/univariate_ts.db`, or `/Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813`.
- Do not modify `experiments/paper_public_baselines` scoring, model, runner, CLI, evaluation, reporting, or artifact-schema code.
- Preserve the existing mixed-method partial root `/Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1` byte-for-byte.
- Official deep configuration is immutable: seed `20260812`, `50` epochs, one outer worker, one PyTorch thread, CPU backend, frozen 1,117-dataset population, alphas `0.005`, `0.01`, `0.02`.
- Assess PyTorch, MLX, and MPS in launch authority. CPU stays canonical because no exact MLX/MPS path preserves frozen PyTorch initialization, optimizer, autograd, score, and routing; an accelerated path would additionally require same-input/seed finite parity, identical candidates at every alpha, score tolerance `1e-5`, and at least `1.1x` end-to-end speed including transfer/synchronization.
- Record detected logical CPU count and actual worker count. The deep stage is frozen sequential work, so the logical-CPU-minus-one policy is recorded as inapplicable rather than changing the official one-worker algorithm.
- Every newly created source, test, design, plan, control, and log filename contains `antigravity`. Existing v1 core runtime filenames such as `prepared.json` and hash-token shards remain unchanged because they are an already-frozen artifact schema.
- The wrapper must not import `experiments.paper_public_baselines.label_access`; verify must reject incomplete or tampered roots before labels can be opened elsewhere.
- Use `apply_patch` for edits. Preserve concurrent/user changes. Commit only owned files.

---

### Task 1: Restore Exp448 code-map coverage before implementation

**Files:**
- Modify together: `docs/codemap/codemap_exp448_antigravity.html`
- Modify together: `docs/codemap/codemap_exp448_antigravity.json`
- Modify together: `docs/codemap/codemap_exp448_antigravity.lock`

**Interfaces:**
- Consumes: active version `Exp448` from `RESEARCH_STATE.json`; the lock’s documented `sha256-v1` module-fingerprint algorithm.
- Produces: a byte-consistent triplet that answers callers, effects, and tests for the existing public-baseline deep closure before any wrapper/test source edit.

- [ ] **Step 1: Confirm the current coverage failure and clean scope**

Run:

```bash
git status --short
jq -e '.nodes | any(.path == "experiments/paper_public_baselines/methods/deep_common.py")' \
  docs/codemap/codemap_exp448_antigravity.json
```

Expected: worktree clean; `jq` exits `1`, proving the active map cannot answer the required module boundary.

- [ ] **Step 2: Regenerate the complete triplet together**

Use `apply_patch` to add these existing boundaries to both the canonical JSON and the exact JSON payload embedded in HTML:

```text
public_baseline_cli
  path: experiments/paper_public_baselines/cli.py
  callers: scratch/run_paper_public_baselines.py
  effects: prepare/assess-mlx/score/freeze dispatch with lazy evaluation imports
  tests: test_cli.py

public_baseline_runner
  path: experiments/paper_public_baselines/runner.py
  callers: cli.py, test_runner.py
  effects: score_published_method -> conformal alerts -> create-only shards -> frozen manifest
  tests: test_runner.py

public_baseline_deep_common
  path: experiments/paper_public_baselines/methods/deep_common.py
  callers: deep_svdd.py, drocc.py, test_deep_methods.py
  effects: deterministic CPU 5-fold TRAIN OOF plus separate full-fit TEST scores
  tests: test_deep_methods.py, test_runner.py source-hash contract

public_baseline_deep_svdd
  path: experiments/paper_public_baselines/methods/deep_svdd.py
  callers: methods/__init__.py, test_deep_methods.py
  effects: Deep SVDD ScoreResult
  tests: test_deep_methods.py

public_baseline_drocc
  path: experiments/paper_public_baselines/methods/drocc.py
  callers: methods/__init__.py, test_deep_methods.py
  effects: DROCC ScoreResult
  tests: test_deep_methods.py
```

Add exact import/call edges:

```text
public_baseline_cli -> public_baseline_runner
public_baseline_runner -> public_baseline_deep_svdd
public_baseline_runner -> public_baseline_drocc
public_baseline_deep_svdd -> public_baseline_deep_common
public_baseline_drocc -> public_baseline_deep_common
```

Add a flow named `Deferred published deep-baseline execution` whose steps are
`public_baseline_cli`, `public_baseline_runner`, `public_baseline_deep_svdd`,
`public_baseline_drocc`, and `public_baseline_deep_common`, and whose outcome states
that 50-epoch CPU score roots remain label-free until frozen evaluation.

Set JSON/HTML `generated_from_commit` and lock `current_commit` to the clean current
HEAD, update the timestamp once, set `working_tree_dirty` to `true` while the triplet
is staged, and recompute all lock module fingerprints exactly as documented in the
lock. `docs/codemap` remains excluded from fingerprint input.

- [ ] **Step 3: Verify structural and hash consistency**

Run this read-only validator:

```bash
python3 - <<'PY'
import hashlib, json, re, subprocess
from pathlib import Path

root = Path('.')
json_path = root / 'docs/codemap/codemap_exp448_antigravity.json'
html_path = root / 'docs/codemap/codemap_exp448_antigravity.html'
lock_path = root / 'docs/codemap/codemap_exp448_antigravity.lock'
data = json.loads(json_path.read_text())
html = html_path.read_text()
match = re.search(r'<script id="codemap-data" type="application/json">(.*?)</script>', html, re.S)
assert match and json.loads(match.group(1)) == data
ids = {node['id'] for node in data['nodes']}
assert len(ids) == len(data['nodes'])
assert all(edge['from'] in ids and edge['to'] in ids for edge in data['edges'])
assert all(set(flow['steps']) <= ids for flow in data['flows'])
required = {
    'public_baseline_cli', 'public_baseline_runner', 'public_baseline_deep_common',
    'public_baseline_deep_svdd', 'public_baseline_drocc',
}
assert required <= ids
lock = json.loads(lock_path.read_text())
assert set(lock['module_fingerprints']) >= {'root','docs','experiments','paper','scratch'}
print('codemap-preflight', len(data['nodes']), len(data['edges']), len(data['flows']))
PY
```

Expected: one `codemap-preflight` line and exit `0`.

- [ ] **Step 4: Commit the pre-edit triplet**

```bash
git add docs/codemap/codemap_exp448_antigravity.html \
  docs/codemap/codemap_exp448_antigravity.json \
  docs/codemap/codemap_exp448_antigravity.lock
git commit -m "docs: map deferred public deep baselines"
```

Expected: exactly the three triplet files committed together.

---

### Task 2: Implement the no-training deferred launcher with TDD

**Files:**
- Create: `scratch/manage_deferred_deep_baselines_antigravity.py`
- Create: `experiments/paper_public_baselines/tests/test_deferred_deep_launcher_antigravity.py`

**Interfaces:**
- Consumes: existing `scratch/run_paper_public_baselines.py`, `canonical_json_bytes`, `read_strict_json`, `sha256_file`, `validate_score_payload`, `verify_score_root`, `EXPECTED_DATASETS`, `EXPECTED_DB_SHA256`, `SEED`.
- Produces: `main(argv: Sequence[str] | None = None) -> int` with public `launch`, `status`, and `verify`; internal `_worker`; `ops_paths(run_root: Path) -> OpsPaths`; `inspect_status(run_root: Path, now_epoch: float | None = None) -> dict[str, object]`.

- [ ] **Step 1: Write the failing launcher boundary tests**

Create the test file first. It must import the future script by path using
`importlib.util.spec_from_file_location` and contain these behavior tests:

```python
def test_import_does_not_create_process_or_files(self):
    before = set(self.root.iterdir())
    self.load_launcher()
    self.assertEqual(set(self.root.iterdir()), before)

def test_parser_accepts_only_two_deep_methods(self):
    module = self.load_launcher()
    self.assertEqual(module.parse_method("deep_svdd_1d"), "deep_svdd_1d")
    self.assertEqual(module.parse_method("drocc_1d"), "drocc_1d")
    with self.assertRaisesRegex(ValueError, "deep method"):
        module.parse_method("knn_distance")

def test_stage_commands_freeze_official_order_and_configuration(self):
    module = self.load_launcher()
    commands = module.stage_commands(
        "deep_svdd_1d", Path("/db.sqlite"), Path("/run"), Path("/python")
    )
    self.assertEqual([stage for stage, _ in commands],
                     ["prepare", "assess_mlx", "score", "freeze"])
    self.assertIn("--epochs", commands[2][1])
    self.assertEqual(commands[2][1][commands[2][1].index("--epochs") + 1], "50")
    self.assertEqual(commands[2][1][commands[2][1].index("--workers") + 1], "1")

def test_status_counts_only_verified_complete_shard_meta_pairs(self):
    # Build a real one-dataset score root with ScoreRunConfig + CountingScorer.
    # Write a matching antigravity authority/state in the sibling ops root.
    # Assert completed_pairs == 1, expected_pairs == 1, abstained_pairs == 0.
    # Then remove only the .meta file and assert status == "failed" with
    # failure_kind == "partial_shard_pair".

def test_live_pid_with_stale_heartbeat_is_not_running(self):
    # Use os.getpid() as a real live PID and heartbeat_epoch = now - 181.
    # Assert status == "failed" and failure_kind == "stale_heartbeat".

def test_duplicate_launch_refuses_live_worker(self):
    # Write authority and pid_antigravity.txt containing os.getpid().
    # Call launch with a spawn callable that raises if reached.
    # Assert ValueError containing "already running" and no spawn call.

def test_verify_never_imports_label_access(self):
    sys.modules.pop("experiments.paper_public_baselines.label_access", None)
    with self.assertRaisesRegex(ValueError, "manifest"):
        self.load_launcher().verify_frozen(self.root / "incomplete")
    self.assertNotIn("experiments.paper_public_baselines.label_access", sys.modules)
```

The fixture must derive expected JSON literals independently and assert real files,
exit state, and return values rather than asserting mock calls.

- [ ] **Step 2: Run RED and confirm the missing-module failure**

Run:

```bash
python3 -m unittest \
  experiments.paper_public_baselines.tests.test_deferred_deep_launcher_antigravity -v
```

Expected: FAIL because
`scratch/manage_deferred_deep_baselines_antigravity.py` does not exist. Record the
exact failure before creating the launcher.

- [ ] **Step 3: Implement immutable constants, paths, and authority**

Create the launcher with these exact constants and path contract:

```python
ALLOWED_METHODS = ("deep_svdd_1d", "drocc_1d")
OFFICIAL_EPOCHS = 50
OFFICIAL_WORKERS = 1
HEARTBEAT_INTERVAL_SECONDS = 15.0
HEARTBEAT_STALE_SECONDS = 180.0

@dataclass(frozen=True, slots=True)
class OpsPaths:
    root: Path
    authority: Path
    authority_sha256: Path
    pid: Path
    state: Path
    progress_log: Path
    worker_lock: Path
```

`ops_paths(Path("/x/run"))` must resolve to sibling directory
`/x/run_ops_antigravity` with filenames:

```text
authority_antigravity.json
authority_sha256_antigravity.txt
pid_antigravity.txt
state_antigravity.json
progress_antigravity.log
worker_lock_antigravity.lock
```

The create-only authority schema is `paper-public-deep-launch-authority-antigravity.v1`
and includes method, resolved DB/run/repository/script paths, DB SHA-256, expected
dataset/pair count 1,117, epochs 50, seed 20260812, workers 1, all three alphas,
resolved executable and Python version, Git commit and clean flag, wrapper SHA-256,
the existing public-baseline scoring closure hashes, detected logical CPU count,
actual worker count, and this backend object:

```python
{
    "selected_backend": "cpu",
    "pytorch_version": version("torch"),
    "mlx_version": version("mlx"),
    "mlx_metal_version": version("mlx-metal"),
    "mps_built": bool(torch.backends.mps.is_built()),
    "mps_available": bool(torch.backends.mps.is_available()),
    "tensor_eligible": True,
    "exact_accelerated_path_available": False,
    "transfer_compile_overhead_benchmarked": False,
    "cpu_fallback": "frozen deterministic PyTorch CPU implementation",
    "fallback_reason": "No exact MLX/MPS path preserves the frozen PyTorch initialization, optimizer, autograd, score, and routing.",
    "canonical_gate": {
        "same_input_and_seed": True,
        "finite_outputs": True,
        "identical_candidates_all_alphas": True,
        "score_absolute_tolerance": 1e-5,
        "minimum_end_to_end_speedup": 1.1,
        "include_transfer_and_synchronization": True,
    },
}
```

Official launch rejects a non-clean Git worktree, DB hash mismatch, an existing
different authority, a non-absolute/equal DB or run-root collision, or an authority
whose current executable/commit/wrapper/scoring closure differs.

- [ ] **Step 4: Implement stage commands and detached launch**

`stage_commands(method, db_path, run_root, executable)` returns four tuples in this
exact order, each invoking the existing `scratch/run_paper_public_baselines.py` with
`-B -u`:

```text
prepare --db-path DB --run-root ROOT --methods METHOD
assess-mlx --run-root ROOT
score --run-root ROOT --methods METHOD --workers 1 --epochs 50 --db-path DB
freeze --run-root ROOT
```

`launch` creates/verifies authority, acquires `fcntl.LOCK_EX|LOCK_NB` on the persistent
worker lock, rejects a live existing PID, and starts this script’s hidden `_worker`
subcommand with `start_new_session=True`, repository cwd, the current exact executable,
and all numerical thread variables set to `1`. It atomically publishes the returned
PID and returns canonical JSON fields `status=launched`, `method`, `pid`, `run_root`,
`ops_root`, and `log_path`. It must not wait for, fit, or score any model in the
launching process.

- [ ] **Step 5: Implement worker monitoring, status, and verification**

The hidden worker holds the long-running `fcntl` lock and executes the four stages.
For prepare, assess-mlx, and freeze it logs command start/finish and captured output.
For score it starts the existing CLI child and every 15 seconds validates visible
shard/meta pairs, atomically replaces `state_antigravity.json`, and appends:

```text
UTC_TIMESTAMP method completed/expected elapsed_seconds eta_seconds recent_dataset state
```

Partial verification must check both files exist, meta exact fields/schema,
`run_config_sha256`, DB SHA, shard SHA, validated score payload, method and dataset.
Any half-pair or mismatch becomes `status=failed`; it is never deleted or counted.
ETA is `null` until at least three new pairs finish and at least 60 seconds elapse;
afterward it uses only this launch session’s completion rate.

The worker sets `score_complete` only after 1,117 verified pairs, runs freeze, calls
the existing `verify_score_root`, then sets `frozen`. Exceptions write `failed`,
`failure_kind`, `last_error`, and child return code while preserving all artifacts.

`status` is read-only. A live PID plus heartbeat age at most 180 seconds is `running`;
a live PID with older heartbeat is `failed/stale_heartbeat`; a dead incomplete worker
is `failed/interrupted`; a verified manifest is `frozen`; no authority is
`not_started`.

`verify_frozen(run_root)` calls only `verify_score_root`, verifies the authority method
and one-method pair count 1,117, and returns `status=frozen_verified`, method,
pair_count, manifest/run-config SHA. It must not import the label module.

- [ ] **Step 6: Run focused GREEN tests**

Run:

```bash
python3 -m unittest \
  experiments.paper_public_baselines.tests.test_deferred_deep_launcher_antigravity -v
```

Expected: every launcher test passes, no process survives, and no path outside each
temporary directory is created.

- [ ] **Step 7: Run existing baseline regressions**

Run:

```bash
python3 -m unittest discover -s experiments/paper_public_baselines/tests -v
```

Expected: all existing 61 tests plus the new launcher tests pass; no Deep SVDD/DROCC
50-epoch process starts.

- [ ] **Step 8: Commit launcher and tests**

```bash
git add scratch/manage_deferred_deep_baselines_antigravity.py \
  experiments/paper_public_baselines/tests/test_deferred_deep_launcher_antigravity.py
git commit -m "feat: prepare deferred deep baseline launches"
```

---

### Task 3: Document operation and close provenance

**Files:**
- Modify: `docs/PAPER_PUBLIC_BASELINES_RUNBOOK_KO.md`
- Modify: `docs/PAPER_PUBLIC_BASELINES_IMPLEMENTATION_REPORT_KO.md`
- Modify together: `docs/codemap/codemap_exp448_antigravity.html`
- Modify together: `docs/codemap/codemap_exp448_antigravity.json`
- Modify together: `docs/codemap/codemap_exp448_antigravity.lock`

**Interfaces:**
- Consumes: the exact launcher CLI from Task 2 and current code hashes.
- Produces: copy/paste commands for two independent future roots, status/tail/verify instructions, and final code-map caller/effect/test coverage.

- [ ] **Step 1: Add exact deferred commands to the runbook**

Document that no command in this section has been executed yet. Use these future roots:

```text
/Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_1d_full_v1_antigravity
/Users/mac/Documents/Dataset/paper_public_baselines_drocc_1d_full_v1_antigravity
```

Add copy/paste launch commands for each method, but state they must run one at a time.
Add these status/log/verify forms:

```bash
python3 scratch/manage_deferred_deep_baselines_antigravity.py status --run-root ROOT
tail -f ROOT_ops_antigravity/progress_antigravity.log
python3 scratch/manage_deferred_deep_baselines_antigravity.py verify --run-root ROOT
```

Explain that relaunching the same command resumes verified shard/meta pairs, while
identity mismatch, partial pair, or tampering fails closed. State that the old
107-pair mixed root is evidence only and is not copied into either official root.

- [ ] **Step 2: Update the implementation report without claiming execution**

Record:

```text
Deep SVDD deferred launcher: prepared, not started, 0/1,117 in the new official root
DROCC deferred launcher: prepared, not started, 0/1,117 in the new official root
Canonical backend: deterministic PyTorch CPU, worker 1, epochs 50
Operational support: independent roots, create-only authority, PID/lock, live log,
partial verified status, exact resume, freeze and verify
```

Do not change the existing classical results or claim deep performance.

- [ ] **Step 3: Regenerate final Exp448 triplet together**

Add nodes for the launcher and its test:

```text
public_baseline_deferred_launcher
  path: scratch/manage_deferred_deep_baselines_antigravity.py
  callers: operator
  effects: immutable authority, detached one-method execution, progress/status,
           frozen score-root verification
  tests: experiments/paper_public_baselines/tests/test_deferred_deep_launcher_antigravity.py

public_baseline_deferred_launcher_tests
  path: experiments/paper_public_baselines/tests/test_deferred_deep_launcher_antigravity.py
  callers: unittest discovery
  effects: protects no-training import, official config, duplicate launch, verified
           partial status, stale heartbeat, and label-free verify boundaries
```

Connect launcher to `public_baseline_cli` and the three existing deep nodes. Update the
deferred flow to start at the launcher. Update JSON and the embedded HTML payload to
exact equality, then recompute the lock fingerprints after all source/test/docs edits.

- [ ] **Step 4: Run final non-training verification**

Run:

```bash
python3 -m unittest discover -s experiments/paper_public_baselines/tests -v
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.lock >/dev/null
git diff --check
pgrep -af 'deep_svdd_1d|drocc_1d' || true
test ! -e /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_1d_full_v1_antigravity
test ! -e /Users/mac/Documents/Dataset/paper_public_baselines_drocc_1d_full_v1_antigravity
```

Expected: tests pass; JSON parses; diff check is clean; no deep training process is
running; neither future official root exists.

Compute and compare a deterministic tree digest for the existing partial root before
and after implementation:

```bash
find /Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1 -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  | shasum -a 256
```

Expected: exact same digest before and after.

- [ ] **Step 5: Commit documentation and final triplet**

```bash
git add docs/PAPER_PUBLIC_BASELINES_RUNBOOK_KO.md \
  docs/PAPER_PUBLIC_BASELINES_IMPLEMENTATION_REPORT_KO.md \
  docs/codemap/codemap_exp448_antigravity.html \
  docs/codemap/codemap_exp448_antigravity.json \
  docs/codemap/codemap_exp448_antigravity.lock
git commit -m "docs: add deferred deep baseline runbook"
```

- [ ] **Step 6: Return structured evidence**

Return the three commit hashes, RED failure, exact GREEN test counts, codemap
node/edge/flow counts, partial-root before/after digest, process/root absence checks,
and any judgment call. A completion claim without this evidence is invalid.

