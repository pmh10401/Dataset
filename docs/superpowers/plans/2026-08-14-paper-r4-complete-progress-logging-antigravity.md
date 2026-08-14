# Paper r4 Complete Progress Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, result-neutral, label-free progress callbacks to every silent paper-r4 parent-side surface so an official launcher can capture flushed stderr, without changing science, workers, the live `bc5ab546` run, or the existing default stderr contract.

**Architecture:** Reuse the `data_paper.SqlObserver` pattern. Home `ProgressCallback`, the seven-key allowlist, the substring firewall, and `emit_progress_paper` in `data_paper.py`. Home the 15.0-second per-stage stderr sink next to `_progress` in `proposal_score_paper.py`. Every instrumented call site invokes `emit_progress_paper(progress, event)`; the official installed callback is the sink; the sink never calls emit. Default `progress=None` keeps today’s bit-identical science and the four bare `Task6`/`Task7` start/finish lines. ProcessPool workers stay `(_score_dataset_task4_task5, raw)`. Live root `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813` is never resumed under a new `code_closure`.

**Tech Stack:** Python 3.14, `unittest`, `io.StringIO`, `time.monotonic` (injectable fake clock), existing paper typed fixtures, canonical JSON/SHA-256, Exp448 `sha256-v1` fingerprints, paper `static_ast_sha256` triplet.

**Approved spec:** `docs/superpowers/specs/2026-08-14-paper-r4-complete-progress-logging-antigravity-design.md` (committed at `6ff4557a922392ced0989ca299256c881956cc46`).

## Global Constraints

- Work only in this Grok worktree. Do not edit `/Users/mac/.codex/worktrees/e1be/Time Series`.
- Do not write `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813` or `/Users/mac/Documents/Dataset/paper_proposal_full_r4_v2_antigravity_20260814_ops_antigravity`.
- Do not attach to, monkeypatch, resume, or stop PIDs 57778 / 57862. Do not run launcher `status` / `verify` / `resume` / `stop` against those live roots.
- Any edit to a file hashed by `paper/equipment_anomaly_detection/codemap_paper.lock` changes `runner_paper._closure()` / journal `code_closure`. An instrumented official run requires a **new** scientific root and a **new** ops root. Do not invent those names here. Do not migrate live checkpoints.
- Every newly created implementation, test, or helper filename contains `antigravity`. Existing paper production modules keep `*_paper.py` names. New test file is `tests_paper/test_progress_logging_antigravity.py`. Do not create a new paper production module. Do not implement `scratch/monitor_paper_r4_live_progress_antigravity.py` in this plan.
- `load_dataset_values` stays values-only with optional `sql_observer` only. No `progress=` on that function.
- `_score_dataset_task4_task5(raw)` stays one positional argument. No callback crosses `ProcessPoolExecutor`.
- Keep exact `Task6 start` / `Task6 finish` / `Task7 start` / `Task7 finish` lines. Structured `task6` / `task7` sink lines appear only when a callback is installed.
- Do not change scores, candidates, indices, thresholds, routing, shard bytes, journal payloads (except the existing non-deterministic `finished_at`), stdout JSON, or exit codes.
- CPU worker policy is unchanged: `WorkerRecord.from_logical_cpu_count` at `acceleration_paper.py:28-32` sets `dataset_workers = max(1, logical_cpu_count - 1)` and `inner_numerical_threads = 1`. Record detected logical CPU count and the resulting worker count in existing acceleration-manifest provenance. Do not add a worker override. Do not combine outer multi-process parallelism with an MLX/MPS job.
- Logging is CPU control-plane only. Do not introduce PyTorch, MLX, or MPS for formatting, allowlisting, or printing. See the acceleration assessment below.
- Strict RED → GREEN per task. Record the exact failing reason before production edits. Each mergeable commit that changes a mapped module or mapped entrypoint signature includes all six Exp448 + paper map files in that same commit. Do not overwrite `docs/codemap/codemap_exp444_antigravity.*` or earlier version maps.

## Acceleration Assessment (mandatory, no-change)

Logging is mapping copies, substring checks, and flushed `print`. It is not a score stage and does not rewrite `acceleration_paper` stage tables.

| Checklist item | Assessment |
| --- | --- |
| Tensor-eligible? | No. No array kernel, distance matrix, projection, or embedding. |
| Current backend availability? | Irrelevant. Existing Task9 inventory from `acceleration_paper._backends` is unchanged. |
| Transfer / compilation overhead? | A hypothetical GPU format path would be dominated by host-to-device copy for a handful of strings every 15 seconds. |
| Deterministic and numerical parity? | Science must remain exactly unchanged. There is no numeric path, so CPU-versus-MLX parity is not applicable. Result identity is proven by typed-result and shard-byte tests. |
| Canonical CPU fallback? | The emit wrapper plus stderr sink is the canonical path. There is no accelerated alternative. |
| Benchmark before an accelerated path becomes canonical? | No accelerated path will be proposed. No MLX/MPS/PyTorch benchmark is required or permitted for logging. |

MLX default-use conditions 1–5 all fail. Retain the CPU implementation. Run only one MLX/MPS-heavy process at a time remains vacuously satisfied because this work adds no GPU job.

## File structure

- Create: `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py`
- Modify: `paper/equipment_anomaly_detection/data_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `paper/equipment_anomaly_detection/runner_paper.py`
- Modify: `paper/equipment_anomaly_detection/pipeline_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/stability_paper.py`
- Modify: `paper/equipment_anomaly_detection/routing_paper.py`
- Modify: `paper/equipment_anomaly_detection/artifacts_paper.py`
- Modify together, every mapped commit: `docs/codemap/codemap_exp448_antigravity.html`, `docs/codemap/codemap_exp448_antigravity.json`, `docs/codemap/codemap_exp448_antigravity.lock`, `paper/equipment_anomaly_detection/codemap_paper.html`, `paper/equipment_anomaly_detection/codemap_paper.json`, `paper/equipment_anomaly_detection/codemap_paper.lock`
- Do not modify: `scratch/manage_paper_r4_proposal_antigravity.py` unless a launcher regression appears (none is expected). Launcher already merges child stderr into stdout at `_spawn_worker` / `_worker` / `_run_pipeline_child` (`stderr=subprocess.STDOUT`) and appends `paper-r4-progress-event-antigravity.v1` records.

## Shared map-resync procedure

Use this exact procedure in Task 1 and in every later commit that changes a mapped module, import edge, or entrypoint signature. Run it from the worktree root.

**Paper triplet** (callers must equal `_ast_import_callers(PAPER_ROOT)`; HTML must equal `_render_codemap_html`; lock hashes every `*_paper.py` plus `lineage_lock_paper.json`, `codemap_paper.html`, `codemap_paper.json`, `sources_paper/source_provenance_paper.json`, `artifact_parity_claim_paper.json`):

```bash
python3 - <<'PY'
from hashlib import sha256
import json
from pathlib import Path

from paper.equipment_anomaly_detection.config_paper import PAPER_ROOT
from paper.equipment_anomaly_detection.lineage_paper import (
    _ast_import_callers,
    _paper_module_inventory,
    _parse_codemap,
    _render_codemap_html,
    read_strict_json,
    sha256_file,
    validate_codemap_triplet,
)

root = PAPER_ROOT
codemap_path = root / "codemap_paper.json"
payload = read_strict_json(codemap_path)
parsed = _parse_codemap(payload)
actual = _ast_import_callers(root)
inventory = _paper_module_inventory(root)
if set(parsed.modules) != inventory:
    raise SystemExit(f"paper inventory drift: {sorted(inventory ^ set(parsed.modules))}")
for module, detail in parsed.modules.items():
    detail["callers"] = sorted(actual[module])
# Keep effects/tests/source_origins unless the current task documents a required text update.
codemap_path.write_text(
    json.dumps(
        {"schema": "paper-codemap.v1", "required_fields": list(parsed.required_fields), "modules": parsed.modules},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ),
    encoding="utf-8",
)
html = _render_codemap_html(_parse_codemap(read_strict_json(codemap_path)))
(root / "codemap_paper.html").write_text(html, encoding="utf-8")
expected_names = set(inventory) | {
    "lineage_lock_paper.json",
    "codemap_paper.html",
    "codemap_paper.json",
    "sources_paper/source_provenance_paper.json",
    "artifact_parity_claim_paper.json",
}
hashes = {name: sha256_file(root / name) for name in sorted(expected_names)}
(root / "codemap_paper.lock").write_text(
    json.dumps({"schema": "paper-codemap-lock.v1", "hashes": hashes}, indent=2) + "\n",
    encoding="utf-8",
)
validate_codemap_triplet(root)
print("paper-triplet-ok", len(hashes))
PY
```

**Exp448 triplet** (HTML `<script id="codemap-data">` payload must equal the JSON file; every edge/flow step must reference a node id; lock fingerprints use the lock’s documented `sha256-v1` algorithm; `docs/codemap` is excluded from fingerprint input):

```bash
python3 - <<'PY'
import hashlib, json, re, subprocess, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

root = Path('.').resolve()
json_path = root / 'docs/codemap/codemap_exp448_antigravity.json'
html_path = root / 'docs/codemap/codemap_exp448_antigravity.html'
lock_path = root / 'docs/codemap/codemap_exp448_antigravity.lock'
data = json.loads(json_path.read_text(encoding='utf-8'))
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
now = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%dT%H:%M:%S+0900')
data['generated_at'] = now
data['generated_from_commit'] = head
data['working_tree_dirty'] = True
ids = {node['id'] for node in data['nodes']}
assert len(ids) == len(data['nodes'])
assert all(edge['from'] in ids and edge['to'] in ids for edge in data['edges'])
assert all(set(flow['steps']) <= ids for flow in data['flows'])
json_path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
html = html_path.read_text(encoding='utf-8')
html = re.sub(
    r'<script id="codemap-data" type="application/json">.*?</script>',
    '<script id="codemap-data" type="application/json">' + json_path.read_text(encoding='utf-8') + '</script>',
    html,
    count=1,
    flags=re.S,
)
html = re.sub(
    r'Generated [^<]+ · source <code>[0-9a-f]{40}</code>',
    f'Generated {now} · source <code>{head}</code>',
    html,
    count=1,
)
html_path.write_text(html, encoding='utf-8')
excluded = {'.git', '.worktrees', 'docs/codemap', 'vendor', 'node_modules', 'build', 'cache', 'dist',
            '__pycache__', '.pytest_cache', '.mypy_cache', '.venv', 'venv', 'outputs', 'results'}
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).split(b'\0')
tracked = [Path(p.decode()) for p in tracked if p]
modules = {
    'root': [], '.superpowers': [], 'docs': [], 'experiments': [], 'operational': [],
    'paper': [], 'reports': [], 'scratch': [], 'scripts': [], 'vercel-dashboard': [],
}
for path in tracked:
    parts = path.parts
    if any(part in excluded or '/'.join(parts[:2]) in excluded for part in parts):
        continue
    if len(parts) == 1:
        modules['root'].append(path)
        continue
    top = parts[0]
    if top in modules:
        modules[top].append(path)

def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        data_bytes = (root / path).read_bytes()
        rel = path.as_posix().encode('utf-8')
        digest.update(rel + b'\0' + str(len(data_bytes)).encode('ascii') + b'\0' + data_bytes + b'\0')
    return digest.hexdigest()

lock = json.loads(lock_path.read_text(encoding='utf-8'))
lock['current_commit'] = head
lock['working_tree_dirty'] = True
lock['generated_at'] = now
lock['module_fingerprints'] = {name: fingerprint(paths) for name, paths in modules.items()}
prior = json.loads(subprocess.check_output(['git', 'show', 'HEAD:docs/codemap/codemap_exp448_antigravity.lock'], cwd=root))
lock['changed_modules_since_prior_lock'] = sorted(
    name for name, digest in lock['module_fingerprints'].items()
    if prior.get('module_fingerprints', {}).get(name) != digest
)
lock_path.write_text(json.dumps(lock, indent=2) + '\n', encoding='utf-8')
embedded = re.search(r'<script id="codemap-data" type="application/json">(.*?)</script>', html_path.read_text(encoding='utf-8'), re.S)
assert embedded and json.loads(embedded.group(1)) == json.loads(json_path.read_text(encoding='utf-8'))
print('exp448-triplet-ok', len(data['nodes']), len(data['edges']), len(data['flows']))
PY
```

After every map rewrite run:

```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper.ConfigLineageTests.test_paper_codemap_triplet_is_complete_and_hash_bound paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper.ConfigLineageTests.test_codemap_callers_are_complete_static_ast_import_edges
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.lock >/dev/null
```

---

### Task 1: Codemap preflight and regenerate Exp448 + paper triplets

**Files:**
- Modify together: `docs/codemap/codemap_exp448_antigravity.html`
- Modify together: `docs/codemap/codemap_exp448_antigravity.json`
- Modify together: `docs/codemap/codemap_exp448_antigravity.lock`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.lock`

**Interfaces:**
- Consumes: active experiment `Exp448` from `RESEARCH_STATE.json`; current paper AST caller graph from `lineage_paper._ast_import_callers`; lock algorithm `sha256-v1`.
- Produces: both complete triplets, generated from current HEAD, that can answer callers / effects / tests for `data_paper.py`, `sources_paper/low_frequency_paper.py`, `sources_paper/stability_paper.py`, `routing_paper.py`, `artifacts_paper.py`, `proposal_score_paper.py`, `runner_paper.py`, and `pipeline_paper.py` **before** any production edit.

Current preflight fact (must be re-proven, not assumed):

| Map | `low_frequency` / `stability` answer? |
| --- | --- |
| `docs/codemap/codemap_exp448_antigravity.json` (39 nodes, generated from `47b67f8c`, HEAD is `6ff4557a`) | **Cannot.** Those path strings are absent. `paper_proposal_score` and `paper_runner` exist; `paper_data`, `paper_low_frequency`, `paper_stability`, and `paper_routing` do not. |
| `paper/equipment_anomaly_detection/codemap_paper.json` | Can. Refresh callers from AST and rebind hashes/HTML in the same commit so the six files stay a matched set. |

Current paper-map answers the executor must copy into the new Exp448 nodes (do not invent different callers):

| Module | Callers (AST + paper map today) | Affects | Tests |
| --- | --- | --- | --- |
| `data_paper.py` | `label_access_paper.py`, `runner_paper.py`, `tests_paper/test_data_boundary_paper.py` | values-only TRAIN + label-free TEST load | `tests_paper/test_data_boundary_paper.py`, `tests_paper/test_hard_sources_paper.py` |
| `sources_paper/low_frequency_paper.py` | `proposal_score_paper.py`, `sources_paper/stability_paper.py`, `tests_paper/test_low_frequency_paper.py`, `tests_paper/test_stability_paper.py` | Task6 `LowFrequencyResult` | `tests_paper/test_low_frequency_paper.py` |
| `sources_paper/stability_paper.py` | `artifacts_paper.py`, `proposal_score_paper.py`, `routing_paper.py`, `tests_paper/test_artifacts_runner_paper.py`, `tests_paper/test_stability_paper.py` | Task7 `StabilityPartition` | `tests_paper/test_stability_paper.py` |
| `routing_paper.py` | `artifacts_paper.py`, `proposal_score_paper.py`, `tests_paper/test_artifacts_runner_paper.py`, `tests_paper/test_stability_paper.py` | hard evidence + four policies | `tests_paper/test_stability_paper.py` |
| `artifacts_paper.py` | `catalog_paper.py`, `proposal_score_paper.py`, `runner_paper.py`, `tests_paper/test_artifacts_runner_paper.py` | shards, freeze, verify, Task4+5 records | `tests_paper/test_artifacts_runner_paper.py` |
| `proposal_score_paper.py` | `runner_paper.py`, `tests_paper/test_artifacts_runner_paper.py`, `tests_paper/test_hard_sources_paper.py` | Task4–7 compose + freeze entry | `tests_paper/test_artifacts_runner_paper.py` |
| `runner_paper.py` | `pipeline_paper.py`, `tests_paper/test_artifacts_runner_paper.py` | catalog raw load, journal, stages | `tests_paper/test_artifacts_runner_paper.py` |
| `pipeline_paper.py` | `tests_paper/test_artifacts_runner_paper.py` (launcher also calls it; paper AST does not import the launcher) | stdout JSON, exit codes | `tests_paper/test_artifacts_runner_paper.py` |

- [ ] **Step 1: Prove the Exp448 coverage failure and a clean production tree**

Run:

```bash
git status --short
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('docs/codemap/codemap_exp448_antigravity.json').read_text(encoding='utf-8'))
blob = json.dumps(data)
ids = {node['id'] for node in data['nodes']}
paths = {node.get('path') for node in data['nodes']}
print('nodes', len(data['nodes']))
print('generated_from_commit', data['generated_from_commit'])
print('has_low_frequency', 'low_frequency' in blob)
print('has_stability_paper', 'stability_paper' in blob)
print('has_data_paper', 'data_paper.py' in blob)
print('has_paper_low_frequency', 'paper_low_frequency' in ids)
print('has_paper_stability', 'paper_stability' in ids)
print('has_paper_routing', 'paper_routing' in ids)
print('has_paper_data', 'paper_data' in ids)
PY
```

Expected: worktree has no production edits; `has_low_frequency` is `False`; `has_paper_low_frequency` / `has_paper_stability` are `False`; `generated_from_commit` is not current HEAD. Do not treat this as authorization to skip regeneration.

- [ ] **Step 2: Add the missing Exp448 nodes, edges, and flow**

Edit `docs/codemap/codemap_exp448_antigravity.json` (then embed the same object in HTML). Add these four nodes. Do not delete existing nodes.

```text
paper_data
  path: paper/equipment_anomaly_detection/data_paper.py
  role: Descriptor-pinned values-only SQLite load and, after later tasks, the result-neutral progress emit wrapper next to SqlObserver.
  entrypoints: load_dataset_values, load_population
  callers: label_access_paper.py, runner_paper.py, tests_paper/test_data_boundary_paper.py
  effects: opens only the SHA256-verified descriptor-pinned read-only database; returns normal TRAIN and label-free TEST float32 values
  tests: tests_paper/test_data_boundary_paper.py, tests_paper/test_hard_sources_paper.py
  constraints: no progress argument on load_dataset_values; TEST_VALUES_SQL stays label-free; this node cannot authorize official r4 or Exp448 lifecycle

paper_low_frequency
  path: paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py
  role: Values-only Task6 exact-length finite-TRAIN grouped low-frequency evidence and Hard-empty top-30 selection.
  entrypoints: score_low_frequency, _score_one_dataset_paper, select_source_candidates
  callers: proposal_score_paper.py, sources_paper/stability_paper.py, tests_paper/test_low_frequency_paper.py, tests_paper/test_stability_paper.py
  effects: replays copied Exp440 evidence for every TEST row and exposes copied Exp441 global 30-candidate selection without TEST-label access
  tests: tests_paper/test_low_frequency_paper.py
  constraints: no import of proposal_score_paper; no per-TEST-row progress; CPU-only; this node cannot authorize official r4

paper_stability
  path: paper/equipment_anomaly_detection/sources_paper/stability_paper.py
  role: Values-only Task7 selected-only TRAIN-omission stability partition.
  entrypoints: classify_stability, _candidate_views
  callers: artifacts_paper.py, proposal_score_paper.py, routing_paper.py, tests_paper/test_artifacts_runner_paper.py, tests_paper/test_stability_paper.py
  effects: refits up to 16 omission views per selected candidate and builds the stable/unstable partition without legacy runtime imports
  tests: tests_paper/test_stability_paper.py
  constraints: no import of proposal_score_paper; no survived/depth/rank progress fields; this node cannot authorize official r4

paper_routing
  path: paper/equipment_anomaly_detection/routing_paper.py
  role: Frozen hard-evidence surface and four process-neutral policy routes.
  entrypoints: build_hard_evidence_surface, build_policies
  callers: artifacts_paper.py, proposal_score_paper.py, tests_paper/test_artifacts_runner_paper.py, tests_paper/test_stability_paper.py
  effects: builds the Task5 five-source 2-of-5 evidence surface and the frozen hard_only / hard_plus_source / hard_plus_stable / hard_plus_unstable routes
  tests: tests_paper/test_stability_paper.py
  constraints: progress must not serialize alerts, p-values, strength, or score arrays; this node cannot authorize official r4
```

Extend existing `paper_task45_success_records` entrypoints to also list `write_typed_proposal_shard`, `freeze_score_root`, and `verify_score_root` so shard/freeze ownership is answerable from Exp448. Extend `paper_proposal_score` entrypoints to include `_progress` (existing). Do not yet add `emit_progress_paper` or `make_stderr_progress_sink_paper`; those symbols do not exist until Task 2.

Add exact edges:

```text
paper_runner -> paper_data                 type: imports   evidence: runner_paper.py · load_dataset_values
paper_proposal_score -> paper_low_frequency type: imports  evidence: proposal_score_paper.py · score_low_frequency
paper_proposal_score -> paper_stability     type: imports  evidence: proposal_score_paper.py · classify_stability
paper_proposal_score -> paper_routing       type: imports  evidence: proposal_score_paper.py · build_hard_evidence_surface, build_policies
paper_stability -> paper_low_frequency      type: imports  evidence: stability_paper.py · score_test_row_paper
paper_routing -> paper_stability            type: imports  evidence: routing_paper.py · StabilityPartition
paper_task45_success_records -> paper_stability type: imports evidence: artifacts_paper.py · StabilityPartition
```

Add one flow:

```text
trigger: Paper r4 complete progress logging (maps only; no production emit yet)
steps: paper_data, paper_runner, paper_proposal_score, paper_low_frequency, paper_stability, paper_routing, paper_task45_success_records, paper_pipeline_cli
outcome: Exp448 can answer callers/effects/tests for every module this plan will instrument. No official root is created and the live bc5ab546 run is untouched.
```

- [ ] **Step 3: Refresh both complete triplets together**

Run the shared paper-triplet script, then the shared Exp448-triplet script. Paper callers stay the current AST set. Paper effects/tests stay as they are in this preflight (progress edges land in later tasks). Do not overwrite `codemap_exp444_antigravity.*`.

- [ ] **Step 4: Verify the three questions now have answers**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from paper.equipment_anomaly_detection.config_paper import PAPER_ROOT
from paper.equipment_anomaly_detection.lineage_paper import validate_codemap_triplet

exp = json.loads(Path('docs/codemap/codemap_exp448_antigravity.json').read_text(encoding='utf-8'))
by_path = {node['path']: node for node in exp['nodes']}
for path in (
    'paper/equipment_anomaly_detection/data_paper.py',
    'paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py',
    'paper/equipment_anomaly_detection/sources_paper/stability_paper.py',
    'paper/equipment_anomaly_detection/routing_paper.py',
):
    node = by_path[path]
    assert node.get('callers') or node.get('entrypoints')
    assert node.get('effects') or node.get('role')
    assert node['tests']
    print('exp448', node['id'], 'callers', node.get('callers'), 'tests', node['tests'])
paper = validate_codemap_triplet(PAPER_ROOT)
for module in (
    'data_paper.py',
    'sources_paper/low_frequency_paper.py',
    'sources_paper/stability_paper.py',
    'routing_paper.py',
    'artifacts_paper.py',
):
    detail = paper.modules[module]
    print('paper', module, 'callers', detail['callers'], 'tests', detail['tests'])
PY
```

Expected: exit `0`. Exp448 now names callers/effects/tests for low_frequency and stability. Paper validation still passes.

- [ ] **Step 5: Commit only the six map files**

```bash
git add docs/codemap/codemap_exp448_antigravity.html \
  docs/codemap/codemap_exp448_antigravity.json \
  docs/codemap/codemap_exp448_antigravity.lock \
  paper/equipment_anomaly_detection/codemap_paper.html \
  paper/equipment_anomaly_detection/codemap_paper.json \
  paper/equipment_anomaly_detection/codemap_paper.lock
git commit -m "$(cat <<'EOF'
docs(codemap): map paper r4 progress-logging surfaces

EOF
)"
```

Expected: exactly those six files. No production modules. No Dataset writes.

---

### Task 2: Cycle-free ProgressCallback, emit helper, and 15s stderr sink

**Files:**
- Create: `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py`
- Modify: `paper/equipment_anomaly_detection/data_paper.py` (after `SqlObserver` at line 18; `__all__` at line 169). Do not change `load_dataset_values` at lines 136–166.
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py` (add `make_stderr_progress_sink_paper` immediately after `_progress` at lines 96–102; add `from .data_paper import ProgressCallback, PROGRESS_TICK_INTERVAL_S`; extend `__all__` at line 256). Do not change `compose_proposal_score` yet.
- Modify together: the six Exp448 + paper map files (new observer edge on `data_paper.py`; new sink entrypoint on `proposal_score_paper.py`; new test file listed under those nodes’ tests).

**Interfaces:**

```python
PROGRESS_TICK_INTERVAL_S = 15.0
ALLOWED_PROGRESS_STAGES = frozenset({
    "raw_preload", "task4_task5", "task6", "task7",
    "hard_evidence", "policies", "shard_write", "freeze_verify", "journal_exit",
})
ALLOWED_PROGRESS_PHASES = frozenset({"start", "tick", "end", "error"})
ALLOWED_PROGRESS_KEYS = frozenset({
    "stage", "phase", "completed", "total", "dataset", "detail", "elapsed_s",
})
FORBIDDEN_PROGRESS_SUBSTRINGS = (
    "label=", "p-value", "pvalue=", "score=", "scores=", "rank=", "ranks=",
    "excess_ratio", "alert_index", "alerts=", "select i.label", "split='test' and i.label",
)
ProgressCallback = Callable[[Mapping[str, Any]], None]

def emit_progress_paper(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None: ...
def make_stderr_progress_sink_paper(
    *,
    min_tick_interval_s: float = PROGRESS_TICK_INTERVAL_S,
    clock=monotonic,
    stream=None,
) -> ProgressCallback: ...
```

`data_paper.py` today imports only `config_paper` and `contracts_paper`. It must not import `proposal_score_paper`, `runner_paper`, any `sources_paper` module, `routing_paper`, or `artifacts_paper`. Inner modules later import emit from `data_paper` only.

- [ ] **Step 1: Write failing tests 3 and 11**

Create `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py` with class `ProgressLoggingAntigravityTests(unittest.TestCase)` matching `tests_paper` unittest style.

```python
"""Result-neutral paper r4 progress-callback contracts."""
from __future__ import annotations

from io import StringIO
import unittest


class ProgressLoggingAntigravityTests(unittest.TestCase):
    def test_rate_gate_emits_start_end_error_immediately_and_ticks_at_most_once_per_15s(self) -> None:
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            make_stderr_progress_sink_paper,
        )

        stamps = iter((0.0, 0.0, 1.0, 14.9, 15.0, 15.1, 20.0))
        stream = StringIO()
        sink = make_stderr_progress_sink_paper(clock=lambda: next(stamps), stream=stream)
        payload = {"stage": "task6", "completed": 0, "total": 3, "elapsed_s": 0.0}
        sink({**payload, "phase": "start", "completed": 0})
        sink({**payload, "phase": "tick", "completed": 0, "elapsed_s": 0.0})
        sink({**payload, "phase": "tick", "completed": 0, "elapsed_s": 1.0})
        sink({**payload, "phase": "tick", "completed": 1, "elapsed_s": 14.9})
        sink({**payload, "phase": "tick", "completed": 2, "elapsed_s": 15.0})
        sink({**payload, "phase": "end", "completed": 3, "elapsed_s": 15.1})
        sink({**payload, "phase": "error", "completed": 3, "elapsed_s": 20.0})
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertEqual(sum(line.startswith("task6 0/3") and "0.0%" in line for line in lines[:2]), 2)
        self.assertTrue(any("task6 2/3" in line for line in lines))
        self.assertTrue(any("task6 3/3" in line and "100.0%" in line for line in lines))
        tick_lines = [line for line in lines if "task6 1/3" in line]
        self.assertEqual(tick_lines, [])
        self.assertGreaterEqual(len(lines), 4)
        self.assertLessEqual(len(lines), 5)

    def test_progress_text_firewall_rejects_label_and_score_fields(self) -> None:
        from paper.equipment_anomaly_detection.data_paper import emit_progress_paper
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            make_stderr_progress_sink_paper,
        )

        received: list[dict[str, object]] = []
        emit_progress_paper(
            received.append,
            {
                "stage": "task6",
                "phase": "tick",
                "completed": 1,
                "total": 2,
                "elapsed_s": 1.0,
                "score": 1.23,
                "pvalue": 0.01,
                "label": 1,
                "alerts": [0, 1],
                "detail": "score=1.23",
            },
        )
        self.assertEqual(len(received), 1)
        self.assertEqual(set(received[0]), {"stage", "phase", "completed", "total", "elapsed_s"})
        self.assertNotIn("detail", received[0])
        stream = StringIO()
        sink = make_stderr_progress_sink_paper(clock=lambda: 0.0, stream=stream)
        sink(received[0])
        text = stream.getvalue().lower()
        for token in ("label=", "p-value", "score=", "alerts=", "rank="):
            self.assertNotIn(token, text)
```

Do not sleep in real time. Import emit from `data_paper` and the sink from `proposal_score_paper`.

- [ ] **Step 2: Run the focused test and capture RED**

```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_rate_gate_emits_start_end_error_immediately_and_ticks_at_most_once_per_15s paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_progress_text_firewall_rejects_label_and_score_fields -v
```

Expected RED: `ImportError` for `emit_progress_paper` and/or `make_stderr_progress_sink_paper`. Record that exact reason. Do not edit production modules before the failure is observed.

- [ ] **Step 3: Implement the minimum helper and sink**

In `data_paper.py`, next to `SqlObserver` (line 18), add the constants, alias, and wrapper. Extend the existing `collections.abc` import with `Mapping` and add `from typing import Any`. Keep `from .config_paper import FROZEN_CONFIG` and `from .contracts_paper import FrozenPaperConfig, RawDataset`.

```python
PROGRESS_TICK_INTERVAL_S = 15.0
ALLOWED_PROGRESS_STAGES = frozenset({
    "raw_preload", "task4_task5", "task6", "task7",
    "hard_evidence", "policies", "shard_write", "freeze_verify", "journal_exit",
})
ALLOWED_PROGRESS_PHASES = frozenset({"start", "tick", "end", "error"})
ALLOWED_PROGRESS_KEYS = frozenset({
    "stage", "phase", "completed", "total", "dataset", "detail", "elapsed_s",
})
FORBIDDEN_PROGRESS_SUBSTRINGS = (
    "label=", "p-value", "pvalue=", "score=", "scores=", "rank=", "ranks=",
    "excess_ratio", "alert_index", "alerts=", "select i.label", "split='test' and i.label",
)
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _progress_int(value: object) -> int | None:
    if type(value) is int:
        return value
    if type(value) is float and value.is_integer() and value == value and value not in (float("inf"), float("-inf")):
        return int(value)
    return None


def emit_progress_paper(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None:
    """Best-effort, result-neutral emit. Never raises into science."""
    if callback is None or not isinstance(event, Mapping):
        return
    filtered: dict[str, Any] = {key: event[key] for key in ALLOWED_PROGRESS_KEYS if key in event}
    if filtered.get("stage") not in ALLOWED_PROGRESS_STAGES or filtered.get("phase") not in ALLOWED_PROGRESS_PHASES:
        return
    completed = _progress_int(filtered.get("completed"))
    total = _progress_int(filtered.get("total"))
    if completed is None or total is None:
        return
    filtered["completed"] = completed
    filtered["total"] = total
    elapsed = filtered.get("elapsed_s")
    if type(elapsed) is bool:
        return
    try:
        elapsed_s = float(elapsed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if elapsed_s != elapsed_s or elapsed_s in (float("inf"), float("-inf")) or elapsed_s < 0.0:
        return
    filtered["elapsed_s"] = elapsed_s
    if "dataset" in filtered and type(filtered["dataset"]) is not str:
        filtered.pop("dataset")
    if "detail" in filtered and type(filtered["detail"]) is not str:
        filtered.pop("detail")
    for key in ("dataset", "detail"):
        value = filtered.get(key)
        if type(value) is str and any(token in value.lower() for token in FORBIDDEN_PROGRESS_SUBSTRINGS):
            filtered.pop(key)
    try:
        callback(filtered)
    except BaseException:
        return
```

The `try` wraps **only** `callback(filtered)`. Do not wrap scoring or I/O around emit. Catch `BaseException` so a callback `KeyboardInterrupt` cannot replace a scientific exception; do not wrap the caller of `emit_progress_paper` in `except BaseException`.

Update `data_paper.__all__` to:

```python
__all__ = [
    "RawDataset", "TEST_VALUES_SQL", "ProgressCallback", "emit_progress_paper",
    "load_dataset_values", "load_population",
]
```

In `proposal_score_paper.py`, after `_progress` (lines 96–102), add:

```python
def make_stderr_progress_sink_paper(
    *,
    min_tick_interval_s: float = PROGRESS_TICK_INTERVAL_S,
    clock=monotonic,
    stream=None,
) -> ProgressCallback:
    """Format one flushed stderr line. Rate-gate ticks only."""
    output = sys.stderr if stream is None else stream
    last_tick: dict[str, float] = {}

    def sink(event: Mapping[str, object]) -> None:
        stage = str(event["stage"])
        phase = str(event["phase"])
        now = float(clock())
        if phase == "start":
            last_tick.pop(stage, None)
        elif phase == "tick":
            previous = last_tick.get(stage)
            if previous is not None and (now - previous) < min_tick_interval_s:
                return
            last_tick[stage] = now
        completed = int(event["completed"])
        total = int(event["total"])
        elapsed = float(event.get("elapsed_s", 0.0))
        pct = (100.0 * completed / total) if total > 0 else 0.0
        rate = (completed / elapsed) if elapsed > 0.0 else 0.0
        eta = ((total - completed) / rate) if rate > 0.0 else 0.0
        suffix = ""
        dataset = event.get("dataset")
        if type(dataset) is str:
            suffix += f" dataset={dataset}"
        detail = event.get("detail")
        if type(detail) is str:
            suffix += f" {detail}"
        print(
            f"{stage} {completed}/{total} {pct:.1f}% elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s{suffix}",
            file=output,
            flush=True,
        )

    return sink
```

Gate is **per stage**, not global. `start` / `end` / `error` always print. First tick of a stage prints immediately. Add `make_stderr_progress_sink_paper` to `proposal_score_paper.__all__`. Do not re-export `emit_progress_paper` from `proposal_score_paper`.

Do not change `compose_proposal_score`, runner, sources, routing, or artifacts in this task.

- [ ] **Step 4: Run focused GREEN**

Repeat the Step 2 command. Expected: both tests PASS. Also run:

```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_data_boundary_paper paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper -v
```

Expected: PASS. `load_dataset_values` still has no `progress` argument.

- [ ] **Step 5: Resync the six maps and commit**

Update Exp448 `paper_data` entrypoints to include `ProgressCallback` and `emit_progress_paper`, and `paper_proposal_score` entrypoints to include `make_stderr_progress_sink_paper`. Add `tests_paper/test_progress_logging_antigravity.py` to those nodes’ tests. Add paper-map tests lists for `data_paper.py` and `proposal_score_paper.py` to include `tests_paper/test_progress_logging_antigravity.py`. Do **not** add the new test file as a paper-map module: inventory is only `*_paper.py`.

Run the shared map-resync scripts. Commit:

```bash
git add paper/equipment_anomaly_detection/data_paper.py \
  paper/equipment_anomaly_detection/proposal_score_paper.py \
  paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py \
  docs/codemap/codemap_exp448_antigravity.html \
  docs/codemap/codemap_exp448_antigravity.json \
  docs/codemap/codemap_exp448_antigravity.lock \
  paper/equipment_anomaly_detection/codemap_paper.html \
  paper/equipment_anomaly_detection/codemap_paper.json \
  paper/equipment_anomaly_detection/codemap_paper.lock
git commit -m "$(cat <<'EOF'
feat: add result-neutral progress emit helper and 15s sink

EOF
)"
```

---

### Task 3: RawPreload plus journal/exit progress without CPU-heavy status

**Files:**
- Modify: `paper/equipment_anomaly_detection/runner_paper.py` (import at line 15; new `_load_catalog_raw`; replace the three identical dict comprehensions at lines 130, 142, and 156; install one sink per official invocation in `_write_complete` / `verify_journal` / `_incomplete_status` / `run_pipeline`)
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py` (`_collect_task4_task5` signature at line 134; `compose_proposal_score` signature at line 211; keep `_progress` at lines 96–102 and the four Task6/Task7 prints at lines 229–240)
- Modify: `paper/equipment_anomaly_detection/pipeline_paper.py` (`main` at lines 22–25)
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py` (tests 1, 2, 4, 5, 10, 12)
- Do not modify: `tests_paper/test_artifacts_runner_paper.py` assertions; `PublicCliContractTests` at lines 1137–1198 must stay green
- Modify together: the six map files

**Interfaces:**

```python
def _load_catalog_raw(
    catalog: dict[str, object],
    db_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, RawDataset]: ...

def _collect_task4_task5(
    raw, backend, *, root=None, context=None, catalog=None,
    acceleration_manifest=None, publish_success=None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, HardSourceBundle], dict[str, HardRouteResult]]: ...

def compose_proposal_score(
    raw_datasets, acceleration_manifest, *, root=None, context=None, catalog=None,
    progress: ProgressCallback | None = None,
) -> ProposalScore: ...
```

`status` / `verify` install the sink so a 1,117-dataset journal reload is no longer silent. That does **not** make official `status` cheap. Do not skip the raw load. Do not invoke `status` against the live root.

- [ ] **Step 1: Write failing tests 1, 2, 4, 5, 10, and 12**

Add these methods to `ProgressLoggingAntigravityTests`. Reuse the two-dataset patched fixture pattern from `Task9ParallelCompositionTests` in `test_artifacts_runner_paper.py` lines 695–716 and 858–901.

Test 1 — default `progress=None` preserves typed proposal and today’s stderr contract:

```python
def test_progress_callback_default_none_preserves_typed_proposal_and_stderr_contract(self) -> None:
    # Drive compose_proposal_score without progress= using the alpha/beta
    # Task9ParallelCompositionTests fixture (patched Task4–7 internals).
    # Redirect stderr. Assert Task4+Task5 start, per-dataset Task4+Task5
    # completions, exact "Task6 start", "Task6 finish", "Task7 start",
    # "Task7 finish". Assert the text does not contain "raw_preload",
    # "task6 0/", or "task7 0/". Capture a baseline ProposalScore with
    # progress=None and assert typed identities equal a second call.
```

Test 2 — raising callback cannot change science or replace a scientific error:

```python
def test_raising_progress_callback_does_not_change_typed_result_or_replace_scientific_error(self) -> None:
    def boom(_event):
        raise RuntimeError("callback boom")
    # Same patched compose as test 1 with progress=boom must return the
    # same source/route/low-frequency/stability/policy-name coordinates
    # as progress=None.
    from paper.equipment_anomaly_detection.proposal_score_paper import compose_proposal_score
    with self.assertRaises(ValueError) as raised:
        compose_proposal_score({}, object(), progress=boom)
    self.assertIsInstance(raised.exception, ValueError)
    self.assertNotIsInstance(raised.exception, RuntimeError)
```

`compose_proposal_score` already raises `ValueError("proposal raw dataset mapping is invalid")` at lines 215–216 for an empty mapping. The callback may also raise on an `error` event; the surfaced exception must remain that `ValueError`.

Test 4 — RawPreload:

```python
def test_raw_preload_emits_dataset_progress_without_label_sql_or_scores(self) -> None:
    from unittest.mock import patch
    from paper.equipment_anomaly_detection import runner_paper as runner
    events: list[dict[str, object]] = []
    catalog = {"datasets": [{"dataset": "alpha"}, {"dataset": "beta"}]}
    with patch.object(runner, "load_dataset_values", side_effect=lambda db_path, name: f"raw-{name}") as loaded:
        raw = runner._load_catalog_raw(catalog, Path("fixture.sqlite"), progress=events.append)
    self.assertEqual(tuple(raw), ("alpha", "beta"))
    self.assertEqual([event["phase"] for event in events], ["start", "tick", "tick", "end"])
    self.assertEqual([event.get("dataset") for event in events], [None, "alpha", "beta", None])
    joined = " ".join(str(value) for event in events for value in event.values())
    for token in ("label", "SELECT", "pvalue", "score", "alert"):
        self.assertNotIn(token.lower(), joined.lower())
    for call in loaded.call_args_list:
        self.assertEqual(call.kwargs, {})
        self.assertEqual(len(call.args), 2)
```

Test 5 — reuse the CompletionExecutor / `as_completed` beta-then-alpha harness from `test_parallel_completion_progress_is_flush_visible_and_keeps_sorted_results` (lines 858–901). Assert the same lines, `flush_calls >= 7`, and catalog-sorted `proposal.sources`. Leave the original method in `test_artifacts_runner_paper.py` unchanged.

Test 10 — journal/exit CLI:

```python
def test_journal_and_exit_progress_does_not_change_stdout_json_or_exit_code(self) -> None:
    from paper.equipment_anomaly_detection import pipeline_paper as pipeline
    payload = {"status": "completed", "completed_stages": ["preflight", "prepare", "acceleration", "proposal_score"]}
    stdout, stderr = StringIO(), StringIO()
    with patch.object(pipeline, "run_pipeline", lambda _config: payload), redirect_stdout(stdout), redirect_stderr(stderr):
        code = pipeline.main(["run", "--db-path", "fixture.sqlite", "--run-root", "run_root_antigravity", "--mode", "smoke", "--smoke-dataset", "alpha"])
    self.assertEqual(code, 0)
    self.assertEqual(stdout.getvalue(), json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    self.assertIn("journal_exit", stderr.getvalue())
    self.assertIn("exit=0", stderr.getvalue())
    self.assertNotIn("{", stderr.getvalue().split("journal_exit", 1)[1].splitlines()[0])
```

Repeat the same pattern for `proposal_score_incomplete` → exit `2`. Do not put the JSON payload on stderr.

Test 12 — worker submit signature:

```python
def test_worker_submit_signature_unchanged_no_callback_in_process_pool(self) -> None:
    from inspect import signature
    from paper.equipment_anomaly_detection.proposal_score_paper import (
        _collect_task4_task5,
        _score_dataset_task4_task5,
    )
    self.assertEqual(list(signature(_score_dataset_task4_task5).parameters), ["raw"])
    self.assertEqual(_score_dataset_task4_task5.__code__.co_argcount, 1)
    # Patch ProcessPoolExecutor.submit to record args/kwargs. Drive
    # _collect_task4_task5 with dataset_workers > 1 using the two-dataset
    # fixture. Every submit is (_score_dataset_task4_task5, raw_dataset)
    # with empty kwargs.
```

- [ ] **Step 2: Run the new methods and capture RED**

```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity -v
```

Expected RED: `_load_catalog_raw` missing; `compose_proposal_score` / `_collect_task4_task5` reject unexpected `progress=`; CLI stderr has no `journal_exit` line. Record the first failing method name and exception.

- [ ] **Step 3: Implement RawPreload, Task4+Task5 wrapper, sink install, and journal/exit**

In `runner_paper.py` change line 15 to:

```python
from .data_paper import ProgressCallback, emit_progress_paper, load_dataset_values
from .proposal_score_paper import ProposalScoreIncomplete, compose_proposal_score, make_stderr_progress_sink_paper, write_and_freeze_proposal_score
```

Add this helper above `verify_journal` (before line 109). Do not add `progress=` to `load_dataset_values`.

```python
def _load_catalog_raw(catalog: dict[str, object], db_path: Path, *, progress: ProgressCallback | None = None) -> dict[str, RawDataset]:
    names = [entry["dataset"] for entry in catalog["datasets"]]
    started = monotonic()
    total = len(names)
    emit_progress_paper(progress, {"stage": "raw_preload", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    raw: dict[str, RawDataset] = {}
    try:
        for completed, name in enumerate(names, 1):
            raw[name] = load_dataset_values(db_path, name)
            emit_progress_paper(progress, {
                "stage": "raw_preload", "phase": "tick",
                "completed": completed, "total": total,
                "dataset": name, "elapsed_s": float(monotonic() - started),
            })
        emit_progress_paper(progress, {
            "stage": "raw_preload", "phase": "end",
            "completed": total, "total": total,
            "elapsed_s": float(monotonic() - started),
        })
    except Exception:
        emit_progress_paper(progress, {
            "stage": "raw_preload", "phase": "error",
            "completed": len(raw), "total": total,
            "elapsed_s": float(monotonic() - started),
        })
        raise
    return raw
```

Add `from time import monotonic` to `runner_paper.py`. Replace the three dict comprehensions:

- `verify_journal` line 130
- `_incomplete_status` line 142
- `_write_complete` proposal_score branch line 156

each with `raw = _load_catalog_raw(catalog, Path(context["paths"]["db_path"]["unresolved_path"]), progress=progress)`.

Construct **one** sink per official invocation:

```python
progress = make_stderr_progress_sink_paper()
```

inside `_write_complete` (proposal_score branch), `verify_journal` (the `proposal_work` raw-load path), and `_incomplete_status`. Pass that same object as `progress=` into `_load_catalog_raw`, `compose_proposal_score`, and (in Task 6) freeze/verify. Do not install a process-global singleton. Do not wrap the sink in a second emit.

In `_write_complete` after freeze and before `write_create_only` of the journal record (line 162):

```python
emit_progress_paper(progress, {
    "stage": "journal_exit", "phase": "start", "completed": 0, "total": 1,
    "elapsed_s": 0.0, "detail": f"journal_stage={stage}",
})
write_create_only(_journal_path(root, index, stage), canonical_json_bytes(_record({...})))
emit_progress_paper(progress, {
    "stage": "journal_exit", "phase": "end", "completed": 1, "total": 1,
    "elapsed_s": 0.0, "detail": f"journal_stage={stage}",
})
```

`detail="journal_stage=proposal_score"` is a journal stage name, not a numeric score.

In `proposal_score_paper.py`:

- Add `progress: ProgressCallback | None = None` as a keyword-only argument on `_collect_task4_task5` and `compose_proposal_score`.
- After the existing `_progress("Task4+Task5 start", 0, total, started)` at line 162, also `emit_progress_paper(progress, {"stage": "task4_task5", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})`.
- After each per-dataset `_progress(...)` (lines 179 and 198), if that dataset is in `failed` emit `{phase: "error", ...}`; elif `completed == total` and not `failed` emit `{phase: "end", ...}`. Do **not** emit `tick` for Task4+Task5 completions.
- If `queue` is empty (lines 159–160), emit nothing new and keep today’s early return.
- Keep `_score_dataset_task4_task5` at lines 90–93 unchanged. Keep both submit sites at lines 188 and 203 as `executor.submit(_score_dataset_task4_task5, raw[dataset])`.
- Forward `progress` from `compose_proposal_score` into `_collect_task4_task5`. Keep the four bare Task6/Task7 prints. Do not auto-install a sink when `progress is None`.

In `pipeline_paper.py` `main` (lines 22–25):

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    value = run_pipeline(_config(args)) if args.command == "run" else verify_completed(_config(args))
    status = value["status"] if isinstance(value.get("status"), str) else "unknown"
    code = exit_code_for_result_paper(value)
    print(f"journal_exit {status} exit={code}", file=sys.stderr, flush=True)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return code
```

Add `import sys`. Do not print the JSON payload on stderr. `exit_code_for_result_paper` stays `completed → 0`, `proposal_score_incomplete → 2`, else `1`.

- [ ] **Step 4: Run focused GREEN plus the existing CLI/parallel tests**

```bash
python3 -m unittest \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity \
  paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper.Task9ParallelCompositionTests.test_parallel_completion_progress_is_flush_visible_and_keeps_sorted_results \
  paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper.PublicCliContractTests \
  -v
```

Expected: PASS. Stdout JSON bytes and exit codes unchanged. No `raw_preload` / `task6 0/` / `task7 0/` lines when `progress` is omitted.

- [ ] **Step 5: Resync maps and commit**

Paper callers for `data_paper.py` now also include `proposal_score_paper.py`. Exp448 `paper_runner` entrypoints add `_load_catalog_raw`. `paper_pipeline_cli` effects mention the stderr `journal_exit` line before stdout JSON. Run the shared map-resync scripts and commit the production files, the test file, and the six maps together.

```bash
git commit -m "$(cat <<'EOF'
feat: surface RawPreload and journal/exit progress

EOF
)"
```

---

### Task 4: Task6 dataset and group progress

**Files:**
- Modify: `paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py` (import `ProgressCallback, emit_progress_paper` from `..data_paper`; `score_low_frequency` at line 389; `_score_one_dataset_paper` at line 349)
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py` (forward `progress` into `score_low_frequency` at line 232; keep prints at lines 229 and 234)
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py` (test 6)
- Optional only if a default-call-site regression appears: `tests_paper/test_low_frequency_paper.py`
- Modify together: the six map files

**Interfaces:**

```python
def score_low_frequency(
    values: RawDataset | Mapping[str, RawDataset],
    hard: Mapping[str, frozenset[int]],
    *,
    progress: ProgressCallback | None = None,
) -> LowFrequencyResult: ...

def _score_one_dataset_paper(
    values: RawDataset,
    *,
    progress: ProgressCallback | None = None,
    parent_completed: int = 0,
    parent_total: int = 1,
    started: float | None = None,
) -> tuple[tuple[LowFrequencyRow, ...], tuple[LowFrequencyCandidate, ...], int]: ...
```

Every `task6` event, including group ticks, uses `total = len(datasets)` and `completed =` number of datasets whose `_score_one_dataset_paper` has already returned. Nested group identity lives only in `detail="group_len={len(rows)}"`. Do not emit per TEST row. Do not put the group index into `completed`/`total`. Do not import `proposal_score_paper` from this module.

- [ ] **Step 1: Write failing test 6**

```python
def test_task6_dataset_and_group_progress_without_scores(self) -> None:
    from paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper import _values
    from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import score_low_frequency

    events: list[dict[str, object]] = []
    values = _values()
    baseline = score_low_frequency(values, {values.dataset: frozenset()})
    observed = score_low_frequency(values, {values.dataset: frozenset()}, progress=events.append)
    self.assertEqual(
        tuple(candidate.coordinate for candidate in observed.selected_candidates),
        tuple(candidate.coordinate for candidate in baseline.selected_candidates),
    )
    self.assertEqual(observed.rows[1].missing_reason, baseline.rows[1].missing_reason)
    self.assertTrue(events[0]["phase"] == "start" and events[0]["completed"] == 0 and events[0]["total"] == 1)
    self.assertTrue(any(event.get("detail", "").startswith("group_len=") and event["completed"] == 0 and event["total"] == 1 for event in events))
    parent_ticks = [event for event in events if event["phase"] == "tick" and not str(event.get("detail", "")).startswith("group_len=")]
    self.assertEqual(parent_ticks[-1]["completed"], 1)
    self.assertEqual(events[-1]["phase"], "end")
    joined = json.dumps(events)
    for token in ("excess_ratio", "empirical_rank", "score", "label="):
        self.assertNotIn(token, joined)
```

Also cover a two-dataset mapping: `{"A": _values("A"), "B": _values("B")}` so every event has `total == 2`, group ticks during the first dataset use `completed == 0`, and the parent tick after the first dataset uses `completed == 1`.

- [ ] **Step 2: Run test 6 and capture RED**

```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_task6_dataset_and_group_progress_without_scores -v
```

Expected RED: `score_low_frequency() got an unexpected keyword argument 'progress'` or zero events.

- [ ] **Step 3: Implement parent-axis Task6 emits**

In `low_frequency_paper.py` add `from time import monotonic` and `from ..data_paper import ProgressCallback, emit_progress_paper`.

`score_low_frequency` (lines 389–417):

```python
started = monotonic()
total = len(datasets)
emit_progress_paper(progress, {"stage": "task6", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
try:
    finished = 0
    for dataset in sorted(datasets):
        dataset_rows, dataset_candidates, dataset_excluded = _score_one_dataset_paper(
            datasets[dataset], progress=progress, parent_completed=finished, parent_total=total, started=started,
        )
        finished += 1
        emit_progress_paper(progress, {
            "stage": "task6", "phase": "tick",
            "completed": finished, "total": total,
            "dataset": dataset, "elapsed_s": float(monotonic() - started),
        })
        ...
    emit_progress_paper(progress, {"stage": "task6", "phase": "end", "completed": total, "total": total, "elapsed_s": float(monotonic() - started)})
except Exception:
    emit_progress_paper(progress, {"stage": "task6", "phase": "error", "completed": finished, "total": total, "elapsed_s": float(monotonic() - started)})
    raise
```

Inside `_score_one_dataset_paper`, in the `for length, rows in finite_groups.items():` loop at lines 361–368, after a successful eligible fit (or once per iterated group before `fit_group_calibration_paper`), emit:

```python
emit_progress_paper(progress, {
    "stage": "task6", "phase": "tick",
    "completed": parent_completed, "total": parent_total,
    "dataset": values.dataset,
    "detail": f"group_len={len(rows)}",
    "elapsed_s": float(monotonic() - (started if started is not None else monotonic())),
})
```

Do not walk TEST rows for progress. `compose_proposal_score` line 232 becomes `low = score_low_frequency(raw, hard_lanes, progress=progress)`. Keep `print("Task6 start")` / `print("Task6 finish")`.

- [ ] **Step 4: Run GREEN**

```bash
python3 -m unittest \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_task6_dataset_and_group_progress_without_scores \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_progress_callback_default_none_preserves_typed_proposal_and_stderr_contract \
  paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper \
  -v
```

Expected: PASS. Default `score_low_frequency(values, hard)` still equals the baseline `LowFrequencyResult`.

- [ ] **Step 5: Resync maps and commit**

Paper callers for `data_paper.py` now also include `sources_paper/low_frequency_paper.py`. Exp448 `paper_low_frequency` effects mention parent-axis dataset/group ticks. Run the shared map-resync scripts.

```bash
git commit -m "$(cat <<'EOF'
feat: add label-free Task6 dataset and group progress

EOF
)"
```

---

### Task 5: Task7 candidate and view progress

**Files:**
- Modify: `paper/equipment_anomaly_detection/sources_paper/stability_paper.py` (import from `..data_paper`; `classify_stability` at lines 315–342; `_candidate_views` at lines 268–312)
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py` (forward `progress` into `classify_stability` at line 238; keep prints at lines 236 and 240)
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py` (test 7)
- Optional only if a default-call-site regression appears: `tests_paper/test_stability_paper.py`
- Modify together: the six map files

**Interfaces:**

```python
def classify_stability(
    low_frequency: LowFrequencyResult,
    raw_datasets: Mapping[str, RawDataset],
    *,
    replay: Replay | None = None,
    progress: ProgressCallback | None = None,
) -> StabilityPartition: ...

def _candidate_views(
    candidate: LowFrequencyCandidate,
    values: RawDataset,
    replay: Replay,
    *,
    progress: ProgressCallback | None = None,
    parent_completed: int = 0,
    parent_total: int = 1,
    started: float | None = None,
) -> tuple[StabilityViewPaper, ...]: ...
```

Every `task7` event, including view ticks, uses `total = len(selected)` and `completed =` number of selected candidates whose `_candidate_views` has already returned. Candidate identity is `(dataset, instance_index)` split across `dataset` and `detail="instance_index={candidate.instance_index}"`. View identity is only `detail="view_index={mask.view_index}"` (0–15). Do not log `survived`, `agreeing_depths`, `missing_reason`, or any `LowFrequencyDepthEvidence` field. Do not import `proposal_score_paper`.

- [ ] **Step 1: Write failing test 7**

Use `_raw` / `_result` from `tests_paper/test_stability_paper.py` lines 28–77 and a recording callback plus a fake-clock sink.

```python
def test_task7_candidate_and_view_progress_is_rate_limited_and_label_free(self) -> None:
    from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability
    from paper.equipment_anomaly_detection.proposal_score_paper import make_stderr_progress_sink_paper
    from paper.equipment_anomaly_detection.tests_paper.test_stability_paper import _raw, _result

    values = _raw("fixture")
    low = _result(values)
    selected = low.selected_candidates
    self.assertGreaterEqual(len(selected), 1)
    events: list[dict[str, object]] = []
    stamps = iter([0.0] * 8 + [15.0, 30.0, 45.0])
    stream = StringIO()
    sink = make_stderr_progress_sink_paper(clock=lambda: next(stamps, 45.0), stream=stream)

    def both(event):
        events.append(dict(event))
        sink(event)

    classify_stability(low, {values.dataset: values}, progress=both)
    total = len(selected)
    self.assertTrue(all(event["total"] == total for event in events if event["stage"] == "task7"))
    self.assertTrue(any(str(event.get("detail", "")).startswith("instance_index=") for event in events))
    self.assertTrue(any(str(event.get("detail", "")).startswith("view_index=") and event["completed"] == 0 for event in events))
    self.assertEqual(events[0]["phase"], "start")
    self.assertEqual(events[-1]["phase"], "end")
    printed = stream.getvalue()
    self.assertIn("task7 0/", printed)
    tick_lines = [line for line in printed.splitlines() if "view_index=" in line]
    self.assertLessEqual(len(tick_lines), 2)
    joined = json.dumps(events)
    for token in ("survived", "agreeing_depths", "excess_ratio", "rank=", "label="):
        self.assertNotIn(token, joined)
```

- [ ] **Step 2: Run test 7 and capture RED**

```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_task7_candidate_and_view_progress_is_rate_limited_and_label_free -v
```

Expected RED: unexpected keyword `progress` or no `view_index=` events.

- [ ] **Step 3: Implement parent-axis Task7 emits**

Replace the generator at `classify_stability` lines 332–335 with an explicit loop:

```python
started = monotonic()
total = len(selected)
finished = 0
emit_progress_paper(progress, {"stage": "task7", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
try:
    results = []
    for candidate in selected:
        emit_progress_paper(progress, {
            "stage": "task7", "phase": "tick",
            "completed": finished, "total": total,
            "dataset": candidate.dataset,
            "detail": f"instance_index={candidate.instance_index}",
            "elapsed_s": float(monotonic() - started),
        })
        views = _candidate_views(
            candidate, raw[candidate.dataset], scorer,
            progress=progress, parent_completed=finished, parent_total=total, started=started,
        )
        results.append(aggregate_stability_views(candidate.coordinate, views))
        finished += 1
        emit_progress_paper(progress, {
            "stage": "task7", "phase": "tick",
            "completed": finished, "total": total,
            "dataset": candidate.dataset,
            "detail": f"instance_index={candidate.instance_index}",
            "elapsed_s": float(monotonic() - started),
        })
    emit_progress_paper(progress, {"stage": "task7", "phase": "end", "completed": total, "total": total, "elapsed_s": float(monotonic() - started)})
except Exception:
    emit_progress_paper(progress, {"stage": "task7", "phase": "error", "completed": finished, "total": total, "elapsed_s": float(monotonic() - started)})
    raise
```

In `_candidate_views`, after each mask is processed in the `for mask in masks:` loop (line 292), emit:

```python
emit_progress_paper(progress, {
    "stage": "task7", "phase": "tick",
    "completed": parent_completed, "total": parent_total,
    "dataset": candidate.dataset,
    "detail": f"view_index={mask.view_index}",
    "elapsed_s": float(monotonic() - (started if started is not None else monotonic())),
})
```

`compose_proposal_score` line 238 becomes `stability = classify_stability(low, raw, progress=progress)`. Keep `print("Task7 start")` / `print("Task7 finish")`.

- [ ] **Step 4: Run GREEN**

```bash
python3 -m unittest \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_task7_candidate_and_view_progress_is_rate_limited_and_label_free \
  paper.equipment_anomaly_detection.tests_paper.test_stability_paper \
  -v
```

Expected: PASS.

- [ ] **Step 5: Resync maps and commit**

Paper callers for `data_paper.py` now also include `sources_paper/stability_paper.py`. Exp448 `paper_stability` effects mention parent-axis candidate/view ticks. Run the shared map-resync scripts.

```bash
git commit -m "$(cat <<'EOF'
feat: add label-free Task7 candidate and view progress

EOF
)"
```

---

### Task 6: Evidence, policy, shard, and freeze progress with one shard owner

**Files:**
- Modify: `paper/equipment_anomaly_detection/routing_paper.py` (`build_hard_evidence_surface` at lines 129–168; `build_policies` at lines 237–269)
- Modify: `paper/equipment_anomaly_detection/artifacts_paper.py` (`write_typed_proposal_shard` at lines 607–609; `freeze_score_root` at lines 613–627; `verify_score_root` at lines 628–643)
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py` (`write_and_freeze_proposal_score` at lines 247–253; `compose_proposal_score` forwards into evidence/policies)
- Modify: `paper/equipment_anomaly_detection/runner_paper.py` (pass the same sink into `write_and_freeze_proposal_score` and `verify_score_root` from `_proposal_artifact`)
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py` (tests 8 and 9)
- Modify together: the six map files

**Interfaces:**

```python
def build_hard_evidence_surface(..., *, progress: ProgressCallback | None = None) -> EvidenceSurface: ...
def build_policies(..., *, progress: ProgressCallback | None = None) -> PolicyBundle: ...
def write_typed_proposal_shard(..., *, progress: ProgressCallback | None = None) -> Path: ...
def freeze_score_root(..., *, progress: ProgressCallback | None = None) -> Task8FrozenManifest: ...
def verify_score_root(..., *, progress: ProgressCallback | None = None) -> Task8FrozenManifest: ...
def write_and_freeze_proposal_score(..., *, progress: ProgressCallback | None = None) -> Task8FrozenManifest: ...
```

Official `shard_write` ticks come only from `write_and_freeze_proposal_score`. That function calls `write_typed_proposal_shard(...)` **without** `progress=`. Direct `write_typed_proposal_shard(..., progress=recording)` emits exactly one `tick` and no `start`/`end`.

- [ ] **Step 1: Write failing tests 8 and 9**

Test 8 uses the existing stability/routing fixtures (`_raw` / `_result` / `_partition_for` from `test_stability_paper.py` plus the typed source/route objects from `Task4Task5SuccessRecordTests._task4_task5_values` if needed). Assert stages `hard_evidence` and `policies`. No event value is a `numpy.ndarray`, a `frozenset` of alert ints, or a key named `alerts` / `pvalue` / `strength`. Policy `detail` is one of `hard_only`, `hard_plus_source`, `hard_plus_stable`, `hard_plus_unstable`. `policies` `total == 4`.

Test 9 uses the typed shard fixture from `test_artifacts_runner_paper.py`. Assert:

- `write_and_freeze_proposal_score(..., progress=recording)` emits `shard_write` start / exactly one tick per dataset / end, with `completed` advancing `1..N` and `total=N`.
- A spy on `write_typed_proposal_shard` shows it was invoked without `progress=` (`"progress" not in kwargs` and no extra positional).
- Direct `write_typed_proposal_shard(..., progress=recording)` emits exactly one `tick` and neither `start` nor `end`.
- `freeze_score_root` / `verify_score_root` emit `freeze_verify` start / per-dataset tick / end.
- `detail` is a `*_paper.json` name or omitted. Do not log shard hashes.

- [ ] **Step 2: Run tests 8 and 9 and capture RED**

```bash
python3 -m unittest \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_hard_evidence_and_policies_progress_has_no_alert_or_score_payload \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_shard_write_and_freeze_verify_progress \
  -v
```

Expected RED: unexpected keyword `progress` on the routing/artifact functions.

- [ ] **Step 3: Implement single-owner shard ticks and the remaining emits**

`build_hard_evidence_surface`: `stage="hard_evidence"`, `total = len(raw_by_dataset)`, start before the sorted loop at line 143, tick after each dataset with `dataset=` only, end after the loop, error then re-raise.

`build_policies`: `stage="policies"`, `total = 4`. After each `_route(...)` at lines 263–267 emit a tick with `detail` equal to that policy name. Do not serialize `alerts`, `strength`, or `scores`.

`write_and_freeze_proposal_score` (lines 247–253):

```python
def write_and_freeze_proposal_score(root, catalog, proposal, *, progress=None):
    ...
    names = tuple(proposal.raw)
    started = monotonic()
    total = len(names)
    emit_progress_paper(progress, {"stage": "shard_write", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    try:
        for completed, dataset in enumerate(names, 1):
            write_typed_proposal_shard(
                root, catalog, proposal.raw[dataset], proposal.sources[dataset],
                proposal.routes[dataset], proposal.low_frequency, proposal.stability,
                proposal.policies, proposal.backend,
            )
            emit_progress_paper(progress, {
                "stage": "shard_write", "phase": "tick",
                "completed": completed, "total": total,
                "dataset": dataset, "detail": f"{dataset}_paper.json",
                "elapsed_s": float(monotonic() - started),
            })
        emit_progress_paper(progress, {"stage": "shard_write", "phase": "end", "completed": total, "total": total, "elapsed_s": float(monotonic() - started)})
    except Exception:
        emit_progress_paper(progress, {"stage": "shard_write", "phase": "error", "completed": completed if names else 0, "total": total, "elapsed_s": float(monotonic() - started)})
        raise
    return freeze_score_root(root, catalog, progress=progress)
```

`write_typed_proposal_shard` after the successful `write_create_only` (line 609):

```python
if progress is not None:
    emit_progress_paper(progress, {
        "stage": "shard_write", "phase": "tick",
        "completed": 1, "total": 1,
        "dataset": raw.dataset, "detail": f"{raw.dataset}_paper.json",
        "elapsed_s": 0.0,
    })
return path
```

`freeze_score_root` / `verify_score_root`: `stage="freeze_verify"`, `total = len(catalog["datasets"])`, tick after each shard validate/hash with `dataset=entry["dataset"]`, end after the manifest write (freeze) or namespace checks (verify). Do not log hashes.

`compose_proposal_score` lines 241–242 become `build_hard_evidence_surface(..., progress=progress)` and `build_policies(..., progress=progress)`.

`_proposal_artifact` (`runner_paper.py` line 96) calls `verify_score_root(..., progress=progress)` with the same sink used for composition.

- [ ] **Step 4: Run GREEN**

```bash
python3 -m unittest \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_hard_evidence_and_policies_progress_has_no_alert_or_score_payload \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_shard_write_and_freeze_verify_progress \
  paper.equipment_anomaly_detection.tests_paper.test_stability_paper \
  paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper.Task4Task5SuccessRecordTests \
  -v
```

Expected: PASS. Exactly one official `shard_write` tick per dataset on the compose path.

- [ ] **Step 5: Resync maps and commit**

Paper callers for `data_paper.py` now also include `routing_paper.py` and `artifacts_paper.py`. Exp448 `paper_routing` and `paper_task45_success_records` effects mention progress without alert/score payloads and single-owner shard ticks. Run the shared map-resync scripts.

```bash
git commit -m "$(cat <<'EOF'
feat: add evidence, policy, shard, and freeze progress

EOF
)"
```

---

### Task 7: Result-byte parity, firewall, launcher capture, closure, and regression

**Files:**
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py` (test 13; keep tests 11 and 14 as verification)
- Modify only if a launcher regression appears: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py`
- Do not modify: `scratch/manage_paper_r4_proposal_antigravity.py` unless that regression is observed
- Modify together: the six map files if this task changes any mapped tests list or if a previous task left map text stale

**Interfaces:**
- Consumes: the completed optional-callback surface from Tasks 2–6.
- Produces: proof that callback presence cannot change candidate coordinates or shard bytes; proof that the existing launcher capture path still records flushed lines as `paper-r4-progress-event-antigravity.v1` with `stream="stdout"`; proof that `code_closure` hashes now differ from live `bc5ab546` and therefore a new official root is required.

- [ ] **Step 1: Write failing test 13**

```python
def test_identical_candidate_indices_and_shard_bytes_with_and_without_callback(self) -> None:
    # Using Task4Task5SuccessRecordTests._task4_task5_values(), run
    # compose_proposal_score + write_typed_proposal_shard twice on the same
    # fixture: progress=None versus a recording callback.
    # Assert:
    #   tuple(low.selected_candidates[i].coordinate) equal
    #   tuple(stability.selected_coordinates / stable / unstable) equal
    #   shard file bytes equal (Path.read_bytes())
    # Recording callback may receive events; it must not mutate them.
```

If the fixture already patches Task6/Task7, still write real shards from the typed `low` / `stability` / `policies` objects so byte comparison is meaningful.

- [ ] **Step 2: Run test 13 and capture RED if the write path is still unwired; otherwise it should already PASS from Task 6**

```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity.ProgressLoggingAntigravityTests.test_identical_candidate_indices_and_shard_bytes_with_and_without_callback -v
```

If it fails because shard bytes differ, stop and fix the callback so it cannot mutate event mappings that science later reads (`emit_progress_paper` already copies an allowlisted dict; call sites must pass fresh literals). Do not “fix” science to match a callback.

- [ ] **Step 3: Re-verify firewall, Task4+Task5, data-boundary, and CLI**

```bash
python3 -m unittest \
  paper.equipment_anomaly_detection.tests_paper.test_progress_logging_antigravity \
  paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper.Task9ParallelCompositionTests.test_parallel_completion_progress_is_flush_visible_and_keeps_sorted_results \
  paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper.PublicCliContractTests \
  paper.equipment_anomaly_detection.tests_paper.test_data_boundary_paper \
  paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper \
  paper.equipment_anomaly_detection.tests_paper.test_stability_paper \
  paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper \
  -v
```

Expected: PASS. Test 14 is the existing `test_parallel_completion_progress_is_flush_visible_and_keeps_sorted_results`. Test 11 still drops `score` / `pvalue` / `label` / `alerts` / forbidden `detail`.

- [ ] **Step 4: Prove launcher capture is unchanged and live closure cannot resume**

```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_operations_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_stdio_isolation_antigravity -v
python3 - <<'PY'
from hashlib import sha256
from pathlib import Path
from paper.equipment_anomaly_detection.config_paper import PAPER_ROOT
from paper.equipment_anomaly_detection.lineage_paper import read_strict_json, sha256_file

lock = read_strict_json(PAPER_ROOT / "codemap_paper.lock")
for name in (
    "data_paper.py", "proposal_score_paper.py", "runner_paper.py", "pipeline_paper.py",
    "sources_paper/low_frequency_paper.py", "sources_paper/stability_paper.py",
    "routing_paper.py", "artifacts_paper.py",
):
    digest = sha256_file(PAPER_ROOT / name)
    print(name, digest, "lock", lock["hashes"][name], "match", digest == lock["hashes"][name])
print("live_reviewed_commit", "bc5ab54677b73e6f177a3afed9dfee816564b48f")
print("new_closure_differs_from_live", lock["hashes"]["proposal_score_paper.py"] != "8774aace")
PY
```

Expected: operations and stdio-isolation tests PASS without launcher edits. Current lock hashes match the files just committed. Those hashes differ from the live journal `000000_context_paper.json` closure (live `proposal_score_paper.py` is `8774aace…`). Therefore same-context resume of `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813` is forbidden. Do not open that journal to rewrite it. Do not print TEST labels.

If a launcher test fails solely because new stderr lines appear in a fixture that mocks a full child, extend `test_paper_r4_operations_antigravity.py` only far enough to treat additional flushed lines as extra `stream="stdout"` events. Do not add a second JSONL schema.

- [ ] **Step 5: Final map sync, compile, and commit**

Confirm Exp448 nodes `paper_data`, `paper_low_frequency`, `paper_stability`, `paper_routing`, `paper_proposal_score`, `paper_runner`, `paper_pipeline_cli`, and `paper_task45_success_records` list `tests_paper/test_progress_logging_antigravity.py` where they grew a progress edge. Confirm paper-map callers for `data_paper.py` are exactly the AST set:

```text
artifacts_paper.py
label_access_paper.py
proposal_score_paper.py
routing_paper.py
runner_paper.py
sources_paper/low_frequency_paper.py
sources_paper/stability_paper.py
tests_paper/test_data_boundary_paper.py
tests_paper/test_progress_logging_antigravity.py
```

`tests_paper/test_progress_logging_antigravity.py` appears as a caller of `data_paper.py` only if that test file contains `from paper.equipment_anomaly_detection.data_paper import ...` or an equivalent AST import. Keep that import. It does **not** itself become a paper-map module.

Run the shared map-resync scripts, then:

```bash
python3 -m py_compile \
  paper/equipment_anomaly_detection/data_paper.py \
  paper/equipment_anomaly_detection/proposal_score_paper.py \
  paper/equipment_anomaly_detection/runner_paper.py \
  paper/equipment_anomaly_detection/pipeline_paper.py \
  paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py \
  paper/equipment_anomaly_detection/sources_paper/stability_paper.py \
  paper/equipment_anomaly_detection/routing_paper.py \
  paper/equipment_anomaly_detection/artifacts_paper.py \
  paper/equipment_anomaly_detection/tests_paper/test_progress_logging_antigravity.py
git diff --check
```

Commit any remaining test/map updates:

```bash
git commit -m "$(cat <<'EOF'
test: prove progress neutrality, closure split, and launcher capture

EOF
)"
```

If this task changes no files because Task 6 already left tests and maps complete, do not create an empty commit. Record that fact in the executor report.

## Live-run hold (not a code task)

Until PID 57778 / 57862 exits or the user stops them:

- Operators monitor with read-only `tail`, `ls`, `stat`, `ps`, and optional `sample` / read-only `py-spy dump`.
- Do not run `scratch/manage_paper_r4_proposal_antigravity.py status|verify|resume|stop` against the live roots from this worktree.
- After the first lock-hashed edit, this worktree’s `code_closure` no longer matches live `000000_context_paper.json`. That mismatch is expected and is not a live-run status.
- Implementation of the optional helper `scratch/monitor_paper_r4_live_progress_antigravity.py` is out of scope.

An official instrumented run after review uses a new scientific root and a new ops root authorized later. Checkpoints are not migrated.

## Self-review

### Spec coverage

| Spec requirement | Plan home |
| --- | --- |
| Optional `ProgressCallback \| None = None` | Tasks 2–6 signatures |
| Surfaces: RawPreload, Task4+Task5 `_progress`, Task6 datasets/groups, Task7 candidates/views, evidence/policies, shards, freeze, journal/exit | Tasks 3–6 |
| One flushed-stderr transport | Task 3 sink install; Task 7 launcher proof |
| 15.0s per-stage tick gate; start/end/error immediate | Task 2 sink |
| Label-free / score-free text | Task 2 firewall; Tasks 4–6 asserts |
| Swallow callback failures only | Task 2 `try` around `callback(filtered)` only; Task 3 test 2 |
| Live `bc5ab546` untouched; new roots after closure change | Global constraints; Task 7 closure proof |
| Cheap write-free monitor specified, not implemented | Live-run hold |
| Exact TDD tests 1–14 | Tasks 2–7 method names match the spec |
| Exp448 + paper triplets same-commit | Task 1 plus every later commit |
| CPU worker policy unchanged | Global constraints; `WorkerRecord.from_logical_cpu_count` |
| PyTorch / MLX / MPS no-change | Acceleration assessment table |
| Cycle-free emit home in `data_paper.py` | Task 2 |
| Parent-axis Task6/Task7 counters | Tasks 4–5 |
| Single official shard-tick owner | Task 6 |
| `load_dataset_values` not extended | Task 3 test 4 |
| Workers stay `(fn, raw)` | Task 3 test 12 |
| Keep exact Task6/Task7 start/finish lines | Task 3 test 1 |
| No Codex worktree / Dataset root edits | Global constraints |
| Every new filename contains `antigravity` | New test file only |

### Placeholder scan

No TODO, TBD, FIXME, XXX, or “to be determined.” Official root names remain user-authorized later; that is an authorization rule, not a missing implementation step. The optional live monitor remains out of scope, matching the spec.

### Type consistency

- `ProgressCallback = Callable[[Mapping[str, Any]], None]` in `data_paper.py`; sink factory returns that alias; `emit_progress_paper(..., event: Mapping[str, Any]) -> None`.
- Every new parameter is keyword-only and defaults to `None`.
- `completed` / `total` are exact `int` (not `bool`); `elapsed_s` is finite `float >= 0.0`; `dataset` / `detail` are `str` or omitted.
- `_score_dataset_task4_task5(raw: RawDataset) -> tuple[HardSourceBundle, HardRouteResult]` is unchanged (`co_argcount == 1`).
- `load_dataset_values(db_path, dataset, *, sql_observer=None) -> RawDataset` is unchanged.
- `classify_stability(..., *, replay=None, progress=None)` keeps `replay` before `progress`.
- `write_typed_proposal_shard(...) -> Path` still returns the same path and still calls `write_create_only` with identical bytes on the official path.
- `pipeline_paper.main` still prints `json.dumps(value, sort_keys=True, separators=(",", ":"))` and still returns `exit_code_for_result_paper(value)`.
- `WorkerRecord.from_logical_cpu_count` still yields `dataset_workers = max(1, logical_cpu_count - 1)` and `inner_numerical_threads = 1`.
- `data_paper` does not import `proposal_score_paper`; `proposal_score_paper` may import `data_paper`; source/routing/artifact modules import emit only from `data_paper`.

### Contradictions checked

- Task4+Task5 completions are not ticks and are not 15-second gated; Task6/Task7 inner loops are.
- Default `progress=None` does not auto-install a sink (required for test 1). Official runner does install a sink (required for RawPreload visibility).
- Official `status` remains an expensive journal reload; this plan does not make it a cheap live probe.
- Regenerating maps in Task 1 does not change `code_closure` until a hashed paper `.py` file changes in Task 2. After Task 2 the live root cannot resume here.

## References

- `docs/superpowers/specs/2026-08-14-paper-r4-complete-progress-logging-antigravity-design.md`
- `paper/equipment_anomaly_detection/data_paper.py` — `SqlObserver` line 18; `load_dataset_values` lines 136–166; `__all__` line 169
- `paper/equipment_anomaly_detection/proposal_score_paper.py` — `_score_dataset_task4_task5` lines 90–93; `_progress` lines 96–102; `_collect_task4_task5` lines 134–208; `compose_proposal_score` lines 211–244; `write_and_freeze_proposal_score` lines 247–253
- `paper/equipment_anomaly_detection/runner_paper.py` — `_closure` lines 38–45; raw-load sites 130 / 142 / 156; `_write_complete` lines 151–162
- `paper/equipment_anomaly_detection/pipeline_paper.py` — `exit_code_for_result_paper` lines 13–21; `main` lines 22–25
- `paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py` — `_score_one_dataset_paper` line 349; `score_low_frequency` line 389
- `paper/equipment_anomaly_detection/sources_paper/stability_paper.py` — `_candidate_views` line 268; `classify_stability` line 315
- `paper/equipment_anomaly_detection/routing_paper.py` — `POLICY_ORDER` line 18; `build_hard_evidence_surface` line 129; `build_policies` line 237
- `paper/equipment_anomaly_detection/artifacts_paper.py` — `write_typed_proposal_shard` line 607; `freeze_score_root` line 613; `verify_score_root` line 628
- `paper/equipment_anomaly_detection/acceleration_paper.py` — `WorkerRecord.from_logical_cpu_count` lines 28–32
- `paper/equipment_anomaly_detection/tests_paper/test_artifacts_runner_paper.py` — parallel progress lines 858–901; `PublicCliContractTests` lines 1137–1198
- `scratch/manage_paper_r4_proposal_antigravity.py` — `PROGRESS_SCHEMA`; `stderr=subprocess.STDOUT`
- Live reviewed commit `bc5ab54677b73e6f177a3afed9dfee816564b48f`
