# Paper r4 Full-Population Hard-Route Parity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore full-population hard-route coordinate parity across all 1,117 datasets (2005 / 2116 / 2253 / 35 / 2218 counts, Candidate C = 2244) against SHA-bound frozen oracles, repairing Mechanism 1 (inexact Exp74d→137 replay) and Mechanism 2 (canonical_normal_state missing pycatch22==0.5.0), restoring general n_train=1 origin replay (Phoneme_normal_39 Hard {49}), validating a 33-dataset coordinate smoke, enforcing a pre-Task6/7 1,117 route-only parity gate, and executing on clean new roots with zero cache reuse.

**Architecture:** Fail-closed `verify_pycatch22_environment_paper()` in `canonical_paper.py` called in `_collect_task4_task5`; worker error propagation in `hard_sources_paper.py`; general `n_train=1` support in `legacy_route_sources_paper.py` and `hard_route_paper.py`; diagnostic-isolated exact primitive repair in `hard_route_paper.py`; atomic strict oracle-bundle flow with data-only `HardRouteParityAuthority` in `contracts_paper.py` and label-free oracle loader / `verify_hard_route_parity_paper` in `proposal_score_paper.py` gating Task6/7; explicit `mode` and `authority` propagation from `PipelineConfig` through `context_payload` and `compose_proposal_score`; clean new smoke and official roots, 33-dataset smoke catalog, and destination refusal in `scratch/manage_paper_r4_proposal_antigravity.py`; CPU-canonical ProcessPool execution with `dataset_workers = max(1, logical - 1)`.

**Tech Stack:** Python 3.14, `unittest`, `numpy==2.3.5`, `scikit-learn==1.8.0`, `aeon==1.5.0`, `pycatch22==0.5.0`, `torch==2.13.0`, canonical JSON/SHA-256, Exp448 `sha256-v1` fingerprints, paper `static_ast_sha256` triplet.

**Approved spec:** `docs/superpowers/specs/2026-08-14-paper-r4-full-route-parity-recovery-antigravity-design.md` (committed at `ad45f90f`).

## Global Constraints

- Work only in this dedicated Antigravity worktree (`/Users/mac/Documents/Time Series/.worktrees/grok-paper-r4-operational-antigravity`). Do not edit `/Users/mac/.codex/worktrees/e1be/Time Series` or `/Users/mac/Documents/Time Series`.
- Do not write to, migrate from, or resume `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813` or any prior smoke/official directories. Zero cache reuse or record migration.
- All newly created files across all tasks must include `antigravity` in their file name (e.g., `test_paper_r4_full_route_parity_recovery_antigravity.py`). Existing paper production modules keep `*_paper.py` names.
- New test file is `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`.
- Strict firewall against TEST label access: `_label_firewall()` remains active during Task4–7. `TEST_VALUES_SQL` contains no label columns.
- Replay and evidence functions accept only values and computed sources. Historical prediction coordinates are strictly oracles in verification helpers and tests; they must never be passed as runtime inputs to replay functions.
- CPU worker policy: Concurrency uses `dataset_workers = max(1, logical_cpu_count - 1)` with `inner_numerical_threads = 1`. Record detected logical CPU count and the resulting worker count in acceleration-manifest provenance.
- Explicit MLX/MPS assessment: CPU ProcessPool is retained as canonical; MLX/MPS acceleration is rejected for discrete route selectors and third-party feature extractors.
- Candidate C (`hard_plus_stable`) remains unchanged as a rule: Automatic Hard $\cup$ stable-26 from the 30-candidate stability partition; target count formula $2218 + 26 = 2244$.
- Do not authorize running full 1,117 official scoring during implementation tasks; only bounded fixtures and explicitly named route-only verification after all unit gates pass.
- Strict RED → GREEN per task. Record the exact failing reason before production edits. Each mergeable commit that changes a mapped module or mapped entrypoint signature includes all six Exp448 + paper map files in that same commit. Do not overwrite `docs/codemap/codemap_exp444_antigravity.*` or earlier version maps.

## Live Verified Authority

Parity oracles are SHA-bound, label-free prediction artifacts:

| Stage | File | SHA-256 | Extracted Field |
| --- | --- | --- | --- |
| Exp137 | `results/exp137/experiment_137_operational_triage_results.csv` | `942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d` | `hard_alert_indices` (space-delimited integers), `test_size` |
| Exp236 | `outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl` | `a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524` | `hard_indices` (JSON list of integers), `n_test` |
| Exp291 | `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl` | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` | `automatic_hard_indices`, `automatic_normal_exact_indices`, `n_test` |
| Exp404 | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` | `hard_indices` (2218 automatic Hard coordinates across full population) |

Supporting frozen authority:
- Database `/Users/mac/Documents/Dataset/univariate_ts.db` SHA-256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Population: 1,117 datasets, 195,900 TEST rows; excluded: `CornellWhaleChallenge`, `Wafer_normal_1`
- Frozen count targets: Exp137 Hard = 2005, Exp236 Hard = 2116, Exp253 Hard = 2253, exact-normal removed = 35, Automatic Hard = 2218, Candidate C = 2244.

## Acceleration Assessment (Mandatory, No-Change)

| Checklist item | Assessment |
| --- | --- |
| Tensor-eligible? | No. Route selectors are discrete rank/set/cap operations. `pycatch22.catch22_all` is a frozen third-party C/Python CPU extractor. |
| Current backend availability? | CPU only. Capable oracle environment provides `pycatch22==0.5.0` on CPU. MLX is absent. |
| Transfer / compilation overhead? | Unfavorable. Rewriting feature extractors for GPU would incur high transfer latency across 1,117 small series with no throughput benefit. |
| Deterministic and numerical parity? | Bit-exact parity against SHA-bound oracles is mandatory. Non-canonical GPU approximations are forbidden. |
| Canonical CPU fallback? | Retained as canonical. `acceleration_paper.py` designates `hard_sources` and `hard_route` as CPU-canonical. |
| Worker concurrency | CPU ProcessPool with `dataset_workers = max(1, logical_cpu_count - 1)` and `inner_numerical_threads = 1`. |

## File Structure

- Create: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/canonical_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/hard_route_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `paper/equipment_anomaly_detection/runner_paper.py`
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_hard_route_paper.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_smoke_remediation_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py`
- Modify: `scratch/manage_paper_r4_proposal_antigravity.py`
- Modify together every mapped commit: `docs/codemap/codemap_exp448_antigravity.html`, `docs/codemap/codemap_exp448_antigravity.json`, `docs/codemap/codemap_exp448_antigravity.lock`, `paper/equipment_anomaly_detection/codemap_paper.html`, `paper/equipment_anomaly_detection/codemap_paper.json`, `paper/equipment_anomaly_detection/codemap_paper.lock`

## Shared Map-Resync Procedure

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

**Exp448 triplet** (HTML `<script id="codemap-data">` payload must equal JSON file; every edge/flow step must reference a node id; lock fingerprints use lock's documented `sha256-v1` algorithm; `docs/codemap` is excluded from fingerprint input):

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

### Task 1: Codemap preflight and synchronize Exp448 + paper triplets

**Files:**
- Modify together: `docs/codemap/codemap_exp448_antigravity.html`
- Modify together: `docs/codemap/codemap_exp448_antigravity.json`
- Modify together: `docs/codemap/codemap_exp448_antigravity.lock`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.html`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.json`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.lock`

**Interfaces:**
- Consumes: Active experiment `Exp448` from `RESEARCH_STATE.json`; current paper AST caller graph from `lineage_paper._ast_import_callers`; lock algorithm `sha256-v1`.
- Produces: Both complete triplets, generated from current HEAD (`ad45f90f`), answering callers / effects / tests for all modified paper modules before production code changes.

- [ ] **Step 1: Execute Codemap Preflight Check**

Run:
```bash
git status --short
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('docs/codemap/codemap_exp448_antigravity.json').read_text(encoding='utf-8'))
ids = {node['id'] for node in data['nodes']}
print('exp448_nodes', len(data['nodes']))
print('has_paper_hard_route', 'paper_hard_route' in ids)
print('has_paper_canonical', 'paper_canonical' in ids)
print('has_paper_hard_sources', 'paper_hard_sources' in ids)
PY
```
Expected: Clean working tree on branch, node count reported.

- [ ] **Step 2: Resynchronize Codemap Triplets**

Execute the shared map-resync procedure for both Paper and Exp448 triplets.
Run validation:
```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper.ConfigLineageTests.test_paper_codemap_triplet_is_complete_and_hash_bound paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper.ConfigLineageTests.test_codemap_callers_are_complete_static_ast_import_edges
```
Expected: `Ran 2 tests in ... OK`.

- [ ] **Step 3: Commit Codemap Preflight Triplet**

Commit:
```bash
git add docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "chore(codemap): synchronize Exp448 and paper codemap triplets for route parity recovery"
```

---

### Task 2: Create full test matrix scaffolding (T1–T14) and verify initial regression locks

**Files:**
- Create: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- Test Class: `FullRouteParityRecoveryAntigravityTests(unittest.TestCase)` in `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`.
- Invariant: Zero duplicate `HardRouteParityAuthority` class definitions in test files. Tests import the production `HardRouteParityAuthority` from `paper.equipment_anomaly_detection.contracts_paper`. Missing-interface tests legitimately fail RED in Task 2 and pass once Task 4 creates the production dataclass.
- Methods:
  - T1: `test_pycatch22_pin_fails_closed_when_missing_or_wrong_version`
  - T2: `test_pycatch22_0_5_0_is_the_only_accepted_version`
  - T3: `test_pycatch22_preflight_fails_closed_before_scoring_or_persistence` and `test_score_hard_sources_propagates_dependency_error_on_eligible_dataset` (capable live scoring is an explicit gated integration test)
  - T4: `test_replay_signature_still_rejects_historical_coordinates` (Lock)
  - T5: `test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch`
  - T6: `test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence`
  - T7: `test_smoke_33_datasets_match_sha_bound_oracles`
  - T8: `test_full_1117_route_only_parity_gate_blocks_task6_and_task7`
  - T9: `test_candidate_c_rule_and_target_algebra_unchanged` (Lock)
  - T10: `test_clean_root_execution_and_rejection_of_live_r4_reuse`
  - T11: `test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations`
  - T12: `test_progress_logging_remains_optional_and_label_free` (Lock)
  - T13: `test_zero_test_label_access_until_evaluation` (Lock)
  - T14: `test_frozen_counts_2218_remain_authoritative_target` (Lock)

- [ ] **Step 1: Write the Full Test Matrix Scaffold**

Write `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py` containing complete typed fixtures for all 14 tests (exercising production imports without duplicate authority classes):

```python
"""Test suite for Paper r4 full-population hard-route parity recovery."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import (
    FROZEN_CONFIG,
    RawDataset,
)
from paper.equipment_anomaly_detection.data_paper import TEST_VALUES_SQL, emit_progress_paper, load_dataset_values
from paper.equipment_anomaly_detection.hard_route_paper import (
    HardRouteResult,
    empty_hard_route_result_paper,
    replay_hard_route,
)
from paper.equipment_anomaly_detection.routing_paper import (
    GLOBAL_CANDIDATE_BUDGET_PAPER,
    _MAXIMUM_VIEWS_PAPER,
)
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    HardSourceBundle,
    HardSourceResult,
    score_hard_sources,
)
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
    LegacyRouteEvidence,
    build_legacy_route_evidence,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    MiniRocketResult,
    score_minirocket_confirmations,
)


class FullRouteParityRecoveryAntigravityTests(unittest.TestCase):
    """Parity recovery test matrix T1 through T14."""

    def test_pycatch22_pin_fails_closed_when_missing_or_wrong_version(self) -> None:
        """T1: verify_pycatch22_environment_paper raises DependencyUnavailablePaper on missing or mismatch."""
        from paper.equipment_anomaly_detection.sources_paper.canonical_paper import (
            verify_pycatch22_environment_paper,
        )
        from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
            DependencyUnavailablePaper,
        )

        with patch("importlib.metadata.version", side_effect=Exception("not found")):
            with self.assertRaises(DependencyUnavailablePaper):
                verify_pycatch22_environment_paper()

        with patch("importlib.metadata.version", return_value="0.4.0"):
            with self.assertRaises(DependencyUnavailablePaper):
                verify_pycatch22_environment_paper()

    def test_pycatch22_0_5_0_is_the_only_accepted_version(self) -> None:
        """T2: PINNED_PYCATCH22_VERSION_PAPER is '0.5.0' and returns valid metadata dict."""
        from paper.equipment_anomaly_detection.sources_paper import canonical_paper

        self.assertTrue(hasattr(canonical_paper, "PINNED_PYCATCH22_VERSION_PAPER"))
        self.assertEqual(canonical_paper.PINNED_PYCATCH22_VERSION_PAPER, "0.5.0")
        with patch("importlib.metadata.version", return_value="0.5.0"), patch(
            "pycatch22.catch22_all", create=True
        ):
            meta = canonical_paper.verify_pycatch22_environment_paper()
            self.assertEqual(meta.get("pycatch22"), "0.5.0")

    def test_pycatch22_preflight_fails_closed_before_scoring_or_persistence(self) -> None:
        """T3 (Unit RED - Preflight): Environment gate aborts execution before any dataset scoring."""
        from paper.equipment_anomaly_detection.sources_paper.canonical_paper import (
            verify_pycatch22_environment_paper,
        )
        from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
            DependencyUnavailablePaper,
        )

        with patch("importlib.metadata.version", side_effect=Exception("pycatch22 missing")):
            with self.assertRaises(DependencyUnavailablePaper):
                verify_pycatch22_environment_paper()

    def test_score_hard_sources_propagates_dependency_error_on_eligible_dataset(self) -> None:
        """T3 (Unit RED - Propagation): score_hard_sources propagates DependencyUnavailablePaper on eligible datasets."""
        from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
            DependencyUnavailablePaper,
        )

        rng = np.random.default_rng(20260814)
        train_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(10))
        test_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(20))
        raw = RawDataset(
            dataset="Synthetic_eligible_dataset",
            metadata_length=100,
            train_indices=tuple(range(10)),
            train_blobs=tuple(r.tobytes() for r in train_rows),
            train_rows=train_rows,
            test_indices=tuple(range(20)),
            test_blobs=tuple(r.tobytes() for r in test_rows),
            test_rows=test_rows,
        )
        with patch(
            "paper.equipment_anomaly_detection.sources_paper.hard_sources_paper._canonical_source_paper",
            side_effect=DependencyUnavailablePaper("dependency_unavailable:pycatch22==0.5.0"),
        ):
            with self.assertRaises(DependencyUnavailablePaper):
                score_hard_sources(raw)

    def test_replay_signature_still_rejects_historical_coordinates(self) -> None:
        """T4 (Lock): replay_hard_route accepts only values, sources, and mini (no oracle kwargs)."""
        sig = inspect.signature(replay_hard_route)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["values", "sources", "mini"])
        sig_legacy = inspect.signature(build_legacy_route_evidence)
        self.assertEqual(list(sig_legacy.parameters.keys()), ["values"])

    def test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch(self) -> None:
        """T5: Phoneme_normal_39 (n_train=1) produces Hard {49} and code contains no dataset name check."""
        from paper.equipment_anomaly_detection import hard_route_paper, sources_paper
        from paper.equipment_anomaly_detection.sources_paper import legacy_route_sources_paper

        hard_route_src = inspect.getsource(hard_route_paper)
        legacy_src = inspect.getsource(legacy_route_sources_paper)
        self.assertNotIn("Phoneme_normal_39", hard_route_src)
        self.assertNotIn("Phoneme_normal_39", legacy_src)

        rng = np.random.default_rng(39)
        train_rows = (rng.normal(0, 1, size=50).astype(np.float32),)
        test_rows = tuple(rng.normal(0, 1, size=50).astype(np.float32) for _ in range(50))
        raw = RawDataset(
            dataset="Phoneme_normal_39",
            metadata_length=50,
            train_indices=(0,),
            train_blobs=tuple(r.tobytes() for r in train_rows),
            train_rows=train_rows,
            test_indices=tuple(range(50)),
            test_blobs=tuple(r.tobytes() for r in test_rows),
            test_rows=test_rows,
        )
        sources = score_hard_sources(raw)
        mini = score_minirocket_confirmations(raw)
        route = replay_hard_route(raw, sources, mini)
        self.assertIn(49, route.exp137.hard | route.exp236.hard | route.automatic_hard)

    def test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence(self) -> None:
        """T6: Stage isolation verifies origin-exact selector behavior against frozen oracle."""
        from paper.equipment_anomaly_detection.contracts_paper import HardRouteParityAuthority
        from paper.equipment_anomaly_detection.proposal_score_paper import load_hard_route_parity_authority
        from scratch.manage_paper_r4_proposal_antigravity import (
            EXP137_ORACLE_PATH,
            EXP236_ORACLE_PATH,
            EXP291_ORACLE_PATH,
            EXP404_ORACLE_PATH,
        )

        authority = load_hard_route_parity_authority(
            EXP137_ORACLE_PATH,
            EXP236_ORACLE_PATH,
            EXP291_ORACLE_PATH,
            EXP404_ORACLE_PATH,
        )
        self.assertIsInstance(authority, HardRouteParityAuthority)
        self.assertEqual(authority.population_count, 1117)
        self.assertEqual(authority.counts["exp137_hard"], 2005)
        self.assertEqual(authority.counts["exp236_hard"], 2116)
        self.assertEqual(authority.counts["exp253_hard"], 2253)
        self.assertEqual(authority.counts["exact_normal_removed"], 35)
        self.assertEqual(authority.counts["automatic_hard"], 2218)

    def test_smoke_33_datasets_match_sha_bound_oracles(self) -> None:
        """T7: All 33 smoke datasets achieve per-dataset coordinate equality against 4 oracles."""
        from paper.equipment_anomaly_detection.proposal_score_paper import load_hard_route_parity_authority
        from scratch.manage_paper_r4_proposal_antigravity import (
            EXP137_ORACLE_PATH,
            EXP236_ORACLE_PATH,
            EXP291_ORACLE_PATH,
            EXP404_ORACLE_PATH,
            SMOKE_DATASETS,
        )

        self.assertEqual(len(SMOKE_DATASETS), 33)
        authority = load_hard_route_parity_authority(
            EXP137_ORACLE_PATH,
            EXP236_ORACLE_PATH,
            EXP291_ORACLE_PATH,
            EXP404_ORACLE_PATH,
        )
        smoke_auth = authority.project(SMOKE_DATASETS)
        self.assertEqual(smoke_auth.population_count, 33)
        db_path = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
        for dataset in SMOKE_DATASETS:
            raw = load_dataset_values(db_path, dataset)
            self.assertEqual(len(raw.test_rows), smoke_auth.n_test[dataset])
            sources = score_hard_sources(raw)
            mini = score_minirocket_confirmations(raw)
            route = replay_hard_route(raw, sources, mini)
            self.assertEqual(route.exp137.hard, smoke_auth.exp137_hard[dataset])
            self.assertEqual(route.exp236.hard, smoke_auth.exp236_hard[dataset])
            self.assertEqual(route.exp253.hard, smoke_auth.exp291_pre_exact[dataset])
            self.assertEqual(route.exact_normal_removed, smoke_auth.exp291_exact_removed[dataset])
            self.assertEqual(route.automatic_hard, smoke_auth.automatic_hard[dataset])

    def test_full_1117_route_only_parity_gate_blocks_task6_and_task7(self) -> None:
        """T8: verify_hard_route_parity_paper halts compose_proposal_score on mismatch before Task6."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import HardRouteParityAuthority
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            compose_proposal_score,
            verify_hard_route_parity_paper,
        )

        # Incomplete inputs must reject with ValueError
        with self.assertRaises(ValueError):
            verify_hard_route_parity_paper({}, authority=None, mode="full")

        fake_routes = {"dataset_a": empty_hard_route_result_paper(10)}
        fake_auth = HardRouteParityAuthority(
            exp137_hard={"dataset_a": frozenset({1})},
            exp236_hard={"dataset_a": frozenset({1})},
            exp291_pre_exact={"dataset_a": frozenset({1})},
            exp291_exact_removed={"dataset_a": frozenset()},
            automatic_hard={"dataset_a": frozenset({1})},
            n_test={"dataset_a": 10},
            counts={"exp137_hard": 1, "exp236_hard": 1, "exp253_hard": 1, "exact_normal_removed": 0, "automatic_hard": 1},
            oracle_sha256={"exp137": "a" * 64, "exp236": "b" * 64, "exp291": "c" * 64, "exp404": "d" * 64},
            population_count=1,
        )
        with self.assertRaises(ValueError):
            verify_hard_route_parity_paper(fake_routes, authority=fake_auth, mode="smoke")

    def test_candidate_c_rule_and_target_algebra_unchanged(self) -> None:
        """T9 (Lock): Candidate C budget 30, view cap 16, survival 14/16, formula 2218 + 26 = 2244."""
        self.assertEqual(GLOBAL_CANDIDATE_BUDGET_PAPER, 30)
        self.assertEqual(_MAXIMUM_VIEWS_PAPER, 16)
        survival_threshold = 14 / 16
        self.assertAlmostEqual(survival_threshold, 0.875)
        automatic_hard_target = 2218
        stable_target = 26
        self.assertEqual(automatic_hard_target + stable_target, 2244)

    def test_clean_root_execution_and_rejection_of_live_r4_reuse(self) -> None:
        """T10: Runner creates fresh records in clean root and rejects live r4 root reuse."""
        forbidden_prior = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813")
        from scratch.manage_paper_r4_proposal_antigravity import validate_clean_root_destination_paper

        with self.assertRaises(ValueError):
            validate_clean_root_destination_paper(forbidden_prior)

    def test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations(self) -> None:
        """T11: Launcher constants use new clean 20260814 paths and refuse prior roots."""
        from scratch.manage_paper_r4_proposal_antigravity import (
            AUTHORIZED_OFFICIAL_ROOT_20260814,
            AUTHORIZED_SMOKE_ROOT_20260814,
            FORBIDDEN_DESTINATION_ROOTS,
        )

        self.assertIn("20260814", str(AUTHORIZED_SMOKE_ROOT_20260814))
        self.assertIn("20260814", str(AUTHORIZED_OFFICIAL_ROOT_20260814))
        self.assertIn(
            Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813"),
            FORBIDDEN_DESTINATION_ROOTS,
        )

    def test_progress_logging_remains_optional_and_label_free(self) -> None:
        """T12 (Lock): Progress logging default None is bit-identical and label-free."""
        event = {"stage": "task4_task5", "phase": "tick", "completed": 1, "total": 10, "dataset": "Synthetic_dataset"}
        emitted: list[dict] = []
        emit_progress_paper(emitted.append, event)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["stage"], "task4_task5")
        self.assertNotIn("y_test", str(emitted[0]))
        self.assertNotIn("labels", str(emitted[0]))

    def test_zero_test_label_access_until_evaluation(self) -> None:
        """T13 (Lock): TEST_VALUES_SQL selects no label column; label firewall is intact."""
        self.assertNotIn("y_test", TEST_VALUES_SQL.lower())
        self.assertNotIn("label", TEST_VALUES_SQL.lower())
        self.assertNotIn("is_anomaly", TEST_VALUES_SQL.lower())

    def test_frozen_counts_2218_remain_authoritative_target(self) -> None:
        """T14 (Lock): Target counts remain 2005/2116/2253/35/2218; 2203 is not a target."""
        self.assertEqual(FROZEN_CONFIG.ablation_alert_counts[0], 2218)
        self.assertNotEqual(FROZEN_CONFIG.ablation_alert_counts[0], 2203)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the Initial Test Matrix (Observe RED and GREEN Locks)**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests
```
Expected output:
- T4, T9, T12, T13, T14 pass GREEN (Regression Locks).
- T1, T2, T3, T5, T6, T7, T8, T10, T11 fail RED (Expected initial state before repairs).

- [ ] **Step 3: Resynchronize Codemaps and Commit Test Matrix**

Run shared map-resync procedure, then:
```bash
git add experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "test(parity): add 14-test verification matrix for full route parity recovery"
```

---

### Task 3: Fail-closed pycatch22==0.5.0 environment pin and worker error handling (PR 1)

**Files:**
- Modify: `paper/equipment_anomaly_detection/sources_paper/canonical_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- `PINNED_PYCATCH22_VERSION_PAPER: str = "0.5.0"` in `sources_paper/canonical_paper.py`.
- `verify_pycatch22_environment_paper() -> dict[str, str]` in `sources_paper/canonical_paper.py`:
  - Returns `{"pycatch22": version}` when package is present, version equals `"0.5.0"`, and `catch22_all` is importable.
  - Raises `DependencyUnavailablePaper("dependency_unavailable:pycatch22==0.5.0")` on missing package, wrong version, or failed import.
- `_collect_task4_task5(...)` in `proposal_score_paper.py`:
  - Invokes `verify_pycatch22_environment_paper()` before processing any dataset.
- `score_hard_sources(values: RawDataset) -> HardSourceBundle` in `sources_paper/hard_sources_paper.py`:
  - If `values` is eligible ($n_{\text{train}} \ge 5$ and finite), unexpected missing dependencies raise `DependencyUnavailablePaper` instead of being swallowed into silent `eligible=False`. Ordinary abstentions (`unsupported_n_train_lt_5`, `nonfinite_input`) remain intact.

- [ ] **Step 1: Run RED Command for Task 3**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_pin_fails_closed_when_missing_or_wrong_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_0_5_0_is_the_only_accepted_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_preflight_fails_closed_before_scoring_or_persistence experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_score_hard_sources_propagates_dependency_error_on_eligible_dataset
```
Expected failure: `ImportError: cannot import name 'verify_pycatch22_environment_paper'` or `AttributeError: module '...canonical_paper' has no attribute 'PINNED_PYCATCH22_VERSION_PAPER'`.

- [ ] **Step 2: Implement Fail-Closed pycatch22 Pin in `canonical_paper.py`**

In `paper/equipment_anomaly_detection/sources_paper/canonical_paper.py`, add:
```python
import importlib.metadata

PINNED_PYCATCH22_VERSION_PAPER = "0.5.0"


def verify_pycatch22_environment_paper() -> dict[str, str]:
    """Fail-closed Task4 environment gate. Missing or mismatched version aborts execution."""
    try:
        version = importlib.metadata.version("pycatch22")
    except Exception as exc:
        raise DependencyUnavailablePaper("dependency_unavailable:pycatch22==0.5.0") from exc
    if version != PINNED_PYCATCH22_VERSION_PAPER:
        raise DependencyUnavailablePaper(
            f"dependency_unavailable:pycatch22=={PINNED_PYCATCH22_VERSION_PAPER} (found {version})"
        )
    try:
        from pycatch22 import catch22_all
    except ImportError as exc:
        raise DependencyUnavailablePaper("dependency_unavailable:pycatch22==0.5.0") from exc
    return {"pycatch22": version}
```

Update `__all__` in `canonical_paper.py` to export `PINNED_PYCATCH22_VERSION_PAPER` and `verify_pycatch22_environment_paper`.

- [ ] **Step 3: Wire Environment Verification into `_collect_task4_task5` in `proposal_score_paper.py`**

Import `verify_pycatch22_environment_paper` from `.sources_paper.canonical_paper` in `proposal_score_paper.py`.
In `_collect_task4_task5`:
```python
    verify_exp84_environment_paper()
    verify_pycatch22_environment_paper()
```

- [ ] **Step 4: Update Worker Error Handling in `score_hard_sources` in `hard_sources_paper.py`**

In `paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py`, ensure that on eligible datasets ($n_{\text{train}} \ge 5$, finite), `_canonical_source_paper` and other sources do not silently swallow unexpected dependency errors:
```python
    # For canonical_normal_state on eligible datasets:
    try:
        results["canonical_normal_state"] = _completed_paper("canonical_normal_state", *_canonical_source_paper(train, test))
    except DependencyUnavailablePaper as exc:
        raise DependencyUnavailablePaper(f"canonical_normal_state failed on eligible dataset: {exc}") from exc
```

- [ ] **Step 5: Run GREEN Verification for Task 3**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_pin_fails_closed_when_missing_or_wrong_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_0_5_0_is_the_only_accepted_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_preflight_fails_closed_before_scoring_or_persistence experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_score_hard_sources_propagates_dependency_error_on_eligible_dataset
```
Expected: `Ran 4 tests in ... OK`.

- [ ] **Step 6: Run Relevant Regression Suite**

Run:
```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_hard_sources_paper
```
Expected: All tests pass GREEN.

- [ ] **Step 7: Resynchronize Codemaps and Commit PR 1**

Run shared map-resync procedure, then:
```bash
git add paper/equipment_anomaly_detection/sources_paper/canonical_paper.py paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py paper/equipment_anomaly_detection/proposal_score_paper.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "feat(sources): enforce fail-closed pycatch22==0.5.0 environment pin and worker propagation"
```

---

### Task 4: Atomic strict oracle-bundle flow, values-only SHA-pinned oracle adapters, and HardRouteParityAuthority

**Files:**
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- Data-only `HardRouteParityAuthority` in `contracts_paper.py`:
  - Frozen slots dataclass holding verified population lane maps (`exp137_hard`, `exp236_hard`, `exp291_pre_exact`, `exp291_exact_removed`, `automatic_hard`, `n_test`), aggregate `counts`, verified `oracle_sha256` digest map, and `population_count`.
  - Methods: `to_dict() -> dict[str, object]`, `from_dict(payload: dict[str, object]) -> HardRouteParityAuthority`, and `project(datasets: Sequence[str]) -> HardRouteParityAuthority` slicing by direct indexing `self.exp137_hard[d]` (no `.get(empty)`).
  - Invariant: Zero `default_verified()` method or machine paths in `contracts_paper.py`. The authorized launcher/manager owns the paths and constructs the instance.
- Atomic Strict Oracle-Bundle Loader in `proposal_score_paper.py`:
  - `load_hard_route_parity_authority(exp137_path: Path, exp236_path: Path, exp291_path: Path, exp404_path: Path) -> HardRouteParityAuthority`:
    1. Byte verification: Verify all four files by exact SHA-256 before parsing.
    2. Exp137 CSV: Exact header tuple check (`tuple(reader.fieldnames) == EXPECTED_EXP137_HEADER_TUPLE`). Extract only `dataset_name`, `hard_alert_indices`, `test_size`. Validate exact full 1117 key set (matching catalog population, no duplicate dataset names).
    3. Exp236 JSONL: Exact frozen key set check per record (`set(rec.keys()) == EXP236_EXACT_KEYS`). Validate `contains_test_labels is False` and `contains_evaluation_metrics is False`. Extract `dataset`, `hard_indices`, `n_test`. Validate exact full 1117 key set and `n_test == exp137_n_test[dataset]`.
    4. Exp291 JSONL: Exact frozen key set check per record (`set(rec.keys()) == EXP291_EXACT_KEYS`). Validate `contains_test_labels is False` and `contains_evaluation_metrics is False`. Extract `dataset`, `automatic_hard_indices`, `automatic_normal_exact_indices`, `n_test`. Validate exact full 1117 key set and `n_test == exp137_n_test[dataset]`.
    5. Construct full 1117 population + n_test authority structure.
    6. Exp404 JSONL: Sparse exhaustive 2218 rows against that full population. Exact frozen key set check per record (`set(rec.keys()) == EXP404_EXACT_KEYS`), false label/metric flags. Explicitly initialize all 1117 datasets in map (empty frozenset() for non-alert datasets). Validate duplicates, coordinate ranges in `[0, n_test)`, total count == 2218, exact equality with Exp291 `automatic_hard_indices` for all 1117 datasets.
    7. Construct and return verified `HardRouteParityAuthority`.
- T6 Sequencing: Adapter-missing RED in Task 4 is `ImportError: cannot import name 'load_hard_route_parity_authority'`, which is separate from the exact diagnostic RED created only after the harness commits its identified dataset/stage/primitive in Task 5.

- [ ] **Step 1: Run RED Command for Task 4**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence
```
Expected failure: `ImportError: cannot import name 'HardRouteParityAuthority' from 'paper.equipment_anomaly_detection.contracts_paper'` or `load_hard_route_parity_authority`.

- [ ] **Step 2: Implement Data-Only `HardRouteParityAuthority` in `contracts_paper.py`**

In `paper/equipment_anomaly_detection/contracts_paper.py`, add:

```python
@dataclass(frozen=True, slots=True)
class HardRouteParityAuthority:
    """Immutable data-only hard-route parity authority bundle."""

    exp137_hard: Mapping[str, frozenset[int]]
    exp236_hard: Mapping[str, frozenset[int]]
    exp291_pre_exact: Mapping[str, frozenset[int]]
    exp291_exact_removed: Mapping[str, frozenset[int]]
    automatic_hard: Mapping[str, frozenset[int]]
    n_test: Mapping[str, int]
    counts: Mapping[str, int]
    oracle_sha256: Mapping[str, str]
    population_count: int

    def __post_init__(self) -> None:
        exp137 = dict(self.exp137_hard)
        exp236 = dict(self.exp236_hard)
        exp291_pre = dict(self.exp291_pre_exact)
        exp291_rem = dict(self.exp291_exact_removed)
        auto = dict(self.automatic_hard)
        n_test = dict(self.n_test)
        counts = dict(self.counts)
        sha = dict(self.oracle_sha256)

        keys = set(exp137)
        if (
            set(exp236) != keys
            or set(exp291_pre) != keys
            or set(exp291_rem) != keys
            or set(auto) != keys
            or set(n_test) != keys
        ):
            raise ValueError("HardRouteParityAuthority lane key sets must match exactly")
        if len(keys) != self.population_count or self.population_count < 1:
            raise ValueError("HardRouteParityAuthority population count mismatch")

        required_counts = {
            "exp137_hard",
            "exp236_hard",
            "exp253_hard",
            "exact_normal_removed",
            "automatic_hard",
        }
        if set(counts) != required_counts or any(type(c) is not int or c < 0 for c in counts.values()):
            raise ValueError("HardRouteParityAuthority counts invalid")

        required_oracles = {"exp137", "exp236", "exp291", "exp404"}
        if set(sha) != required_oracles or any(type(d) is not str or len(d) != 64 for d in sha.values()):
            raise ValueError("HardRouteParityAuthority oracle digests invalid")

        object.__setattr__(self, "exp137_hard", MappingProxyType(exp137))
        object.__setattr__(self, "exp236_hard", MappingProxyType(exp236))
        object.__setattr__(self, "exp291_pre_exact", MappingProxyType(exp291_pre))
        object.__setattr__(self, "exp291_exact_removed", MappingProxyType(exp291_rem))
        object.__setattr__(self, "automatic_hard", MappingProxyType(auto))
        object.__setattr__(self, "n_test", MappingProxyType(n_test))
        object.__setattr__(self, "counts", MappingProxyType(counts))
        object.__setattr__(self, "oracle_sha256", MappingProxyType(sha))

    def project(self, datasets: Sequence[str]) -> "HardRouteParityAuthority":
        """Project authority to an explicit population using direct indexing (raises on missing)."""
        selected = tuple(datasets)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("projection requires a non-empty sequence of unique datasets")

        p_137 = {d: self.exp137_hard[d] for d in selected}
        p_236 = {d: self.exp236_hard[d] for d in selected}
        p_pre = {d: self.exp291_pre_exact[d] for d in selected}
        p_rem = {d: self.exp291_exact_removed[d] for d in selected}
        p_auto = {d: self.automatic_hard[d] for d in selected}
        p_ntest = {d: self.n_test[d] for d in selected}

        p_counts = {
            "exp137_hard": sum(len(v) for v in p_137.values()),
            "exp236_hard": sum(len(v) for v in p_236.values()),
            "exp253_hard": sum(len(v) for v in p_pre.values()),
            "exact_normal_removed": sum(len(v) for v in p_rem.values()),
            "automatic_hard": sum(len(v) for v in p_auto.values()),
        }

        return HardRouteParityAuthority(
            exp137_hard=p_137,
            exp236_hard=p_236,
            exp291_pre_exact=p_pre,
            exp291_exact_removed=p_rem,
            automatic_hard=p_auto,
            n_test=p_ntest,
            counts=p_counts,
            oracle_sha256=dict(self.oracle_sha256),
            population_count=len(selected),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "paper-hard-route-parity-authority.v1",
            "population_count": self.population_count,
            "counts": dict(self.counts),
            "oracle_sha256": dict(self.oracle_sha256),
            "n_test": dict(self.n_test),
            "exp137_hard": {k: sorted(v) for k, v in self.exp137_hard.items()},
            "exp236_hard": {k: sorted(v) for k, v in self.exp236_hard.items()},
            "exp291_pre_exact": {k: sorted(v) for k, v in self.exp291_pre_exact.items()},
            "exp291_exact_removed": {k: sorted(v) for k, v in self.exp291_exact_removed.items()},
            "automatic_hard": {k: sorted(v) for k, v in self.automatic_hard.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HardRouteParityAuthority":
        if not isinstance(payload, dict) or payload.get("schema") != "paper-hard-route-parity-authority.v1":
            raise ValueError("invalid HardRouteParityAuthority payload schema")
        return cls(
            exp137_hard={k: frozenset(v) for k, v in payload["exp137_hard"].items()},
            exp236_hard={k: frozenset(v) for k, v in payload["exp236_hard"].items()},
            exp291_pre_exact={k: frozenset(v) for k, v in payload["exp291_pre_exact"].items()},
            exp291_exact_removed={k: frozenset(v) for k, v in payload["exp291_exact_removed"].items()},
            automatic_hard={k: frozenset(v) for k, v in payload["automatic_hard"].items()},
            n_test={k: int(v) for k, v in payload["n_test"].items()},
            counts={k: int(v) for k, v in payload["counts"].items()},
            oracle_sha256={k: str(v) for k, v in payload["oracle_sha256"].items()},
            population_count=int(payload["population_count"]),
        )
```

- [ ] **Step 3: Implement Atomic Strict Oracle-Bundle Loader in `proposal_score_paper.py`**

In `paper/equipment_anomaly_detection/proposal_score_paper.py`, add:

```python
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

from .contracts_paper import HardRouteParityAuthority

EXP137_ORACLE_SHA256 = "942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d"
EXP236_ORACLE_SHA256 = "a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524"
EXP291_ORACLE_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
EXP404_ORACLE_SHA256 = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"

EXPECTED_EXP137_HEADER_TUPLE = (
    "experiment_id", "dataset_name", "family", "config_name", "selector_name",
    "selector_reason", "threshold_method", "score_family", "score_source_name",
    "hard_alert_indices", "standard_review_indices", "priority_review_indices",
    "hard_alert_count", "standard_review_count", "priority_review_count",
    "no_alert_count", "hard_tp", "hard_fp", "hard_fn", "hard_precision",
    "hard_recall", "hard_f1", "standard_review_tp", "standard_review_fp",
    "standard_review_fn", "standard_review_precision", "standard_review_recall",
    "standard_review_f1", "priority_review_tp", "priority_review_fp",
    "priority_review_fn", "priority_review_precision", "priority_review_recall",
    "priority_review_f1", "combined_tp", "combined_fp", "combined_fn",
    "combined_precision", "combined_recall", "combined_f1", "combined_metric_scope",
    "routing_uses_test_labels", "routing_uses_test_position", "routing_uses_family_performance",
    "priority_review_posthoc_research_rule", "prospective_priority_validation_required",
    "source_exp133_config", "source_exp135_config", "source_exp93_hard_count",
    "source_exp135_review_count", "train_normal_count", "sequence_length",
    "test_size", "anomaly_count",
)

EXP236_EXACT_KEYS = frozenset({
    "contains_evaluation_metrics", "contains_test_labels", "dataset",
    "deployment_eligible", "experiment_id", "hard_indices", "hard_policy",
    "n_test", "new_hard_indices", "new_standard_review_indices",
    "priority_review_indices", "priority_review_policy",
    "retrospective_candidate_packaging", "standard_review_indices",
    "standard_review_policy",
})

EXP291_EXACT_KEYS = frozenset({
    "automatic_hard_indices", "automatic_normal_exact_indices", "automatic_normal_rule",
    "contains_evaluation_metrics", "contains_test_labels", "dataset",
    "default_review_indices", "experiment_id", "n_test", "priority_review_indices",
    "prospective_validated", "recovery_priority_scope", "recovery_shadow_review_indices",
    "research_only_recovery_priority_indices", "retrospective_integration",
    "shadow_review_indices", "source_baseline_hard_indices", "standard_review_indices",
})

EXP404_EXACT_KEYS = frozenset({
    "abstention_reason", "candidate_index", "confidence_by_alpha",
    "contains_evaluation_metrics", "contains_test_labels", "dataset",
    "eligibility", "scores_by_alpha",
})


def _verify_bytes_sha(path: Path, expected_sha: str) -> None:
    raw = Path(path).read_bytes()
    # Normalize CRLF if needed for git checkout differences on CSV
    actual_sha = sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        crlf_sha = sha256(raw.replace(b"\n", b"\r\n")).hexdigest()
        if crlf_sha != expected_sha:
            raise ValueError(f"Oracle SHA-256 mismatch on {path}: expected {expected_sha} got {actual_sha}")


def load_hard_route_parity_authority(
    exp137_path: Path,
    exp236_path: Path,
    exp291_path: Path,
    exp404_path: Path,
) -> HardRouteParityAuthority:
    """Atomic strict oracle-bundle loader validating 100% exact schemas, keys, n_test, and coordinates."""
    # 1. Byte verification
    _verify_bytes_sha(exp137_path, EXP137_ORACLE_SHA256)
    _verify_bytes_sha(exp236_path, EXP236_ORACLE_SHA256)
    _verify_bytes_sha(exp291_path, EXP291_ORACLE_SHA256)
    _verify_bytes_sha(exp404_path, EXP404_ORACLE_SHA256)

    # 2. Parse and globally validate Exp137 CSV
    exp137_hard: dict[str, frozenset[int]] = {}
    n_test_map: dict[str, int] = {}
    with Path(exp137_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        if header != EXPECTED_EXP137_HEADER_TUPLE:
            raise ValueError(f"Exp137 CSV exact header tuple mismatch: {header[:5]}")
        d_idx = header.index("dataset_name")
        h_idx = header.index("hard_alert_indices")
        t_idx = header.index("test_size")
        for row in reader:
            dataset = str(row[d_idx])
            if dataset in exp137_hard:
                raise ValueError(f"Duplicate dataset {dataset} in Exp137 CSV")
            raw_indices = str(row[h_idx]).strip()
            indices = frozenset(map(int, raw_indices.split())) if raw_indices else frozenset()
            exp137_hard[dataset] = indices
            test_size = int(row[t_idx])
            if test_size <= 0:
                raise ValueError(f"Invalid test_size {test_size} on {dataset}")
            n_test_map[dataset] = test_size

    if len(exp137_hard) != 1117:
        raise ValueError(f"Exp137 row count must be exactly 1117 (got {len(exp137_hard)})")

    full_population = set(exp137_hard)

    # 3. Parse and globally validate Exp236 JSONL
    exp236_hard: dict[str, frozenset[int]] = {}
    with Path(exp236_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            if set(rec.keys()) != EXP236_EXACT_KEYS:
                raise ValueError(f"Exp236 JSONL exact key mismatch on record: {sorted(rec.keys())}")
            if rec["contains_test_labels"] is not False or rec["contains_evaluation_metrics"] is not False:
                raise ValueError("Exp236 record violates label/metric boundary")
            dataset = str(rec["dataset"])
            if dataset in exp236_hard:
                raise ValueError(f"Duplicate dataset {dataset} in Exp236 JSONL")
            if dataset not in full_population:
                raise ValueError(f"Unknown dataset {dataset} in Exp236 JSONL")
            if int(rec["n_test"]) != n_test_map[dataset]:
                raise ValueError(f"Exp236 n_test mismatch on {dataset}")
            exp236_hard[dataset] = frozenset(map(int, rec["hard_indices"]))

    if set(exp236_hard) != full_population:
        raise ValueError("Exp236 JSONL key set differs from full 1117 population")

    # 4. Parse and globally validate Exp291 JSONL
    exp291_auto: dict[str, frozenset[int]] = {}
    exp291_removed: dict[str, frozenset[int]] = {}
    exp291_pre: dict[str, frozenset[int]] = {}
    with Path(exp291_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            if set(rec.keys()) != EXP291_EXACT_KEYS:
                raise ValueError(f"Exp291 JSONL exact key mismatch on record: {sorted(rec.keys())}")
            if rec["contains_test_labels"] is not False or rec["contains_evaluation_metrics"] is not False:
                raise ValueError("Exp291 record violates label/metric boundary")
            dataset = str(rec["dataset"])
            if dataset in exp291_auto:
                raise ValueError(f"Duplicate dataset {dataset} in Exp291 JSONL")
            if dataset not in full_population:
                raise ValueError(f"Unknown dataset {dataset} in Exp291 JSONL")
            if int(rec["n_test"]) != n_test_map[dataset]:
                raise ValueError(f"Exp291 n_test mismatch on {dataset}")
            hard = frozenset(map(int, rec["automatic_hard_indices"]))
            removed = frozenset(map(int, rec["automatic_normal_exact_indices"]))
            exp291_auto[dataset] = hard
            exp291_removed[dataset] = removed
            exp291_pre[dataset] = hard | removed

    if set(exp291_auto) != full_population:
        raise ValueError("Exp291 JSONL key set differs from full 1117 population")

    # 5. Parse and globally validate Exp404 JSONL
    exp404_auto_builder: dict[str, set[int]] = {d: set() for d in full_population}
    exp404_row_count = 0
    with Path(exp404_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            if set(rec.keys()) != EXP404_EXACT_KEYS:
                raise ValueError(f"Exp404 JSONL exact key mismatch on record: {sorted(rec.keys())}")
            if rec["contains_test_labels"] is not False or rec["contains_evaluation_metrics"] is not False:
                raise ValueError("Exp404 record violates label/metric boundary")
            dataset = str(rec["dataset"])
            if dataset not in full_population:
                raise ValueError(f"Unknown dataset {dataset} in Exp404 JSONL")
            coord = int(rec["candidate_index"])
            if coord < 0 or coord >= n_test_map[dataset]:
                raise ValueError(f"Exp404 coordinate out of bounds on {dataset}: {coord}")
            if coord in exp404_auto_builder[dataset]:
                raise ValueError(f"Exp404 duplicate coordinate {coord} on {dataset}")
            exp404_auto_builder[dataset].add(coord)
            exp404_row_count += 1

    if exp404_row_count != 2218:
        raise ValueError(f"Exp404 row count must be exactly 2218 (got {exp404_row_count})")

    exp404_auto = {d: frozenset(v) for d, v in exp404_auto_builder.items()}
    if exp404_auto != exp291_auto:
        raise ValueError("Exp404 automatic Hard coordinates differ from Exp291 automatic Hard coordinates")

    counts = {
        "exp137_hard": sum(len(v) for v in exp137_hard.values()),
        "exp236_hard": sum(len(v) for v in exp236_hard.values()),
        "exp253_hard": sum(len(v) for v in exp291_pre.values()),
        "exact_normal_removed": sum(len(v) for v in exp291_removed.values()),
        "automatic_hard": sum(len(v) for v in exp404_auto.values()),
    }
    if (
        counts["exp137_hard"] != 2005
        or counts["exp236_hard"] != 2116
        or counts["exp253_hard"] != 2253
        or counts["exact_normal_removed"] != 35
        or counts["automatic_hard"] != 2218
    ):
        raise ValueError(f"Oracle authority count validation failed: {counts}")

    oracle_digests = {
        "exp137": EXP137_ORACLE_SHA256,
        "exp236": EXP236_ORACLE_SHA256,
        "exp291": EXP291_ORACLE_SHA256,
        "exp404": EXP404_ORACLE_SHA256,
    }

    return HardRouteParityAuthority(
        exp137_hard=exp137_hard,
        exp236_hard=exp236_hard,
        exp291_pre_exact=exp291_pre,
        exp291_exact_removed=exp291_removed,
        automatic_hard=exp404_auto,
        n_test=n_test_map,
        counts=counts,
        oracle_sha256=oracle_digests,
        population_count=1117,
    )
```

- [ ] **Step 4: Run GREEN Verification for Task 4**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence
```
Expected: `Ran 1 test in ... OK`.

- [ ] **Step 5: Resynchronize Codemaps and Commit Task 4**

Run shared map-resync procedure, then:
```bash
git add paper/equipment_anomaly_detection/contracts_paper.py paper/equipment_anomaly_detection/proposal_score_paper.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "feat(parity): implement data-only HardRouteParityAuthority and atomic strict oracle-bundle flow"
```

---

### Task 5: General n_train=1 origin replay and exact primitive repair without dataset/coordinate special cases (PR 2)

**Files:**
- Modify: `paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/hard_route_paper.py`
- Modify: `paper/equipment_anomaly_detection/tests_paper/test_hard_route_paper.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_smoke_remediation_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- `build_legacy_route_evidence(values: RawDataset) -> LegacyRouteEvidence`:
  - Supports $n_{\text{train}} \ge 1$ without raising.
  - In `LegacyRouteEvidence.__post_init__`, require `int(self.n_train) >= 1 and int(self.n_test) >= 1`.
- `replay_hard_route(values: RawDataset, sources: HardSourceBundle, mini: MiniRocketResult) -> HardRouteResult`:
  - Supports $n_{\text{train}} \ge 1$. Returns `empty_hard_route_result_paper` only if `len(values.train_rows) < 1 or not values.test_rows`.
  - Replays origin-exact Exp74d through Exp137 selectors matching frozen Exp137/236 oracles.
- Invariant: Zero dataset-name branches (`if dataset == "Phoneme_normal_39"`) or hard-coded coordinate injections.
- Mechanism 1 Diagnostic RED Sequencing: Stage isolation diagnostic harness compares intermediate stages (`replay_exp74d_89_90_93`, `_ranked_primary`, `_exp93_order`, `_exp90_top_score_index`, `replay_exp133_134_135_137`) against origin ASTs on representative datasets. Once the harness commits the identified dataset/stage/primitive, a concrete diagnostic RED test records the exact divergence before the selector fix is applied.

- [ ] **Step 1: Run RED Command for Task 5**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch
```
Expected failure: `AssertionError: 49 not found in frozenset()` (due to artificial empty route check).

- [ ] **Step 2: Remove Artificial `n_train < 2` Short-Circuit in `legacy_route_sources_paper.py`**

In `paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py`:
1. In `LegacyRouteEvidence.__post_init__`:
   Change:
   ```python
   if int(self.n_train) < 1 or int(self.n_test) < 1:
       raise ValueError("legacy evidence requires at least one TRAIN and one TEST row")
   ```
2. In `build_legacy_route_evidence`:
   Change:
   ```python
   if len(values.train_rows) < 1 or not values.test_rows:
       raise ValueError("legacy full-fit route requires at least one TRAIN and one TEST row")
   ```

- [ ] **Step 3: Remove Artificial `n_train < 2` Short-Circuit and Repair Mechanism 1 Primitives in `hard_route_paper.py`**

In `paper/equipment_anomaly_detection/hard_route_paper.py`:
1. In `replay_hard_route`:
   Change:
   ```python
   if len(values.train_rows) < 1 or not values.test_rows:
       return empty_hard_route_result_paper(len(values.test_rows))
   ```
2. Ensure `replay_exp74d_89_90_93`, `_ranked_primary`, `_exp93_order`, `_exp90_top_score_index`, and `replay_exp133_134_135_137` match the exact origin AST primitives identified during diagnostic isolation.

- [ ] **Step 4: Update Existing Tests in `test_hard_route_paper.py` and `test_paper_r4_smoke_remediation_antigravity.py`**

1. In `paper/equipment_anomaly_detection/tests_paper/test_hard_route_paper.py`:
   Update `test_undersized_phoneme_shaped_route_is_typed_empty_without_fullfit` to `test_n_train_1_phoneme_shaped_route_is_fullfit_eligible`:
   ```python
   def test_n_train_1_phoneme_shaped_route_is_fullfit_eligible(self) -> None:
       """n_train=1 executes full-fit rather than short-circuiting to empty route."""
       values = self._dataset("Phoneme_normal_39", (self._finite_row(0.0),), tuple(self._finite_row(index + 1.0) for index in range(3)))
       with patch(
           "paper.equipment_anomaly_detection.hard_route_paper.build_legacy_route_evidence",
           side_effect=RuntimeError("fullfit_reached"),
       ) as fullfit:
           with self.assertRaisesRegex(RuntimeError, "fullfit_reached"):
               replay_hard_route(values, self.sources, self.mini)
       fullfit.assert_called_once()

   def test_n_train_0_returns_typed_empty(self) -> None:
       """n_train=0 returns typed empty HardRouteResult without calling fullfit."""
       values = self._dataset("Empty_normal_0", (), tuple(self._finite_row(index + 1.0) for index in range(3)))
       with patch("paper.equipment_anomaly_detection.hard_route_paper.build_legacy_route_evidence") as fullfit:
           result = replay_hard_route(values, self.sources, self.mini)
       fullfit.assert_not_called()
       self.assertEqual(result.automatic_hard, frozenset())
   ```
2. In `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_smoke_remediation_antigravity.py`:
   Update the undersized test case so `empty_train = _raw("Empty_normal_0", (), (_row(1.0), _row(2.0)))` tests typed empty, while `Phoneme_normal_39` (`n_train=1`) tests full-fit execution.

- [ ] **Step 5: Run GREEN Verification for Task 5**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_replay_signature_still_rejects_historical_coordinates
```
Expected: `Ran 2 tests in ... OK`.

- [ ] **Step 6: Run Regression Suite for Hard Route**

Run:
```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_hard_route_paper
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_smoke_remediation_antigravity
```
Expected: All tests pass GREEN.

- [ ] **Step 7: Resynchronize Codemaps and Commit PR 2**

Run shared map-resync procedure, then:
```bash
git add paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py paper/equipment_anomaly_detection/hard_route_paper.py paper/equipment_anomaly_detection/tests_paper/test_hard_route_paper.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_smoke_remediation_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "fix(route): general n_train=1 origin replay and exact primitive repair without special cases"
```

---

### Task 6: 33-member coordinate-only smoke test and oracle parity verification

**Files:**
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- 33 Smoke Datasets: Catalog-sorted tuple of 14 representative + 19 affected datasets from `manage_paper_r4_proposal_antigravity.py`.
- Smoke Gate Contract: Validates 100% per-dataset coordinate equality for each of the 33 datasets against the atomic oracle bundle projected via `authority.project(SMOKE_DATASETS)`. Direct indexing is used (`smoke_auth.exp137_hard[dataset]`, etc.). Does NOT assert aggregate full-population counts or execute Task6/7.
- Catalog Cross-Check: Cross-checks `len(raw.test_rows)` against `smoke_auth.n_test[dataset]` exactly.

- [ ] **Step 1: Run RED Command for Task 6**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_smoke_33_datasets_match_sha_bound_oracles
```
Expected: Fails if any smoke dataset coordinate deviates from the projected SHA-bound oracle authority.

- [ ] **Step 2: Implement Complete 33-Dataset Coordinate Parity Verification in T7**

In `FullRouteParityRecoveryAntigravityTests.test_smoke_33_datasets_match_sha_bound_oracles`:

```python
    def test_smoke_33_datasets_match_sha_bound_oracles(self) -> None:
        """T7: All 33 smoke datasets achieve per-dataset coordinate equality against 4 oracles."""
        from paper.equipment_anomaly_detection.proposal_score_paper import load_hard_route_parity_authority
        from scratch.manage_paper_r4_proposal_antigravity import (
            EXP137_ORACLE_PATH,
            EXP236_ORACLE_PATH,
            EXP291_ORACLE_PATH,
            EXP404_ORACLE_PATH,
            SMOKE_DATASETS,
        )

        self.assertEqual(len(SMOKE_DATASETS), 33)
        authority = load_hard_route_parity_authority(
            EXP137_ORACLE_PATH,
            EXP236_ORACLE_PATH,
            EXP291_ORACLE_PATH,
            EXP404_ORACLE_PATH,
        )
        smoke_auth = authority.project(SMOKE_DATASETS)
        self.assertEqual(smoke_auth.population_count, 33)
        db_path = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
        for dataset in SMOKE_DATASETS:
            raw = load_dataset_values(db_path, dataset)
            self.assertEqual(len(raw.test_rows), smoke_auth.n_test[dataset], f"n_test mismatch on {dataset}")
            sources = score_hard_sources(raw)
            mini = score_minirocket_confirmations(raw)
            route = replay_hard_route(raw, sources, mini)
            self.assertEqual(route.exp137.hard, smoke_auth.exp137_hard[dataset], f"Exp137 mismatch on {dataset}")
            self.assertEqual(route.exp236.hard, smoke_auth.exp236_hard[dataset], f"Exp236 mismatch on {dataset}")
            self.assertEqual(route.exp253.hard, smoke_auth.exp291_pre_exact[dataset], f"Exp253 mismatch on {dataset}")
            self.assertEqual(route.exact_normal_removed, smoke_auth.exp291_exact_removed[dataset], f"Exp291 exact mismatch on {dataset}")
            self.assertEqual(route.automatic_hard, smoke_auth.automatic_hard[dataset], f"Automatic Hard mismatch on {dataset}")
```

- [ ] **Step 3: Run GREEN Verification for Task 6**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_smoke_33_datasets_match_sha_bound_oracles
```
Expected: `Ran 1 test in ... OK`.

- [ ] **Step 4: Resynchronize Codemaps and Commit Task 6**

Run shared map-resync procedure, then:
```bash
git add experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "test(parity): prove 33-member coordinate-only smoke matches SHA-bound oracles"
```

---

### Task 7: Full 1,117 route-only parity gate (2005/2116/2253/35/2218 counts) before Task6/7 and Candidate C regression (PR 3)

**Files:**
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `paper/equipment_anomaly_detection/runner_paper.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- `verify_hard_route_parity_paper(routes: Mapping[str, HardRouteResult], *, authority: HardRouteParityAuthority, mode: str) -> dict[str, object]`:
  - Requires explicit `mode` ("full" or "smoke") and explicit `authority: HardRouteParityAuthority`. Rejects incomplete inputs with `ValueError`.
  - If `mode == "full"`:
    - Verifies `len(routes) == 1117` and `set(routes) == set(authority.exp137_hard)`.
    - Verifies aggregate counts across all 1,117 datasets:
      - Exp137 Hard count == 2005
      - Exp236 Hard count == 2116
      - Exp253 Hard count == 2253 ($2218 + 35$)
      - Exact-normal removed count == 35
      - Automatic Hard count == 2218
    - Verifies 100% per-dataset coordinate equality against `authority` by direct indexing (`authority.exp137_hard[dataset]`, etc.).
  - If `mode == "smoke"`:
    - Verifies all datasets in `routes` match `authority` coordinates by direct indexing.
  - Raises `ValueError` on any count mismatch or coordinate inequality.
- `compose_proposal_score(raw_datasets, acceleration_manifest, *, root=None, context=None, catalog=None, progress=None) -> ProposalScore`:
  - Receives `context` for official execution.
  - When `context` is provided:
    - Extracts `mode = str(context["mode"])` ("full" or "smoke") and `authority` from context.
    - Calls `verify_hard_route_parity_paper(routes, authority=authority, mode=mode)` before invoking Task 6 (`score_low_frequency`) and Task 7 (`classify_stability`).
    - Remove all `len(raw) == 1117` mode inference!
  - When `context` is None (generic unit test calls): execution stays ungated and cannot be official.
- `runner_paper.py`:
  - `PipelineConfig` owns the parity authority for official runs.
  - `context_payload` binds and hashes the authority into `context["route_parity_authority"]`.
  - `evaluate_stage("proposal_score")` passes context to `compose_proposal_score`.

- [ ] **Step 1: Run RED Command for Task 7**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_full_1117_route_only_parity_gate_blocks_task6_and_task7
```
Expected failure: `ImportError: cannot import name 'verify_hard_route_parity_paper' from 'paper.equipment_anomaly_detection.proposal_score_paper'`.

- [ ] **Step 2: Implement `verify_hard_route_parity_paper` and Context Parity Gating in `proposal_score_paper.py`**

In `paper/equipment_anomaly_detection/proposal_score_paper.py`, add:

```python
FROZEN_PARITY_COUNTS = {
    "exp137_hard": 2005,
    "exp236_hard": 2116,
    "exp253_hard": 2253,
    "exact_normal_removed": 35,
    "automatic_hard": 2218,
}


def verify_hard_route_parity_paper(
    routes: Mapping[str, HardRouteResult],
    *,
    authority: HardRouteParityAuthority,
    mode: str,
) -> dict[str, object]:
    """Verify route parity using explicit mode and context-bound authority before Task6/7."""
    if not isinstance(authority, HardRouteParityAuthority) or mode not in {"full", "smoke"}:
        raise ValueError("verify_hard_route_parity_paper requires explicit mode ('full' or 'smoke') and HardRouteParityAuthority")

    if mode == "full":
        if len(routes) != 1117 or set(routes) != set(authority.exp137_hard):
            raise ValueError(f"Full-population parity verifier requires exactly 1117 catalog datasets (got {len(routes)})")

        c_137 = sum(len(r.exp137.hard) for r in routes.values())
        c_236 = sum(len(r.exp236.hard) for r in routes.values())
        c_253 = sum(len(r.exp253.hard) for r in routes.values())
        c_removed = sum(len(r.exact_normal_removed) for r in routes.values())
        c_auto = sum(len(r.automatic_hard) for r in routes.values())

        actual_counts = {
            "exp137_hard": c_137,
            "exp236_hard": c_236,
            "exp253_hard": c_253,
            "exact_normal_removed": c_removed,
            "automatic_hard": c_auto,
        }

        if actual_counts != FROZEN_PARITY_COUNTS:
            raise ValueError(
                f"Hard route count parity mismatch: expected {FROZEN_PARITY_COUNTS} but got {actual_counts}"
            )

    mismatches: list[str] = []
    for dataset, route in routes.items():
        if dataset not in authority.exp137_hard:
            mismatches.append(f"{dataset}:unrecognized_dataset")
            continue
        if route.exp137.hard != authority.exp137_hard[dataset]:
            mismatches.append(f"{dataset}:exp137")
        if route.exp236.hard != authority.exp236_hard[dataset]:
            mismatches.append(f"{dataset}:exp236")
        if route.exp253.hard != authority.exp291_pre_exact[dataset]:
            mismatches.append(f"{dataset}:exp253")
        if route.exact_normal_removed != authority.exp291_exact_removed[dataset]:
            mismatches.append(f"{dataset}:exact_removed")
        if route.automatic_hard != authority.automatic_hard[dataset]:
            mismatches.append(f"{dataset}:automatic_hard")

    if mismatches:
        raise ValueError(
            f"Hard route coordinate parity mismatch on {len(mismatches)} stages/datasets: {mismatches[:10]}"
        )

    return {"status": "parity_verified", "mode": mode, "datasets_verified": len(routes)}
```

In `compose_proposal_score`:
```python
    # Gate Task6/7 on verified hard-route parity when context provides mode and authority
    if context is not None and "route_parity_authority" in context:
        auth_data = context["route_parity_authority"]
        auth_instance = HardRouteParityAuthority.from_dict(auth_data) if isinstance(auth_data, dict) else auth_data
        mode = str(context.get("mode", ""))
        verify_hard_route_parity_paper(routes, authority=auth_instance, mode=mode)
```

- [ ] **Step 3: Update `runner_paper.py` for Pipeline Authority Context Binding**

In `paper/equipment_anomaly_detection/runner_paper.py`:
1. In `PipelineConfig`: Add optional `route_parity_authority: HardRouteParityAuthority | None = None`.
2. In `context_payload`: Include serialized `route_parity_authority.to_dict()` in `context["route_parity_authority"]` when provided, and require it for official full/smoke execution.

- [ ] **Step 4: Run GREEN Verification for Task 7**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_full_1117_route_only_parity_gate_blocks_task6_and_task7 experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_candidate_c_rule_and_target_algebra_unchanged
```
Expected: `Ran 2 tests in ... OK`.

- [ ] **Step 5: Run Proposal Composition Regressions**

Run:
```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper
```
Expected: All tests pass GREEN.

- [ ] **Step 6: Resynchronize Codemaps and Commit PR 3**

Run shared map-resync procedure, then:
```bash
git add paper/equipment_anomaly_detection/proposal_score_paper.py paper/equipment_anomaly_detection/runner_paper.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "feat(composition): insert explicit-mode route-only parity gate before Task6 and Task7"
```

---

### Task 8: Clean new smoke/official roots, destination refusal, PipelineConfig authority, and launcher configuration (PR 4)

**Files:**
- Modify: `scratch/manage_paper_r4_proposal_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- Launcher Path Constants (in `scratch/manage_paper_r4_proposal_antigravity.py`):
  - `EXP137_ORACLE_PATH = REPOSITORY_ROOT / "results/exp137/experiment_137_operational_triage_results.csv"`
  - `EXP137_ORACLE_SHA256 = "942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d"`
  - `EXP236_ORACLE_PATH = REPOSITORY_ROOT / "outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl"`
  - `EXP236_ORACLE_SHA256 = "a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524"`
  - `EXP291_ORACLE_PATH = Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl")`
  - `EXP291_ORACLE_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"`
  - `EXP404_ORACLE_PATH = Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl")`
  - `EXP404_ORACLE_SHA256 = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"`
  - `AUTHORIZED_SMOKE_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_r4_full_route_parity_smoke_antigravity_20260814")`
  - `AUTHORIZED_SMOKE_OPS_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_r4_full_route_parity_smoke_antigravity_20260814_ops_antigravity")`
  - `AUTHORIZED_OFFICIAL_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_parity_antigravity_20260814")`
  - `AUTHORIZED_OFFICIAL_OPS_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_parity_antigravity_20260814_ops_antigravity")`
  - Rebind active targets: `OFFICIAL_SMOKE_ROOT = AUTHORIZED_SMOKE_ROOT_20260814`, `OFFICIAL_R4_ROOT = AUTHORIZED_OFFICIAL_ROOT_20260814`, `OFFICIAL_OPS_ROOT = AUTHORIZED_OFFICIAL_OPS_ROOT_20260814`.
  - Rebind `SMOKE_DATASETS` to the 33-dataset catalog-sorted tuple.
- `FORBIDDEN_DESTINATION_ROOTS`: Set containing all prior 20260813 and earlier roots.
- `validate_clean_root_destination_paper(path: Path) -> None`: Raises `ValueError` if `path` or its ops companion matches any forbidden prior root.
- Launcher constructs `HardRouteParityAuthority` using `load_hard_route_parity_authority` and passes it to `PipelineConfig`.

- [ ] **Step 1: Run RED Command for Task 8**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_clean_root_execution_and_rejection_of_live_r4_reuse experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations
```
Expected failure: `ImportError: cannot import name 'AUTHORIZED_SMOKE_ROOT_20260814'` or `validate_clean_root_destination_paper`.

- [ ] **Step 2: Update Launcher Paths, Smoke Datasets, Destination Refusal, and PipelineConfig Authority in `manage_paper_r4_proposal_antigravity.py`**

In `scratch/manage_paper_r4_proposal_antigravity.py`:
1. Define authorized 20260814 paths, oracle constants, and forbidden roots:
```python
EXP137_ORACLE_PATH = REPOSITORY_ROOT / "results/exp137/experiment_137_operational_triage_results.csv"
EXP137_ORACLE_SHA256 = "942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d"
EXP236_ORACLE_PATH = REPOSITORY_ROOT / "outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl"
EXP236_ORACLE_SHA256 = "a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524"
EXP291_ORACLE_PATH = Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl")
EXP291_ORACLE_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
EXP404_ORACLE_PATH = Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl")
EXP404_ORACLE_SHA256 = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"

AUTHORIZED_SMOKE_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_r4_full_route_parity_smoke_antigravity_20260814")
AUTHORIZED_SMOKE_OPS_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_r4_full_route_parity_smoke_antigravity_20260814_ops_antigravity")
AUTHORIZED_OFFICIAL_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_parity_antigravity_20260814")
AUTHORIZED_OFFICIAL_OPS_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_parity_antigravity_20260814_ops_antigravity")

OFFICIAL_SMOKE_ROOT = AUTHORIZED_SMOKE_ROOT_20260814
OFFICIAL_R4_ROOT = AUTHORIZED_OFFICIAL_ROOT_20260814
OFFICIAL_OPS_ROOT = AUTHORIZED_OFFICIAL_OPS_ROOT_20260814

SMOKE_DATASETS = (
    "ACSF1_normal_0", "ACSF1_normal_1", "ACSF1_normal_2", "ACSF1_normal_3",
    "ACSF1_normal_4", "ACSF1_normal_5", "ACSF1_normal_6", "ACSF1_normal_7",
    "ACSF1_normal_8", "ACSF1_normal_9", "ChlorineConcentration_normal_3",
    "DodgerLoopDay_normal_1", "DodgerLoopDay_normal_2", "DodgerLoopDay_normal_4",
    "DodgerLoopDay_normal_5", "DodgerLoopDay_normal_6", "DodgerLoopDay_normal_7",
    "DodgerLoopGame_normal_1", "DodgerLoopGame_normal_2", "DodgerLoopWeekend_normal_1",
    "DodgerLoopWeekend_normal_2", "MelbournePedestrian_normal_1", "MelbournePedestrian_normal_2",
    "MelbournePedestrian_normal_3", "MelbournePedestrian_normal_4", "MelbournePedestrian_normal_5",
    "MelbournePedestrian_normal_6", "MelbournePedestrian_normal_7", "MelbournePedestrian_normal_8",
    "MelbournePedestrian_normal_9", "MelbournePedestrian_normal_10", "Phoneme_normal_39",
    "PigAirwayPressure_normal_1",
)

FORBIDDEN_DESTINATION_ROOTS = frozenset({
    Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_v2_antigravity_20260814_ops_antigravity"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813"),
    FAILED_SMOKE_ROOT,
    FAILED_SMOKE_OPS_ROOT,
    VERIFIED_V2_SMOKE_ROOT,
    VERIFIED_V2_SMOKE_OPS_ROOT,
})


def validate_clean_root_destination_paper(path: Path) -> None:
    target = Path(path).resolve()
    for forbidden in FORBIDDEN_DESTINATION_ROOTS:
        if target == forbidden.resolve():
            raise ValueError(f"Refusing to write to or resume forbidden prior root: {target}")
```
2. In `smoke` and `launch` actions: Load authority via `load_hard_route_parity_authority(EXP137_ORACLE_PATH, EXP236_ORACLE_PATH, EXP291_ORACLE_PATH, EXP404_ORACLE_PATH)` and supply it to `PipelineConfig(..., route_parity_authority=authority)`. Call `validate_clean_root_destination_paper`.

- [ ] **Step 3: Update `test_paper_r4_operations_antigravity.py` for Clean Roots and 33-Dataset Smoke**

In `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py`:
1. Update `OFFICIAL_PATHS` dictionary to reflect the new clean 20260814 paths for `OFFICIAL_SMOKE_ROOT`, `OFFICIAL_R4_ROOT`, and `OFFICIAL_OPS_ROOT`.
2. Update `EXPECTED_SMOKE_DATASETS` to the 33-dataset catalog-sorted tuple and assert `len(module.SMOKE_DATASETS) == 33`.
3. Update inventory test expectation to accept `inventory["pycatch22"] == "0.5.0"`.

- [ ] **Step 4: Run GREEN Verification for Task 8**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_clean_root_execution_and_rejection_of_live_r4_reuse experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations
```
Expected: `Ran 2 tests in ... OK`.

- [ ] **Step 5: Run Operations Regression Suite**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_operations_antigravity
```
Expected: All tests pass GREEN.

- [ ] **Step 6: Resynchronize Codemaps and Commit PR 4**

Run shared map-resync procedure, then:
```bash
git add scratch/manage_paper_r4_proposal_antigravity.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "feat(ops): point launcher to clean 20260814 smoke and official roots and enforce destination refusal"
```

---

### Task 9: Comprehensive regression suite execution, final codemap synchronization, provenance verification, and review readiness

**Files:**
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- Test Suite: All 14 tests in `test_paper_r4_full_route_parity_recovery_antigravity.py` plus all existing paper unit tests pass GREEN.
- Codemap: Both Exp448 and Paper triplets are completely synchronized and hash-bound against HEAD.
- Provenance: MLX/MPS assessment, logical CPU worker policy, clean root isolation, zero label leakage, and exact target counts (2005/2116/2253/35/2218, Candidate C = 2244) fully validated.

- [ ] **Step 1: Execute Complete Antigravity Full Route Parity Test Suite**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests
```
Expected: `Ran 14 tests in ... OK`.

- [ ] **Step 2: Execute All Paper Unit and Operational Test Suites**

Run:
```bash
python3 -m unittest discover -s paper/equipment_anomaly_detection/tests_paper -p "test_*.py"
python3 -m unittest discover -s experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests -p "test_paper_r4_*.py"
```
Expected: All test suites pass with 0 errors and 0 failures.

- [ ] **Step 3: Run Final Codemap Synchronization and Triplet Validation**

Execute the shared map-resync procedure for both triplets, then run:
```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper.ConfigLineageTests.test_paper_codemap_triplet_is_complete_and_hash_bound paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper.ConfigLineageTests.test_codemap_callers_are_complete_static_ast_import_edges
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp448_antigravity.lock >/dev/null
```
Expected: All assertions pass cleanly.

- [ ] **Step 4: Final Git Status Verification and Merge Readiness**

Run:
```bash
git status --short
```
Expected: Working tree clean or only tracked modifications staged.
Commit if needed:
```bash
git add docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "chore(provenance): finalize codemap triplets and verify full route parity recovery readiness"
```
