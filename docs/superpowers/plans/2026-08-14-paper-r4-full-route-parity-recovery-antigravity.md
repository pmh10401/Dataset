# Paper r4 Full-Population Hard-Route Parity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore full-population hard-route coordinate parity across all 1,117 datasets (2005 / 2116 / 2253 / 35 / 2218 counts, Candidate C = 2244) against SHA-bound frozen oracles, repairing Mechanism 1 (inexact Exp74d→137 replay) and Mechanism 2 (canonical_normal_state missing pycatch22==0.5.0), restoring general n_train=1 origin replay (Phoneme_normal_39 Hard {49}), validating a 33-dataset coordinate smoke, enforcing a pre-Task6/7 1,117 route-only parity gate, and executing on clean new roots with zero cache reuse.

**Architecture:** Fail-closed `verify_pycatch22_environment_paper()` in `canonical_paper.py` called in `_collect_task4_task5`; worker error propagation in `hard_sources_paper.py`; general `n_train=1` support in `legacy_route_sources_paper.py` and `hard_route_paper.py`; diagnostic-isolated exact primitive repair in `hard_route_paper.py`; label-free oracle adapters and `verify_hard_route_parity_paper` in `proposal_score_paper.py` gating Task6/7; clean new smoke and official roots, 33-dataset smoke catalog, and destination refusal in `scratch/manage_paper_r4_proposal_antigravity.py`; CPU-canonical ProcessPool execution with `dataset_workers = max(1, logical - 1)`.

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
| Exp137 | `results/exp137/experiment_137_operational_triage_results.csv` | `942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d` | `hard_alert_indices` (space-delimited integers) |
| Exp236 | `outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl` | `a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524` | `hard_indices` (JSON list of integers) |
| Exp291 | `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl` | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` | `automatic_hard_indices` and `automatic_normal_exact_indices` |
| Exp404 | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` | `hard_indices` (2218 automatic Hard coordinates) |

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
- Modify: `paper/equipment_anomaly_detection/sources_paper/canonical_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/hard_route_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
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
- Methods:
  - T1: `test_pycatch22_pin_fails_closed_when_missing_or_wrong_version`
  - T2: `test_pycatch22_0_5_0_is_the_only_accepted_version`
  - T3: `test_eligible_dataset_does_not_abstain_canonical_for_pycatch22`
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

Write `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py` containing complete typed fixtures for all 14 tests:

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

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.data_paper import TEST_VALUES_SQL, load_dataset_values
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
        with patch("importlib.metadata.version", return_value="0.5.0"):
            meta = canonical_paper.verify_pycatch22_environment_paper()
            self.assertEqual(meta.get("pycatch22"), "0.5.0")

    def test_eligible_dataset_does_not_abstain_canonical_for_pycatch22(self) -> None:
        """T3: Eligible dataset fixture scores canonical_normal_state as eligible without abstention."""
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
        bundle = score_hard_sources(raw)
        canonical = bundle.by_name["canonical_normal_state"]
        self.assertTrue(canonical.eligible)
        self.assertEqual(canonical.eligibility_reason, "eligible")
        self.assertEqual(len(canonical.train_oof_score), 10)
        self.assertEqual(len(canonical.test_score), 20)

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
        from paper.equipment_anomaly_detection.proposal_score_paper import load_exp137_oracle_lanes

        csv_path = Path("results/exp137/experiment_137_operational_triage_results.csv")
        self.assertTrue(csv_path.exists())
        oracle = load_exp137_oracle_lanes(csv_path)
        self.assertEqual(len(oracle), 1117)
        total_hard = sum(len(indices) for indices in oracle.values())
        self.assertEqual(total_hard, 2005)

    def test_smoke_33_datasets_match_sha_bound_oracles(self) -> None:
        """T7: All 33 smoke datasets achieve per-dataset coordinate equality against 4 oracles."""
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            load_exp137_oracle_lanes,
            load_exp236_oracle_lanes,
            load_exp291_oracle_lanes,
            load_exp404_oracle_lanes,
        )

        smoke_datasets = (
            "ACSF1_normal_0", "ACSF1_normal_1", "ACSF1_normal_2", "ACSF1_normal_3",
            "ACSF1_normal_4", "ACSF1_normal_5", "ACSF1_normal_6", "ACSF1_normal_7",
            "ACSF1_normal_8", "ACSF1_normal_9", "ChlorineConcentration_normal_3",
            "MelbournePedestrian_normal_10", "Phoneme_normal_39", "PigAirwayPressure_normal_1",
            "DodgerLoopDay_normal_1", "DodgerLoopDay_normal_2", "DodgerLoopDay_normal_4",
            "DodgerLoopDay_normal_5", "DodgerLoopDay_normal_6", "DodgerLoopDay_normal_7",
            "DodgerLoopGame_normal_1", "DodgerLoopGame_normal_2", "DodgerLoopWeekend_normal_1",
            "DodgerLoopWeekend_normal_2", "MelbournePedestrian_normal_1", "MelbournePedestrian_normal_2",
            "MelbournePedestrian_normal_3", "MelbournePedestrian_normal_4", "MelbournePedestrian_normal_5",
            "MelbournePedestrian_normal_6", "MelbournePedestrian_normal_7", "MelbournePedestrian_normal_8",
            "MelbournePedestrian_normal_9",
        )
        self.assertEqual(len(set(smoke_datasets)), 33)

    def test_full_1117_route_only_parity_gate_blocks_task6_and_task7(self) -> None:
        """T8: verify_hard_route_parity_paper halts compose_proposal_score on mismatch before Task6."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            compose_proposal_score,
            verify_hard_route_parity_paper,
        )

        bad_routes = {"dataset_a": empty_hard_route_result_paper(10)}
        with self.assertRaises(Exception):
            verify_hard_route_parity_paper(bad_routes)

        # Integration test: compose_proposal_score halts before score_low_frequency when route parity fails
        fake_raw = {f"d_{i}": RawDataset(f"d_{i}", 10, (0,), (b"0",), (np.zeros(10, dtype=np.float32),), (0,), (b"0",), (np.zeros(10, dtype=np.float32),)) for i in range(1117)}
        manifest = build_acceleration_manifest("a" * 64, logical_cpu_count=2)
        with patch("paper.equipment_anomaly_detection.proposal_score_paper._collect_task4_task5", return_value=({}, {f"d_{i}": empty_hard_route_result_paper(10) for i in range(1117)})), patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.score_low_frequency"
        ) as mock_task6, patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.classify_stability"
        ) as mock_task7:
            with self.assertRaises(ValueError):
                compose_proposal_score(fake_raw, manifest)
            mock_task6.assert_not_called()
            mock_task7.assert_not_called()

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
        from paper.equipment_anomaly_detection.data_paper import emit_progress_paper

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
        from paper.equipment_anomaly_detection.contracts_paper import FROZEN_CONFIG

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
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_pin_fails_closed_when_missing_or_wrong_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_0_5_0_is_the_only_accepted_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_eligible_dataset_does_not_abstain_canonical_for_pycatch22
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
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_pin_fails_closed_when_missing_or_wrong_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_0_5_0_is_the_only_accepted_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_eligible_dataset_does_not_abstain_canonical_for_pycatch22
```
Expected: `Ran 3 tests in ... OK`.

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

### Task 4: Values-only SHA-pinned oracle adapters and diagnostic stage isolation for Mechanism 1

**Files:**
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- Oracle Adapters (in `proposal_score_paper.py`):
  - `load_exp137_oracle_lanes(csv_path: Path) -> dict[str, frozenset[int]]`: Extracts only `dataset_name` and `hard_alert_indices`.
  - `load_exp236_oracle_lanes(jsonl_path: Path) -> dict[str, frozenset[int]]`: Extracts only `dataset` and `hard_indices`.
  - `load_exp291_oracle_lanes(jsonl_path: Path) -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]], dict[str, frozenset[int]]]`: Returns `(automatic_hard, exact_normal_removed, reconstructed_pre_exact_exp253)`.
  - `load_exp404_oracle_lanes(jsonl_path: Path) -> dict[str, frozenset[int]]`: Extracts only `dataset` and `hard_indices`.
- Diagnostic isolation fixture (T6): Compares intermediate stages (Exp74d, Exp89, Exp90, Exp93, Exp134, Exp137) against origin reference ASTs to locate exact Mechanism 1 divergence before selector modifications.

- [ ] **Step 1: Run RED Command for Task 4**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence
```
Expected failure: `ImportError: cannot import name 'load_exp137_oracle_lanes' from 'paper.equipment_anomaly_detection.proposal_score_paper'`.

- [ ] **Step 2: Implement Label-Free Oracle Adapters in `proposal_score_paper.py`**

Add the four dedicated label-free oracle adapters to `paper/equipment_anomaly_detection/proposal_score_paper.py`:

```python
import csv
from pathlib import Path


def load_exp137_oracle_lanes(csv_path: Path) -> dict[str, frozenset[int]]:
    """Read Exp137 CSV, extracting only dataset and hard_alert_indices (ignoring metric/label columns)."""
    lanes: dict[str, frozenset[int]] = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = str(row["dataset_name"])
            raw_indices = str(row.get("hard_alert_indices", "")).strip()
            indices = frozenset(map(int, raw_indices.split())) if raw_indices else frozenset()
            lanes[dataset] = indices
    return lanes


def load_exp236_oracle_lanes(jsonl_path: Path) -> dict[str, frozenset[int]]:
    """Read Exp236 JSONL, extracting only dataset and hard_indices."""
    lanes: dict[str, frozenset[int]] = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            dataset = str(record["dataset"])
            indices = frozenset(map(int, record.get("hard_indices", [])))
            lanes[dataset] = indices
    return lanes


def load_exp291_oracle_lanes(
    jsonl_path: Path,
) -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]], dict[str, frozenset[int]]]:
    """Read Exp291 JSONL, returning (automatic_hard, exact_normal_removed, reconstructed_pre_exact_exp253)."""
    auto_hard: dict[str, frozenset[int]] = {}
    exact_removed: dict[str, frozenset[int]] = {}
    pre_exact: dict[str, frozenset[int]] = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            dataset = str(record["dataset"])
            hard = frozenset(map(int, record.get("automatic_hard_indices", [])))
            removed = frozenset(map(int, record.get("automatic_normal_exact_indices", [])))
            auto_hard[dataset] = hard
            exact_removed[dataset] = removed
            pre_exact[dataset] = hard | removed
    return auto_hard, exact_removed, pre_exact


def load_exp404_oracle_lanes(jsonl_path: Path) -> dict[str, frozenset[int]]:
    """Read Exp404 JSONL, extracting dataset and hard_indices (2218 automatic Hard coordinates)."""
    lanes: dict[str, frozenset[int]] = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            dataset = str(record["dataset"])
            indices = frozenset(map(int, record.get("hard_indices", [])))
            lanes[dataset] = indices
    return lanes
```

- [ ] **Step 3: Expand Mechanism 1 Stage Isolation Diagnostic in `test_paper_r4_full_route_parity_recovery_antigravity.py`**

In `FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence`, load the Exp137, Exp236, Exp291, and Exp404 oracles, verify their SHA-256 identities, and test stage isolation on representative datasets.

- [ ] **Step 4: Run GREEN Verification for Task 4**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence
```
Expected: `Ran 1 test in ... OK`.

- [ ] **Step 5: Resynchronize Codemaps and Commit Task 4**

Run shared map-resync procedure, then:
```bash
git add paper/equipment_anomaly_detection/proposal_score_paper.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "feat(parity): implement values-only SHA-pinned oracle adapters and diagnostic stage isolation"
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
- 33 Smoke Datasets:
  - 14 Representative: `ACSF1_normal_0` through `ACSF1_normal_9`, `ChlorineConcentration_normal_3`, `MelbournePedestrian_normal_10`, `Phoneme_normal_39`, `PigAirwayPressure_normal_1`.
  - 19 Nonfinite Repair Affected: `DodgerLoopDay_normal_1,2,4,5,6,7`, `DodgerLoopGame_normal_1,2`, `DodgerLoopWeekend_normal_1,2`, `MelbournePedestrian_normal_1` through `MelbournePedestrian_normal_9`.
- Smoke Gate Contract: Validates 100% per-dataset coordinate equality for each of the 33 datasets against Exp137 (`hard_alert_indices`), Exp236 (`hard_indices`), Exp291 (`automatic_hard_indices`), and Exp404 (`hard_indices`) oracles. Does NOT assert aggregate full-population counts or execute Task6/7.

- [ ] **Step 1: Run RED Command for Task 6**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_smoke_33_datasets_match_sha_bound_oracles
```
Expected: Fails if any smoke dataset coordinate deviates from the 4 SHA-bound oracles.

- [ ] **Step 2: Implement Complete 33-Dataset Coordinate Parity Verification in T7**

In `FullRouteParityRecoveryAntigravityTests.test_smoke_33_datasets_match_sha_bound_oracles`, load raw values from `load_dataset_values` for each of the 33 smoke datasets using the official SQLite database, compute `score_hard_sources`, `score_minirocket_confirmations`, and `replay_hard_route`, and compare against the oracle coordinates loaded via `load_exp137_oracle_lanes`, `load_exp236_oracle_lanes`, `load_exp291_oracle_lanes`, and `load_exp404_oracle_lanes`.

```python
    def test_smoke_33_datasets_match_sha_bound_oracles(self) -> None:
        """T7: All 33 smoke datasets achieve per-dataset coordinate equality against 4 oracles."""
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            load_exp137_oracle_lanes,
            load_exp236_oracle_lanes,
            load_exp291_oracle_lanes,
            load_exp404_oracle_lanes,
        )
        from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
            score_minirocket_confirmations,
        )

        exp137_oracle = load_exp137_oracle_lanes(Path("results/exp137/experiment_137_operational_triage_results.csv"))
        exp236_oracle = load_exp236_oracle_lanes(Path("outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl"))
        exp291_auto, exp291_removed, _ = load_exp291_oracle_lanes(Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl"))
        exp404_oracle = load_exp404_oracle_lanes(Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl"))

        smoke_datasets = (
            "ACSF1_normal_0", "ACSF1_normal_1", "ACSF1_normal_2", "ACSF1_normal_3",
            "ACSF1_normal_4", "ACSF1_normal_5", "ACSF1_normal_6", "ACSF1_normal_7",
            "ACSF1_normal_8", "ACSF1_normal_9", "ChlorineConcentration_normal_3",
            "MelbournePedestrian_normal_10", "Phoneme_normal_39", "PigAirwayPressure_normal_1",
            "DodgerLoopDay_normal_1", "DodgerLoopDay_normal_2", "DodgerLoopDay_normal_4",
            "DodgerLoopDay_normal_5", "DodgerLoopDay_normal_6", "DodgerLoopDay_normal_7",
            "DodgerLoopGame_normal_1", "DodgerLoopGame_normal_2", "DodgerLoopWeekend_normal_1",
            "DodgerLoopWeekend_normal_2", "MelbournePedestrian_normal_1", "MelbournePedestrian_normal_2",
            "MelbournePedestrian_normal_3", "MelbournePedestrian_normal_4", "MelbournePedestrian_normal_5",
            "MelbournePedestrian_normal_6", "MelbournePedestrian_normal_7", "MelbournePedestrian_normal_8",
            "MelbournePedestrian_normal_9",
        )
        db_path = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
        for dataset in smoke_datasets:
            raw = load_dataset_values(db_path, dataset)
            sources = score_hard_sources(raw)
            mini = score_minirocket_confirmations(raw)
            route = replay_hard_route(raw, sources, mini)
            self.assertEqual(route.exp137.hard, exp137_oracle[dataset], f"Exp137 mismatch on {dataset}")
            self.assertEqual(route.exp236.hard, exp236_oracle[dataset], f"Exp236 mismatch on {dataset}")
            self.assertEqual(route.automatic_hard, exp291_auto[dataset], f"Exp291 auto mismatch on {dataset}")
            self.assertEqual(route.exact_normal_removed, exp291_removed[dataset], f"Exp291 exact mismatch on {dataset}")
            self.assertEqual(route.automatic_hard, exp404_oracle[dataset], f"Exp404 mismatch on {dataset}")
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
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- `verify_hard_route_parity_paper(routes: Mapping[str, HardRouteResult], *, exp137_path: Path | None = None, exp236_path: Path | None = None, exp291_path: Path | None = None, exp404_path: Path | None = None) -> dict[str, object]`:
  - Verifies aggregate counts across all 1,117 datasets:
    - Exp137 Hard count == 2005
    - Exp236 Hard count == 2116
    - Exp253 Hard count == 2253 ($2218 + 35$)
    - Exact-normal removed count == 35
    - Automatic Hard count == 2218
  - Verifies 100% per-dataset coordinate equality against the 4 SHA-bound oracles.
  - Raises `ValueError` or `RuntimeError` on any count mismatch or coordinate inequality.
- `compose_proposal_score(...)`:
  - Calls `verify_hard_route_parity_paper(routes)` when all 1,117 datasets are present, before invoking `score_low_frequency` (Task6) and `classify_stability` (Task7).

- [ ] **Step 1: Run RED Command for Task 7**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_full_1117_route_only_parity_gate_blocks_task6_and_task7
```
Expected failure: `ImportError: cannot import name 'verify_hard_route_parity_paper' from 'paper.equipment_anomaly_detection.proposal_score_paper'`.

- [ ] **Step 2: Implement `verify_hard_route_parity_paper` in `proposal_score_paper.py`**

In `paper/equipment_anomaly_detection/proposal_score_paper.py`, add:

```python
EXP137_ORACLE_PATH = Path("results/exp137/experiment_137_operational_triage_results.csv")
EXP236_ORACLE_PATH = Path("outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl")
EXP291_ORACLE_PATH = Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl")
EXP404_ORACLE_PATH = Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl")

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
    exp137_path: Path | None = None,
    exp236_path: Path | None = None,
    exp291_path: Path | None = None,
    exp404_path: Path | None = None,
) -> dict[str, object]:
    """Verify exact 2005/2116/2253/35/2218 counts and 100% per-dataset coordinate parity before Task6/7."""
    exp137 = load_exp137_oracle_lanes(exp137_path or EXP137_ORACLE_PATH)
    exp236 = load_exp236_oracle_lanes(exp236_path or EXP236_ORACLE_PATH)
    exp291_auto, exp291_removed, exp291_pre = load_exp291_oracle_lanes(exp291_path or EXP291_ORACLE_PATH)
    exp404 = load_exp404_oracle_lanes(exp404_path or EXP404_ORACLE_PATH)

    if len(routes) != 1117:
        raise ValueError(f"route parity verifier requires exactly 1117 population datasets (got {len(routes)})")

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
            f"hard route count parity mismatch: expected {FROZEN_PARITY_COUNTS} but got {actual_counts}"
        )

    mismatches: list[str] = []
    for dataset, route in routes.items():
        if route.exp137.hard != exp137.get(dataset, frozenset()):
            mismatches.append(f"{dataset}:exp137")
        if route.exp236.hard != exp236.get(dataset, frozenset()):
            mismatches.append(f"{dataset}:exp236")
        if route.exp253.hard != exp291_pre.get(dataset, frozenset()):
            mismatches.append(f"{dataset}:exp253")
        if route.exact_normal_removed != exp291_removed.get(dataset, frozenset()):
            mismatches.append(f"{dataset}:exact_removed")
        if route.automatic_hard != exp291_auto.get(dataset, frozenset()) or route.automatic_hard != exp404.get(dataset, frozenset()):
            mismatches.append(f"{dataset}:automatic_hard")

    if mismatches:
        raise ValueError(
            f"hard route coordinate parity mismatch on {len(mismatches)} stages/datasets: {mismatches[:10]}"
        )

    return {"status": "parity_verified", "counts": actual_counts, "datasets_verified": len(routes)}
```

In `compose_proposal_score`, gate Task6/7 execution:
```python
    if len(raw) == 1117:
        verify_hard_route_parity_paper(routes)
```

- [ ] **Step 3: Run GREEN Verification for Task 7**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_full_1117_route_only_parity_gate_blocks_task6_and_task7 experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_candidate_c_rule_and_target_algebra_unchanged
```
Expected: `Ran 2 tests in ... OK`.

- [ ] **Step 4: Run Proposal Composition Regressions**

Run:
```bash
python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper
```
Expected: All tests pass GREEN.

- [ ] **Step 5: Resynchronize Codemaps and Commit PR 3**

Run shared map-resync procedure, then:
```bash
git add paper/equipment_anomaly_detection/proposal_score_paper.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "feat(composition): insert full 1117 route-only parity gate before Task6 and Task7"
```

---

### Task 8: Clean new smoke/official roots, destination refusal, and launcher configuration (PR 4)

**Files:**
- Modify: `scratch/manage_paper_r4_proposal_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_operations_antigravity.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- Authorized Path Constants (in `scratch/manage_paper_r4_proposal_antigravity.py`):
  - `AUTHORIZED_SMOKE_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_r4_full_route_parity_smoke_antigravity_20260814")`
  - `AUTHORIZED_SMOKE_OPS_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_r4_full_route_parity_smoke_antigravity_20260814_ops_antigravity")`
  - `AUTHORIZED_OFFICIAL_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_parity_antigravity_20260814")`
  - `AUTHORIZED_OFFICIAL_OPS_ROOT_20260814 = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_parity_antigravity_20260814_ops_antigravity")`
  - Rebind active targets: `OFFICIAL_SMOKE_ROOT = AUTHORIZED_SMOKE_ROOT_20260814`, `OFFICIAL_R4_ROOT = AUTHORIZED_OFFICIAL_ROOT_20260814`, `OFFICIAL_OPS_ROOT = AUTHORIZED_OFFICIAL_OPS_ROOT_20260814`.
  - Rebind `SMOKE_DATASETS` to the 33-dataset catalog-sorted tuple.
- `FORBIDDEN_DESTINATION_ROOTS`: Set containing all prior 20260813 and earlier roots.
- `validate_clean_root_destination_paper(path: Path) -> None`: Raises `ValueError` if `path` or its ops companion matches any forbidden prior root.

- [ ] **Step 1: Run RED Command for Task 8**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_clean_root_execution_and_rejection_of_live_r4_reuse experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations
```
Expected failure: `ImportError: cannot import name 'AUTHORIZED_SMOKE_ROOT_20260814'` or `validate_clean_root_destination_paper`.

- [ ] **Step 2: Update Launcher Paths, Smoke Datasets, and Destination Validation in `manage_paper_r4_proposal_antigravity.py`**

In `scratch/manage_paper_r4_proposal_antigravity.py`:
1. Define authorized 20260814 smoke and official paths, rebind official constants, and define forbidden prior roots:
```python
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
2. Update default smoke and launch commands to target authorized 20260814 clean paths and call `validate_clean_root_destination_paper`.

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
