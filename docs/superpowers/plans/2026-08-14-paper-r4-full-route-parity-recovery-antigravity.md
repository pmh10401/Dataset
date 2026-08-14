# Paper r4 Full-Population Hard-Route Parity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore full-population hard-route coordinate parity across all 1,117 datasets (2005 / 2116 / 2253 / 35 / 2218 counts, Candidate C = 2244) against SHA-bound frozen oracles, repairing Mechanism 1 (inexact Exp74d→137 replay via systematic diagnostic isolation) and Mechanism 2 (`canonical_normal_state` missing `pycatch22==0.5.0`), restoring general $n_{\text{train}}=1$ origin full-fit replay (`Phoneme_normal_39` Hard `{49}` without special-casing), validating a 33-dataset coordinate smoke (including the 8 known Exp404-empty smoke datasets), enforcing a pre-Task6/7 1,117 route-only parity gate (exact counts, exact key set, exact $n_{\text{test}}$, zero `.get(..., empty)` masking) bound to a typed context authority `HardRouteParityAuthority`, and executing on clean new 20260814 roots with zero cache reuse.

**Architecture:** Fail-closed `verify_pycatch22_environment_paper()` in `canonical_paper.py` called in `_collect_task4_task5`; worker error propagation on eligible datasets in `hard_sources_paper.py`; general $n_{\text{train}}=1$ full-fit support in `legacy_route_sources_paper.py` and `hard_route_paper.py`; diagnostic-isolated exact primitive repair in `hard_route_paper.py` (preceded by diagnostic artifact); typed data-only `HardRouteParityAuthority` with absolute paths and verified SHAs, single strict `load_hard_route_oracle_bundle`, and `verify_hard_route_parity_paper` in `proposal_score_paper.py` gating Task6/7 from explicit runner context; clean new smoke and official roots, 33-dataset smoke catalog, and destination refusal in `scratch/manage_paper_r4_proposal_antigravity.py`; CPU-canonical ProcessPool execution with `dataset_workers = max(1, logical - 1)` and `inner_numerical_threads = 1`.

**Tech Stack:** Python 3.14, `unittest`, `numpy==2.3.5`, `scikit-learn==1.8.0`, `aeon==1.5.0`, `pycatch22==0.5.0`, `torch==2.13.0`, canonical JSON/SHA-256, Exp448 `sha256-v1` fingerprints, paper `static_ast_sha256` triplet. Capable execution interpreter: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`.

**Approved spec:** `docs/superpowers/specs/2026-08-14-paper-r4-full-route-parity-recovery-antigravity-design.md` (committed at `ad45f90f`).

## Global Constraints

- Work only in this dedicated Antigravity worktree (`/Users/mac/Documents/Time Series/.worktrees/grok-paper-r4-operational-antigravity`). Do not edit `/Users/mac/.codex/worktrees/e1be/Time Series`, `/Users/mac/Documents/Time Series`, or any other worktree.
- Do not write to, migrate from, or resume `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813` or any prior smoke/official directories. Zero cache reuse or record migration. Clean create-only execution on 20260814 paths.
- All newly created files across all tasks must include `antigravity` in their file name (e.g., `test_paper_r4_full_route_parity_recovery_antigravity.py`, `diagnose_paper_r4_mechanism1_parity_antigravity.py`, `paper_r4_mechanism1_parity_diagnosis_antigravity.md`). Existing paper production modules keep `*_paper.py` names.
- New test file is `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`.
- Strict firewall against TEST label access: `_label_firewall()` remains active during Task4–7. `TEST_VALUES_SQL` contains no label columns.
- Replay and evidence functions accept only values and computed sources. Historical prediction coordinates are strictly oracles in verification helpers and tests; they must never be passed as runtime inputs to replay functions.
- Gate authority paths and hashes through typed data-only `HardRouteParityAuthority` (no `default_verified`), with launcher explicitly constructing the four path+hash fields, bound into verified `PipelineConfig` and immutable run context, then passed by runner into compose/verification. There are zero cwd-relative paths or implicit module defaults that silently find an oracle by cwd or local machine path.
- One strict `load_hard_route_oracle_bundle(authority, selected_population, selected_n_test)` SHA-verifies all 4 files, globally parses and validates exact 1,117 Exp137/236/291 populations and builds global $n_{\text{test}}$ maps requiring equality across all three, validates ALL coordinates including unselected rows against global $n_{\text{test}}$, initializes all 1,117 explicit Exp404 keys from sparse 2,218 rows (888 nonempty + 229 implicit-empty datasets) and validates coordinate bounds against global $n_{\text{test}}$, then requires `selected_n_test` (if supplied) matches global values for `selected_population`, and direct-index projects to 33 smoke or 1,117 full datasets. Zero `allow_unselected` and zero `.get(..., empty)` masking anywhere. The 33 smoke must work for the 8 known Exp404-empty smoke datasets.
- Gate verification is triggered from explicit verified context/mode/authority passed through runner, never from `len(raw)` mode inference. Official context supplies explicit full/smoke mode and authority; direct optional calls require both explicit authority and `route_parity_mode`. Official context without authority fails closed for both full and smoke (`elif context is not None: raise`). Unit/local calls without official context or authority may execute composition ungated, but cannot complete an official journal.
- CPU worker policy: Concurrency uses `dataset_workers = max(1, logical_cpu_count - 1)` with `inner_numerical_threads = 1`. Record detected logical CPU count and the resulting worker count in acceleration-manifest provenance.
- Explicit MLX/MPS assessment: CPU ProcessPool is retained as canonical; MLX/MPS acceleration is rejected for discrete route selectors and third-party feature extractors.
- Candidate C (`hard_plus_stable`) remains unchanged as a rule: Automatic Hard $\cup$ stable-26 from the 30-candidate stability partition; target count formula $2218 + 26 = 2244$. Task6 low-frequency filtering consumes pre-guard Exp253 Hard (2253), policy construction consumes post-guard Automatic Hard (2218).
- Do not authorize running full 1,117 official scoring during implementation tasks; only bounded fixtures, diagnostic harnesses, and explicitly named route-only verification after all unit gates pass.
- Strict RED → GREEN per task: No production route edits until an isolated RED test demonstrates the exact diagnosed primitive divergence.
- Each mergeable commit that changes a mapped module or mapped entrypoint signature includes all six Exp448 + paper map files in that same commit. Do not overwrite `docs/codemap/codemap_exp444_antigravity.*` or earlier version maps.

## Live Verified Authority

Parity oracles are SHA-bound, label-free prediction artifacts:

| Stage | File | SHA-256 | Extracted Field & Parsing Schema |
| --- | --- | --- | --- |
| Exp137 | `/Users/mac/Documents/Dataset/experiments/exp234_recovery_antigravity/experiment_137_operational_triage_results_antigravity.csv` *(Note: workspace copy `results/exp137/...` has sha `7bedd1bd...` and is not the SHA authority)* | `942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d` | Live full header is exact 54-field tuple (including `dataset_name`, `hard_alert_indices`, `test_size`, `family`, `hard_alert_count`, etc.); parse only `dataset_name`, space-delimited `hard_alert_indices`, and integer `test_size` (1,117 rows, 2,005 coordinates, global $n_{\text{test}}$ map) |
| Exp236 | `/Users/mac/Documents/Dataset/experiments/exp236_direct_run_antigravity/exp236_frozen_predictions.jsonl` | `a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524` | Exact 15 live row keys (`contains_evaluation_metrics`, `contains_test_labels`, `dataset`, `deployment_eligible`, `experiment_id`, `hard_indices`, `hard_policy`, `n_test`, `new_hard_indices`, `new_standard_review_indices`, `priority_review_indices`, `priority_review_policy`, `retrospective_candidate_packaging`, `standard_review_indices`, `standard_review_policy`); require `contains_test_labels is False` and `contains_evaluation_metrics is False`; parse `dataset`, integer list `hard_indices` (1,117 rows, 2,116 coordinates), and integer `n_test` |
| Exp291 | `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl` | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` | Exact 18 live row keys (`automatic_hard_indices`, `automatic_normal_exact_indices`, `automatic_normal_rule`, `contains_evaluation_metrics`, `contains_test_labels`, `dataset`, `default_review_indices`, `experiment_id`, `n_test`, `priority_review_indices`, `prospective_validated`, `recovery_priority_scope`, `recovery_shadow_review_indices`, `research_only_recovery_priority_indices`, `retrospective_integration`, `shadow_review_indices`, `source_baseline_hard_indices`, `standard_review_indices`); require `contains_test_labels is False` and `contains_evaluation_metrics is False`; parse `dataset`, `automatic_hard_indices` (2,218 coordinates), `automatic_normal_exact_indices` (35 coordinates), and integer `n_test`; pre-exact Exp253 = union of both (2,253 coordinates) |
| Exp404 | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` | Sparse exhaustive coordinate authority: 2,218 unique rows across 888 nonempty datasets; 229 implicit-empty datasets (888 + 229 = 1,117 total). Exact 8 row keys (`abstention_reason`, `candidate_index`, `confidence_by_alpha`, `contains_evaluation_metrics`, `contains_test_labels`, `dataset`, `eligibility`, `scores_by_alpha`); require `contains_test_labels is False` and `contains_evaluation_metrics is False`; grouped by `dataset` with `candidate_index`. Explicitly initialized from verified population, checked for $0 \le \text{candidate\_index} < \text{global\_n\_test}$, direct indexing only. |

Supporting frozen authority:
- Database `/Users/mac/Documents/Dataset/univariate_ts.db` SHA-256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Population: Exactly 1,117 datasets, 195,900 TEST rows; excluded: `CornellWhaleChallenge`, `Wafer_normal_1`
- Frozen count targets: Exp137 Hard = 2005, Exp236 Hard = 2116, Exp253 Hard = 2253 ($2218 + 35$), exact-normal removed = 35, Automatic Hard = 2218, Candidate C = 2244 ($2218 + 26$).

## Acceleration Assessment (Mandatory, No-Change)

| Checklist item | Assessment |
| --- | --- |
| Tensor-eligible? | No. Route selectors are discrete rank/set/cap operations. `pycatch22.catch22_all` is a frozen third-party C/Python CPU extractor. |
| Current backend availability? | CPU only. Capable oracle environment (`/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`) provides `pycatch22==0.5.0`, `aeon==1.5.0`, `numpy==2.3.5`, `scikit-learn==1.8.0`, `torch==2.13.0` on CPU. MLX is absent. |
| Transfer / compilation overhead? | Unfavorable. Rewriting feature extractors for GPU would incur high transfer latency across 1,117 small series with no throughput benefit. |
| Deterministic and numerical parity? | Bit-exact parity against SHA-bound oracles is mandatory. Non-canonical GPU approximations are forbidden. |
| Canonical CPU fallback? | Retained as canonical. `acceleration_paper.py` designates `hard_sources` and `hard_route` as CPU-canonical. |
| Worker concurrency | CPU ProcessPool with `dataset_workers = max(1, logical_cpu_count - 1)` and `inner_numerical_threads = 1`. |

## File Structure

- Create: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Create (Task 4 Diagnostic): `scratch/diagnose_paper_r4_mechanism1_parity_antigravity.py`
- Create (Task 4 Report): `reports/paper_r4_mechanism1_parity_diagnosis_antigravity.md`
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/canonical_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py`
- Modify: `paper/equipment_anomaly_detection/hard_route_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `paper/equipment_anomaly_detection/runner_paper.py`
- Modify: `paper/equipment_anomaly_detection/pipeline_paper.py`
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
data['working_tree_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip())
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
lock['working_tree_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip())
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
- Produces: Both complete triplets, generated from current `git rev-parse HEAD`, reflecting actual diff and answering callers / effects / tests for all modified paper modules before production code changes.

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
- Pinned capable interpreter: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python` verified for `pycatch22==0.5.0`, `aeon==1.5.0`, `torch==2.13.0`.
- Typed Data-Only Authority: `HardRouteParityAuthority` in `contracts_paper.py` (data-only, no `default_verified`), with absolute paths and exact SHA-256 hashes.
- Single Strict Oracle Bundle Loader: `load_hard_route_oracle_bundle(authority, selected_population=..., selected_n_test=...)` returning `HardRouteOracleBundle`.
- Module-level imports avoid not-yet-created symbols (`HardRouteParityAuthority`, `HardRouteOracleBundle`, `load_hard_route_oracle_bundle`, `verify_hard_route_parity_paper`); local imports are used inside RED tests (T6, T7, T8), keeping existing regression locks (T4, T9, T12, T13, T14) importable and GREEN at test module load.
- Methods:
  - T1: `test_pycatch22_pin_fails_closed_when_missing_or_wrong_version` (Missing-interface RED)
  - T2: `test_pycatch22_0_5_0_is_the_only_accepted_version` (Missing-interface RED)
  - T3: `test_pycatch22_dependency_failure_propagates_without_swallowing` (Error-propagation RED: asserts dependency failure on eligible datasets raises `DependencyUnavailablePaper` and propagates rather than silently swallowing into `eligible=False`)
  - Integration: `test_capable_venv_eligible_canonical_scoring_with_pycatch22` (Capable-venv GREEN: proves eligible canonical scoring with real `pycatch22==0.5.0`)
  - T4: `test_replay_signature_still_rejects_historical_coordinates` (Regression Lock: currently GREEN)
  - T5: `test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch` (Origin Replay RED)
  - T6: `test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence` (Diagnostic RED: asserts exact diagnosed divergence on representative datasets against frozen origin / SHA authority)
  - T7: `test_smoke_33_datasets_match_sha_bound_oracles` (Capable integration: 33 smoke datasets match 4 SHA-bound oracles bit-for-bit via `load_hard_route_oracle_bundle`, direct indexing on `bundle.exp404[dataset]` works for all 33 including the 8 empty datasets, zero `allow_unselected` and zero `.get`)
  - T8: `test_full_1117_route_only_parity_gate_blocks_task6_and_task7` (Missing-interface / Fail-closed RED: official context without authority fails closed for both full and smoke; mismatch fails closed before Task6/7; passes on verified authority)
  - T9: `test_candidate_c_rule_and_target_algebra_unchanged` (Regression Lock: currently GREEN)
  - T10: `test_clean_root_execution_and_rejection_of_live_r4_reuse` (Missing-interface RED)
  - T11: `test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations` (Missing-interface RED)
  - T12: `test_progress_logging_remains_optional_and_label_free` (Regression Lock: currently GREEN)
  - T13: `test_zero_test_label_access_until_evaluation` (Regression Lock: currently GREEN)
  - T14: `test_frozen_counts_2218_remain_authoritative_target` (Regression Lock: currently GREEN)

- [ ] **Step 1: Write the Full Test Matrix Scaffold**

Write `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py` containing complete typed fixtures for all 14 tests:

```python
"""Test suite for Paper r4 full-population hard-route parity recovery."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sqlite3
import sys
from typing import Mapping, Sequence
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from paper.equipment_anomaly_detection.catalog_paper import build_catalog
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
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import score_hard_sources
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import build_legacy_route_evidence
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import score_minirocket_confirmations
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import DependencyUnavailablePaper


DB_PATH = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
VERIFIED_EXP137_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/experiments/exp234_recovery_antigravity/experiment_137_operational_triage_results_antigravity.csv"
)
VERIFIED_EXP137_AUTHORITY_SHA256 = "942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d"

VERIFIED_EXP236_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/experiments/exp236_direct_run_antigravity/exp236_frozen_predictions.jsonl"
)
VERIFIED_EXP236_AUTHORITY_SHA256 = "a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524"

VERIFIED_EXP291_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl"
)
VERIFIED_EXP291_AUTHORITY_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"

VERIFIED_EXP404_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl"
)
VERIFIED_EXP404_AUTHORITY_SHA256 = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"


def _build_test_authority():
    from paper.equipment_anomaly_detection.contracts_paper import HardRouteParityAuthority

    return HardRouteParityAuthority(
        exp137_path=VERIFIED_EXP137_AUTHORITY_PATH,
        exp137_sha256=VERIFIED_EXP137_AUTHORITY_SHA256,
        exp236_path=VERIFIED_EXP236_AUTHORITY_PATH,
        exp236_sha256=VERIFIED_EXP236_AUTHORITY_SHA256,
        exp291_path=VERIFIED_EXP291_AUTHORITY_PATH,
        exp291_sha256=VERIFIED_EXP291_AUTHORITY_SHA256,
        exp404_path=VERIFIED_EXP404_AUTHORITY_PATH,
        exp404_sha256=VERIFIED_EXP404_AUTHORITY_SHA256,
    )


class FullRouteParityRecoveryAntigravityTests(unittest.TestCase):
    """Parity recovery test matrix T1 through T14."""

    def test_pycatch22_pin_fails_closed_when_missing_or_wrong_version(self) -> None:
        """T1 (Missing-interface RED): verify_pycatch22_environment_paper raises DependencyUnavailablePaper on missing or mismatch."""
        from paper.equipment_anomaly_detection.sources_paper.canonical_paper import (
            verify_pycatch22_environment_paper,
        )

        with patch("importlib.metadata.version", side_effect=Exception("not found")):
            with self.assertRaises(DependencyUnavailablePaper):
                verify_pycatch22_environment_paper()

        with patch("importlib.metadata.version", return_value="0.4.0"):
            with self.assertRaises(DependencyUnavailablePaper):
                verify_pycatch22_environment_paper()

    def test_pycatch22_0_5_0_is_the_only_accepted_version(self) -> None:
        """T2 (Missing-interface RED): PINNED_PYCATCH22_VERSION_PAPER is '0.5.0' and returns valid metadata dict."""
        from paper.equipment_anomaly_detection.sources_paper import canonical_paper

        self.assertTrue(hasattr(canonical_paper, "PINNED_PYCATCH22_VERSION_PAPER"))
        self.assertEqual(canonical_paper.PINNED_PYCATCH22_VERSION_PAPER, "0.5.0")
        with patch("importlib.metadata.version", return_value="0.5.0"), patch.dict(
            "sys.modules", {"pycatch22": type("m", (), {"catch22_all": lambda *a: ()})()}
        ):
            meta = canonical_paper.verify_pycatch22_environment_paper()
            self.assertEqual(meta["pycatch22"], "0.5.0")

    def test_pycatch22_dependency_failure_propagates_without_swallowing(self) -> None:
        """T3 (Error-propagation RED): Missing pycatch22 on eligible dataset raises DependencyUnavailablePaper and propagates through _collect_task4_task5."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        from paper.equipment_anomaly_detection.proposal_score_paper import _collect_task4_task5

        rng = np.random.default_rng(20260814)
        train_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(10))
        test_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(20))
        raw_ds = RawDataset(
            dataset="Synthetic_eligible_dataset",
            metadata_length=100,
            train_indices=tuple(range(10)),
            train_blobs=tuple(r.tobytes() for r in train_rows),
            train_rows=train_rows,
            test_indices=tuple(range(20)),
            test_blobs=tuple(r.tobytes() for r in test_rows),
            test_rows=test_rows,
        )

        manifest = build_acceleration_manifest("a" * 64, logical_cpu_count=2)
        backend = FrozenTask8BackendDecision.from_acceleration_manifest(manifest)

        # On an eligible dataset, forcing pycatch22 dependency error must raise DependencyUnavailablePaper and propagate
        with patch(
            "paper.equipment_anomaly_detection.sources_paper.canonical_paper._canonical_source_paper",
            side_effect=DependencyUnavailablePaper("dependency_unavailable:pycatch22==0.5.0"),
        ):
            with self.assertRaises(DependencyUnavailablePaper):
                score_hard_sources(raw_ds)

            with self.assertRaises(DependencyUnavailablePaper):
                _collect_task4_task5({"Synthetic_eligible_dataset": raw_ds}, backend)

    def test_capable_venv_eligible_canonical_scoring_with_pycatch22(self) -> None:
        """Integration (Capable venv): Eligible dataset scores canonical_normal_state as eligible=True with pycatch22==0.5.0."""
        try:
            version = importlib.metadata.version("pycatch22")
        except Exception:
            self.skipTest("pycatch22 not installed in current interpreter; run under capable venv")
        if version != "0.5.0":
            self.skipTest(f"pycatch22 version is {version}, expected 0.5.0")

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
        """T4 (Regression Lock): replay_hard_route accepts only values, sources, and mini (no oracle kwargs)."""
        sig = inspect.signature(replay_hard_route)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["values", "sources", "mini"])
        sig_legacy = inspect.signature(build_legacy_route_evidence)
        self.assertEqual(list(sig_legacy.parameters.keys()), ["values"])

    def test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch(self) -> None:
        """T5 (Origin Replay RED): Phoneme_normal_39 (n_train=1) produces Hard {49} using values-only DB data and zero name branching."""
        from paper.equipment_anomaly_detection import hard_route_paper
        from paper.equipment_anomaly_detection.sources_paper import legacy_route_sources_paper

        hard_route_src = inspect.getsource(hard_route_paper)
        legacy_src = inspect.getsource(legacy_route_sources_paper)
        self.assertNotIn("Phoneme_normal_39", hard_route_src)
        self.assertNotIn("Phoneme_normal_39", legacy_src)

        raw = load_dataset_values(DB_PATH, "Phoneme_normal_39")
        self.assertEqual(len(raw.train_rows), 1)
        self.assertEqual(len(raw.test_rows), 50)
        sources = score_hard_sources(raw)
        mini = score_minirocket_confirmations(raw)
        route = replay_hard_route(raw, sources, mini)
        self.assertEqual(route.exp137.hard, frozenset({49}))
        self.assertEqual(route.exp236.hard, frozenset({49}))
        self.assertEqual(route.automatic_hard, frozenset({49}))

    def test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence(self) -> None:
        """T6 (Diagnostic RED): Diagnostic stage isolation proves exact primitive parity against SHA-bound Exp137 authority."""
        from paper.equipment_anomaly_detection.proposal_score_paper import load_hard_route_oracle_bundle

        authority = _build_test_authority()
        self.assertTrue(authority.exp137_path.exists())
        file_sha = hashlib.sha256(authority.exp137_path.read_bytes()).hexdigest()
        self.assertEqual(file_sha, authority.exp137_sha256)

        rep_datasets = (
            "ACSF1_normal_0", "ACSF1_normal_1", "ACSF1_normal_2", "ACSF1_normal_3",
            "ACSF1_normal_4", "ACSF1_normal_5", "ACSF1_normal_6", "ACSF1_normal_7",
            "ACSF1_normal_8", "ACSF1_normal_9", "ChlorineConcentration_normal_3",
            "MelbournePedestrian_normal_10", "Phoneme_normal_39", "PigAirwayPressure_normal_1",
        )
        catalog = build_catalog(DB_PATH, mode="smoke", smoke_datasets=rep_datasets)
        n_test_map = {entry["dataset"]: int(entry["test"]["count"]) for entry in catalog["datasets"]}
        bundle = load_hard_route_oracle_bundle(
            authority=authority,
            selected_population=rep_datasets,
            selected_n_test=n_test_map,
        )
        for ds in rep_datasets:
            raw = load_dataset_values(DB_PATH, ds)
            sources = score_hard_sources(raw)
            mini = score_minirocket_confirmations(raw)
            route = replay_hard_route(raw, sources, mini)
            self.assertEqual(route.exp137.hard, bundle.exp137[ds], f"Exp137 legacy route replay diverged on {ds}")

    def test_smoke_33_datasets_match_sha_bound_oracles(self) -> None:
        """T7 (Capable Integration): All 33 smoke datasets achieve per-dataset coordinate equality against the 4 SHA-bound oracles; direct indexing works on Exp404."""
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            load_hard_route_oracle_bundle,
        )

        authority = _build_test_authority()
        smoke_datasets = (
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
        self.assertEqual(len(set(smoke_datasets)), 33)

        catalog = build_catalog(DB_PATH, mode="smoke", smoke_datasets=smoke_datasets)
        n_test_map = {entry["dataset"]: int(entry["test"]["count"]) for entry in catalog["datasets"]}

        bundle = load_hard_route_oracle_bundle(
            authority=authority,
            selected_population=smoke_datasets,
            selected_n_test=n_test_map,
        )

        for dataset in smoke_datasets:
            raw = load_dataset_values(DB_PATH, dataset)
            sources = score_hard_sources(raw)
            mini = score_minirocket_confirmations(raw)
            route = replay_hard_route(raw, sources, mini)
            self.assertEqual(route.exp137.hard, bundle.exp137[dataset], f"Exp137 mismatch on {dataset}")
            self.assertEqual(route.exp236.hard, bundle.exp236[dataset], f"Exp236 mismatch on {dataset}")
            self.assertEqual(route.exp253.hard, bundle.exp253[dataset], f"Exp253 mismatch on {dataset}")
            self.assertEqual(route.automatic_hard, bundle.automatic_hard[dataset], f"Exp291 auto mismatch on {dataset}")
            self.assertEqual(route.exact_normal_removed, bundle.exact_normal_removed[dataset], f"Exp291 exact mismatch on {dataset}")
            self.assertEqual(route.automatic_hard, bundle.exp404[dataset], f"Exp404 mismatch on {dataset}")

    def test_full_1117_route_only_parity_gate_blocks_task6_and_task7(self) -> None:
        """T8 (Missing-interface / Fail-closed RED): verify_hard_route_parity_paper halts compose_proposal_score on mismatch before Task6; official context without authority fails for both full and smoke."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            compose_proposal_score,
            verify_hard_route_parity_paper,
        )

        authority = _build_test_authority()
        bad_routes = {"dataset_a": empty_hard_route_result_paper(10)}
        with self.assertRaises(Exception):
            verify_hard_route_parity_paper(bad_routes, authority=authority, mode="full")

        fake_raw = {
            f"d_{i}": RawDataset(
                f"d_{i}", 10, (0,), (b"0",), (np.zeros(10, dtype=np.float32),), (0,), (b"0",), (np.zeros(10, dtype=np.float32),)
            )
            for i in range(1117)
        }
        manifest = build_acceleration_manifest("a" * 64, logical_cpu_count=2)

        # Official context without authority must fail for both full and smoke
        with self.assertRaises(ValueError):
            compose_proposal_score(fake_raw, manifest, context={"mode": "full"})
        with self.assertRaises(ValueError):
            compose_proposal_score(fake_raw, manifest, context={"mode": "smoke"})

        with patch(
            "paper.equipment_anomaly_detection.proposal_score_paper._collect_task4_task5",
            return_value=({}, {f"d_{i}": empty_hard_route_result_paper(10) for i in range(1117)}),
        ), patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.score_low_frequency"
        ) as mock_task6, patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.classify_stability"
        ) as mock_task7:
            context = {"mode": "full", "hard_route_parity_authority": authority.to_dict()}
            with self.assertRaises(ValueError):
                compose_proposal_score(fake_raw, manifest, context=context)
            mock_task6.assert_not_called()
            mock_task7.assert_not_called()

    def test_candidate_c_rule_and_target_algebra_unchanged(self) -> None:
        """T9 (Regression Lock): Candidate C budget 30, view cap 16, survival 14/16, formula 2218 + 26 = 2244."""
        self.assertEqual(GLOBAL_CANDIDATE_BUDGET_PAPER, 30)
        self.assertEqual(_MAXIMUM_VIEWS_PAPER, 16)
        survival_threshold = 14 / 16
        self.assertAlmostEqual(survival_threshold, 0.875)
        automatic_hard_target = 2218
        stable_target = 26
        self.assertEqual(automatic_hard_target + stable_target, 2244)

    def test_clean_root_execution_and_rejection_of_live_r4_reuse(self) -> None:
        """T10 (Missing-interface RED): Runner creates fresh records in clean root and rejects live r4 root reuse."""
        forbidden_prior = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813")
        from scratch.manage_paper_r4_proposal_antigravity import validate_clean_root_destination_paper

        with self.assertRaises(ValueError):
            validate_clean_root_destination_paper(forbidden_prior)

    def test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations(self) -> None:
        """T11 (Missing-interface RED): Launcher constants use new clean 20260814 paths and refuse prior roots."""
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
        """T12 (Regression Lock): Progress logging default None is bit-identical and label-free."""
        event = {"stage": "task4_task5", "phase": "tick", "completed": 1, "total": 10, "dataset": "Synthetic_dataset"}
        emitted: list[dict] = []
        emit_progress_paper(emitted.append, event)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["stage"], "task4_task5")
        self.assertNotIn("y_test", str(emitted[0]))
        self.assertNotIn("labels", str(emitted[0]))

    def test_zero_test_label_access_until_evaluation(self) -> None:
        """T13 (Regression Lock): TEST_VALUES_SQL selects no label column; label firewall is intact."""
        self.assertNotIn("y_test", TEST_VALUES_SQL.lower())
        self.assertNotIn("label", TEST_VALUES_SQL.lower())
        self.assertNotIn("is_anomaly", TEST_VALUES_SQL.lower())

    def test_frozen_counts_2218_remain_authoritative_target(self) -> None:
        """T14 (Regression Lock): Target counts remain 2005/2116/2253/35/2218; 2203 is not a target."""
        self.assertEqual(FROZEN_CONFIG.ablation_alert_counts[0], 2218)
        self.assertNotEqual(FROZEN_CONFIG.ablation_alert_counts[0], 2203)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the Initial Test Matrix (Verify Initial Lock and RED Sequencing)**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests
```
Expected output:
- T4, T9, T12, T13, T14 pass GREEN (Regression Locks).
- T1, T2, T8, T10, T11 fail RED (Missing interfaces/constants).
- T3 fails RED (Current code swallows dependency errors on eligible datasets instead of propagating).
- T5, T6 fail RED (Current code short-circuits $n_{\text{train}}=1$ and legacy replay is inexact).
- T7 skipped or fails RED under default interpreter without capable environment.

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
- Pinned interpreter preflight: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`.
- `PINNED_PYCATCH22_VERSION_PAPER: str = "0.5.0"` in `sources_paper/canonical_paper.py`.
- `verify_pycatch22_environment_paper() -> dict[str, str]` in `sources_paper/canonical_paper.py`:
  - Returns `{"pycatch22": version}` when package is present, version equals `"0.5.0"`, and `catch22_all` is importable.
  - Raises `DependencyUnavailablePaper("dependency_unavailable:pycatch22==0.5.0")` on missing package, wrong version, or failed import.
- `_collect_task4_task5(...)` in `proposal_score_paper.py`:
  - Invokes `verify_pycatch22_environment_paper()` before processing any dataset.
- `score_hard_sources(values: RawDataset) -> HardSourceBundle` in `sources_paper/hard_sources_paper.py`:
  - If `values` is eligible ($n_{\text{train}} \ge 5$ and finite), unexpected missing dependencies raise `DependencyUnavailablePaper` instead of being swallowed into silent `eligible=False`. Ordinary abstentions (`unsupported_n_train_lt_5`, `nonfinite_input`) remain intact.

- [ ] **Step 1: Run Preflight Environment Verification and RED Command for Task 3**

Run preflight interpreter check:
```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -c "
import importlib.metadata, pycatch22
assert importlib.metadata.version('pycatch22') == '0.5.0'
assert hasattr(pycatch22, 'catch22_all')
print('pycatch22-capable-env-ok')
"
```
Expected: `pycatch22-capable-env-ok`.

Run RED command:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_pin_fails_closed_when_missing_or_wrong_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_0_5_0_is_the_only_accepted_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_dependency_failure_propagates_without_swallowing
```
Expected failure: `ImportError: cannot import name 'verify_pycatch22_environment_paper'` or `AttributeError: module '...canonical_paper' has no attribute 'PINNED_PYCATCH22_VERSION_PAPER'` or `AssertionError: DependencyUnavailablePaper not raised`.

- [ ] **Step 2: Implement Fail-Closed pycatch22 Pin in `canonical_paper.py`**

In `paper/equipment_anomaly_detection/sources_paper/canonical_paper.py`, add:
```python
import importlib.metadata
from .preprocessing_paper import DependencyUnavailablePaper

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

In `paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py`, ensure that on eligible datasets ($n_{\text{train}} \ge 5$ and finite), `_canonical_source_paper` and other sources do not silently swallow unexpected dependency errors:
```python
    # For canonical_normal_state on eligible datasets:
    try:
        results["canonical_normal_state"] = _completed_paper("canonical_normal_state", *_canonical_source_paper(train, test))
    except DependencyUnavailablePaper as exc:
        raise DependencyUnavailablePaper(f"canonical_normal_state failed on eligible dataset: {exc}") from exc
```

- [ ] **Step 5: Run GREEN Verification for Task 3**

Run under default interpreter (tests mocking/fail-fast behavior):
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_pin_fails_closed_when_missing_or_wrong_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_0_5_0_is_the_only_accepted_version experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_pycatch22_dependency_failure_propagates_without_swallowing
```

Run under pinned capable interpreter (tests live package integration):
```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_capable_venv_eligible_canonical_scoring_with_pycatch22
```
Expected: All tests pass GREEN (`OK`).

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

### Task 4: Values-only SHA-pinned oracle bundle loader and Mechanism 1 diagnostic isolation phase

**Files:**
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Create: `scratch/diagnose_paper_r4_mechanism1_parity_antigravity.py`
- Create: `reports/paper_r4_mechanism1_parity_diagnosis_antigravity.md`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- `HardRouteParityAuthority` dataclass (in `contracts_paper.py`):
  - Pure data-only dataclass (`exp137_path, exp137_sha256, exp236_path, exp236_sha256, exp291_path, exp291_sha256, exp404_path, exp404_sha256`).
  - No `default_verified` classmethod.
  - `to_dict() -> dict[str, str]` and `from_dict(data: Mapping[str, object]) -> HardRouteParityAuthority` with exact equality schema check `set(data.keys()) == required`.
- `HardRouteOracleBundle` dataclass (in `contracts_paper.py`):
  - Contains projected `{dataset: frozenset(coords)}` mappings for `exp137`, `exp236`, `exp253`, `exact_normal_removed`, `automatic_hard`, and `exp404`.
- Single Strict Oracle Bundle Loader `load_hard_route_oracle_bundle(authority, *, selected_population=None, selected_n_test=None)` (in `proposal_score_paper.py`):
  - Validates byte SHA-256 for all 4 files via `_verify_file_sha256`.
  - Globally parses and validates exact 1,117 populations for Exp137, Exp236, Exp291 against LIVE FULL schemas:
    - Exp137 CSV: header schema equality `tuple(reader.fieldnames) == EXP137_EXACT_HEADER` (exact 54 live fields), checks no duplicate datasets, parses `dataset_name`, space-delimited `hard_alert_indices`, and integer `test_size`, builds `exp137_n_test`, verifies exact 1,117 rows and 2,005 hard alert count.
    - Exp236 JSONL: line schema equality `set(record.keys()) == EXP236_EXACT_ROW_KEYS` (exact 15 live keys including `n_test` and flags), requires `contains_test_labels is False` and `contains_evaluation_metrics is False`, checks no duplicate datasets, parses integer list `hard_indices` and integer `n_test`, builds `exp236_n_test`, verifies exact 1,117 rows and 2,116 hard alert count.
    - Exp291 JSONL: line schema equality `set(record.keys()) == EXP291_EXACT_ROW_KEYS` (exact 18 live keys including `n_test` and flags), requires `contains_test_labels is False` and `contains_evaluation_metrics is False`, checks no duplicate datasets, parses integer lists `automatic_hard_indices`, `automatic_normal_exact_indices`, and integer `n_test`, builds `exp291_n_test`, constructs pre-exact Exp253 as union, verifies exact 1,117 rows and aggregate counts (pre-exact == 2253, exact-removed == 35, auto-hard == 2218).
    - Checks key set equality: `set(exp137_global.keys()) == set(exp236_global.keys()) == set(auto_global.keys())` (all 1,117 datasets).
    - Checks global $n_{\text{test}}$ map equality across all three: `exp137_n_test == exp236_n_test == exp291_n_test` and sets `global_n_test = exp137_n_test`.
    - Validates ALL coordinates across ALL 1,117 datasets (including unselected rows) against `global_n_test`: $0 \le c < \text{global\_n\_test}[ds]$ for Exp137, Exp236, and Exp291.
  - Globally parses Exp404 JSONL:
    - Initializes all 1,117 population keys explicitly to empty sets (`{ds: set() for ds in pop_1117}`).
    - Row schema equality `set(record.keys()) == EXP404_EXACT_ROW_KEYS` (exact 8 live keys), requires `contains_test_labels is False` and `contains_evaluation_metrics is False`.
    - Validates each coordinate row: `dataset` in 1,117 population, $0 \le cand < \text{global\_n\_test}[dataset]$, no duplicate coordinates per dataset.
    - Verifies exactly 2,218 rows seen across the entire file (888 nonempty datasets + 229 implicit-empty datasets = 1,117).
    - Verifies total coordinates == 2218.
  - Selected population verification and direct-index projection:
    - If `selected_population` is provided (e.g. 33 smoke datasets or 1,117 full datasets):
      - If `selected_n_test` is provided, requires `selected_n_test` exactly matches `global_n_test` for every dataset in `selected_population`.
      - Direct indexing `[dataset]` projects each map. Any unselected dataset is unreachable and missing dataset raises `KeyError`.
    - If `selected_population` is None, validates `selected_n_test == global_n_test` (if provided) and returns the full 1,117 bundle.
    - Zero `allow_unselected` and zero `.get(..., empty)` masking anywhere.
- Diagnostic Isolation Phase:
  - `scratch/diagnose_paper_r4_mechanism1_parity_antigravity.py` executes values-only against DB, comparing current paper replay intermediate selectors against independently executed frozen-origin AST primitives.
  - Fail-closed diagnostic execution discovers and records the exact divergent dataset(s), stage(s), and primitive function(s) into `reports/paper_r4_mechanism1_parity_diagnosis_antigravity.md`.
  - Isolates and locks the failing RED assertion in T6 without guessing.

- [ ] **Step 1: Run RED Command for Task 4 Adapter Interface**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence
```
Expected failure: `ImportError: cannot import name 'load_hard_route_oracle_bundle' from 'paper.equipment_anomaly_detection.proposal_score_paper'` (missing adapter RED).

- [ ] **Step 2: Implement `HardRouteParityAuthority`, `HardRouteOracleBundle`, and `load_hard_route_oracle_bundle`**

In `paper/equipment_anomaly_detection/contracts_paper.py`, define `HardRouteParityAuthority` and `HardRouteOracleBundle`.
In `paper/equipment_anomaly_detection/proposal_score_paper.py`, implement `load_hard_route_oracle_bundle`:

```python
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .contracts_paper import HardRouteOracleBundle, HardRouteParityAuthority

EXP137_EXACT_HEADER = (
    "experiment_id", "dataset_name", "family", "config_name", "selector_name",
    "selector_reason", "threshold_method", "score_family", "score_source_name",
    "hard_alert_indices", "standard_review_indices", "priority_review_indices",
    "hard_alert_count", "standard_review_count", "priority_review_count", "no_alert_count",
    "hard_tp", "hard_fp", "hard_fn", "hard_precision", "hard_recall", "hard_f1",
    "standard_review_tp", "standard_review_fp", "standard_review_fn", "standard_review_precision",
    "standard_review_recall", "standard_review_f1", "priority_review_tp", "priority_review_fp",
    "priority_review_fn", "priority_review_precision", "priority_review_recall", "priority_review_f1",
    "combined_tp", "combined_fp", "combined_fn", "combined_precision", "combined_recall", "combined_f1",
    "combined_metric_scope", "routing_uses_test_labels", "routing_uses_test_position",
    "routing_uses_family_performance", "priority_review_posthoc_research_rule",
    "prospective_priority_validation_required", "source_exp133_config", "source_exp135_config",
    "source_exp93_hard_count", "source_exp135_review_count", "train_normal_count",
    "sequence_length", "test_size", "anomaly_count",
)
EXP236_EXACT_ROW_KEYS = frozenset({
    "contains_evaluation_metrics", "contains_test_labels", "dataset", "deployment_eligible",
    "experiment_id", "hard_indices", "hard_policy", "n_test", "new_hard_indices",
    "new_standard_review_indices", "priority_review_indices", "priority_review_policy",
    "retrospective_candidate_packaging", "standard_review_indices", "standard_review_policy",
})
EXP291_EXACT_ROW_KEYS = frozenset({
    "automatic_hard_indices", "automatic_normal_exact_indices", "automatic_normal_rule",
    "contains_evaluation_metrics", "contains_test_labels", "dataset", "default_review_indices",
    "experiment_id", "n_test", "priority_review_indices", "prospective_validated",
    "recovery_priority_scope", "recovery_shadow_review_indices", "research_only_recovery_priority_indices",
    "retrospective_integration", "shadow_review_indices", "source_baseline_hard_indices", "standard_review_indices",
})
EXP404_EXACT_ROW_KEYS = frozenset({
    "abstention_reason", "candidate_index", "confidence_by_alpha", "contains_evaluation_metrics",
    "contains_test_labels", "dataset", "eligibility", "scores_by_alpha",
})


def _verify_file_sha256(path: Path, expected_sha256: str) -> None:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Oracle authority file not found: {resolved}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Oracle authority file SHA-256 mismatch for {resolved}: expected {expected_sha256}, got {digest}"
        )


def load_hard_route_oracle_bundle(
    authority: HardRouteParityAuthority,
    *,
    selected_population: Sequence[str] | None = None,
    selected_n_test: Mapping[str, int] | None = None,
) -> HardRouteOracleBundle:
    """Strictly SHA-verifies and parses all 4 frozen oracle files against full live schemas,
    builds global n_test maps from Exp137/236/291 requiring exact equality, validates ALL coordinates
    including unselected rows against global n_test, builds all 1117 Exp404 explicit keys from sparse 2218 rows,
    validates selected_n_test against global values, and direct-index projects to selected_population."""
    if not isinstance(authority, HardRouteParityAuthority):
        raise TypeError("HardRouteParityAuthority instance is required")

    # 1. SHA-256 verify all 4 oracle files
    _verify_file_sha256(authority.exp137_path, authority.exp137_sha256)
    _verify_file_sha256(authority.exp236_path, authority.exp236_sha256)
    _verify_file_sha256(authority.exp291_path, authority.exp291_sha256)
    _verify_file_sha256(authority.exp404_path, authority.exp404_sha256)

    # 2. Parse and validate Exp137 CSV globally
    exp137_global: dict[str, frozenset[int]] = {}
    exp137_n_test: dict[str, int] = {}
    with Path(authority.exp137_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != EXP137_EXACT_HEADER:
            raise ValueError(f"Exp137 CSV header schema mismatch: expected exact 54 live fields {EXP137_EXACT_HEADER}, got {reader.fieldnames}")
        for row in reader:
            dataset = str(row["dataset_name"])
            if dataset in exp137_global:
                raise ValueError(f"Exp137 CSV contains duplicate dataset: {dataset}")
            exp137_n_test[dataset] = int(row["test_size"])
            raw_indices = str(row["hard_alert_indices"]).strip()
            exp137_global[dataset] = frozenset(map(int, raw_indices.split())) if raw_indices else frozenset()

    if len(exp137_global) != 1117:
        raise ValueError(f"Exp137 dataset population count mismatch: expected 1117, got {len(exp137_global)}")
    if sum(len(v) for v in exp137_global.values()) != 2005:
        raise ValueError(f"Exp137 total hard alert count mismatch: expected 2005, got {sum(len(v) for v in exp137_global.values())}")

    # 3. Parse and validate Exp236 JSONL globally
    exp236_global: dict[str, frozenset[int]] = {}
    exp236_n_test: dict[str, int] = {}
    with Path(authority.exp236_path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            if not isinstance(record, dict) or set(record.keys()) != EXP236_EXACT_ROW_KEYS:
                raise ValueError(f"Exp236 JSONL line {line_num} schema mismatch: expected exact 15 live keys {EXP236_EXACT_ROW_KEYS}")
            if record.get("contains_test_labels") is not False or record.get("contains_evaluation_metrics") is not False:
                raise ValueError(f"Exp236 line {line_num} security violation: contains_test_labels or contains_evaluation_metrics is not False")
            dataset = str(record["dataset"])
            if dataset in exp236_global:
                raise ValueError(f"Exp236 JSONL contains duplicate dataset: {dataset}")
            raw_indices = record["hard_indices"]
            if not isinstance(raw_indices, list):
                raise ValueError(f"Exp236 hard_indices must be a list on line {line_num}")
            exp236_n_test[dataset] = int(record["n_test"])
            exp236_global[dataset] = frozenset(map(int, raw_indices))

    if len(exp236_global) != 1117:
        raise ValueError(f"Exp236 dataset population count mismatch: expected 1117, got {len(exp236_global)}")
    if sum(len(v) for v in exp236_global.values()) != 2116:
        raise ValueError(f"Exp236 total hard alert count mismatch: expected 2116, got {sum(len(v) for v in exp236_global.values())}")

    # 4. Parse and validate Exp291 JSONL globally
    auto_global: dict[str, frozenset[int]] = {}
    removed_global: dict[str, frozenset[int]] = {}
    pre_global: dict[str, frozenset[int]] = {}
    exp291_n_test: dict[str, int] = {}
    with Path(authority.exp291_path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            if not isinstance(record, dict) or set(record.keys()) != EXP291_EXACT_ROW_KEYS:
                raise ValueError(f"Exp291 JSONL line {line_num} schema mismatch: expected exact 18 live keys {EXP291_EXACT_ROW_KEYS}")
            if record.get("contains_test_labels") is not False or record.get("contains_evaluation_metrics") is not False:
                raise ValueError(f"Exp291 line {line_num} security violation: contains_test_labels or contains_evaluation_metrics is not False")
            dataset = str(record["dataset"])
            if dataset in auto_global:
                raise ValueError(f"Exp291 JSONL contains duplicate dataset: {dataset}")
            raw_hard = record["automatic_hard_indices"]
            raw_removed = record["automatic_normal_exact_indices"]
            if not isinstance(raw_hard, list) or not isinstance(raw_removed, list):
                raise ValueError(f"Exp291 indices must be lists on line {line_num}")
            exp291_n_test[dataset] = int(record["n_test"])
            hard = frozenset(map(int, raw_hard))
            removed = frozenset(map(int, raw_removed))
            auto_global[dataset] = hard
            removed_global[dataset] = removed
            pre_global[dataset] = hard | removed

    if len(auto_global) != 1117:
        raise ValueError(f"Exp291 dataset population count mismatch: expected 1117, got {len(auto_global)}")
    if sum(len(v) for v in pre_global.values()) != 2253:
        raise ValueError(f"Exp253 pre-exact count mismatch: expected 2253, got {sum(len(v) for v in pre_global.values())}")
    if sum(len(v) for v in removed_global.values()) != 35:
        raise ValueError(f"Exp291 exact-normal removed count mismatch: expected 35, got {sum(len(v) for v in removed_global.values())}")
    if sum(len(v) for v in auto_global.values()) != 2218:
        raise ValueError(f"Exp291 automatic hard count mismatch: expected 2218, got {sum(len(v) for v in auto_global.values())}")

    # Check key set and global n_test agreement across Exp137, Exp236, Exp291
    pop_1117 = set(exp137_global.keys())
    if set(exp236_global.keys()) != pop_1117 or set(auto_global.keys()) != pop_1117:
        raise ValueError("Population discrepancy across Exp137, Exp236, and Exp291 oracle files")
    if exp137_n_test != exp236_n_test or exp137_n_test != exp291_n_test:
        raise ValueError("Global n_test map mismatch across Exp137, Exp236, and Exp291")
    global_n_test = exp137_n_test

    # Validate ALL coordinates across ALL 1,117 datasets against global_n_test
    for ds in pop_1117:
        limit = global_n_test[ds]
        if any(idx < 0 or idx >= limit for idx in exp137_global[ds]):
            raise ValueError(f"Exp137 coordinate out of range for {ds} (n_test={limit})")
        if any(idx < 0 or idx >= limit for idx in exp236_global[ds]):
            raise ValueError(f"Exp236 coordinate out of range for {ds} (n_test={limit})")
        if any(idx < 0 or idx >= limit for idx in pre_global[ds]):
            raise ValueError(f"Exp291 coordinate out of range for {ds} (n_test={limit})")

    # 5. Parse Exp404 JSONL globally, explicitly initializing all 1,117 population keys
    exp404_global: dict[str, set[int]] = {ds: set() for ds in pop_1117}
    seen_rows = 0
    with Path(authority.exp404_path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            if not isinstance(record, dict) or set(record.keys()) != EXP404_EXACT_ROW_KEYS:
                raise ValueError(f"Exp404 JSONL line {line_num} schema mismatch: expected exact 8 live keys {EXP404_EXACT_ROW_KEYS}")
            if record.get("contains_test_labels") is not False or record.get("contains_evaluation_metrics") is not False:
                raise ValueError(f"Exp404 line {line_num} security violation: contains_test_labels or contains_evaluation_metrics is not False")
            dataset = str(record["dataset"])
            cand = int(record["candidate_index"])
            seen_rows += 1
            if dataset not in exp404_global:
                raise ValueError(f"Exp404 JSONL contains unknown dataset not in verified population: {dataset}")
            limit = global_n_test[dataset]
            if cand < 0 or cand >= limit:
                raise ValueError(f"Exp404 out-of-range candidate_index {cand} for dataset {dataset} (n_test={limit})")
            if cand in exp404_global[dataset]:
                raise ValueError(f"Exp404 duplicate coordinate ({dataset}, {cand}) on line {line_num}")
            exp404_global[dataset].add(cand)

    if seen_rows != 2218:
        raise ValueError(f"Exp404 row count drift: expected 2218 rows, found {seen_rows}")
    exp404_frozen = {ds: frozenset(coords) for ds, coords in exp404_global.items()}
    if sum(len(v) for v in exp404_frozen.values()) != 2218:
        raise ValueError(f"Exp404 total automatic hard count mismatch: expected 2218, got {sum(len(v) for v in exp404_frozen.values())}")
    nonempty_count = sum(1 for v in exp404_frozen.values() if v)
    empty_count = sum(1 for v in exp404_frozen.values() if not v)
    if nonempty_count != 888 or empty_count != 229:
        raise ValueError(f"Exp404 partition drift: expected 888 nonempty / 229 empty, got {nonempty_count} / {empty_count}")

    # 6. Direct-index projection
    if selected_population is not None:
        if not selected_population:
            raise ValueError("selected_population cannot be empty")
        if selected_n_test is not None:
            for ds in selected_population:
                if ds not in selected_n_test or selected_n_test[ds] != global_n_test[ds]:
                    raise ValueError(f"selected_n_test mismatch for {ds}: expected {global_n_test[ds]}, got {selected_n_test.get(ds)}")
        return HardRouteOracleBundle(
            exp137={ds: exp137_global[ds] for ds in selected_population},
            exp236={ds: exp236_global[ds] for ds in selected_population},
            exp253={ds: pre_global[ds] for ds in selected_population},
            exact_normal_removed={ds: removed_global[ds] for ds in selected_population},
            automatic_hard={ds: auto_global[ds] for ds in selected_population},
            exp404={ds: exp404_frozen[ds] for ds in selected_population},
        )

    if selected_n_test is not None:
        for ds in pop_1117:
            if ds not in selected_n_test or selected_n_test[ds] != global_n_test[ds]:
                raise ValueError(f"selected_n_test mismatch for {ds}: expected {global_n_test[ds]}, got {selected_n_test.get(ds)}")

    return HardRouteOracleBundle(
        exp137=exp137_global,
        exp236=exp236_global,
        exp253=pre_global,
        exact_normal_removed=removed_global,
        automatic_hard=auto_global,
        exp404=exp404_frozen,
    )
```

- [ ] **Step 3: Execute Diagnostic Harness and Commit Diagnostic Report**

Create `scratch/diagnose_paper_r4_mechanism1_parity_antigravity.py` to compare intermediate stages (Exp74d, Exp89, Exp90, Exp93, Exp134, Exp137) of `hard_route_paper.py` against origin reference selectors and the SHA-bound Exp137 authority.

Run diagnostic harness:
```bash
python3 scratch/diagnose_paper_r4_mechanism1_parity_antigravity.py
```
Commit the output report as `reports/paper_r4_mechanism1_parity_diagnosis_antigravity.md` documenting:
1. Exact intermediate stages where paper replay diverges from frozen origin selectors.
2. The specific primitive functions responsible for the divergence (identified through diagnostic comparison).
3. The exact per-dataset differences between current paper output (2074) and frozen authority (2005).

- [ ] **Step 4: Lock Mechanism 1 Diagnostic Exact Primitive Assertion in `test_paper_r4_full_route_parity_recovery_antigravity.py`**

In `FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence`, assert the exact diagnosed primitive divergence discovered in Step 3 in RED before proceeding to Task 5 production code modifications.

- [ ] **Step 5: Resynchronize Codemaps and Commit Task 4**

Run shared map-resync procedure, then:
```bash
git add paper/equipment_anomaly_detection/contracts_paper.py paper/equipment_anomaly_detection/proposal_score_paper.py scratch/diagnose_paper_r4_mechanism1_parity_antigravity.py reports/paper_r4_mechanism1_parity_diagnosis_antigravity.md experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "feat(parity): implement values-only SHA-pinned oracle bundle loader and Mechanism 1 diagnostic isolation"
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
  - Replays origin-exact Exp74d through Exp137 selectors matching frozen Exp137/236 oracles by modifying only the functions identified and documented in the committed Task 4 diagnostic report (`reports/paper_r4_mechanism1_parity_diagnosis_antigravity.md`).
- Invariant: Zero dataset-name branches (`if dataset == "Phoneme_normal_39"`) or hard-coded coordinate injections. Task 5 edits only diagnosed functions.

- [ ] **Step 1: Run RED Command for Task 5**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence
```
Expected failure: `AssertionError: 49 not found` (on Phoneme_normal_39) and diagnostic isolation mismatch.

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

- [ ] **Step 3: Remove Artificial `n_train < 2` Short-Circuit and Apply Diagnosed Primitive Repairs in `hard_route_paper.py`**

In `paper/equipment_anomaly_detection/hard_route_paper.py`:
1. In `replay_hard_route`:
   Change:
   ```python
   if len(values.train_rows) < 1 or not values.test_rows:
       return empty_hard_route_result_paper(len(values.test_rows))
   ```
2. Apply the exact primitive repair diagnosed in Task 4 only to the diagnosed functions named in `reports/paper_r4_mechanism1_parity_diagnosis_antigravity.md` (no edits to undiagnosed functions).

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
   Update the undersized test case so `empty_train = _raw("Empty_normal_0", (), (_row(1.0), _row(2.0)))` tests typed empty, while `n_train=1` tests full-fit execution.

- [ ] **Step 5: Run GREEN Verification for Task 5**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_replay_signature_still_rejects_historical_coordinates
```
Expected: `Ran 3 tests in ... OK`.

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
  - Includes exactly 25 nonempty Exp404 datasets and 8 known Exp404-empty datasets (`ACSF1_normal_5`, `DodgerLoopDay_normal_1`, `DodgerLoopDay_normal_2`, `DodgerLoopDay_normal_4`, `DodgerLoopGame_normal_2`, `DodgerLoopWeekend_normal_1`, `MelbournePedestrian_normal_6`, `MelbournePedestrian_normal_8`).
- Gated Capable Integration Execution:
  - The real 33-dataset values-only smoke is an explicit gated integration command with progress logging using the capable interpreter (`/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`).
  - Validates 100% per-dataset coordinate equality against the 4 SHA-bound oracles via `HardRouteParityAuthority`:
    - Exp137: `VERIFIED_EXP137_AUTHORITY_PATH` (SHA `942e2f5f...`)
    - Exp236: `VERIFIED_EXP236_AUTHORITY_PATH` (SHA `a03ce4ea...`)
    - Exp291: `VERIFIED_EXP291_AUTHORITY_PATH` (SHA `73dff56e...`)
    - Exp404: `VERIFIED_EXP404_AUTHORITY_PATH` (SHA `437dc637...`)
  - Direct indexing `exp404[dataset]` works for all 33 smoke datasets without any `.get` fallback.
  - Smoke contract: Does NOT assert aggregate full-population counts or execute Task6/7.

- [ ] **Step 1: Execute Gated 33-Dataset Smoke Integration Command**

Run with pinned capable interpreter and progress logging:
```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_smoke_33_datasets_match_sha_bound_oracles
```
Expected: `Ran 1 test in ... OK` with all 33 datasets matching Exp137, Exp236, Exp291, and Exp404 coordinates bit-for-bit.

- [ ] **Step 2: Resynchronize Codemaps and Commit Task 6**

Run shared map-resync procedure, then:
```bash
git add experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
git commit -m "test(parity): prove 33-member coordinate-only smoke matches SHA-bound oracles"
```

---

### Task 7: Full 1,117 route-only parity gate (2005/2116/2253/35/2218 counts) before Task6/7 and Candidate C regression (PR 3)

**Files:**
- Modify: `paper/equipment_anomaly_detection/contracts_paper.py`
- Modify: `paper/equipment_anomaly_detection/runner_paper.py`
- Modify: `paper/equipment_anomaly_detection/pipeline_paper.py`
- Modify: `paper/equipment_anomaly_detection/proposal_score_paper.py`
- Modify: `experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp448_antigravity.*`
- Modify together: `paper/equipment_anomaly_detection/codemap_paper.*`

**Interfaces:**
- `verify_hard_route_parity_paper(routes: Mapping[str, HardRouteResult], *, authority: HardRouteParityAuthority, raw: Mapping[str, RawDataset] | None = None, catalog: dict[str, object] | None = None, mode: str) -> dict[str, object]`:
  - `mode` must be explicitly `"full"` or `"smoke"`.
  - Invokes `bundle = load_hard_route_oracle_bundle(authority, selected_population=verified_population, selected_n_test=n_test_by_dataset)`.
  - Enforces exact catalog key set match (all 1,117 datasets in full mode, 33 in smoke mode) and exact $n_{\text{test}}$ agreement per dataset.
  - Direct indexing without `.get(dataset, empty)` masking anywhere.
  - In full mode, verifies aggregate counts across all 1,117 datasets:
    - Exp137 Hard count == 2005
    - Exp236 Hard count == 2116
    - Exp253 Hard count == 2253 ($2218 + 35$)
    - Exact-normal removed count == 35
    - Automatic Hard count == 2218
  - Verifies 100% per-dataset coordinate equality against the 4 SHA-bound oracles.
  - Raises `ValueError` or `RuntimeError` on any count mismatch, key set discrepancy, or coordinate inequality.
- `compose_proposal_score(...)`:
  - Triggered from explicit verified `context` / `authority` passed through runner (never inferred from `len(raw)`).
  - Official context supplies explicit `mode` and `hard_route_parity_authority`.
  - Direct optional calls require both explicit `route_parity_authority` AND `route_parity_mode`.
  - Fails closed if official context (full or smoke) lacks verified `HardRouteParityAuthority`.
  - Executes `verify_hard_route_parity_paper(routes, authority=authority, raw=raw, catalog=catalog, mode=mode)` before invoking `score_low_frequency` (Task6) and `classify_stability` (Task7).
- Candidate C (`hard_plus_stable`):
  - Task6 uses pre-guard Exp253 Hard (2253).
  - Policy construction uses post-guard Automatic Hard (2218) $\cup$ stable-26 = 2244.
  - Candidate C rule, budget (30), view cap (16), and survival threshold (14/16) remain unchanged.

- [ ] **Step 1: Run RED Command for Task 7**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_full_1117_route_only_parity_gate_blocks_task6_and_task7
```
Expected failure: `ImportError: cannot import name 'verify_hard_route_parity_paper' from 'paper.equipment_anomaly_detection.proposal_score_paper'`.

- [ ] **Step 2: Implement `verify_hard_route_parity_paper` and Wire into `compose_proposal_score` and Runner**

In `paper/equipment_anomaly_detection/proposal_score_paper.py`:

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
    raw: Mapping[str, RawDataset] | None = None,
    catalog: dict[str, object] | None = None,
    mode: str,
) -> dict[str, object]:
    """Verify exact 2005/2116/2253/35/2218 counts, exact key set, and 100% per-dataset coordinate parity before Task6/7."""
    if not isinstance(authority, HardRouteParityAuthority):
        raise TypeError("HardRouteParityAuthority instance is required for parity verification")
    if mode not in ("full", "smoke"):
        raise ValueError(f"route parity verification mode must be 'full' or 'smoke', got {mode!r}")

    if raw is not None:
        verified_population = tuple(raw.keys())
        n_test_by_dataset = {k: len(v.test_rows) for k, v in raw.items()}
    elif catalog is not None:
        verified_population = tuple(entry["dataset"] for entry in catalog["datasets"])
        n_test_by_dataset = {entry["dataset"]: int(entry["test"]["count"]) for entry in catalog["datasets"]}
    else:
        verified_population = tuple(routes.keys())
        n_test_by_dataset = {k: r.n_test for k, r in routes.items()}

    bundle = load_hard_route_oracle_bundle(
        authority,
        selected_population=verified_population,
        selected_n_test=n_test_by_dataset,
    )

    route_keys = set(routes.keys())
    pop_set = set(verified_population)
    if route_keys != pop_set:
        raise ValueError(f"route parity verifier key set mismatch against verified population: {sorted(route_keys ^ pop_set)}")

    if mode == "full":
        if len(routes) != 1117:
            raise ValueError(f"Full mode route parity verifier requires exactly 1117 population datasets (got {len(routes)})")
        if (
            route_keys != set(bundle.exp137.keys())
            or route_keys != set(bundle.exp236.keys())
            or route_keys != set(bundle.automatic_hard.keys())
            or route_keys != set(bundle.exp404.keys())
        ):
            raise ValueError("route parity verifier catalog key set mismatch with oracle key sets")

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
    else:
        if len(routes) != 33:
            raise ValueError(f"Smoke mode route parity verifier requires exactly 33 smoke datasets (got {len(routes)})")
        actual_counts = {
            "exp137_hard": sum(len(r.exp137.hard) for r in routes.values()),
            "exp236_hard": sum(len(r.exp236.hard) for r in routes.values()),
            "exp253_hard": sum(len(r.exp253.hard) for r in routes.values()),
            "exact_normal_removed": sum(len(r.exact_normal_removed) for r in routes.values()),
            "automatic_hard": sum(len(r.automatic_hard) for r in routes.values()),
        }

    mismatches: list[str] = []
    for dataset, route in routes.items():
        if route.exp137.hard != bundle.exp137[dataset]:
            mismatches.append(f"{dataset}:exp137")
        if route.exp236.hard != bundle.exp236[dataset]:
            mismatches.append(f"{dataset}:exp236")
        if route.exp253.hard != bundle.exp253[dataset]:
            mismatches.append(f"{dataset}:exp253")
        if route.exact_normal_removed != bundle.exact_normal_removed[dataset]:
            mismatches.append(f"{dataset}:exact_removed")
        if route.automatic_hard != bundle.automatic_hard[dataset]:
            mismatches.append(f"{dataset}:exp291_auto")
        if route.automatic_hard != bundle.exp404[dataset]:
            mismatches.append(f"{dataset}:exp404")

    if mismatches:
        raise ValueError(
            f"hard route coordinate parity mismatch on {len(mismatches)} stages/datasets: {mismatches[:10]}"
        )

    return {"status": "parity_verified", "mode": mode, "counts": actual_counts, "datasets_verified": len(routes)}
```

In `compose_proposal_score`, gate Task6/7 execution from explicit context authority or explicit optional arguments without `len(raw)` inference:
```python
    if context is not None and "hard_route_parity_authority" in context:
        authority = HardRouteParityAuthority.from_dict(context["hard_route_parity_authority"])
        mode = context.get("mode")
        if mode not in ("full", "smoke"):
            raise ValueError(f"Official context requires explicit mode ('full' or 'smoke'), got {mode!r}")
        verify_hard_route_parity_paper(routes, authority=authority, raw=raw, catalog=catalog, mode=mode)
    elif route_parity_authority is not None or route_parity_mode is not None:
        if route_parity_authority is None or route_parity_mode is None:
            raise ValueError("Direct parity verification requires both route_parity_authority and route_parity_mode")
        if route_parity_mode not in ("full", "smoke"):
            raise ValueError(f"Explicit route_parity_mode must be 'full' or 'smoke', got {route_parity_mode!r}")
        verify_hard_route_parity_paper(routes, authority=route_parity_authority, raw=raw, catalog=catalog, mode=route_parity_mode)
    elif context is not None:
        raise ValueError(f"Official context ({context.get('mode')!r}) requires verified hard_route_parity_authority in context")
```

In `runner_paper.py` & `pipeline_paper.py`:
- `PipelineConfig` accepts `hard_route_parity_authority: HardRouteParityAuthority | None = None`.
- `context_payload` serializes `authority.to_dict()` into `context["hard_route_parity_authority"]` and hashes it into `context_sha256`.

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
git add paper/equipment_anomaly_detection/contracts_paper.py paper/equipment_anomaly_detection/runner_paper.py paper/equipment_anomaly_detection/pipeline_paper.py paper/equipment_anomaly_detection/proposal_score_paper.py experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/test_paper_r4_full_route_parity_recovery_antigravity.py docs/codemap/codemap_exp448_antigravity.* paper/equipment_anomaly_detection/codemap_paper.*
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
- Verified Hard Route Parity Authority Construction (in `scratch/manage_paper_r4_proposal_antigravity.py`):
  - Explicitly instantiates `OFFICIAL_HARD_ROUTE_PARITY_AUTHORITY = HardRouteParityAuthority(...)` from constant path and SHA strings (no `default_verified`), and binds it to `PipelineConfig`.
- `FORBIDDEN_DESTINATION_ROOTS`: Set containing all prior 20260813 and earlier roots.
- `validate_clean_root_destination_paper(path: Path) -> None`: Raises `ValueError` if `path` or its ops companion matches any forbidden prior root.

- [ ] **Step 1: Run RED Command for Task 8**

Run:
```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_clean_root_execution_and_rejection_of_live_r4_reuse experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations
```
Expected failure: `ImportError: cannot import name 'AUTHORIZED_SMOKE_ROOT_20260814'` or `validate_clean_root_destination_paper`.

- [ ] **Step 2: Update Launcher Paths, Smoke Datasets, Authority Binding, and Destination Validation in `manage_paper_r4_proposal_antigravity.py`**

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

EXP137_PATH = Path("/Users/mac/Documents/Dataset/experiments/exp234_recovery_antigravity/experiment_137_operational_triage_results_antigravity.csv")
EXP137_SHA256 = "942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d"

EXP236_PATH = Path("/Users/mac/Documents/Dataset/experiments/exp236_direct_run_antigravity/exp236_frozen_predictions.jsonl")
EXP236_SHA256 = "a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524"

EXP291_PATH = Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl")
EXP291_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"

EXP404_PATH = Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl")
EXP404_SHA256 = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"

OFFICIAL_HARD_ROUTE_PARITY_AUTHORITY = HardRouteParityAuthority(
    exp137_path=EXP137_PATH,
    exp137_sha256=EXP137_SHA256,
    exp236_path=EXP236_PATH,
    exp236_sha256=EXP236_SHA256,
    exp291_path=EXP291_PATH,
    exp291_sha256=EXP291_SHA256,
    exp404_path=EXP404_PATH,
    exp404_sha256=EXP404_SHA256,
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
2. Update default smoke and launch commands to target authorized 20260814 clean paths, pass `hard_route_parity_authority=OFFICIAL_HARD_ROUTE_PARITY_AUTHORITY` to `PipelineConfig`, and call `validate_clean_root_destination_paper`.

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
- Provenance: MLX/MPS assessment, logical CPU worker policy (`dataset_workers = max(1, logical - 1)`), clean root isolation, zero label leakage, and exact target counts (2005/2116/2253/35/2218, Candidate C = 2244) fully validated.

- [ ] **Step 1: Execute Complete Antigravity Full Route Parity Test Suite**

Run under capable interpreter:
```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests
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
Expected: Working tree clean.
