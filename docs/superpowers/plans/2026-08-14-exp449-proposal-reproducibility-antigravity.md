# Exp449 Proposal Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned Exp449 label-free proposal pipeline that freezes the current Task4–Task7/Candidate C algorithm, survives per-dataset failures, proves independent smoke A/B reproducibility, and produces an independently verifiable proposal without historical-coordinate gates.

**Architecture:** Add a new namespace-package boundary under `experiments/exp449_reproducible_rebaseline_antigravity/` that orchestrates existing paper scientific primitives without changing them. The versioned runner inserts an exact catalog-bijection gate between dataset-local Task4/5 and population-global Task6/7, while versioned operations, manifests, and verifiers bind clean Exp449 roots and forbid Exp448 artifacts. Existing `paper/equipment_anomaly_detection/` remains the scientific implementation; Exp448 files and intentional RED tests remain immutable evidence.

**Tech Stack:** Python 3.14.6, NumPy, SQLite, existing paper Task4–Task9 modules, unittest, strict canonical JSON, SHA-256, create-only filesystem publication.

## Global Constraints

- The active new scientific version is `Exp449`; never overwrite the Exp448 codemap triplet or artifacts.
- Every newly created filename contains `antigravity`; do not add `__init__.py` files.
- Task4 TRAIN-only preprocessing and Task5 label-free TRAIN+TEST-value-transductive length selection remain distinct frozen behaviors.
- Do not change Task4–Task7 algorithms, thresholds, seeds, routes, Candidate C, or serialized score definitions.
- Do not call `load_hard_route_oracle_bundle`, `load_acceptance`, `verify_parity`, or any Exp448 prediction/replay authority from Exp449 production paths.
- Historical coordinates and counts are diagnostic only and are not Exp449 success criteria.
- CPU-parallel dataset stages use `max(1, os.cpu_count() - 1)` workers and record detected and actual counts.
- Assess PyTorch, MLX, and MPS; accelerated scoring remains noncanonical unless coordinates are identical, score delta is at most `1e-5`, and end-to-end speedup is at least `1.1x` including transfer, compilation, and synchronization.
- Run only one MLX/MPS-heavy process at a time; never combine an outer process pool with concurrent MLX/MPS work unless a separately preregistered parity and end-to-end benchmark passes.
- Same-CPU smoke A/B requires identical coordinates and continuous-score maximum absolute difference at most `1e-12`.
- TEST labels remain inaccessible through proposal freeze and independent verification.
- No smoke or official run root is created by implementation tasks in this plan.
- Do not run the 23-dataset smoke or 1,117-dataset official experiment without a later explicit execution authorization.

## File Structure

- `experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py`: frozen Exp449 IDs, roots, smoke inventory, tolerances, and policy identity.
- `experiments/exp449_reproducible_rebaseline_antigravity/contracts_antigravity.py`: immutable context, outer manifest, terminal status, and smoke-comparison contracts.
- `experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py`: Task4/5 collection, exact catalog gate, Task6/7/Candidate C composition, and freeze wrapper.
- `experiments/exp449_reproducible_rebaseline_antigravity/artifacts_antigravity.py`: Exp449-only typed shard publication, manifest freeze, and strict antigravity filename inventory.
- `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`: fixed-root readiness, two-wave retry lifecycle, status, and execution refusal gates.
- `experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py`: scorer-free proposal-root and smoke A/B verification.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_config_contracts_antigravity.py`: frozen identity and defensive-contract tests.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py`: catalog gate, failure batching, retry terminal, and Task6/7 blocking tests.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_manifest_verifier_antigravity.py`: outer manifest, tamper, extra/missing inventory, and smoke comparison tests.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_readiness_antigravity.py`: root safety, dependency, acceleration, label firewall, and no-execution tests.
- `scratch/run_exp449_reproducible_rebaseline_antigravity.py`: thin CLI entrypoint; import-safe and inactive unless an explicit subcommand is invoked.
- `docs/EXP449_PREREGISTRATION_antigravity_KO.md`: frozen scientific and reproducibility declaration.
- `docs/EXP449_RUNBOOK_antigravity_KO.md`: later execution commands and explicit authorization gates.
- `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`: implementation evidence only; no run-result claims.
- `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`: complete Exp449 map triplet.

---

### Task 1: Establish Exp449 State and Code Map Before Code Changes

**Files:**
- Modify: `RESEARCH_STATE.json`
- Create: `docs/codemap/codemap_exp449_antigravity.json`
- Create: `docs/codemap/codemap_exp449_antigravity.html`
- Create: `docs/codemap/codemap_exp449_antigravity.lock`
- Test: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_config_contracts_antigravity.py`

**Interfaces:**
- Consumes: approved design commit `26830622`; current Exp448 and paper codemap structures.
- Produces: active experiment `Exp449`, versioned map node IDs, and a lock whose `sha256-v1` fingerprints match the current tracked tree outside excluded directories.

- [ ] **Step 1: Write the failing version-boundary test**

```python
class Exp449VersionBoundaryTests(unittest.TestCase):
    def test_exp449_state_and_triplet_exist_without_mutating_exp448(self):
        state = json.loads(Path("RESEARCH_STATE.json").read_text())
        self.assertEqual(state["active_experiment"], "Exp449")
        self.assertEqual(state["status"], "design_approved_implementation_planned")
        for suffix in ("html", "json", "lock"):
            self.assertTrue(Path(f"docs/codemap/codemap_exp449_antigravity.{suffix}").is_file())
        frozen = exp448_frozen_blob_inventory_at_design_commit("26830622")
        self.assertTrue(frozen)
        self.assertEqual(current_exp448_blob_inventory(), frozen)
```

The frozen inventory is the sorted tracked path/blob-hash mapping at design commit `26830622` for the complete `experiments/exp448_exp447_text_label_boundary_repair_antigravity/` tree, all three `docs/codemap/codemap_exp448_antigravity.*` files, and the intentional RED full-route test. The comparison hashes current bytes and rejects added, removed, or changed paths; checking only file existence is insufficient. Repeat this assertion in Task 8 before completion.

- [ ] **Step 2: Run the test and observe the expected failure**

Run:

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_config_contracts_antigravity.Exp449VersionBoundaryTests.test_exp449_state_and_triplet_exist_without_mutating_exp448 -v
```

Expected: FAIL because Exp449 state and triplet are absent.

- [ ] **Step 3: Update research state with an explicit non-execution boundary**

Add an `exp449_reproducible_rebaseline_antigravity` object whose exact initial values are:

```json
{
  "experiment": "Exp449",
  "status": "design_approved_implementation_planned",
  "design": "docs/superpowers/specs/2026-08-14-exp449-reproducible-rebaseline-antigravity-design.md",
  "plan": "docs/superpowers/plans/2026-08-14-exp449-proposal-reproducibility-antigravity.md",
  "dataset_count": 1117,
  "proposal_policy": "hard_plus_stable",
  "historical_coordinate_gate": false,
  "smoke_started": false,
  "official_run_started": false,
  "test_labels_opened": false,
  "official_label_access_count": 0,
  "next_action": "implement_exp449_without_execution"
}
```

Set the top-level `active_experiment`, `active_phase`, and `status` to `Exp449`, `implementation_planned`, and `design_approved_implementation_planned`. Point `active_documents.design` and `active_documents.plan` to the two Exp449 documents. Do not change historical Exp448 sections.

- [ ] **Step 4: Create all three Exp449 map files together**

Use the current Exp448 JSON/HTML as a structural seed, change schema metadata to Exp449, add planned nodes for every file in this plan, retain existing paper dependency nodes, and add flows for:

```text
Exp449 readiness -> Task4/5 dataset-local collection -> exact catalog gate
-> Task6 -> Task7 -> Candidate C -> typed shards -> Exp449 outer manifest
-> independent verifier
```

The lock must use the documented `sha256-v1` path/length/content framing and exclude `.git`, `.worktrees`, `docs/codemap`, virtual environments, caches, outputs, and results. Its metadata records the current commit and dirty status at generation time.

- [ ] **Step 5: Validate structural and fingerprint integrity**

Run:

```bash
python3 -m json.tool docs/codemap/codemap_exp449_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp449_antigravity.lock >/dev/null
python3 - <<'PY'
import json, re
from pathlib import Path
j = json.loads(Path('docs/codemap/codemap_exp449_antigravity.json').read_text())
h = Path('docs/codemap/codemap_exp449_antigravity.html').read_text()
m = re.search(r'<script id="codemap-data" type="application/json">(.*?)</script>', h, re.S)
assert m and json.loads(m.group(1)) == j
ids = {node['id'] for node in j['nodes']}
assert len(ids) == len(j['nodes'])
assert all(edge['from'] in ids and edge['to'] in ids for edge in j['edges'])
assert all(set(flow['steps']) <= ids for flow in j['flows'])
print('exp449-map-ok', len(j['nodes']), len(j['edges']), len(j['flows']))
PY
```

- [ ] **Step 6: Run the version-boundary test and commit**

Expected: PASS.

```bash
git add RESEARCH_STATE.json \
  docs/codemap/codemap_exp449_antigravity.html \
  docs/codemap/codemap_exp449_antigravity.json \
  docs/codemap/codemap_exp449_antigravity.lock \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_config_contracts_antigravity.py
git commit -m "chore(exp449): establish reproducible rebaseline boundary"
```

### Task 2: Freeze Exp449 Configuration and Immutable Contracts

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/contracts_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_config_contracts_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: `FrozenTask8BackendDecision`, strict canonical JSON, the fixed 23-dataset smoke inventory.
- Produces: `FROZEN_EXP449_CONFIG`, `Exp449Context.from_verified_inputs(...)`, `Exp449ProposalManifest`, `Exp449TerminalStatus`, and `SmokeComparison`.

- [ ] **Step 1: Add failing contract tests**

Test exact constants, tuple sorting/uniqueness, roots, tolerances, policy ID, dataset count, defensive copying, finite numeric values, exact schema fields, and rejection of historical acceptance paths.

```python
def test_frozen_config_is_exact():
    self.assertEqual(FROZEN_EXP449_CONFIG.experiment_id, "Exp449")
    self.assertEqual(FROZEN_EXP449_CONFIG.dataset_count, 1117)
    self.assertEqual(FROZEN_EXP449_CONFIG.policy_id, "hard_plus_stable")
    self.assertEqual(FROZEN_EXP449_CONFIG.cpu_repeat_tolerance, 1e-12)
    self.assertEqual(FROZEN_EXP449_CONFIG.accelerated_tolerance, 1e-5)
    self.assertEqual(FROZEN_EXP449_CONFIG.minimum_accelerated_speedup, 1.1)
    self.assertEqual(len(FROZEN_EXP449_CONFIG.smoke_datasets), 23)
    self.assertIn("MelbournePedestrian_normal_10", FROZEN_EXP449_CONFIG.smoke_datasets)
    self.assertEqual(FROZEN_EXP449_CONFIG.preprocessing_contracts, (
        "task4_train_only",
        "task5_label_free_train_test_value_length_transductive",
    ))

def test_context_rejects_historical_parity_authority():
    with self.assertRaisesRegex(ValueError, "historical authority"):
        Exp449Context.from_verified_inputs(
            run_root=Path("/tmp/exp449_fixture_antigravity"),
            mode="smoke",
            catalog={},
            acceleration={},
            acceptance_path=Path("old.json"),
        )
```

`from_verified_inputs` must reject every non-`None` `acceptance_path` before inspecting the other inputs, so this test cannot accidentally reach a historical authority.

- [ ] **Step 2: Run tests and observe import failures**

Run:

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_config_contracts_antigravity -v
```

Expected: FAIL because config and contracts modules are absent.

- [ ] **Step 3: Implement the frozen config**

Define a frozen dataclass with these exact fields:

```python
@dataclass(frozen=True, slots=True)
class FrozenExp449Config:
    experiment_id: str
    dataset_count: int
    test_row_count: int
    policy_id: str
    cpu_repeat_tolerance: float
    accelerated_tolerance: float
    minimum_accelerated_speedup: float
    maximum_waves: int
    preprocessing_contracts: tuple[str, str]
    smoke_datasets: tuple[str, ...]
    smoke_a_scientific_root: Path
    smoke_a_operations_root: Path
    smoke_b_scientific_root: Path
    smoke_b_operations_root: Path
    official_scientific_root: Path
    official_operations_root: Path
```

Set `test_row_count=195900`, `maximum_waves=2`, and use the 23 sorted names from `scratch/manage_paper_r4_proposal_antigravity.py:117-141`. Freeze these six exact, independently absent paths from the approved design: smoke A scientific and operations, smoke B scientific and operations, and official scientific and operations. Do not derive or accept caller-supplied alternatives.

- [ ] **Step 4: Implement immutable exact-schema contracts**

`Exp449Context` must bind experiment ID, mode, resolved run root, database and catalog hashes, code closure, lineage/codemap hashes, interpreter/environment, backend record, seeds/constants, and `historical_parity_status="not_used"`. `Exp449ProposalManifest` must bind every shard path/size/hash/schema plus policy coordinate digests and Candidate C score digests. `Exp449TerminalStatus` permits only `dataset_local_complete` and `proposal_score_incomplete`. `SmokeComparison` records both manifest hashes, compared dataset count, coordinate identity, maximum score delta, and `passed`.

Every mapping and sequence is copied into immutable containers in `__post_init__`; every SHA is exactly 64 lowercase hexadecimal characters; all counts are exact nonnegative integers; all floats are finite.

- [ ] **Step 5: Run tests, synchronize the Exp449 triplet, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_config_contracts_antigravity -v
git diff --check
git add experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/contracts_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_config_contracts_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): freeze proposal identities and contracts"
```

### Task 3: Add the Exact Dataset-Local Completion Gate

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: paper `_collect_task4_task5`, `ProposalScore`, `score_low_frequency`, `classify_stability`, `build_hard_evidence_surface`, and `build_policies` primitives. It does not consume any filesystem writer.
- Produces: `verify_dataset_local_bijection_antigravity(...) -> tuple[str, ...]` and `compose_exp449_proposal_score_antigravity(...) -> ProposalScore`.

- [ ] **Step 1: Write failing gate tests**

Cover missing, duplicate, extra, catalog-order drift, source/route mismatch, full-mode count not equal to 1,117, and a success case. Patch Task6/7 functions to raise and prove they are not called when the gate fails.

```python
def test_missing_success_blocks_task6_and_task7(self):
    with patch.object(runner, "score_low_frequency", side_effect=AssertionError("Task6 called")), \
         patch.object(runner, "classify_stability", side_effect=AssertionError("Task7 called")):
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.compose_exp449_proposal_score_antigravity(
                raw, acceleration, root=root, context=context,
                catalog=full_catalog, collector=fake_collector_missing_one,
            )
```

- [ ] **Step 2: Run tests and verify missing API failures**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_runner_recovery_antigravity -v
```

- [ ] **Step 3: Implement the pure bijection verifier**

```python
def verify_dataset_local_bijection_antigravity(
    catalog: dict[str, object],
    raw: Mapping[str, RawDataset],
    sources: Mapping[str, HardSourceBundle],
    routes: Mapping[str, HardRouteResult],
    *,
    mode: str,
) -> tuple[str, ...]:
    expected = tuple(entry["dataset"] for entry in catalog["datasets"])
    if expected != tuple(sorted(set(expected))):
        raise ValueError("Exp449 catalog bijection differs")
    if tuple(raw) != expected or tuple(sources) != expected or tuple(routes) != expected:
        raise ValueError("Exp449 dataset-local catalog bijection differs")
    if mode == "full" and len(expected) != 1117:
        raise ValueError("Exp449 full catalog count differs")
    return expected
```

- [ ] **Step 4: Implement the versioned composition order**

The function validates the label firewall, derives `FrozenTask8BackendDecision`, calls the collector, sorts returned mappings into catalog order, runs the exact gate, then and only then calls Task6, Task7, hard evidence, and `build_policies`. It constructs the existing immutable `ProposalScore`. It must not import or call any historical oracle/acceptance/parity API.

- [ ] **Step 5: Prove `ProposalScore` semantic parity with existing composition on a complete fixture**

Use a small typed fixture and patch the expensive scientific functions. Compare every in-memory `ProposalScore` field produced by existing and Exp449 composition. Do not call a paper or Exp449 filesystem writer in this task; typed-shard payload parity begins only after `artifacts_antigravity.py` is introduced in Task 5. The only intended composition difference is the pre-Task6 explicit gate.

- [ ] **Step 6: Run focused and paper regression tests, update map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_runner_recovery_antigravity \
  paper.equipment_anomaly_detection.tests_paper.test_low_frequency_paper \
  paper.equipment_anomaly_detection.tests_paper.test_stability_paper -v
git diff --check
git add experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): gate global scoring on exact local coverage"
```

### Task 4: Enforce Two-Wave Recovery and Terminal Semantics

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: verified Task4/5 success and attempt namespaces, `ProposalScoreIncomplete`, fixed Exp449 roots, `compose_exp449_proposal_score_antigravity`.
- Produces: `run_exp449_wave_antigravity(config) -> Exp449TerminalStatus`, `status_exp449_antigravity(config) -> Exp449TerminalStatus`, and label-free timestamped progress events containing wave/completed/failed/pending/elapsed/ETA only.

- [ ] **Step 1: Write failing interrupted/resume tests**

Test an initial wave with one failure, a second wave that succeeds, and a second wave that fails terminally. Assert successful datasets are revalidated and not recomputed, attempt records never count as success, Task6/7 never run while incomplete, and a third call is rejected without scoring.

- [ ] **Step 2: Observe RED**

Run the focused module and require failures for missing operations APIs.

- [ ] **Step 3: Implement exact two-wave state derivation**

Read and verify all success and attempt records before scheduling. Derive the next wave as `max(attempt.wave, default=0) + 1`. Permit waves 1 and 2 only. On an incomplete second wave, create one terminal create-only artifact named `proposal_score_incomplete_antigravity.json` containing context hash, catalog hash, verified success tuple, failed tuple, never-attempted tuple, attempt-ledger hash, and record hash.

```python
if next_wave > FROZEN_EXP449_CONFIG.maximum_waves:
    return verify_terminal_incomplete_antigravity(config)
```

No terminal artifact is written for wave 1. A complete catalog writes no incomplete terminal and may proceed to population-global composition.

- [ ] **Step 4: Add exact resume reconciliation**

At every call, require the on-disk success tokens to be a subset of the catalog, re-read every record, verify typed payload and raw identity, and reject extras, duplicates, half-published files, symlinks, or schema drift. Compute the pending queue as `catalog - verified_success`, never from attempt records.

Emit operational progress through the existing result-neutral callback and one append-only operations JSONL owned outside the scientific root. Each event has exact keys `timestamp`, `experiment`, `mode`, `wave`, `completed`, `failed`, `pending`, `elapsed_seconds`, `eta_seconds`, and optional `dataset`; reject labels, scores, p-values, ranks, and coordinates. Logging exceptions are swallowed only by the logging boundary and can never publish scientific success.

- [ ] **Step 5: Verify byte-identical interrupted/resumed output**

With deterministic fixtures, compare uninterrupted and fail-once/resumed final typed proposal shards and outer scientific payloads byte-for-byte after excluding root-bound context fields explicitly identified by schema.

- [ ] **Step 6: Run tests, synchronize map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_runner_recovery_antigravity -v
git add experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): bound retries and terminal recovery state"
```

### Task 5: Add Exp449 Outer Manifest and Scorer-Free Verification

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_manifest_verifier_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: in-memory `ProposalScore`, the frozen paper typed-payload schema as a scientific compatibility contract, Exp449 context, and catalog. It does not call the paper filesystem writer because that writer publishes non-Exp449 filenames.
- Produces: `canonical_float64_bytes_antigravity(values) -> bytes`, `freeze_exp449_proposal_antigravity(...) -> Exp449ProposalManifest`, and `verify_exp449_proposal_antigravity(...) -> Exp449ProposalManifest`.

- [ ] **Step 1: Write failing manifest and tamper tests**

Cover exact 1,117/full and smoke inventories, relative-path confinement, file size/hash/schema, context/catalog/backend identities, per-policy coordinate digests, Candidate C score digest, extra files, missing files, altered score, altered alert, symlink swap, old Exp448 root rejection, and rejection of every runtime filename that lacks `antigravity`. Add a fixture parity test proving the Exp449 serializer preserves the paper typed shard's scientific payload and score semantics while changing only schema/version-bound envelope fields and filenames.

- [ ] **Step 2: Observe RED for absent verifier**

- [ ] **Step 3: Implement create-only outer freeze**

Define the score-digest primitive explicitly so every platform hashes the same little-endian bytes:

```python
def canonical_float64_bytes_antigravity(values: object) -> bytes:
    array = np.asarray(values, dtype=np.dtype("<f8"))
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Exp449 score vector differs")
    return np.ascontiguousarray(array).tobytes(order="C")
```

Implement an Exp449-only create-only writer. It serializes the frozen typed proposal payload directly from `ProposalScore`, validates the exact policy/alert/score invariants, and writes only `<dataset_digest>_antigravity/<dataset>_proposal_shard_antigravity.json`. It never creates `proposal_manifest_paper.json` or `<dataset>_paper.json`. After all Exp449 shards verify, build one canonical manifest entry:

```python
{
    "dataset": dataset,
    "relative_path": relative_path,
    "size": size,
    "sha256": shard_sha,
    "schema": shard["schema"],
    "policy_coordinate_sha256": coordinate_hashes,
    "candidate_c_score_sha256": sha256(canonical_float64_bytes_antigravity(scores)).hexdigest(),
}
```

The top-level schema is `exp449-proposal-manifest-antigravity.v1`; it binds the Exp449 context hash, catalog hash, backend record, ordered entries, and `historical_parity_status="not_used"`. Publish as `proposal_manifest_antigravity.json` with a strict create-only helper. Do not create or bind a paper manifest.

- [ ] **Step 4: Implement a scorer-free independent verifier**

The verifier imports Exp449 contracts/artifact validation, catalog validation, and canonical JSON only. Static tests reject imports from `proposal_score_paper`, source modules, `hard_route_paper`, paper filesystem writers, or any Exp448 module. It rehashes the outer manifest and every declared file, validates the exact root namespace, invokes the Exp449 shard semantic validator, recomputes coordinate and score digests, and rejects any undeclared or non-`antigravity` file.

- [ ] **Step 5: Prove the label capability is not imported**

Patch `builtins.__import__` to raise on any module containing `label_access`; verify a complete fixture root successfully.

- [ ] **Step 6: Run tests, map synchronization, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_manifest_verifier_antigravity \
  paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -v
git add experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/artifacts_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_manifest_verifier_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): freeze and verify proposal authority"
```

### Task 6: Add Independent Smoke A/B Reproducibility Verification

**Files:**
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_manifest_verifier_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: two independently verified `Exp449ProposalManifest` values.
- Produces: `compare_exp449_smokes_antigravity(root_a, root_b, authority) -> SmokeComparison`.

- [ ] **Step 1: Write failing A/B comparison tests**

Cover distinct roots, no hard links, no checkpoint reuse, identical catalog/raw identities, identical environment/backend/seed/scientific closure, exact policy coordinate equality, finite scores, `max_abs <= 1e-12`, score length equality, tie preservation, and one-coordinate/one-score mismatch failures.

- [ ] **Step 2: Observe RED**

- [ ] **Step 3: Implement comparison after complete independent verification**

Verify both roots fully before opening cross-run entries. Reject equal roots, symlinks, shared inodes for scientific files, or any manifest declaring the other root. Compare all four policy alerts and scores in catalog order; Candidate C is mandatory and no root-bound context field participates in scientific equality.

```python
delta = float(np.max(np.abs(scores_a - scores_b))) if scores_a.size else 0.0
if delta > FROZEN_EXP449_CONFIG.cpu_repeat_tolerance:
    raise ValueError("Exp449 smoke score reproducibility differs")
```

- [ ] **Step 4: Add a create-only comparison artifact**

Write `smoke_comparison_antigravity.json` only in the frozen smoke B operations root after both scientific roots verify. Bind both scientific-manifest hashes, both operations-root identities, coordinate digests, maximum score delta, and `passed=true`. Never write into either scientific root or smoke A operations root.

- [ ] **Step 5: Run tests, synchronize map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_manifest_verifier_antigravity -v
git add experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_manifest_verifier_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): verify independent smoke reproducibility"
```

### Task 7: Add Readiness, Acceleration, Root-Safety, and Label-Firewall Gates

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_readiness_antigravity.py`
- Create: `scratch/run_exp449_reproducible_rebaseline_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: paper acceleration manifest/verifier, fixed Exp449 config, context closure, root safety, dependency preflights.
- Produces: `readiness_exp449_antigravity(reviewed_commit) -> dict[str, object]` and an import-safe CLI with `readiness`, `status`, `verify-proposal`, and later-authorized execution subcommands.

- [ ] **Step 1: Write failing readiness tests**

Require clean reviewed commit, exact capable interpreter, database/catalog identities, 12→11 detected worker policy on the fixture host abstraction, PyTorch/MLX/MPS facts, CPU canonical fallback, transfer/compile/synchronization-inclusive benchmark facts, the one-heavy-process/no-outer-pool rule, exact pycatch22/Hydra preflights, all six exact absent non-symlink roots, no label module import, no run-root creation, and refusal of every known Exp448/r4 root.

Also require the frozen context to disclose both `task4_train_only` and `task5_label_free_train_test_value_length_transductive`; a context that collapses both to TRAIN-only fails readiness.

- [ ] **Step 2: Observe RED**

- [ ] **Step 3: Implement readiness as a read-only operation**

It verifies the reviewed 40-hex commit equals HEAD, worktree is clean, recomputes the Exp449 context and closure, calls dependency and acceleration verifiers, checks roots with `lstat` without creating them, confirms label-module absence from `sys.modules`, and returns canonical JSON-ready facts. It must not call any scoring function.

- [ ] **Step 4: Implement the thin CLI without launching experiments**

The entrypoint imports `operations_antigravity.main` only under `if __name__ == "__main__"`. `readiness`, `status`, and `verify-proposal` are usable. `run-smoke-a`, `run-smoke-b`, and `run-official` require both a reviewed commit and a separate create-only execution-authority artifact; without it they fail before root creation.

- [ ] **Step 5: Prove no execution side effect**

Run readiness against temporary absent roots and assert all remain absent. Patch scoring and label functions to raise if called. Test every historical destination refusal.

- [ ] **Step 6: Run focused and regression tests, update map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_readiness_antigravity \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_runner_recovery_antigravity \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_manifest_verifier_antigravity -v
python3 -m unittest discover \
  -s paper/equipment_anomaly_detection/tests_paper -p 'test_*_paper.py' -v
git add experiments/exp449_reproducible_rebaseline_antigravity/tests/test_readiness_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): gate proposal execution on readiness"
```

### Task 8: Freeze Documentation, Full Verification, and Independent Review

**Files:**
- Create: `docs/EXP449_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP449_RUNBOOK_antigravity_KO.md`
- Create: `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: all Task1–Task7 APIs, test evidence, and final clean commit.
- Produces: reviewed implementation-ready state; no smoke or official execution authorization.

- [ ] **Step 1: Write exact preregistration and runbook**

The preregistration repeats the approved scientific definition, no historical gate, 23-dataset smoke inventory, A/B criteria, tolerances, worker policy, acceleration policy, two-wave recovery, 1,117 exact gate, and label firewall. The runbook contains exact commands but marks smoke A, smoke B, and official execution as separately unauthorized until a fresh review and user approval.

- [ ] **Step 2: Write an evidence-only implementation report**

Record commits, interpreter/package facts, test commands and counts, map/closure hashes, expected roots still absent, and `labels_opened=false`. Do not claim smoke, official, AUROC, AP, or scientific results.

- [ ] **Step 3: Run the complete non-execution verification set**

```bash
python3 -m unittest discover \
  -s experiments/exp449_reproducible_rebaseline_antigravity/tests \
  -p 'test_*_antigravity.py' -v
python3 -m unittest discover \
  -s paper/equipment_anomaly_detection/tests_paper \
  -p 'test_*_paper.py' -v
python3 -m unittest discover \
  -s experiments/paper_public_baselines/tests \
  -p 'test_*.py' -v
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp449_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp449_antigravity.lock >/dev/null
git diff --check
```

The known Exp448 Task4 intentional RED is not included in these green suites and is not modified.

- [ ] **Step 4: Obtain fresh independent result-integrity review**

The reviewer checks no historical oracle enters Exp449, no Task6/7 runs on partial coverage, failure recovery is context-bound, smoke A/B cannot share records, root creation is unauthorized, acceleration facts are accurate, labels remain unopened, and all map fingerprints match.

- [ ] **Step 5: Fix only blocker/high/medium findings, rerun verification, and commit**

```bash
git add docs/EXP449_PREREGISTRATION_antigravity_KO.md \
  docs/EXP449_RUNBOOK_antigravity_KO.md \
  docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md \
  RESEARCH_STATE.json docs/codemap/codemap_exp449_antigravity.*
git commit -m "docs(exp449): preregister proposal readiness"
```

Update state to `implementation_verified_awaiting_smoke_authorization`. Keep `smoke_started=false`, `official_run_started=false`, `test_labels_opened=false`, and all six new roots absent. Recompute the design-commit Exp448 blob inventory and require exact equality before making this state transition.

## Execution Gates Outside This Plan

Implementation completion does not authorize execution. After Task 8 and a clean independent SHIP review:

1. request explicit smoke A authorization;
2. execute and verify smoke A;
3. request explicit smoke B authorization;
4. execute and verify smoke B;
5. compare A/B and require exact coordinate identity plus score delta `<=1e-12`;
6. request explicit 1,117 official authorization;
7. execute at most two dataset-local waves;
8. freeze and independently verify the proposal;
9. only then proceed to the separate public-baseline/evaluation plan and its label gate.
