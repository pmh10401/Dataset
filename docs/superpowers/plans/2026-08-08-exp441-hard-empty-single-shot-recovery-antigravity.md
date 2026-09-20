# Exp441 Hard-Empty Single-Shot Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, label-free Exp441 selector that deterministically replays the frozen Exp440 provisional pool, adds at most one candidate to each Hard-empty dataset and 30 candidates globally, then reproduces the already-observed TEST evaluation only after a new prediction freeze.

**Architecture:** Prediction is a short, read-only replay pipeline: verify every frozen Exp440 checkpoint/publication byte and the frozen Exp404 Hard source, parse 1,117 records with 11 CPU workers, reduce candidates through a pure selector, and publish six create-only review artifacts. Evaluation lives behind a lazy import, opens only descriptor-bound TEST identities and labels after publication inspection, and uses one fully pinned NumPy/PCG64 bootstrap implementation; no prediction module imports the evaluator or opens SQLite.

**Tech Stack:** Python 3.14.6, NumPy 2.4.6, SQLite descriptor-bound read-only evaluation, `unittest`, `ProcessPoolExecutor` with 11 record workers, canonical JSON/JSONL, SHA256, Darwin `renamex_np(source, destination, RENAME_EXCL)`, optional environment inspection of PyTorch 2.13.0, MLX 0.32.0, mlx-metal 0.32.0, and MPS availability without tensor execution.

## Global Constraints

- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Experiment identity is `Exp441`; run identity is `exp441-hard-empty-single-shot-recovery-antigravity-20260808`.
- The final reviewed design source is `docs/superpowers/specs/2026-08-08-exp441-hard-empty-single-shot-recovery-antigravity-design.md` in the same commit as this plan, with exact SHA256 `c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da`; the older `99af339a` revision is historical only.
- Run root is `/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808`.
- Prediction inputs are limited to the frozen Exp440 checkpoint/publication, frozen Exp404 Hard prediction/manifest, the current clean reviewed Exp441 code/config, and no other source.
- Prediction never opens SQLite, TEST labels, anomaly locations, Exp440 evaluation, past family TEST performance, or a label-capable module.
- Evaluation is a separate lazy-import command and may read only TEST `instance_index,label` after prediction publication is frozen and replayed.
- Exp440 checkpoint context/state SHA256 values are `d598a159d798f0f1dbdcbf684ef7994c89c982b60bce62b877a82c7e04ba7dfc` and `486235b70856247912a02c115748e22477ea11224d3a50de088187903da56268`.
- Exp440 checkpoint retained-inventory SHA256 is `c31c0414dced3f463d187e8e17adf84a50eb0c7cf86a6e8399f9074fda94ccf2`; it is integrity-checked but never used for selection.
- Exp440 publication prediction/manifest/artifact-map SHA256 values are `06d905d9d6f96fa67704111b984ed87b74229e13fd0690401db67a4a0fdd79af`, `3778a0e8770e06f4be87d1b87281c858518169c377ee6c53b78f32ce01d39e85`, and `7a083fa8f1287205d3469e36f2f88b95f72adecd90b3b79167aad673b3439a1a`.
- Exp404 Hard prediction/manifest/coordinate-identity SHA256 values are `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`, `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8`, and `858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322`.
- Hard identity and Exp441 selected identity are distinct: Hard uses Exp416's coordinate-sorted, newline-free compact JSON array of `[dataset,index]` pairs and must hash to `858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322`; selected coordinates use coordinate-sorted compact JSONL objects `{"dataset":"...","instance_index":n}` with one newline per row and must hash to `db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`.
- Database SHA256 is `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`; prediction must not open it.
- Fixed database path is `/Users/mac/Documents/Dataset/univariate_ts.db`.
- Fixed Exp440 checkpoint root is `/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity`, containing only `context_antigravity.json`, `state_antigravity.json`, `retained_candidates_antigravity.json`, and `records_antigravity/{0000..1116}_record_antigravity.json`.
- Fixed Exp440 publication root is `/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/publication_antigravity`; require exactly the artifact map plus its nine declared regular files and replay all ten bytes.
- Fixed Hard paths are `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` and `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json`.
- Prediction/readiness forbidden paths are the entire `/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/evaluation_antigravity` subtree and `experiments/exp440_train_only_low_frequency_precision_gate_antigravity/evaluation_antigravity.py`; resolved-path allowlisting fails closed.
- The exact Exp440 publication filenames are `abstention_ledger_antigravity.jsonl`, `artifact_sha256_antigravity.json`, `backend_decision_antigravity.json`, `dataset_summary_antigravity.jsonl`, `depth_evidence_antigravity.jsonl`, `manifest_antigravity.json`, `prediction_antigravity.jsonl`, `retained_candidate_ranking_antigravity.jsonl`, `status_antigravity.json`, and `train_exclusion_audit_antigravity.jsonl`; each path is the fixed publication root joined with that basename, with no recursive or alternate path resolution.
- Expected label-free counts are exactly `1117/840/394/73/49/30` for datasets/provisional/outside-Hard/Hard-empty eligible/eligible datasets/retained candidates.
- Selection admits only datasets with zero frozen Hard coordinates, retains one candidate per dataset, ranks by frozen raw Python floats, and uses a hard global budget of 30.
- Dataset-local order is `(-second_excess_ratio, second_empirical_rank, instance_index)`; global order is `(-second_excess_ratio, second_empirical_rank, dataset, instance_index)`.
- `agreement_count` is eligibility evidence only and never a ranking key; no threshold, alpha, depth, source, or budget is retuned.
- Never call Exp440 `ProvisionalCandidateAntigravity.ranking_key` or `validate_retained_candidates_antigravity`; both encode Exp440's agreement-first/budget-100 policy.
- Official CPU workers are `max(1, logical CPUs - 1)`, expected exactly 11 on this 12-logical-CPU Mac; each worker has one numerical thread.
- The nearest worktree `AGENTS.md` and the user's later max-minus-one instruction govern Exp441 over the ancestor checkout's older seven-worker fallback. There is no seven-worker fallback or worker override; 11 is the current topology result, not a scientific constant.
- PyTorch, MLX, and MPS are recorded as `inapplicable_no_tensor_stage`; no acceleration benchmark or tensor rewrite is allowed in Exp441.
- Runtime creates a fresh immutable attempt for every `new`; there is no `resume` command or mutable checkpoint.
- Publication and evaluation use private same-parent staging, create-only files, fsync, Darwin no-clobber rename, and complete byte replay.
- NumPy is exactly `2.4.6`; bootstrap uses `Generator(PCG64(44120260808))`, 20,000 calls of one `integers(low=0, high=1117, size=1117, dtype=int64, endpoint=False)`, project harmonic F1, and linear 2.5/97.5 percentiles.
- Structural values compare exactly; the two point deltas and four CI endpoints use `rel_tol=0.0`, `abs_tol=1e-15`.
- `governance_payload_antigravity()` returns exactly nine flat top-level fields with exact types and values: `official_run_purpose="deterministic_retrospective_reproduction"`, `bootstrap_interpretation="descriptive_post_selection"`, `adaptive_followup=true`, `independent_validation=false`, `external_generalization_supported=false`, `confirmatory_statistical_claim=false`, `scientific_winner_selected=false`, `review_only=true`, and `operational_alert_eligible=false`. Reject missing, extra, nested, truthy-equivalent, or mismatched governance.
- Governance-bearing records are the preregistration, runbook, implementation report, Exp441 catalog/state/champion entries, frozen config payload, readiness JSON, terminal selection/failure evidence, publication manifest/status, and evaluation evaluation/bootstrap/decision JSON. Data-only JSONL, source audit, confusion, telemetry/PID/logs, and artifact maps do not duplicate governance; the governing manifest/decision binds them transitively.
- Exp404 Hard audited by Exp416 remains the verified reference; Exp441 never changes `CHAMPION_LEDGER` to a new champion.
- Code-map HTML, JSON, and lock are regenerated together before implementation and whenever boundaries, dependencies, or major flows change.
- Use per-boundary code-map synchronization: every boundary-changing task stages the regenerated HTML/JSON/lock triplet in the same commit as its source and tests.

## File Structure

- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/config_antigravity.py`: all frozen literals, fixed paths, CPU policy, and JSON-safe configuration.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/contracts_antigravity.py`: immutable source, candidate, selection, attempt, publication, bootstrap, and evaluation value objects.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/artifacts_antigravity.py`: canonical serialization, pinned regular-file reads, create-only writes, fsync, and exclusive directory rename.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/sources_antigravity.py`: strict Exp440 checkpoint/publication and Exp404 Hard verification with ordered 11-worker record parsing.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/selector_antigravity.py`: pure Hard-empty, one-per-dataset, global-budget-30 selection.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/progress_antigravity.py`: process-owned atomic JSON, JSONL/text logs, PID, heartbeat, throughput, and ETA.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/runtime_antigravity.py`: immutable attempt allocation, signal handling, source replay, selection, and completed-attempt evidence.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/reporting_antigravity.py`: exact six-file prediction publication and read-only inspection.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/readiness_antigravity.py`: clean-HEAD/source/backend proof and atomic run-parent ownership.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/bootstrap_antigravity.py`: the one allowed project-F1 and PCG64 paired-dataset bootstrap implementation.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/evaluation_antigravity.py`: descriptor-bound post-freeze labels, confusion/dataset metrics, reproduction status, and exact eight-file evaluation publication.
- `experiments/exp441_hard_empty_single_shot_recovery_antigravity/cli_antigravity.py`: fixed four-command lazy-import interface.
- `scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py`: outside-CWD-safe wrapper.
- `scratch/verify_exp441_production_replay_antigravity.py`: read-only 1-worker/11-worker source and selector parity audit with no SQLite access.
- Matching `test_*_antigravity.py` files under the Exp441 `tests` directory cover each production boundary.

---

### Task 1: Freeze Exp441 governance, preregistration, runbook, and map scope

**Files:**
- Create: `docs/EXP441_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP441_RUNBOOK_antigravity_KO.md`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `FAILURE_LEDGER.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: the same-commit reviewed Exp441 design at exact content SHA256 `c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da` and frozen Exp440/Hard artifacts.
- Produces: exact identity, source hashes, selector rule, retrospective-only status, commands, log paths, and implementation state used by every later task.

- [ ] **Step 1: Write the exact preregistration contract**

Verify the design file SHA before copying all Global Constraints into the preregistration, record that SHA in preregistration/runbook, and include this non-promotion block verbatim:

```text
official_run_purpose = deterministic_retrospective_reproduction
bootstrap_interpretation = descriptive_post_selection
adaptive_followup = true
independent_validation = false
external_generalization_supported = false
confirmatory_statistical_claim = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
```

- [ ] **Step 2: Write the fixed four-command runbook**

The runbook exposes only these commands and their fixed absolute artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py readiness
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py new
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py inspect
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py evaluate
```

List `readiness_antigravity.json`, `attempts_antigravity/attempt_0000_antigravity`, its progress/event/text/PID paths, `publication_antigravity`, and `evaluation_antigravity`. State that `new` always allocates the smallest absent attempt sequence and never resumes an older attempt.

- [ ] **Step 3: Update the research ledgers without promoting Exp441**

First reconcile the stale Exp440 state: set its catalog/control status to `official_completed_evaluated_not_promoted`, prediction started/completed true, TEST labels opened only after freeze, and bind its frozen publication/evaluation hashes. Append Exp440 to `FAILURE_LEDGER.json.entries` with prediction SHA `06d905d9d6f96fa67704111b984ed87b74229e13fd0690401db67a4a0fdd79af`, evaluation SHA `6cb023b40e02a5caa813c46884b921b76442ef39a736409bff7a32decb546b84`, 100 candidates, TP/FP `40/60`, micro CI lower `-0.000874817656299559`, and maximum strong loss `0.5`. Add Exp441 to the catalog as `design_approved_implementation_planned`; set `RESEARCH_STATE` active experiment/phase accordingly and create one dedicated `exp441_reproduction` object. Append one Exp441 non-promotion review entry to `CHAMPION_LEDGER.history`; do not store Exp441 in `pending_candidate`, do not promote it, and leave Exp404/Exp416 as the verified reference. An unrelated pre-existing pending candidate may remain, but `pending_candidate` must be null or name an experiment other than Exp441.

The three new Exp441 control records use exact allowlisted schemas. The catalog entry has only `id`, `kind`, `status`, `documents`, `artifact_root`, `dataset_count`, `reviewed_design_sha256`, `run_identity`, `prediction_sha256`, `evaluation_sha256`, `test_labels_opened`, `notes`, and the nine governance fields. `RESEARCH_STATE.exp441_reproduction` has only `experiment`, `status`, `reviewed_design_sha256`, `run_identity`, `run_root`, `prediction_sha256`, `evaluation_sha256`, `prediction_started`, `test_labels_opened`, `exact_next_command`, and the nine governance fields. The Exp441 `CHAMPION_LEDGER.history` entry has only `experiment`, `status`, `reason`, `reviewed_design_sha256`, `prediction_sha256`, `evaluation`, and the nine governance fields. Exact key-set equality rejects nested governance, alternate governance aliases, and unrelated extra keys; Exp440 and older records retain their historical schema.

- [ ] **Step 4: Regenerate the code-map triplet before product code**

Extend the existing `research_control` evidence with the new preregistration/runbook and the frozen retrospective status. Preserve 20 primary nodes; do not invent code edges before source files exist.

- [ ] **Step 5: Validate governance and commit**

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
python3 - <<'PY'
import json
expected={
  "official_run_purpose":"deterministic_retrospective_reproduction",
  "bootstrap_interpretation":"descriptive_post_selection",
  "adaptive_followup":True,"independent_validation":False,
  "external_generalization_supported":False,
  "confirmatory_statistical_claim":False,
  "scientific_winner_selected":False,"review_only":True,
  "operational_alert_eligible":False,
}
# Locate the three dedicated Exp441 records and require exact per-record schemas.
catalog=json.load(open("EXPERIMENT_CATALOG.json"))
state=json.load(open("RESEARCH_STATE.json"))
champion=json.load(open("CHAMPION_LEDGER.json"))
catalog_records=[row for row in catalog["experiments"] if row.get("id")=="Exp441"]
champion_records=[row for row in champion["history"] if row.get("experiment")=="Exp441"]
assert len(catalog_records)==len(champion_records)==1
catalog_record=catalog_records[0]
state_record=state["exp441_reproduction"]
champion_record=champion_records[0]
assert state["active_experiment"]=="Exp441"
pending=champion["pending_candidate"]
assert pending is None or (type(pending) is dict and pending.get("experiment")!="Exp441")
governance_keys=set(expected)
schemas=[
  (catalog_record, {"id","kind","status","documents","artifact_root","dataset_count",
    "reviewed_design_sha256","run_identity","prediction_sha256","evaluation_sha256",
    "test_labels_opened","notes"}),
  (state_record, {"experiment","status","reviewed_design_sha256","run_identity","run_root",
    "prediction_sha256","evaluation_sha256","prediction_started","test_labels_opened",
    "exact_next_command"}),
  (champion_record, {"experiment","status","reason","reviewed_design_sha256",
    "prediction_sha256","evaluation"}),
]
for record,base_keys in schemas:
  assert set(record)==base_keys | governance_keys
  for key,value in expected.items():
    assert type(record[key]) is type(value) and record[key]==value
PY
git diff --check
git add docs/EXP441_PREREGISTRATION_antigravity_KO.md \
  docs/EXP441_RUNBOOK_antigravity_KO.md EXPERIMENT_CATALOG.json \
  FAILURE_LEDGER.json CHAMPION_LEDGER.json RESEARCH_STATE.json \
  docs/codemap/codemap.html docs/codemap/codemap.json \
  docs/codemap/codemap.lock
git commit -m "docs(exp441): freeze retrospective selector contract"
```

---

### Task 2: Create frozen configuration, immutable contracts, and artifact primitives

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/config_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/contracts_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_contracts_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_artifacts_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: public `experiments.exp403_408.artifact_io.canonical_json_bytes` and `sha256_bytes`; do not use its replace-style writers.
- Produces: `Exp441FrozenConfig`, `FROZEN_CONFIG`, `cpu_worker_count_antigravity`, `governance_payload_antigravity`, `CandidateEvidenceAntigravity`, `RankedCandidateAntigravity`, `ParsedRecordAntigravity`, `SelectionResultAntigravity`, `FrozenSourceSnapshotAntigravity`, `BackendAssessmentAntigravity`, `ReadinessBindingAntigravity`, `CompletedAttemptAntigravity`, `FrozenPublicationAntigravity`, `BootstrapResultAntigravity`, `EvaluationFactsAntigravity`, `FrozenEvaluationAntigravity`, and safe artifact helpers.

- [ ] **Step 1: Write RED configuration tests**

```python
self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp441")
self.assertEqual(FROZEN_CONFIG.reviewed_design_sha256, "c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da")
self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
self.assertEqual(FROZEN_CONFIG.candidate_budget, 30)
self.assertEqual(FROZEN_CONFIG.candidate_per_dataset, 1)
self.assertEqual(FROZEN_CONFIG.bootstrap_seed, 44120260808)
self.assertEqual(FROZEN_CONFIG.bootstrap_samples, 20000)
self.assertEqual(FROZEN_CONFIG.numpy_version, "2.4.6")
self.assertEqual(cpu_worker_count_antigravity(12), 11)
self.assertEqual(FROZEN_CONFIG.official_run_purpose, "deterministic_retrospective_reproduction")
self.assertEqual(FROZEN_CONFIG.bootstrap_interpretation, "descriptive_post_selection")
self.assertFalse(FROZEN_CONFIG.external_generalization_supported)
self.assertFalse(FROZEN_CONFIG.tensor_eligible)
self.assertFalse(FROZEN_CONFIG.scientific_winner_selected)
self.assertEqual(set(governance_payload_antigravity()), {
    "official_run_purpose", "bootstrap_interpretation", "adaptive_followup",
    "independent_validation", "external_generalization_supported",
    "confirmatory_statistical_claim", "scientific_winner_selected",
    "review_only", "operational_alert_eligible",
})
```

Run and confirm an import failure because the package does not yet exist:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_config_antigravity -v
```

- [ ] **Step 2: Implement the frozen config with no override surface**

Use a frozen dataclass and validate every literal at construction:

```python
@dataclass(frozen=True)
class Exp441FrozenConfig:
    experiment_id: str = "Exp441"
    run_identity: str = "exp441-hard-empty-single-shot-recovery-antigravity-20260808"
    reviewed_design_sha256: str = "c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da"
    dataset_count: int = 1117
    candidate_budget: int = 30
    candidate_per_dataset: int = 1
    bootstrap_seed: int = 44120260808
    bootstrap_samples: int = 20000
    numpy_version: str = "2.4.6"
    tensor_eligible: bool = False
    official_run_purpose: str = "deterministic_retrospective_reproduction"
    bootstrap_interpretation: str = "descriptive_post_selection"
    adaptive_followup: bool = True
    independent_validation: bool = False
    external_generalization_supported: bool = False
    confirmatory_statistical_claim: bool = False
    review_only: bool = True
    operational_alert_eligible: bool = False
    scientific_winner_selected: bool = False
```

Add only the exact positive allowlisted prediction source paths/hashes from Global Constraints: checkpoint context/state/records/retained inventory, ten-file Exp440 publication, and Hard prediction/manifest. Do not carry either forbidden Exp440 evaluation literal in production config, module constants, serialized config, logs, or manifests; those exact negative fixtures live only in this plan, design, and tests. Production accepts no path overrides and rejects any resolved path outside the positive allowlist. Expose only `frozen_config_payload_antigravity()`, `governance_payload_antigravity()`, and `cpu_worker_count_antigravity()`. Test that governance has exactly nine flat keys, `type(value) is bool` for seven booleans, exact strings for the other two, and no nested representation; separately test that serialized production config contains neither forbidden evaluation string.

- [ ] **Step 3: Write RED contract tests**

Reject unknown payload keys, nonfinite ranking floats, `agreement_count < 2`, unsorted/duplicate depth evidence, duplicate coordinates, a selected candidate from a Hard-active dataset, more than one selected coordinate per dataset, more than 30 selected coordinates, wrong structure counts, and mutable caller-owned containers. Prove exact roundtrips:

```python
candidate = CandidateEvidenceAntigravity(
    dataset="A", instance_index=7, raw_length=32,
    depths=(1, 2), agreement_count=2,
    second_excess_ratio=1.5, second_empirical_rank=0.1,
)
self.assertEqual(candidate.dataset_key, (-1.5, 0.1, 7))
self.assertEqual(candidate.global_key, (-1.5, 0.1, "A", 7))
self.assertEqual(
    CandidateEvidenceAntigravity.from_payload(candidate.to_payload()),
    candidate,
)
```

- [ ] **Step 4: Implement immutable contracts**

`ParsedRecordAntigravity` explicitly exposes `dataset: str`, `test_row_count: int`, and the adapted `CandidateEvidenceAntigravity` tuple. `FrozenSourceSnapshotAntigravity` owns the context dataset tuple, immutable `test_row_count_by_dataset`, a read-only record-digest map, frozen Hard coordinates translated from source `candidate_index` to internal `instance_index`, parsed provisional candidates, all frozen hashes, and label access count zero. `SelectionResultAntigravity` owns tuples for all provisional/outside/eligible/winner/selected collections and enforces structural invariants for both sparse fixtures and production. A separate `validate_production_selection_antigravity(result)` enforces exact `840/394/73/49/30`; tests never weaken those production counts or pad sparse fixtures. `RankedCandidateAntigravity.global_rank` is always 1 through the fixture/production winner count; in production it is 1 through 49. `selection_rank: int | None` is 1 through the selected count exactly when selected and otherwise `None`; production requires 1 through 30.

- [ ] **Step 5: Write RED artifact safety tests**

Test duplicate JSON keys, noncanonical JSON, symlink/nonregular reads, `O_EXCL|O_NOFOLLOW` create-only writes, partial-write loops, file and directory fsync, existing destination, and a destination appearing during staging. Verify Darwin exclusive rename never replaces a winner.

- [ ] **Step 6: Implement the shared artifact boundary**

Expose these exact helpers:

```python
def canonical_json_bytes_antigravity(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"

def canonical_jsonl_bytes_antigravity(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes_antigravity(dict(row)) for row in rows)

def read_pinned_regular_bytes_antigravity(path: Path) -> tuple[bytes, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source is not a regular file")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        stable_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        stable_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if stable_before != stable_after or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("source changed while pinned")
        payload = b"".join(blocks)
        return payload, hashlib.sha256(payload).hexdigest()
    finally:
        os.close(descriptor)

def write_create_only_antigravity(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written == 0:
                raise OSError(errno.EIO, "create-only write made no progress", str(path))
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def fsync_directory_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def exclusive_rename_antigravity(source: Path, destination: Path) -> None:
    renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(source), os.fsencode(destination), ctypes.c_uint(0x00000004)) != 0:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, os.strerror(code), str(destination))
```

The implementation uses canonical sorted compact JSON plus one trailing newline, descriptor hashing, `O_NOFOLLOW`, progress-checked full write loops, `fsync`, and Darwin `renamex_np` flag `0x00000004`. The artifact tests patch `os.write` to return zero before completion and require one immediate `EIO`, no spin, no false success, and no `fsync` call.

- [ ] **Step 7: Add the new package/artifact boundary to all three code-map files**

Extend the existing frequency-runtime node with config/contracts/artifact source symbols and their tests. Recompute the docs/experiments fingerprints and embedded HTML payload before staging.

- [ ] **Step 8: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_config_antigravity \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_contracts_antigravity \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_artifacts_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): freeze selector contracts and artifacts"
```

---

### Task 3: Verify and load every frozen source without labels

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/sources_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_sources_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: `Exp440DatasetResultAntigravity.from_payload`, fixed checkpoint/publication/Hard paths and hashes, `cpu_worker_count_antigravity()`.
- Produces: `verify_frozen_sources_antigravity() -> FrozenSourceSnapshotAntigravity`, `hard_coordinate_identity_sha256_antigravity(coordinates) -> str`; internal `_parse_record_task_antigravity(task) -> ParsedRecordAntigravity` is safe for `ProcessPoolExecutor`, internal `_verify_frozen_sources_with_observer_antigravity(observer)` emits parent-ordered immutable progress events through the fixed 11-worker path without exposing a public override, and private `_verify_frozen_sources_single_worker_reference_antigravity() -> FrozenSourceSnapshotAntigravity` performs the same checks sequentially only for source tests and `scratch/verify_exp441_production_replay_antigravity.py`.

- [ ] **Step 1: Write RED source identity and firewall tests**

Reject a wrong context/state/retained-inventory/publication/Hard hash, missing/extra record, wrong checkpoint or ten-file publication inventory, sequence gap, duplicate dataset, record digest mismatch, record dataset mismatch, symlink/FIFO/directory artifact, unknown payload field, noncanonical dataset order, state count other than 1,117, Hard count other than 2,218, Hard coordinate identity mismatch, and test-supplied paths resolving outside the exact positive allowlist. The two forbidden Exp440 evaluation strings exist only in negative test fixtures; production source/config contains neither literal.

```python
with self.assertRaisesRegex(ValueError, "evaluation source is forbidden"):
    _validate_allowed_prediction_path_antigravity(
        Path("exp440/evaluation_antigravity/evaluation_antigravity.json")
    )
```

- [ ] **Step 2: Write RED ordered 11-worker parsing tests**

Use delayed fixture tasks that finish out of order. Assert parent results are ordered by context sequence, the worker count must equal 11 for production, each worker receives only path/dataset/digest/sequence, and numerical-thread environment variables are set to `1`. Add AST and call-spy assertions that the public verifier and observer-enabled runtime verifier always enter the fixed 11-worker implementation and never call the single-worker reference. Assert no CLI, readiness, runtime, selector, reporting, or publication module imports or resolves the reference helper; only `test_sources_antigravity.py` and `scratch/verify_exp441_production_replay_antigravity.py` may use it.

- [ ] **Step 3: Implement strict source verification**

The verifier performs this order before returning:

```text
pin and hash context/state
pin and hash the known retained inventory without parsing or selecting from it
validate exact 1,117 sorted unique context datasets
validate state completed/record map
pin and hash the exact ten-file Exp440 publication inventory
replay the artifact map and all nine declared file hashes
pin and hash Hard manifest/prediction
parse 2,218 unique Hard coordinates, 888 Hard-active datasets, 229 Hard-empty datasets
hash sorted Hard as one newline-free compact JSON array of [dataset,candidate_index] pairs and require 858e24e...54322
enumerate exactly 1,117 regular record files
parse records in 11 processes, return in context order
adapt every Exp440 provisional candidate into CandidateEvidenceAntigravity by explicitly copying the frozen fields
build immutable per-record test_row_count_by_dataset
translate Hard source candidate_index to internal instance_index
validate every provisional and Hard instance_index against that dataset record's test_row_count
reverify every source descriptor/path snapshot
freeze the snapshot with label_access_count = 0
```

The production function has no path, worker, expected-count, or observer arguments. It delegates to the internal observer form with a no-op observer; runtime supplies its process-owned progress writer to that internal form, while readiness uses the public wrapper. Both production entrances select the constant 11-worker implementation and have no worker override or route to the reference helper. `_verify_frozen_sources_single_worker_reference_antigravity()` is a separate zero-argument, sequential context-order implementation with no `ProcessPoolExecutor`, observer, worker argument, environment switch, or production caller; it exists solely for tests and the final read-only parity script. Events are emitted only as the parent commits the next context-ordered record, never from workers or completion order. Tests exercise internal pure helpers with temporary fixtures.

- [ ] **Step 4: Add exact label-free count validation**

For every parsed record require `type(test_row_count) is int`, `test_row_count >= 0`, candidate dataset equality with the record, and `type(instance_index) is int` with `0 <= instance_index < test_row_count`. After all 1,117 records, build immutable `test_row_count_by_dataset`; require every Hard dataset to exist and every translated Hard `candidate_index` to satisfy the same bound. Test `-1`, `count - 1`, `count`, `bool`, unknown dataset, and zero-row cases. Malformed candidate/Hard ranges must fail before importing or calling Exp417's connection helper. Then flatten the adapted candidates in context order and assert provisional `840`, coordinates unique `840`, outside-Hard `394`, and no source payload contains a non-false label or evaluation field. Do not compute Hard-empty filtering here; that belongs to the selector.

- [ ] **Step 5: Map the evidenced Exp440/Hard source dependencies**

Add source-symbol evidence and import/read edges to the existing frequency-runtime node. Regenerate all three map files and include them in this task's commit.

- [ ] **Step 6: Run GREEN and inherited parser regression, then commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_sources_antigravity -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_contracts_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_predecessor_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): verify frozen selector sources"
```

---

### Task 4: Implement the pure Hard-empty single-shot selector

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/selector_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_selector_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: `FrozenSourceSnapshotAntigravity`.
- Produces: `select_hard_empty_single_shot_antigravity(snapshot) -> SelectionResultAntigravity` and `selected_coordinate_identity_sha256_antigravity(candidates) -> str`.

`selected_coordinate_identity_sha256_antigravity` sorts selected coordinates by `(dataset, instance_index)`, serializes rows shaped exactly as `{"dataset":"A","instance_index":7}` in compact canonical JSONL with one trailing newline per coordinate, and hashes those bytes. It is never used to validate Hard. Add a fixture proving this JSONL hash differs from `hard_coordinate_identity_sha256_antigravity`, whose Exp416 contract is one newline-free compact JSON array of `[dataset,index]` pairs.

- [ ] **Step 1: Write RED eligibility tests**

Construct three datasets: no Hard coordinate, one Hard coordinate, and a candidate exactly overlapping Hard. Assert only the first dataset can contribute. `agreement_count` values 2 and 3 with otherwise identical severity must not change rank.

- [ ] **Step 2: Write RED dataset-local and global tie tests**

```python
self.assertEqual(
    tuple(item.instance_index for item in select_fixture("A")),
    (7,),
)
self.assertEqual(
    tuple((item.dataset, item.instance_index) for item in result.selected),
    expected_global_top_30,
)
```

Cover excess ratio descending, empirical rank ascending, index ascending inside a dataset, then dataset case-sensitive ascending and index ascending globally. Shuffle input and worker-completion order 20 times and require identical payload bytes.

- [ ] **Step 3: Implement the minimal pure selector**

```python
hard_datasets = frozenset(dataset for dataset, _ in snapshot.hard_coordinates)
outside = tuple(
    item for item in snapshot.provisional_candidates
    if item.coordinate not in snapshot.hard_coordinates
)
eligible = tuple(item for item in outside if item.dataset not in hard_datasets)
grouped = group_by_dataset_antigravity(eligible)
winners = tuple(
    min(items, key=lambda item: item.dataset_key)
    for _, items in sorted(grouped.items())
)
selected = tuple(sorted(winners, key=lambda item: item.global_key)[:30])
```

The pure selector validates unique coordinates and at most one selected item per dataset, assigns complete `global_rank`/`selection_rank`, and never pads a sparse fixture. Call `validate_production_selection_antigravity` only at the production boundary to require exactly 49 winners, 30 retained candidates, and the full `840/394/73/49/30` structure.

- [ ] **Step 4: Add the real 1,117-record label-free replay test**

Call `verify_frozen_sources_antigravity()` and the selector without opening SQLite. Independently assert the actual Hard array identity `858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322`, structure `840/394/73/49/30`, label access zero, and selected-coordinate JSONL identity `db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`. Cross-feeding either serialized payload to the other identity contract must fail.

- [ ] **Step 5: Map the selector call and evidence symbols**

Update the frequency-runtime node evidence and source-verifier-to-selector call flow, regenerate the triplet, and keep the primary node count at 20.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_selector_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): select hard-empty single shots"
```

---

### Task 5: Add immutable attempts, process-owned progress, and observable runtime

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/progress_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/runtime_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_progress_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_runtime_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: source verifier and pure selector.
- Produces: `allocate_attempt_antigravity() -> Path`, `run_exp441_new_antigravity(*, reviewed_commit: str) -> CompletedAttemptAntigravity`, immutable attempt evidence, and self-written logs.

- [ ] **Step 1: Write RED progress schema and atomicity tests**

Require every atomic progress JSON to contain:

```text
pid, active, current_dataset, current_record_name, active_datasets,
train_row_count, test_row_count, raw_length,
dimension_status, stage, backend, completed, total, imported, recomputed,
success, error, provisional_candidates, outside_hard_candidates,
eligible_candidates, selected_candidates, throughput, elapsed_seconds,
eta_seconds, source_checkpoint, heartbeat, latest_error
```

Assert `train_row_count` and unknown `raw_length` are `null` with `dimension_status="not_available_in_frozen_selector_source"`; JSONL/text logs append, progress replaces atomically, and PID becomes inactive at every terminal state.
While record verification is active, `current_dataset` is the next context-ordered result awaiting parent acceptance and is never null; `active_datasets` truthfully lists all submitted unfinished records. Terminal states clear both fields.

- [ ] **Step 2: Write RED attempt allocation and immutability tests**

Assert the first attempt is `attempt_0000_antigravity`, gaps choose the smallest absent sequence, symlink/non-directory attempts fail, existing attempts never change, interrupted/failed attempts remain inspectable, and no `resume` API exists. Every attempt-level `failure_antigravity.json` fixture must carry the exact nine flat governance fields; test each missing key, extra governance alias, wrong type/value, and nested replacement.

- [ ] **Step 3: Write RED runtime order, signal, and failure tests**

Patch the source parser with delayed records. Assert 11 workers remain bounded, the runtime-owned observer receives one immutable event only when the parent accepts the next context-ordered record, progress advances only in that order, SIGTERM becomes `interrupted`, source/schema failure becomes `failed`, no publication appears on failure, and a second `new` uses a new attempt rather than repairing the first. Before runtime implementation, add RED assertions that successful terminal `selection_antigravity.json` and each failed-attempt `failure_antigravity.json` contain all and only the exact flat governance names with strict key/type/value checks.

- [ ] **Step 4: Implement the runtime sequence**

```text
allocate create-only attempt
create PID/progress/events/text artifacts
verify exact clean reviewed commit again
verify all frozen sources with 11 workers through the internal parent-ordered progress observer
select 30 candidates
write source_snapshot_antigravity.json create-only
write selection_antigravity.json create-only
reverify sources and clean HEAD
mark attempt complete and PID inactive
return CompletedAttemptAntigravity
```

The attempt snapshot stores only hashes, structure counts, record digest-map identity, ranking evidence, and candidate coordinates; it contains no labels or metrics. Terminal `selection_antigravity.json` and every attempt `failure_antigravity.json` carry the exact flat nine-field governance payload, with artifact-specific RED roundtrip/tamper assertions.

- [ ] **Step 5: Map attempt/progress/runtime ownership**

Add the process-owned log, 11-worker source replay, selector, and completed-attempt flow with exact source symbols; regenerate the three map files.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_progress_antigravity \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_runtime_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): add immutable observable attempts"
```

---

### Task 6: Publish and replay the exact six-file label-free prediction

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/reporting_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_reporting_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: a terminal inactive `CompletedAttemptAntigravity`, reverified source snapshot, and frozen config.
- Produces: `publish_exp441_antigravity(attempt) -> FrozenPublicationAntigravity` and fixed-root `inspect_exp441_publication_antigravity() -> FrozenPublicationAntigravity`.

- [ ] **Step 1: Write RED inventory, ordering, and firewall tests**

Require exactly:

```text
prediction_antigravity.jsonl
candidate_ranking_antigravity.jsonl
source_audit_antigravity.json
manifest_antigravity.json
status_antigravity.json
artifact_sha256_antigravity.json
```

Prediction contains the 30 selected rows sorted by `(dataset, instance_index)`. Ranking contains all 49 dataset winners in global ranking order with `global_rank`, `selected`, and `selection_rank`. Reject duplicate coordinates, wrong order, budget over 30, more than one candidate per dataset, Hard-active datasets, forbidden label/evaluation keys, and any governance value inconsistent with Global Constraints. Require the exact flat nine-field governance payload in `manifest_antigravity.json` and `status_antigravity.json`; reject missing, extra, nested, truthy-equivalent, or partial governance. Prediction/ranking JSONL, source audit, and artifact map carry no governance keys and are bound transitively.

- [ ] **Step 2: Write RED hash-graph and source-coherence tests**

Freeze this acyclic graph:

```text
manifest -> prediction + ranking + source audit + frozen source/config identities
status -> manifest + prediction + completed retrospective-only governance
artifact map -> all five non-self publication files
artifact map -> self excluded explicitly
```

Tamper each artifact, attempt snapshot, Exp440 record, source state, Hard file, runtime progress, or PID and require replay/publication failure.

- [ ] **Step 3: Write RED create-only race tests**

Assert private mode `0700` staging, file mode `0600`, `O_EXCL|O_NOFOLLOW`, fsync, source revalidation immediately before rename, Darwin `RENAME_EXCL`, no adoption of a destination that appears during staging, and replay-only handling of a destination present on entry. A failed staging tree is renamed to a unique `failed_staging_antigravity` evidence path with `failure_antigravity.json` and preserved without creating the publication destination. That failure file carries the exact nine flat governance fields; tamper every key/type/value.

- [ ] **Step 4: Implement deterministic publication and inspection**

`inspect_exp441_publication_antigravity()` accepts no path or expectation overrides. It checks exact inventory, regular files, canonical bytes, all hashes, all selection invariants, source identities, `label_access_count=0`, and `contains_test_labels=false` without importing evaluation or opening SQLite.

- [ ] **Step 5: Map publication and inspection flows**

Extend the publication/evaluation map node with runtime-to-publication writes and inspect reads, including tests and exact symbols; regenerate all three files.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_reporting_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): freeze label-free selector publication"
```

---

### Task 7: Add atomic readiness, fixed CLI, and outside-CWD wrapper

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/readiness_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/cli_antigravity.py`
- Create: `scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_readiness_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_cli_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Produces: `run_readiness_antigravity(reviewed_commit: str) -> ReadinessBindingAntigravity`, `validate_readiness_for_run_antigravity(reviewed_commit: str) -> ReadinessBindingAntigravity`, `build_parser_antigravity()`, and `main_antigravity(argv=None)`.

- [ ] **Step 1: Write RED readiness evidence tests**

Require exact clean 40-character HEAD, absent run parent/output attempts/publication/evaluation, all prediction-allowed source hashes/counts, the DB path/hash only as unevaluated frozen literals, 12 logical CPUs/11 workers, label access zero, NumPy 2.4.6, and environment evidence for Python, PyTorch, MPS built/available, MLX, and mlx-metal. Readiness must not verify or open the DB. The readiness JSON carries the exact flat nine-field governance payload. The backend decision core fields must be:

```json
{
  "canonical_backend": "cpu",
  "workers": 11,
  "tensor_eligible": false,
  "acceleration_benchmark_status": "not_run_inapplicable",
  "pytorch": "inapplicable_no_tensor_stage",
  "mlx": "inapplicable_no_tensor_stage",
  "mps": "inapplicable_no_tensor_stage"
}
```

The same evidence object also records Python/PyTorch/MLX/mlx-metal versions, MPS built/available booleans, MLX default device, and the no-benchmark reason without changing the core decision fields.

- [ ] **Step 2: Write RED atomic run-parent ownership tests**

Stage `readiness_antigravity.json` in an antigravity-named sibling directory, fsync it, and rename exclusively to the absent fixed parent. Reject an existing parent, symlink parent, concurrent winner, source mutation, and dirty HEAD. Failure must not create the fixed run parent; instead preserve the private staging tree under a unique failed antigravity name with canonical `failure_antigravity.json`. Both readiness and failure fixtures carry the exact nine flat governance fields; tamper every key/type/value and reject nesting.

- [ ] **Step 3: Implement readiness and exact byte validation**

Readiness performs one full label-free source/selector replay before publishing its evidence. `validate_readiness_for_run_antigravity()` recomputes canonical bytes and requires byte equality, current clean HEAD equality, source hash equality, worker equality, and absent publication/evaluation before `new`.

- [ ] **Step 4: Write RED CLI/wrapper tests**

Parser commands are exactly `readiness`, `new`, `inspect`, and `evaluate`, with no flags for paths, workers, backend, seed, budget, source, threshold, or bootstrap. Assert no `resume` command. Prove `new` rechecks HEAD after readiness and immediately before source replay; `inspect` is byte read-only; prediction commands never import `evaluation_antigravity`; `evaluate` imports only the exact evaluator lazily; wrapper help works outside the repository.

In fresh-interpreter route tests for `--help`, `readiness`, `new`, and `inspect`, make `sqlite3.connect` raise and install an import sentinel rejecting the Exp441 evaluator, Exp440 evaluator, and `experiments.exp417_causal_multihorizon_shadow.inputs`; import the wrapper/CLI and invoke the mocked route. All four routes must complete without either sentinel firing, with byte-identical pre/post trees where the route is read-only. Only `evaluate` may perform the exact branch-local evaluator import. Inside evaluation, import `verified_read_only_connection` function-locally only after publication/source/selector/range preflight. Static AST auditing rejects `sqlite3`, Exp417-input, or evaluator imports from config/contracts/artifacts/sources/selector/progress/runtime/reporting/readiness and the wrapper; the CLI's branch-local Exp441 evaluator import is the sole exception. A unit test of source/selector alone is not a substitute for these route-level firewalls.

- [ ] **Step 5: Implement CLI lifecycle**

```python
if command == "readiness":
    result = run_readiness_antigravity(current_clean_head_antigravity())
elif command == "new":
    binding = validate_readiness_for_run_antigravity(current_clean_head_antigravity())
    attempt = run_exp441_new_antigravity(reviewed_commit=binding.reviewed_commit)
    result = publish_exp441_antigravity(attempt)
elif command == "inspect":
    result = inspect_exp441_publication_antigravity()
else:
    from .evaluation_antigravity import evaluate_exp441_antigravity
    result = evaluate_exp441_antigravity()
print(canonical_cli_status_antigravity(command, result))
return 0
```

`main_antigravity` returns integer exit code `0` after every successful route and prints only canonical JSON status to stdout. The wrapper calls `raise SystemExit(main_antigravity())`; domain objects are never passed to `SystemExit`.

- [ ] **Step 6: Map readiness and four-command routes**

Add readiness ownership, CLI-to-runtime/publication calls, read-only inspect, and the lazy evaluator route to the existing nodes. Regenerate HTML/JSON/lock together.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_readiness_antigravity \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_cli_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): bind fixed selector lifecycle"
```

---

### Task 8: Freeze project F1 and the exact paired bootstrap

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/bootstrap_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_bootstrap_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Produces: `project_f1_antigravity(confusion: Sequence[int | float]) -> float`, `require_numpy_2_4_6_antigravity()`, and `paired_dataset_bootstrap_antigravity(rows: np.ndarray) -> BootstrapResultAntigravity`.

- [ ] **Step 1: Write RED F1 and deterministic-bootstrap tests**

`project_f1_antigravity` accepts exactly four values in `(tp, fp, fn, tn)` order, rejects bool, negative, nonfinite, or wrong-length input, and returns a built-in `float`. Assert zero-denominator F1 is zero and freeze the precision/recall harmonic formula rather than direct `2TP` arithmetic. Use a small hand-derived matrix to prove paired row sampling. Patch a spy generator and require exactly 20,000 calls with this signature:

```python
rng.integers(
    low=0, high=1117, size=1117,
    dtype=np.int64, endpoint=False,
)
```

Reject NumPy versions other than `2.4.6`, any other bit generator, dtype/shape/order drift, nonfinite rows, and a quantile method other than `linear`.

- [ ] **Step 2: Implement the one bootstrap path from Global Constraints**

Use `Generator(PCG64(44120260808))`, one `integers` call per repetition, float64 sums/means, and `np.quantile(..., method="linear")` exactly as preregistered. Convert all NumPy scalar CI endpoints to built-in `float` before contract construction or JSON serialization. No evaluator or SQLite import is permitted in this module.

```python
def paired_dataset_bootstrap_antigravity(rows: np.ndarray) -> BootstrapResultAntigravity:
    require_numpy_2_4_6_antigravity()
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (1117, 9) or not np.isfinite(values).all():
        raise ValueError("bootstrap rows differ")
    rng = np.random.Generator(np.random.PCG64(44120260808))
    micro = np.empty(20000, dtype=np.float64)
    macro = np.empty(20000, dtype=np.float64)
    for repetition in range(20000):
        indices = rng.integers(
            low=0, high=1117, size=1117,
            dtype=np.int64, endpoint=False,
        )
        sample = values[indices]
        baseline = sample[:, :4].sum(axis=0, dtype=np.float64)
        union = sample[:, 4:8].sum(axis=0, dtype=np.float64)
        micro[repetition] = project_f1_antigravity(union) - project_f1_antigravity(baseline)
        macro[repetition] = sample[:, 8].mean(dtype=np.float64)
    q = np.asarray([0.025, 0.975], dtype=np.float64)
    micro_ci = np.quantile(micro, q, method="linear")
    macro_ci = np.quantile(macro, q, method="linear")
    return BootstrapResultAntigravity(
        micro_ci=tuple(float(value) for value in micro_ci),
        macro_ci=tuple(float(value) for value in macro_ci),
    )
```

- [ ] **Step 3: Map and verify the pure calculation boundary**

Add exact bootstrap symbols/tests to the existing publication/evaluation node without adding a database edge. Regenerate all three code-map files, run the focused tests, and commit:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_bootstrap_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): freeze retrospective bootstrap"
```

---

### Task 9: Enforce the frozen-context post-freeze label boundary

**Files:**
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: fully inspected Exp441 publication and the reverified `FrozenSourceSnapshotAntigravity`.
- Produces: internal frozen-context label snapshot with one descriptor-bound connection; it does not yet compute metrics or publish evaluation files.

- [ ] **Step 1: Write RED preflight-before-import tests**

Require publication replay, source/hash replay, selector/count replay, immutable row-count map, and candidate/Hard range validity before importing or calling any label-capable helper. For every preflight failure, install an import sentinel for `experiments.exp417_causal_multihorizon_shadow.inputs` and a connection function that explodes; neither may fire. Import `verified_read_only_connection` function-locally only after all preflight gates pass.

- [ ] **Step 2: Freeze the only permitted database statement sequence**

The helper verifies the DB descriptor/hash when its one Exp417 context manager opens, then executes exactly:

```sql
PRAGMA quick_check
```

once, requiring `fetchall() == [("ok",)]`, followed by this sole label-bearing SELECT exactly once for each dataset in the frozen 1,117-name context tuple:

```sql
SELECT i.instance_index, i.label
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
```

The parameter tuple must be `(dataset,)` in context order. Do not enumerate the DB dataset table and do not issue a bulk TEST query. Assert one connection, one quick-check, 1,117 parameterized label queries, and zero parameters outside context. Explicitly prove `CornellWhaleChallenge` and `Wafer_normal_1` are never queried even though the current DB contains them.

- [ ] **Step 3: Validate actual TEST identities without inventing contiguity**

For each dataset, require returned indices to be strictly increasing unique integers, row count equal to `test_row_count_by_dataset[dataset]`, labels exactly binary after strict normalization, and every Hard/Exp441 coordinate present in the actual identity set. Preserve noncontiguous indices as returned; never substitute `range(test_row_count)`. Record one descriptor-bound connection, 1,117 label-query calls, and zero out-of-scope label rows.

- [ ] **Step 4: Map, run GREEN, and commit**

Add the lazy evaluator read edge from the publication/source preflight through Exp417's descriptor-bound helper to SQLite. Do not map evaluation writes yet. Regenerate the map triplet and run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_evaluation_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): enforce post-freeze label boundary"
```

---

### Task 10: Reproduce pure retrospective evaluation facts

**Files:**
- Modify: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/evaluation_antigravity.py`
- Modify: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Produces pure confusion, per-dataset metrics, rescue/regression facts, exact bootstrap inputs/results, and one of three retrospective reproduction statuses; no filesystem publication occurs yet.

- [ ] **Step 1: Write RED hand-derived confusion and status tests**

Verify baseline/union confusion, candidate TP/FP, project F1, dataset macro F1, and bootstrap-row order. A rescued dataset has baseline TP `== 0` and union TP `>= 1`; a regression has union dataset F1 below baseline; strong loss is measured only when baseline F1 is at least `0.8`. Status rules are:

```text
source/hash/population/selector/confusion/label-firewall mismatch -> engineering_reproduction_failed
correct structure but any of six finite float values exceeds abs 1e-15 -> retrospective_reproduction_discrepancy
all exact/tolerant values match -> retrospective_reproduction_complete
all three -> scientific_winner_selected=false
```

- [ ] **Step 2: Add the frozen full-population expected-value test**

Require exactly:

```text
baseline = TP 1915 / FP 303 / FN 2003 / TN 191679
union = TP 1930 / FP 318 / FN 1988 / TN 191664
candidate TP/FP = 15/15
rescued/regressed = 15/0
baseline F1 = 0.6241851368970013
union F1 = 0.6260136230943887
micro delta = 0.0018284861973874023
macro delta = 0.012444046553267682
micro CI = [0.00005373870255999115, 0.003814109586073529]
macro CI = [0.006535362578334826, 0.019158460161145925]
maximum strong loss = 0.0
micro delta hex = 0x1.df53a1ad05800p-10
macro delta hex = 0x1.97c43a7cf3ab8p-7
micro CI hex = [0x1.c2cafc29c4668p-15, 0x1.f3ec47deffdf7p-9]
macro CI hex = [0x1.ac4d308a9c7cep-8, 0x1.39e467f55da52p-6]
```

- [ ] **Step 3: Implement the pure facts/status layer**

Require exact integer structural fields and finite built-in float values. Apply `rel_tol=0.0, abs_tol=1e-15` only to the two point deltas and four CI endpoints; do not round artifacts. All three statuses carry the same no-winner governance. Tests cover every decision boundary and prove no TEST result can retune source, order, threshold, alpha, or budget.

- [ ] **Step 4: Map, run GREEN, and commit**

Add facts/status symbols and tests to the existing evaluation node, regenerate the triplet, run the focused bootstrap/evaluation tests, and commit:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_bootstrap_antigravity \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_evaluation_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): reproduce retrospective evaluation facts"
```

---

### Task 11: Freeze and replay the exact eight-file evaluation publication

**Files:**
- Modify: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/evaluation_antigravity.py`
- Modify: `experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests/test_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Produces: `evaluate_exp441_antigravity() -> FrozenEvaluationAntigravity`, exactly eight create-only artifacts, replay-only existing-evaluation handling, and preserved failure evidence.

- [ ] **Step 1: Write RED inventory, governance, and hash-graph tests**

Create only:

```text
evaluation_antigravity.json
dataset_metrics_antigravity.jsonl
rescued_datasets_antigravity.jsonl
regressed_datasets_antigravity.jsonl
confusion_antigravity.json
bootstrap_antigravity.json
decision_antigravity.json
artifact_sha256_antigravity.json
```

`evaluation_antigravity.json`, `bootstrap_antigravity.json`, and `decision_antigravity.json` each carry the exact flat nine-field governance payload. Data JSONL, confusion, and artifact map carry no governance keys. The artifact map hashes exactly the other seven files and explicitly excludes only itself. Tamper every file/hash and every governance key/type/value/nesting form and require replay failure.

- [ ] **Step 2: Write RED create-only race and failure-preservation tests**

Use the shared `O_EXCL|O_NOFOLLOW`, fsync, and Darwin `RENAME_EXCL` boundary. A destination present on entry is replayed completely without reopening labels; a destination appearing later is a concurrency failure and is never adopted. Preserve a uniquely named failed evaluation staging tree plus canonical `failure_antigravity.json` while the fixed evaluation destination remains absent. Evaluation failure fixtures carry the exact nine flat governance fields and have artifact-specific missing/extra/type/value/nesting tests.

- [ ] **Step 3: Implement the exact revalidation and publication order**

Execute:

```text
inspect publication -> reverify source/selector/ranges -> load frozen-context labels
-> compute facts/bootstrap -> reverify all source/publication bytes
-> stage eight artifacts -> reverify immediately before exclusive rename
-> reopen and byte-replay all eight artifacts -> reverify source/publication again
```

No evaluation destination exists before every post-label revalidation passes. An existing valid evaluation returns by complete replay without importing Exp417 inputs or opening SQLite.

- [ ] **Step 4: Map, run GREEN, and commit**

Add evaluation write/replay edges and exact source symbols to the map, regenerate all three files, run focused tests, and commit:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_bootstrap_antigravity \
  experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_evaluation_antigravity -v
git add experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp441): freeze retrospective evaluation publication"
```

---

### Task 12: Run full verification, synchronize the code map, and bind readiness handoff

**Files:**
- Create: `docs/EXP441_IMPLEMENTATION_REPORT_antigravity.md`
- Create: `scratch/verify_exp441_production_replay_antigravity.py`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify: `CHAMPION_LEDGER.json`
- Modify together: `docs/codemap/codemap.html`
- Modify together: `docs/codemap/codemap.json`
- Modify together: `docs/codemap/codemap.lock`

**Interfaces:**
- Produces: reviewed clean implementation commit, synchronized dependency map, exact readiness command, and no official Exp441 run yet.

- [ ] **Step 1: Run all Exp441 tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp441_hard_empty_single_shot_recovery_antigravity/tests \
  -p 'test_*_antigravity.py' -v
```

Record exact test count, failures, errors, duration, Python version, NumPy version, logical CPUs, and worker count.

- [ ] **Step 2: Run inherited contract and input regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_config_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_contracts_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp440_train_only_low_frequency_precision_gate_antigravity.tests.test_predecessor_antigravity \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs -v
```

- [ ] **Step 3: Run naming, import-firewall, and CLI-surface audits**

```bash
find experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  -type f ! -path '*/__pycache__/*' ! -name '*antigravity*' -print
git grep -nE 'exp440.*evaluation_antigravity|SELECT .*label|load.*label|groundtruth|anomaly_location' -- \
  experiments/exp441_hard_empty_single_shot_recovery_antigravity \
  scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py --help
```

The naming command prints nothing. Classify the label SQL and evaluator-only references; no prediction/readiness/runtime/reporting module may import or contain a label query.

- [ ] **Step 4: Implement and run the production-shaped real label-free replay twice**

The verifier imports the public source/selector APIs plus the explicitly test-only `_verify_frozen_sources_single_worker_reference_antigravity()` interface, patches `sqlite3.connect` to raise, and replays all 1,117 records once through the immutable public 11-worker route and once through that named sequential reference. Require identical selection payloads, Exp416-array Hard identity `858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322`, Exp441-JSONL selected identity `db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`, counts `1117/840/394/73/49/30`, finite evidence, and label access zero. Record elapsed time, verify the official path used 11 workers, and rerun the AST/call-spy guard proving no production entrypoint can select or invoke the reference interface.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/verify_exp441_production_replay_antigravity.py
```

- [ ] **Step 5: Run synthetic end-to-end publication/evaluation-order smoke**

Use a temporary fixed-path fixture through CLI-shaped internal helpers. Prove readiness ownership, immutable attempt, six-file publication replay, evaluation loader explosion before freeze, first label-capable import only after source/range preflight, one quick-check followed by context-scoped label queries, exact eight-file evaluation, and zero changes to publication/attempt bytes.

- [ ] **Step 6: Regenerate and fully validate the code-map triplet**

Keep 20 primary nodes by extending the current frequency runtime and publication/evaluation groups with Exp441 source symbols. Add evidenced edges for CLI → runtime → sources/selector → publication and lazy evaluator → SQLite. Verify all node paths/symbols, edge evidence, flow references, HTML embedded JSON identity, current parent commit/dirty state, module fingerprints, and zero unknown relationships.

- [ ] **Step 7: Write the implementation report and control state**

Write the exact flat nine-field governance payload into the implementation report and every governance-bearing Exp441 control record. Preserve the exact Task 1 allowlisted key sets while setting the catalog entry and `RESEARCH_STATE.exp441_reproduction` to `implementation_complete_readiness_pending`, `run_identity` null, prediction not started, TEST labels unopened, review-only, and no champion. Do not add PID or attempt fields because no process or attempt exists yet. Update the Exp441 `CHAMPION_LEDGER.history` row as a non-promotion implementation review record; never set `pending_candidate` to Exp441, and assert it remains null or names another experiment. The exact next command is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py readiness
```

- [ ] **Step 8: Final verification and commit**

Rerun the complete Task 1 exact control-record schema validator unchanged after these final ledger edits, then run:

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git status --short --branch
git add docs/EXP441_IMPLEMENTATION_REPORT_antigravity.md \
  scratch/verify_exp441_production_replay_antigravity.py \
  EXPERIMENT_CATALOG.json CHAMPION_LEDGER.json RESEARCH_STATE.json \
  docs/codemap/codemap.html docs/codemap/codemap.json \
  docs/codemap/codemap.lock
git commit -m "docs(exp441): bind verified readiness handoff"
```

- [ ] **Step 9: Obtain a fresh Sol/High final review before readiness**

The reviewer must inspect the complete committed implementation, rerun focused tests and the real label-free replay, confirm no worktree mutation, and return `proceed` before readiness or `new` is executed. Any review fix invalidates the verdict and requires a fresh reviewer.

After a clean `proceed` verdict, run readiness once. Do not start `new` until the readiness artifact has been replayed and its reviewed commit, source hashes, 11-worker evidence, NumPy 2.4.6, and zero label access are confirmed.
