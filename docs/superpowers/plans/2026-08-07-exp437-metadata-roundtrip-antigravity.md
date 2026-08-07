# Exp437 Metadata Roundtrip Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Exp437 as a thin, fixed-identity successor that reuses Exp436's label-free detector, accepts canonical JSON length histograms independent of insertion order, and is ready for a fresh 1,117-dataset run under a new root.

**Architecture:** Keep the Exp436 feature, scoring, backend, input, checkpoint, progress, reporting, and evaluation modules as the shared engine. Add one immutable execution-profile module and pass either the Exp436 or Exp437 profile through identity-sensitive boundaries; keep existing public defaults bound to Exp436 for regression compatibility. Artifact schema strings remain the shared engine's `exp436.*.v1` format version, while persisted `experiment_id`, `run_identity`, fixed roots, manifest identity, and frozen-config identity distinguish Exp437.

**Tech Stack:** Python 3, `unittest`, NumPy, SQLite read-only inputs, optional PyTorch/MPS and MLX readiness benchmarks, canonical JSON/SHA256 artifacts, Darwin `RENAME_EXCL`, 11 CPU workers with one numerical thread each.

## Global Constraints

- Use only the frozen local database and lane artifacts; no new external data.
- Prediction must not read TEST labels, anomaly locations, or historical family-level TEST performance.
- TEST labels may be opened only after the Exp437 review-only prediction and hashes are frozen.
- Exp437 changes execution identity, fixed roots, and metadata persistence only; feature definitions, scores, `candidate_alpha=0.01`, thresholds, lanes, candidate budget 100, and success gates remain unchanged.
- The failed Exp436 root is immutable and must never be deleted, repaired, imported, or resumed.
- Exp437 recomputes all 1,117 datasets; none of Exp436's 47 committed records are reused.
- The official CPU policy is `max(1, logical_cpus - 1)`, currently 11 workers, with one numerical thread per worker.
- MLX/MPS is adopted only if the existing exact-candidate parity and 1.1x end-to-end speed gates pass; only one accelerator-heavy process may run.
- Every newly created project filename contains `antigravity`; never create `__init__.py`. The only filename exception is the Git-ignored `.superpowers/sdd/<plan-basename>/` control workspace required by `subagent-driven-development` (`progress.md`, generated task briefs/reports, and review packages); these files are temporary, untracked, and deleted after final review.
- Long execution writes atomic JSON progress, JSONL events, text logs, PID, current dataset metadata, checkpoint state, speed, elapsed time, ETA, and latest error itself.
- Regenerate `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, and `docs/codemap/codemap.lock` together whenever the module boundary changes.

## File Structure

- Create `experiments/exp436_missing_safe_strict_consensus_antigravity/execution_profile_antigravity.py`: the only definition of the two allowed execution identities and their fixed roots.
- Create `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_execution_profile_antigravity.py`: identity immutability, path isolation, and scientific-config equality tests.
- Modify `experiments/exp436_missing_safe_strict_consensus_antigravity/checkpoint_antigravity.py`: profile-bound context/store validation and insertion-order-independent length histograms.
- Modify `experiments/exp436_missing_safe_strict_consensus_antigravity/progress_antigravity.py`: insertion-order-independent length histograms; run identity continues to bind the shared progress schema.
- Modify `experiments/exp436_missing_safe_strict_consensus_antigravity/runtime_antigravity.py`: pass the selected profile through readiness, context, checkpoint, and inspection while leaving scoring unchanged.
- Modify `experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py`: publish and replay the selected experiment identity.
- Modify `experiments/exp436_missing_safe_strict_consensus_antigravity/evaluation_antigravity.py`: derive fixed paths and identity from the selected profile after prediction freeze.
- Modify `experiments/exp436_missing_safe_strict_consensus_antigravity/cli_antigravity.py`: expose a fixed-profile entrypoint without new user flags.
- Create `scratch/run_exp437_metadata_roundtrip_antigravity.py`: outside-CWD-safe Exp437 wrapper.
- Modify the existing Exp436 test modules named in each task; they remain the shared-engine regression suite.
- Create `docs/EXP437_PREREGISTRATION_antigravity_KO.md`: frozen scientific and transport commitments before readiness.
- Create `docs/EXP437_RUNBOOK_antigravity_KO.md`: exact readiness/new/resume/inspect/evaluate commands and immutable-failure handling.
- Create `docs/EXP437_IMPLEMENTATION_REPORT_antigravity.md`: final implementation evidence and readiness handoff.
- Modify `RESEARCH_STATE.json` and `EXPERIMENT_CATALOG.json`: phase, verified commit, exact next command, and Exp436/Exp437 lineage.
- Regenerate the three `docs/codemap/` files together.

---

### Task 1: Immutable Exp436/Exp437 execution profiles

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/execution_profile_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_execution_profile_antigravity.py`
- Regenerate together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: `Exp436FrozenConfig` and `FROZEN_CONFIG` from `config_antigravity.py`.
- Produces: `ExecutionProfileAntigravity`, `EXP436_PROFILE`, `EXP437_PROFILE`, `validated_profile_antigravity(profile)`, `frozen_config_for_profile_antigravity(profile)`, `frozen_config_payload_for_profile_antigravity(profile)`, and `validate_frozen_config_for_profile_antigravity(config, profile)`.
- Preserves: existing `FROZEN_CONFIG` and all scientific modules that import it.

- [ ] **Step 1: Write profile tests that freeze identity, roots, and the two-field config difference**

```python
from dataclasses import asdict
from pathlib import Path
import unittest

from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
    EXP436_PROFILE,
    EXP437_PROFILE,
    ExecutionProfileAntigravity,
    frozen_config_for_profile_antigravity,
    validated_profile_antigravity,
)


class ExecutionProfileAntigravityTests(unittest.TestCase):
    def test_exp437_identity_and_fixed_sibling_roots_are_frozen(self):
        self.assertEqual(EXP437_PROFILE.experiment_id, "Exp437")
        self.assertEqual(
            EXP437_PROFILE.run_identity,
            "exp437-metadata-roundtrip-antigravity-20260807",
        )
        self.assertEqual(
            EXP437_PROFILE.run_parent,
            Path("/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807"),
        )
        self.assertEqual(EXP437_PROFILE.output, EXP437_PROFILE.run_parent / "output_antigravity")
        self.assertEqual(EXP437_PROFILE.checkpoint, EXP437_PROFILE.run_parent / "checkpoint_antigravity")
        self.assertEqual(EXP437_PROFILE.readiness, EXP437_PROFILE.run_parent / "readiness_antigravity.json")
        self.assertEqual(EXP437_PROFILE.evaluation, EXP437_PROFILE.run_parent / "evaluation_antigravity")

    def test_profiles_differ_only_in_execution_identity_fields(self):
        left = asdict(frozen_config_for_profile_antigravity(EXP436_PROFILE))
        right = asdict(frozen_config_for_profile_antigravity(EXP437_PROFILE))
        self.assertEqual(
            {key for key in left if left[key] != right[key]},
            {"experiment_id", "run_identity"},
        )

    def test_arbitrary_profile_is_rejected(self):
        forged = ExecutionProfileAntigravity(
            experiment_id="Exp999",
            run_identity="forged-antigravity",
            cli_program="forged_antigravity.py",
            run_parent=Path("/tmp/forged_antigravity"),
        )
        with self.assertRaisesRegex(ValueError, "frozen execution profile"):
            validated_profile_antigravity(forged)
```

- [ ] **Step 2: Run the new test and verify the missing module failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_execution_profile_antigravity -v
```

Expected: FAIL with `ModuleNotFoundError` for `execution_profile_antigravity`.

- [ ] **Step 3: Implement the frozen profile and config projection**

```python
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .config_antigravity import Exp436FrozenConfig, FROZEN_CONFIG


@dataclass(frozen=True)
class ExecutionProfileAntigravity:
    experiment_id: str
    run_identity: str
    cli_program: str
    run_parent: Path

    @property
    def output(self) -> Path:
        return self.run_parent / "output_antigravity"

    @property
    def checkpoint(self) -> Path:
        return self.run_parent / "checkpoint_antigravity"

    @property
    def readiness(self) -> Path:
        return self.run_parent / "readiness_antigravity.json"

    @property
    def evaluation(self) -> Path:
        return self.run_parent / "evaluation_antigravity"


EXP436_PROFILE = ExecutionProfileAntigravity(
    "Exp436",
    "exp436-missing-safe-strict-consensus-antigravity-20260806",
    "run_exp436_missing_safe_strict_consensus_antigravity.py",
    Path("/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806"),
)
EXP437_PROFILE = ExecutionProfileAntigravity(
    "Exp437",
    "exp437-metadata-roundtrip-antigravity-20260807",
    "run_exp437_metadata_roundtrip_antigravity.py",
    Path("/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807"),
)
_ALLOWED_PROFILES = (EXP436_PROFILE, EXP437_PROFILE)
_DATA_ROOT = Path("/Users/mac/Documents/Dataset")


def validated_profile_antigravity(profile: object) -> ExecutionProfileAntigravity:
    if type(profile) is not ExecutionProfileAntigravity or profile not in _ALLOWED_PROFILES:
        raise ValueError("frozen execution profile is invalid")
    if (
        not profile.run_parent.is_absolute()
        or profile.run_parent.parent != _DATA_ROOT
        or profile.run_parent.is_symlink()
    ):
        raise ValueError("frozen execution profile path is invalid")
    return profile


def frozen_config_for_profile_antigravity(profile: object) -> Exp436FrozenConfig:
    selected = validated_profile_antigravity(profile)
    return replace(
        FROZEN_CONFIG,
        experiment_id=selected.experiment_id,
        run_identity=selected.run_identity,
    )


def frozen_config_payload_for_profile_antigravity(profile: object) -> dict[str, object]:
    config = frozen_config_for_profile_antigravity(profile)
    return json.loads(canonical_json_bytes(asdict(config)))


def validate_frozen_config_for_profile_antigravity(
    config: object,
    profile: object,
) -> Exp436FrozenConfig:
    expected = frozen_config_for_profile_antigravity(profile)
    if type(config) is not Exp436FrozenConfig or config != expected:
        raise ValueError("frozen config does not match execution profile")
    return config
```

- [ ] **Step 4: Run profile and frozen-config regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_execution_profile_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_config_antigravity -v
```

Expected: all tests PASS; the Exp436 profile produces the same `asdict(FROZEN_CONFIG)` bytes as before.

- [ ] **Step 5: Synchronize the code map with the new profile module**

Group `execution_profile_antigravity.py` and its focused test under the existing `exp436_contract_boundary` node. Add the exact source path and these symbols to node evidence: `ExecutionProfileAntigravity`, `EXP436_PROFILE`, and `EXP437_PROFILE`. Do not add a twenty-first primary node or invent callers before Task 5 wires the profile into the CLI. Regenerate HTML's embedded payload and lock fingerprints from the same JSON state, then run `python3 -m json.tool` on both JSON files and `git diff --check`.

- [ ] **Step 6: Commit the profile boundary and synchronized map**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/execution_profile_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_execution_profile_antigravity.py \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "refactor(exp437): add frozen execution profiles"
```

---

### Task 2: Canonical metadata roundtrip repair

**Files:**
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/checkpoint_antigravity.py:269-290`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/progress_antigravity.py:117-136`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_checkpoint_antigravity.py`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_progress_antigravity.py`

**Interfaces:**
- Consumes: canonical JSON maps whose keys are lexicographically ordered after decoding.
- Produces: `_dataset_metadata(value)` and `_validate_metadata(value)` that validate semantic contents and normalize with `dict(sorted(histogram.items()))`.
- Preserves: exact metadata keys, positive decimal length keys, nonnegative integer counts, and exact count sums.

- [ ] **Step 1: Add checkpoint and progress RED tests using lengths 29 and 102**

Add to the checkpoint tests:

```python
def test_dataset_metadata_survives_canonical_lexicographic_key_order(self) -> None:
    metadata = {
        "train_count": 2,
        "test_count": 2,
        "train_length_histogram": {"29": 1, "102": 1},
        "test_length_histogram": {"29": 1, "102": 1},
    }
    payload = _result_payload(_result("D0000", 2), 0, metadata)
    decoded = json.loads(canonical_json_bytes(payload))
    _result_from_payload(decoded, expected_sequence=0)
    self.assertEqual(
        list(decoded["dataset_metadata"]["train_length_histogram"]),
        ["102", "29"],
    )
```

Add to the progress tests:

```python
def test_progress_metadata_accepts_canonical_lexicographic_key_order(self) -> None:
    metadata = {
        "train_count": 2,
        "test_count": 2,
        "train_length_histogram": {"102": 1, "29": 1},
        "test_length_histogram": {"102": 1, "29": 1},
    }
    normalized = progress_antigravity._validate_metadata(metadata)
    self.assertEqual(list(normalized["train_length_histogram"]), ["102", "29"])
```

- [ ] **Step 2: Run both focused tests and verify the original failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_progress_antigravity -v
```

Expected: the new tests FAIL with `dataset record metadata is invalid` or `dataset metadata is invalid` because the reader still requires numeric insertion order.

- [ ] **Step 3: Replace order validation with semantic validation and lexical normalization**

Use this rule in both validators:

```python
pairs = list(histogram.items())
if any(
    type(key) is not str
    or not key.isdecimal()
    or key.startswith("0")
    or int(key) < 1
    or type(item) is not int
    or item < 0
    for key, item in pairs
):
    raise ValueError("dataset metadata is invalid")
if sum(item for _, item in pairs) != count:
    raise ValueError("dataset metadata is invalid")
result[name] = dict(sorted(pairs))
```

Remove only the numeric insertion-order comparisons. Do not relax the allowed fields, types, positive length keys, nonnegative counts, or sum checks.

- [ ] **Step 4: Verify GREEN tests and the actual immutable Exp436 record**

Run the two focused test modules again, then run this read-only check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json
from pathlib import Path
from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import _result_from_payload

record = Path(
    "/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/"
    "checkpoint_antigravity/records_antigravity/"
    "1f10eac76bc530cbb48ab776cf8fe1817d08acab7ce4976ccf2c58d0d4c04af0.json"
)
payload = json.loads(record.read_bytes())
result = _result_from_payload(payload)
print(result.dataset)
PY
```

Expected: focused tests PASS and the read-only check prints `AllGestureWiimoteX_normal_1`. The command must not write anywhere under the Exp436 root.

- [ ] **Step 5: Commit the persistence repair**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/checkpoint_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/progress_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_checkpoint_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_progress_antigravity.py
git commit -m "fix(exp437): accept canonical metadata key order"
```

---

### Task 3: Profile-bound checkpoint and runtime core

**Files:**
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/checkpoint_antigravity.py:44-127,325-587`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/runtime_antigravity.py:45-126,180-569`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_checkpoint_antigravity.py`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_runtime_antigravity.py`
- Regenerate together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: `ExecutionProfileAntigravity`, defaulting to `EXP436_PROFILE` on existing public calls.
- Produces: profile-aware `Exp436RunContext.from_payload`, `Exp436CheckpointStore` open/create methods, `_canonical_readiness`, `_build_run_context`, `run_exp436_new`, `resume_exp436`, and `inspect_exp436`.
- Preserves: shared `exp436.*.v1` transport schemas, scoring functions, CPU schedule, candidate budget, and ordered checkpoint authority.

- [ ] **Step 1: Add RED tests proving cross-profile artifacts are rejected before mutation**

```python
def test_exp437_context_roundtrip_is_profile_bound(self) -> None:
    payload = _context(EXP437_PROFILE).to_payload()
    parsed = Exp436RunContext.from_payload(payload, profile=EXP437_PROFILE)
    self.assertEqual(parsed.to_payload()["experiment_id"], "Exp437")
    with self.assertRaisesRegex(ValueError, "identity"):
        Exp436RunContext.from_payload(payload, profile=EXP436_PROFILE)

def test_exp437_readiness_and_context_are_profile_bound(self) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        path = _readiness_antigravity(
            runtime_antigravity,
            root,
            profile=EXP437_PROFILE,
        )
        readiness = runtime_antigravity._canonical_readiness(
            path,
            "a" * 40,
            root / "output_antigravity",
            root / "checkpoint_antigravity",
            profile=EXP437_PROFILE,
        )
        context = runtime_antigravity._build_run_context(
            session=_session_antigravity(),
            readiness=readiness,
            readiness_path=path,
            run_root=root / "output_antigravity",
            checkpoint_root=root / "checkpoint_antigravity",
            decision=BackendDecision.cpu_fallback(),
            profile=EXP437_PROFILE,
        )
        self.assertEqual(context.to_payload()["run_identity"], EXP437_PROFILE.run_identity)
        with self.assertRaisesRegex(ValueError, "identity"):
            runtime_antigravity._canonical_readiness(
                path,
                "a" * 40,
                root / "output_antigravity",
                root / "checkpoint_antigravity",
                profile=EXP436_PROFILE,
            )
```

Change the checkpoint helper signature to `def _context(profile=EXP436_PROFILE)` and build its config with `frozen_config_payload_for_profile_antigravity(profile)`, `profile.experiment_id`, and `profile.run_identity`. Change the runtime helper signature to `def _readiness_antigravity(runtime, root, *, profile=EXP436_PROFILE)` and build the same three fields from the selected profile. All existing calls keep the Exp436 default.

- [ ] **Step 2: Run checkpoint/runtime tests and verify the missing profile argument failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity -v
```

Expected: the new tests FAIL because `profile` is not accepted or Exp437 identity is compared with `FROZEN_CONFIG.run_identity`.

- [ ] **Step 3: Thread an explicit validated profile through the context and store**

Use keyword-only defaults to keep predecessor callers unchanged:

```python
@classmethod
def from_payload(
    cls,
    payload: Mapping[str, object],
    *,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> "Exp436RunContext":
    selected = validated_profile_antigravity(profile)
    config = frozen_config_payload_for_profile_antigravity(selected)
    if (
        value["experiment_id"] != selected.experiment_id
        or value["run_identity"] != selected.run_identity
    ):
        raise ValueError("run context identity is invalid")
    if value["frozen_config"] != config:
        raise ValueError("run context frozen config is invalid")
```

Store `self.profile = selected` in `Exp436CheckpointStore`. Add `profile=EXP436_PROFILE` to `create`, `_load_locked`, `open_existing`, `open_read_only`, `open_classified_read_only`, and `inspect_read_only`; pass it to every nested parser and constructor. Reject a stored Exp437 context when opened with the Exp436 default and vice versa before recovery or file mutation.

- [ ] **Step 4: Thread the same profile through readiness and runtime**

```python
def _frozen_config_payload(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> dict[str, object]:
    return frozen_config_payload_for_profile_antigravity(profile)

def run_exp436_new(
    *,
    readiness_path: Path,
    run_root: Path,
    checkpoint_root: Path,
    reviewed_commit: str,
    input_session_factory: Callable[[], object],
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Exp436RunSummary:
    selected = validated_profile_antigravity(profile)
    readiness = _canonical_readiness(
        readiness_path,
        reviewed_commit,
        run_root,
        checkpoint_root,
        profile=selected,
    )
    with _session_from_factory(input_session_factory) as session:
        return _run(
            run_root=Path(run_root),
            checkpoint_root=Path(checkpoint_root),
            readiness_path=Path(readiness_path),
            readiness=readiness,
            input_session=session,
            decision=_decision(readiness),
            resume=False,
            profile=selected,
        )
```

Give `resume_exp436` and `inspect_exp436` the identical keyword-only `profile: ExecutionProfileAntigravity = EXP436_PROFILE` parameter and begin each with `selected = validated_profile_antigravity(profile)`. `_build_run_context(..., profile=selected)` writes `selected.experiment_id`, `selected.run_identity`, and `frozen_config_payload_for_profile_antigravity(selected)`. Every `Exp436CheckpointStore` open/create call includes `profile=selected`. Worker tasks and scoring functions receive no profile and retain their existing signatures.

- [ ] **Step 5: Run focused and shared scientific regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_execution_profile_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_progress_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_features_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_scoring_antigravity -v
```

Expected: all tests PASS; Exp436 default behavior is unchanged, Exp437 context identity is accepted only with `EXP437_PROFILE`, and no scoring signature changed.

- [ ] **Step 6: Synchronize the code map with the profile-aware runtime API**

Update the existing `exp436_contract_boundary` node's role, entrypoints, tests, constraints, and source evidence for the profile-aware context/store/runtime signatures. Update only edges whose exact calls or imports appear in the changed source. Regenerate all three map files together and validate JSON, paths, symbols, edge endpoints/types, HTML payload identity, and lock fingerprints.

- [ ] **Step 7: Commit the profile-bound engine core and synchronized map**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/checkpoint_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/runtime_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_checkpoint_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_runtime_antigravity.py \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "refactor(exp437): bind runtime to execution profile"
```

---

### Task 4: Profile-bound review-only publication and post-freeze evaluation

**Files:**
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py:250-792`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/evaluation_antigravity.py:40-66,730-820,1150-1180,1510-1575`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_reporting_antigravity.py`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_evaluation_antigravity.py`
- Regenerate together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: a checkpoint store already bound to the selected execution profile.
- Produces: `publish_exp436(..., profile=EXP436_PROFILE)`, `inspect_publication(..., profile=EXP436_PROFILE)`, and `evaluate_exp436(..., profile=EXP436_PROFILE)`.
- Preserves: review-only publication, forbidden-key scan, no-clobber rename, prediction hashes, delayed label import, paired bootstrap, and frozen success gates.

- [ ] **Step 1: Add RED tests for Exp437 manifest identity and fixed evaluation siblings**

```python
def test_exp437_publication_manifest_uses_profile_identity(self) -> None:
    publication = publish_exp436(
        completed_store,
        output,
        profile=EXP437_PROFILE,
    )
    self.assertEqual(publication.manifest["experiment_id"], "Exp437")
    self.assertEqual(publication.manifest["run_identity"], EXP437_PROFILE.run_identity)
    with self.assertRaisesRegex(ValueError, "identity"):
        inspect_publication(output, completed_store, profile=EXP436_PROFILE)

def test_exp437_evaluation_rejects_exp436_fixed_paths_before_labels(self) -> None:
    with patch.object(evaluation_antigravity, "_load_test_labels") as labels:
        with self.assertRaisesRegex(ValueError, "fixed paths"):
            evaluation_antigravity.evaluate_exp436(
                EXP436_PROFILE.output,
                EXP436_PROFILE.evaluation,
                profile=EXP437_PROFILE,
            )
    labels.assert_not_called()

def test_public_evaluation_keeps_only_a_keyword_profile_extension(self) -> None:
    signature = inspect.signature(evaluation_antigravity.evaluate_exp436)
    self.assertEqual(tuple(signature.parameters), ("output_root", "evaluation_root", "profile"))
    self.assertEqual(
        signature.parameters["profile"].kind,
        inspect.Parameter.KEYWORD_ONLY,
    )
```

- [ ] **Step 2: Run reporting/evaluation tests and verify profile support is absent**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_evaluation_antigravity -v
```

Expected: the new tests FAIL because these entrypoints do not yet accept `profile`.

- [ ] **Step 3: Make publication identity and config hashes profile-aware**

```python
def publish_exp436(
    completed_store: object,
    output: Path,
    *,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> FrozenPublication:
    selected = validated_profile_antigravity(profile)
    return _publish_exp436(completed_store, output, profile=selected)
```

Give `inspect_publication` and `_inspect_publication` the same keyword-only profile default. At the start of `_open_store`, `_publication_payloads`, `_inspect_publication`, and `_publish_exp436`, set `selected = validated_profile_antigravity(profile)` and require `store.profile == selected`, `store.run_identity == selected.run_identity`, and matching context identity. Build `config_hash` from `frozen_config_payload_for_profile_antigravity(selected)` and write `selected.experiment_id`/`selected.run_identity` to the manifest. Semantic rows and candidate accounting retain their current code and signatures.

- [ ] **Step 4: Make evaluation derive exact paths from the profile**

```python
def evaluate_exp436(
    output_root: Path,
    evaluation_root: Path,
    *,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Exp436Evaluation:
    selected = validated_profile_antigravity(profile)
    fixed_output = selected.output
    fixed_checkpoint = selected.checkpoint
    fixed_evaluation = selected.evaluation
    run_parent = selected.run_parent
```

Open the checkpoint and inspect the publication with `profile=selected`. Use `frozen_config_for_profile_antigravity(selected)` wherever evaluation validates config identity or hashes. Preserve the lazy evaluator import from the CLI, and prove fixed-path/profile mismatches occur before `_load_test_labels` or the verified SQLite label reader can run.

Replace both production `config.validate()` calls with `validate_frozen_config_for_profile_antigravity(config, selected)`. Update the existing public-signature assertion from two parameters to the three-parameter tuple shown in Step 1; `profile` must remain keyword-only so no alternate path or scientific value becomes positional CLI input.

- [ ] **Step 5: Run reporting/evaluation and label-firewall regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_evaluation_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity -v
```

Expected: all tests PASS; Exp437 manifests carry only Exp437 execution identity, Exp436 defaults replay unchanged, and mismatched profile/path calls never open labels.

- [ ] **Step 6: Synchronize the code map with publication/evaluation profile flow**

Update the existing Exp436 boundary node and post-freeze flow with the exact profile-aware reporting/evaluation symbols and calls. Preserve the TEST-label boundary and do not add a relationship without source evidence. Regenerate all three map files together and validate JSON, source symbols, flow nodes, HTML payload identity, and lock fingerprints.

- [ ] **Step 7: Commit profile-aware publication/evaluation and synchronized map**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/evaluation_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_reporting_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_evaluation_antigravity.py \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "refactor(exp437): bind publication and evaluation profile"
```

---

### Task 5: Fixed five-command Exp437 wrapper

**Files:**
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/cli_antigravity.py:31-418`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_cli_antigravity.py`
- Create: `scratch/run_exp437_metadata_roundtrip_antigravity.py`
- Regenerate together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: `EXP436_PROFILE`, `EXP437_PROFILE`, and all profile-aware shared-engine entrypoints.
- Produces: `main_for_profile_antigravity(profile, argv=None)`; existing `main(argv=None)` remains Exp436-only.
- CLI surface: exactly `readiness`, `new`, `resume`, `inspect`, and `evaluate`; no path, worker, backend, seed, threshold, alpha, lane, or budget flags.

- [ ] **Step 1: Add RED tests for parser identity, fixed roots, and zero override flags**

```python
def test_exp437_parser_and_dependencies_are_bound_to_the_new_profile(self):
    parser = module.build_parser(profile=EXP437_PROFILE)
    self.assertIn(EXP437_PROFILE.cli_program, parser.prog)
    deps = module._production_dependencies(EXP437_PROFILE)
    self.assertEqual(deps.output, EXP437_PROFILE.output)
    self.assertEqual(deps.checkpoint, EXP437_PROFILE.checkpoint)
    self.assertEqual(deps.readiness, EXP437_PROFILE.readiness)
    with self.assertRaises(SystemExit):
        parser.parse_args(["new", "--workers", "1"])

def test_exp437_wrapper_runs_from_outside_repository_with_no_override_flags(self):
    wrapper = Path(__file__).resolve().parents[3] / "scratch/run_exp437_metadata_roundtrip_antigravity.py"
    completed = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)
    self.assertIn("run_exp437_metadata_roundtrip_antigravity.py", completed.stdout)
    self.assertIn("{readiness,new,resume,inspect,evaluate}", completed.stdout)
    self.assertNotIn("--workers", completed.stdout)
    self.assertNotIn("--backend", completed.stdout)
```

- [ ] **Step 2: Run CLI tests and verify the profile-aware entrypoint is absent**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity -v
```

Expected: the new tests FAIL because `build_parser(profile=...)`, `_production_dependencies(profile)`, and the Exp437 wrapper do not exist.

- [ ] **Step 3: Refactor CLI constants into selected profile paths**

```python
def build_parser(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> argparse.ArgumentParser:
    selected = validated_profile_antigravity(profile)
    parser = argparse.ArgumentParser(prog=selected.cli_program, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "new", "resume", "inspect", "evaluate"):
        commands.add_parser(name, allow_abbrev=False)
    return parser


def main_for_profile_antigravity(
    profile: ExecutionProfileAntigravity,
    argv: list[str] | None = None,
) -> int:
    selected = validated_profile_antigravity(profile)
    return _main_with_dependencies(argv, _production_dependencies(selected), profile=selected)


def main(argv: list[str] | None = None) -> int:
    return main_for_profile_antigravity(EXP436_PROFILE, argv)
```

Make readiness generation, validation, fixed-root preflight, runtime calls, publication replay, and lazy evaluation receive the selected profile. Keep `RUN_PARENT`, `FROZEN_OUTPUT`, `FROZEN_CHECKPOINT`, `FROZEN_READINESS`, and `FROZEN_EVALUATION` as Exp436 compatibility aliases only; no Exp437 code may read those aliases.

- [ ] **Step 4: Create the outside-CWD-safe Exp437 wrapper**

```python
"""Outside-CWD-safe fixed Exp437 CLI wrapper."""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.exp436_missing_safe_strict_consensus_antigravity.cli_antigravity import (
    main_for_profile_antigravity,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
    EXP437_PROFILE,
)

if __name__ == "__main__":
    raise SystemExit(main_for_profile_antigravity(EXP437_PROFILE))
```

- [ ] **Step 5: Verify both wrappers and the label-free import boundary**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_evaluation_antigravity -v
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py --help
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import sys
from experiments.exp436_missing_safe_strict_consensus_antigravity.cli_antigravity import main_for_profile_antigravity
from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import EXP437_PROFILE
assert "experiments.exp436_missing_safe_strict_consensus_antigravity.evaluation_antigravity" not in sys.modules
print(EXP437_PROFILE.run_parent)
PY
```

Expected: tests PASS; help shows only five commands; importing the Exp437 prediction boundary does not import evaluation; the printed path is the fixed Exp437 parent.

- [ ] **Step 6: Synchronize the code map with the fixed Exp437 CLI flow**

Add the fixed wrapper path and `main_for_profile_antigravity` source evidence to the existing Exp436 boundary node. Add or update one Exp437 readiness/new/evaluate flow only where every step references an existing node and every edge has an exact source symbol. Keep the map at 20 primary nodes or fewer, regenerate all three files together, and validate JSON, HTML payload identity, evidence, endpoints, edge types, flows, and fingerprints.

- [ ] **Step 7: Commit the Exp437 command boundary and synchronized map**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/cli_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_cli_antigravity.py \
  scratch/run_exp437_metadata_roundtrip_antigravity.py \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "feat(exp437): add fixed profile command wrapper"
```

---

### Task 6: Freeze Exp437 governance before readiness

**Files:**
- Create: `docs/EXP437_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP437_RUNBOOK_antigravity_KO.md`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Regenerate together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: final implemented commands and fixed profile constants.
- Produces: an auditable `implementation_complete_readiness_pending` state and exact operator commands.
- Preserves: Exp436 failed root, readiness artifact hash, 47-record authority, one orphan record, and zero TEST-label access as immutable history.

- [ ] **Step 1: Write the preregistration with exact frozen values**

The document must state these exact commitments in Korean:

```text
Experiment: Exp437
Run identity: exp437-metadata-roundtrip-antigravity-20260807
Dataset population: 1,117 sorted unique datasets
Candidate alpha: 0.01
Candidate budget: 100
CPU workers: max(1, logical_cpus - 1), expected 11 on this Mac
Scientific delta from Exp436: none
Transport delta: canonical length-histogram validation ignores insertion order and normalizes lexicographically
Prediction label access: zero
Exp436 checkpoint reuse: forbidden
Evaluation: only after prediction, manifest, and artifact hashes are frozen
```

Use the sections `Prediction boundary`, `Frozen scientific rule`, `Backend readiness`, `Candidate budget`, and `Post-freeze evaluation` from `docs/EXP436_PREREGISTRATION_antigravity_KO.md` as the normative source. In each Exp437 section, state that the corresponding Exp436 rule and numeric value is unchanged, then add only the new identity/root and the lexical histogram-normalization rule listed above. Verify with `git diff --word-diff` that no alpha, threshold, lane, budget, seed, bootstrap, rescue, or false-positive value differs from Exp436.

- [ ] **Step 2: Write the runbook with exact commands and immutable failure rules**

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery'
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py readiness
PYTHONDONTWRITEBYTECODE=1 nohup python3 scratch/run_exp437_metadata_roundtrip_antigravity.py new \
  > /Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/launcher_antigravity.log 2>&1 &
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py inspect
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py resume
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py evaluate
```

The runbook must explain that `resume` is legal only for an `interrupted` Exp437 root. A `failed` root is immutable and requires a new experiment number. It must list the progress JSON, event JSONL, parent log, PID file, checkpoint directory, publication directory, and evaluation directory.

- [ ] **Step 3: Update research state and experiment catalog**

Set:

```json
{
  "active_experiment": "Exp437",
  "active_phase": "implementation_complete_readiness_pending",
  "prediction_phase_started": false,
  "test_labels_opened": false
}
```

Record the final implementation commit only after Task 7's verified commit exists. Until then, use the last verified implementation commit field already present and make `next_action` the clean-HEAD readiness review. Keep the complete Exp436 failure record unchanged.

- [ ] **Step 4: Regenerate and validate the code map triplet**

Add the execution-profile boundary and Exp437 fixed-wrapper flow to the existing 20-node map by grouping them under the current `exp436_contract_boundary`; do not exceed 20 primary nodes. Attach real source paths and symbols to every changed node/edge. Regenerate HTML's embedded payload and lock fingerprints from the same JSON state.

Run:

```bash
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
```

Expected: all commands exit 0; HTML and JSON carry identical nodes/edges/flows; node/evidence paths and symbols resolve; every edge and flow step references an existing node; edge types are limited to `imports`, `calls`, `reads`, `writes`, `publishes`, `subscribes`; unsupported relationships are marked `unknown`.

- [ ] **Step 5: Commit the frozen governance state**

```bash
git add docs/EXP437_PREREGISTRATION_antigravity_KO.md \
  docs/EXP437_RUNBOOK_antigravity_KO.md \
  RESEARCH_STATE.json EXPERIMENT_CATALOG.json \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(exp437): freeze preregistration and runbook"
```

---

### Task 7: Full regression, immutable-record proof, and readiness handoff

**Files:**
- Create: `docs/EXP437_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Regenerate together if state evidence changes: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: all Tasks 1-6 commits.
- Produces: one clean reviewed commit eligible for exactly one official Exp437 readiness invocation.
- Does not produce: readiness, prediction, publication, or evaluation artifacts.

- [ ] **Step 1: Run the complete Exp436 shared-engine suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p 'test_*_antigravity.py' -v
```

Expected: all tests PASS with zero errors and zero failures.

- [ ] **Step 2: Run predecessor boundary regressions**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_progress_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_runtime_antigravity \
  experiments.exp425_selective_recovery.tests.test_database -v
```

Expected: all selected predecessor tests PASS; Exp435/Exp425 behavior and database read-only boundaries are unchanged.

- [ ] **Step 3: Prove Exp436 failure evidence is unchanged and now readable**

```bash
shasum -a 256 \
  /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/readiness_antigravity.json
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json
from pathlib import Path
from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import _result_from_payload

root = Path("/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806")
record = root / "checkpoint_antigravity/records_antigravity/1f10eac76bc530cbb48ab776cf8fe1817d08acab7ce4976ccf2c58d0d4c04af0.json"
before = record.read_bytes()
parsed = _result_from_payload(json.loads(before))
after = record.read_bytes()
assert before == after
assert parsed.dataset == "AllGestureWiimoteX_normal_1"
print(parsed.dataset, len(before))
PY
```

Expected: readiness SHA256 is `5def8dbe785163dc69e80e0bb6a5a1cdba1d3663058f4d73625a6deda048c43a`; the record parser prints the failed dataset; bytes are unchanged.

- [ ] **Step 4: Verify Exp437 roots are absent and the wrapper is fixed**

```bash
test ! -e /Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py --help
git grep -nE 'test_label|groundtruth|anomaly_location|family.*test' -- \
  experiments/exp436_missing_safe_strict_consensus_antigravity/cli_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/runtime_antigravity.py \
  scratch/run_exp437_metadata_roundtrip_antigravity.py
```

Expected: the root absence check passes; help exposes only the five fixed commands; any grep hits are rejection counters, comments, or delayed-evaluation guards and are listed in the report with their source symbol.

- [ ] **Step 5: Write the implementation report and final state**

Record:

```text
Reviewed implementation commit
Test commands and exact counts
Actual failed-record read-only proof
Exp436 readiness SHA256
Exp437 root absence
Exp436 versus Exp437 frozen-config field diff
CPU topology expectation 12/11/1
MLX/MPS policy and readiness-pending status
Prediction started: false
TEST labels opened: false
Exact next command: python3 scratch/run_exp437_metadata_roundtrip_antigravity.py readiness
```

Set `last_verified_commit` to the clean implementation commit being reviewed, keep `active_phase` as `implementation_complete_readiness_pending`, and regenerate the code-map triplet together if the changed state evidence is mapped.

- [ ] **Step 6: Validate report/state/map and commit the handoff**

```bash
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git status --short
git add docs/EXP437_IMPLEMENTATION_REPORT_antigravity.md \
  RESEARCH_STATE.json EXPERIMENT_CATALOG.json \
  docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(exp437): record readiness handoff"
```

Expected: the commit succeeds and `git status --short` is empty afterward. Do not run official readiness in this task; readiness must start from this final clean reviewed commit.

---

## Final implementation gate

Before the official readiness command is allowed, confirm all of the following from fresh command output:

- The full shared-engine suite and selected predecessor regressions have zero failures and zero errors.
- Exp436's readiness hash and failed record bytes are unchanged.
- The failed record parses only because histogram object order is no longer treated as meaning.
- Exp436 defaults still bind the Exp436 profile; Exp437 wrapper binds only the Exp437 profile.
- Exp437 frozen config differs from Exp436 only in `experiment_id` and `run_identity`.
- Exp437 run parent is absent.
- Prediction and evaluation have not started, and TEST-label access remains zero.
- The worktree is clean and the code-map triplet represents the committed module boundary.
