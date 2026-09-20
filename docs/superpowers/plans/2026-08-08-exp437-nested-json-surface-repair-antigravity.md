# Exp437 Nested JSON Surface Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the official Exp437 publication and inspection path accept its valid recursively frozen JSON backend metadata without changing checkpoint, prediction, or scientific bytes.

**Architecture:** Keep `BackendDecision.from_payload` strict and canonical. Add one private recursive thaw helper at the Exp436 reporting boundary so mapping proxies become dictionaries and tuple/list JSON arrays become lists before strict reconstruction.

**Tech Stack:** Python 3.14, `collections.abc.Mapping`, immutable checkpoint adapters, canonical JSON, SHA256, `unittest`.

## Global Constraints

- Every newly created filename contains `antigravity`.
- Do not modify the Exp437 checkpoint or publication artifact bytes.
- Do not open SQLite, TEST values, TEST labels, or anomaly locations.
- Preserve `BackendDecision.from_payload` strict derived-field validation.
- Follow TDD: focused RED, minimal GREEN, focused and full regression verification.
- Regenerate `docs/codemap/codemap.html`, `codemap.json`, and `codemap.lock` together if source evidence changes.

---

### Task 1: Reproduce nested tuple/list rejection and repair the reporting adapter

**Files:**
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_reporting_antigravity.py`
- Modify: `experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py`

**Interfaces:**
- Consumes: `BackendDecision.from_payload(payload: Mapping[str, object])` and recursively frozen `store.context`.
- Produces: `_canonical_json_surface_antigravity(value: object) -> object`; `_backend_from_context` still returns `BackendDecision`.

- [ ] **Step 1: Write the failing canonical-surface test**

Import `_backend_from_context` into the reporting test and add:

```python
def test_backend_context_accepts_recursively_frozen_json_arrays(self) -> None:
    raw = _frozen_context(EXP437_PROFILE).to_payload()["backend_decision"]

    def freeze(value):
        if isinstance(value, dict):
            return MappingProxyType({key: freeze(child) for key, child in value.items()})
        if isinstance(value, list):
            return tuple(freeze(child) for child in value)
        return value

    restored = _backend_from_context({"backend_decision": freeze(raw)})
    self.assertEqual(restored.to_payload(), raw)
```

Add `MappingProxyType` and `_backend_from_context` imports. The production change that makes this pass is recursive list restoration; changing any scientific value is not sufficient.

- [ ] **Step 2: Run RED and confirm the exact reason**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity.ReportingAntigravityTests.test_backend_context_accepts_recursively_frozen_json_arrays -v
```

Expected: FAIL with `checkpoint backend decision is invalid`, caused by `backend decision derived fields are invalid`.

- [ ] **Step 3: Add the tampering RED case**

Copy the canonical payload, increase the first MLX timing pair's CPU duration by `1.0`, recursively freeze it, and assert `_backend_from_context` raises `ValueError`. This proves the repair preserves derived median/speed validation.

```python
tampered = copy.deepcopy(raw)
tampered["mlx"]["batches"][0]["timing_pairs"][0][0] += 1.0
with self.assertRaisesRegex(ValueError, "checkpoint backend decision is invalid"):
    _backend_from_context({"backend_decision": freeze(tampered)})
```

- [ ] **Step 4: Implement the minimal recursive JSON surface**

In `reporting_antigravity.py`, add:

```python
def _canonical_json_surface_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _canonical_json_surface_antigravity(child)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_json_surface_antigravity(child) for child in value]
    return value
```

Replace the current shallow `batches` conversion with:

```python
raw = _canonical_json_surface_antigravity(payload)
if not isinstance(raw, Mapping):
    raise ValueError("checkpoint backend decision is invalid")
return BackendDecision.from_payload(raw)
```

Do not change the exception wrapper or the public decoder.

- [ ] **Step 5: Run GREEN and the full reporting suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity -v
```

Expected: all reporting tests PASS, including the new acceptance and tampering tests.

- [ ] **Step 6: Commit the isolated repair**

```bash
git diff --check
git add \
  experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_reporting_antigravity.py
git commit -m "fix(exp437): restore nested json metadata surface"
```

---

### Task 2: Verify the real immutable Exp437 artifact and downstream evaluation boundary

**Files:**
- Read only: `/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/checkpoint_antigravity`
- Read only: `/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity/publication_antigravity`
- Modify together if source evidence changes: `docs/codemap/codemap.html`
- Modify together if source evidence changes: `docs/codemap/codemap.json`
- Modify together if source evidence changes: `docs/codemap/codemap.lock`

**Interfaces:**
- Consumes: fixed `inspect_publication`, `EXP437_PROFILE`, real completed store.
- Produces: reproducible read-only verification evidence and unchanged prediction SHA256.

- [ ] **Step 1: Run shared-engine regression tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p 'test_*_antigravity.py'
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Inspect the real publication without an adapter**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path
from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import Exp436CheckpointStore
from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import EXP437_PROFILE
from experiments.exp436_missing_safe_strict_consensus_antigravity.reporting_antigravity import inspect_publication

store = Exp436CheckpointStore.open_read_only(
    EXP437_PROFILE.checkpoint,
    profile=EXP437_PROFILE,
)
publication = inspect_publication(
    EXP437_PROFILE.output,
    store,
    profile=EXP437_PROFILE,
)
assert publication.prediction_sha256 == "f4a824eea284ecc67ebe5284669b81059a1c0ce52530a69c90c493dd42e699ef"
print(publication.prediction_sha256)
PY
```

Expected: exact SHA printed, no file writes, and no TEST-label operation.

- [ ] **Step 3: Run Exp439 evaluator-focused regression without recreating evaluation**

Run only unit tests with temporary fixtures:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp439_scale_aware_reconstruction_repair_antigravity.tests.test_evaluation_antigravity \
  experiments.exp439_scale_aware_reconstruction_repair_antigravity.tests.test_cli_antigravity -v
```

Expected: zero failures and zero errors. Do not invoke the real Exp439 `evaluate` command because its no-clobber evaluation root already exists.

- [ ] **Step 4: Synchronize code-map evidence and commit if required**

Add `_canonical_json_surface_antigravity` to the existing `exp436_contract_boundary` node evidence. Keep all 20 primary nodes and existing edges. Regenerate HTML's embedded base64 payload and lock fingerprints from the same JSON state, then run:

```bash
python3 -m json.tool docs/codemap/codemap.json >/dev/null
python3 -m json.tool docs/codemap/codemap.lock >/dev/null
git diff --check
git add docs/codemap/codemap.html docs/codemap/codemap.json docs/codemap/codemap.lock
git commit -m "docs(codemap): record nested json repair"
```
