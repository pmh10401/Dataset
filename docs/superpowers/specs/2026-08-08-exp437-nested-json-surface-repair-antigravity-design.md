# Exp437 Nested JSON Surface Repair Design

## Purpose

Repair the operational reader that rejects the completed Exp437 checkpoint and
publication when recursively frozen JSON arrays are represented as tuples in
memory. This is a transport-only repair. It must not change Exp437 prediction
coordinates, scores, thresholds, candidate budgets, checkpoint bytes, or the
TEST-label boundary.

## Observed failure

The completed Exp437 checkpoint contains a valid canonical JSON
`backend_decision`. `Exp436CheckpointStore.open_read_only` recursively freezes
JSON objects and arrays as mapping proxies and tuples. During publication replay,
`reporting_antigravity._backend_from_context` restores the top-level `batches`
collection to a list but leaves each batch's nested `timing_pairs` as tuples.
`BackendDecision.from_payload` reconstructs canonical list-valued timing pairs
and then correctly rejects the noncanonical tuple/list mismatch with:

```text
ValueError: checkpoint backend decision is invalid
caused by: backend decision derived fields are invalid
```

The Exp437 checkpoint is complete at 1,117/1,117 with zero prediction errors.
The failure occurs only while replaying metadata for publication inspection.

## Chosen approach

Add one narrow recursive JSON-surface decoder inside the Exp436 reporting
boundary. It converts `Mapping` objects to ordinary dictionaries and tuple/list
arrays to ordinary lists before calling the already strict
`BackendDecision.from_payload` decoder.

The strict decoder remains authoritative. Unknown keys, invalid derived fields,
nonfinite timing values, incorrect medians, incorrect backend selection, and
changed hashes must continue to fail closed.

### Alternatives rejected

1. Editing the Exp437 checkpoint in place is forbidden because the completed
   checkpoint is immutable evidence.
2. Relaxing `BackendDecision.from_payload` to accept arbitrary tuples would
   weaken the public canonical JSON contract for every caller.
3. Keeping an evaluation-only monkey patch would leave the official CLI broken
   and make the recovery non-reproducible.

## Scope and interfaces

Modify only the shared reporting adapter and its focused tests:

- `experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py`
- `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_reporting_antigravity.py`
- focused evaluation/CLI regression tests where needed

The private helper accepts a JSON-compatible recursively frozen value and
returns a canonical mutable JSON surface. `_backend_from_context` passes the
decoded backend decision to `BackendDecision.from_payload` without changing any
scientific value.

## Data and security boundaries

- Do not write to the Exp437 checkpoint or existing publication.
- Do not open the SQLite database, TEST values, TEST labels, or anomaly
  locations while testing the decoder.
- Do not read Exp439 evaluation artifacts from prediction code.
- Preserve exact SHA256 verification for context, records, publication, and
  prediction artifacts.
- Keep `review_only=true` and `operational_alert_eligible=false`.

## Tests

TDD must demonstrate the failure before production code changes.

1. A recursively frozen valid backend payload with tuple-valued
   `batches[*].timing_pairs` is accepted and reconstructs the exact canonical
   JSON payload.
2. A tampered nested timing value is rejected by the existing derived-field
   validation.
3. A completed synthetic Exp437 publication can be inspected from a read-only
   checkpoint without a compatibility proxy.
4. Existing Exp436/Exp437 reporting, evaluation, and CLI tests remain green.
5. A read-only live check verifies the real Exp437 publication and confirms its
   prediction SHA256 remains
   `f4a824eea284ecc67ebe5284669b81059a1c0ce52530a69c90c493dd42e699ef`.

## Code-map impact

The existing `exp436_contract_boundary` node remains the correct module. The
repair changes no module boundary. Its callers include Exp437 publication/CLI
and Exp439 retrospective evaluation; its downstream effect is strict frozen
publication validation. Related tests are the Exp436 reporting, evaluation, and
CLI suites. The code map must be regenerated with the repair commit.
