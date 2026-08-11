# Exp446 Task 3 report: closed profile and runtime contracts

## Scope and map contract

- Base commit: `a295bd63c2233bc1a0de56db0d25b73be339879f`.
- Caller: the shared Exp444 lifecycle modules are the first concrete Exp446
  consumer; `readiness_for_profile_antigravity` and
  `_verify_exp446_predecessor_antigravity` import the Task 4 verifier.
- Effect: the closed singleton routes Exp446 checkpoint context, readiness,
  fixed-new runtime, and reporting profile dispatch. The later Exp446 lifecycle
  and publication wrapper remains intentionally unmapped/pending.
- Tests: the new focused profile-contract suite; existing profile, contract,
  and readiness suites; and the Task 4 predecessor suite.

## TDD evidence

RED command:

```bash
python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_profile_contracts_antigravity
```

RED result: 3 failures, each correctly reporting the missing
`EXP446_PROFILE_ANTIGRAVITY` contract before production implementation.

GREEN commands and results:

```bash
python3 -m unittest -v experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_predecessor_antigravity experiments.exp446_exp445_publication_materialization_repair_antigravity.tests.test_profile_contracts_antigravity
# 10 tests OK

python3 -m unittest -v experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_profiles_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity
# 18 tests OK
```

The existing profile test had an exact-two-export assertion incompatible with
the approved exact-three singleton design and with the required regression
command. The task owner explicitly authorized the bounded update of that one
test to assert Exp444/Exp445/Exp446 only.

## Compatibility evidence

- `test_exp444_public_signatures_remain_frozen_antigravity` passed: the public
  Exp444 config/context/CLI signatures are unchanged.
- `test_legacy_checkpoint_context_payload_remains_eight_fields_antigravity`
  passed: Exp444 context serialization retains its original eight fields.
- `ProfileCheckpointContextAntigravity` now selects Exp445 or Exp446 document
  authority internally; Exp445's field set and Exp444's legacy class are not
  changed.
- New tests reject a caller-minted equal Exp446 object at readiness, runtime,
  and reporting boundaries; reject altered identity/root/schema; reject an
  Exp445-shaped context; and reject a label-capable context field.

## Readiness gates

| Gate before allocation | Contract evidence |
| --- | --- |
| Current clean reviewed HEAD | `validate_execution_identity_antigravity` runs before profile readiness continues. |
| Exp446 design and preregistration bytes | `_profile_document_sha256_antigravity` pins `3fce...d281` and `440d...5838`. |
| Scientific/environment topology | frozen scientific/environment authority requires CPU, 12 logical CPUs, outer 11, inner 1. |
| Source DB/link and labels | values-only source verification plus Task 4 descriptor-pinned DB single-link verification; readiness evidence requires label access `0`. |
| Immutable Exp445 predecessor | Exp446 selects the real Task 4 `verify_exp445_predecessor_antigravity` before source materialization. |
| Exp446 root | profile readiness rejects an existing or symlinked fixed root before any allocation. |

## Read-only operational evidence

No official Exp446 readiness, `new`, lifecycle, publication, evaluation, or
label-capable operation was run. The Task 4 verifier was called twice read-only
and returned identical values-only authority: 30 record SHA values and aggregate
`af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`.
The official Exp446 root was explicitly verified absent and non-symlink after
both calls.

## Code-map and final checks

- Regenerated `codemap_exp446_antigravity.{json,html,lock}` together.
- Validation passed: `evidence=1090 nodes=20 edges=60 flows=6 fingerprints=9`;
  JSON parsed, HTML embedded bytes matched JSON, every evidence path/symbol and
  edge/flow endpoint resolved, and deterministic fingerprints matched the lock.
- `git diff --check --cached` passed.

## Commit and concerns

Implementation commit: `22f1957cb7eed90535ff16540a219f7692f982f5`

Concern: the official readiness command is intentionally not green-run here:
its contract requires a clean reviewed HEAD, while this implementation task has
uncommitted work. The read-only predecessor/root check provides the permitted
operational evidence; a later lifecycle task must run readiness from its exact
reviewed clean commit.
