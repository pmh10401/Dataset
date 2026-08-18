# Exp449 Smoke A pinned-grant lifecycle — implementation I review evidence

## State

```text
implementation_status = implementation_ready_awaiting_grant_authorization
smoke_started = false
official_run_started = false
test_labels_opened = false
official_label_access_count = 0
all_six_fixed_roots = absent
smoke_a_claim = absent
smoke_a_grant = absent
smoke_a_grant_digest = absent
```

This record covers implementation-review evidence only. It does not claim a Smoke A,
Smoke B, official run, TEST-label access, scoring result, AP, AUROC, or scientific
performance result.

## I boundary and the next authorization boundary

The final reviewed implementation HEAD is `I`: it contains the pinned Smoke A
authority parser/verifier, exact A→I delta checks, process-local capability,
claim/flock consumption, fresh/resume lifecycle, durable progress, fixed-path worker
handoff, and scorer-free status/proposal verification. `I` contains **no authority
JSON** and must remain execution-inert without one.

The following stage is deliberately separate:

```text
clean I + independent Sol SHIP
  → read-only prepare-smoke-a-grant at I
  → exactly two canonical authority JSON files
  → direct-child A + independent Sol SHIP
  → user says “Smoke A 실행 승인” for exact A
  → separate execution turn may launch Smoke A
```

A pinned digest limits accidental reuse or tampering of an exact approved Git delta;
it is not a signature system and does not protect against an attacker who can both
rewrite repository history and impersonate the user's approval. The exact `A` SHA,
`I` SHA, grant SHA-256, roots, claim, CPU provenance, and launch command must be
presented immediately before any launch.

## Frozen operational facts

- The six fixed production roots and the Smoke A claim path are those listed in the
  runbook. At implementation review they are lstat-absent and are never created by
  the bounded tests.
- The only future authority filenames are
  `smoke_a_execution_grant_antigravity.json` and
  `smoke_a_execution_grant_digest_antigravity.json` under `docs/authorities/`; both
  are absent at implementation review and none is tracked by `I`.
- The canonical backend is CPU. The host provenance is logical CPU `12`, dataset
  workers `11` (`max(1, 12 - 1)`). PyTorch/MPS and MLX remain noncanonical unless a
  separate exact-stage parity and end-to-end speed gate is recorded; CPU is the
  deterministic fallback.
- Fresh execution allows at most two Task4/5 waves. It cannot bypass the exact
  authorization delta, fixed roots, claim, receipt, or process-local capability.
- Resume creates no root and never recomputes a verified success. It rejects changed
  authority, receipt, claim, context, catalog, database, closure, environment,
  backend, worker, checkpoint, or attempt facts. A verified freeze interrupted before
  complete terminal can only perform scorer-free terminal recovery.
- Progress append/fsync failure is terminal for further scheduling. Derived status is
  read-only/non-authoritative cache data; its observation opens neither scorer nor DB.
- Two-wave noncompletion records `proposal_score_incomplete` and cannot enter
  Task6/7, Candidate C, freeze, or a complete terminal.

## Verification scope

The Task 7 implementation-review run uses the pinned Python 3.14.6 interpreter and
temporary fixture seams. It runs Exp449 discovery, the curated paper suite while
excluding exactly the known production-DB raw-smoke test, fresh import label/scorer
firewalls, repeated lifecycle/authority subprocess tests, changed-Python compilation,
codemap semantic/fingerprint checks, diff whitespace checks, and lstat absence checks.
It does not invoke real-DB `readiness`, prepare a grant, create roots/claim/logs,
score, import TEST labels, or launch any smoke/official target. The exact command
outputs and final `I` SHA are recorded in the ignored Task 7 implementation report.

## Still blocked

Smoke B, official execution, all TEST-label access/evaluation, and every non-A target
remain blocked. Neither this report, a codemap hash, an artifact self-hash, nor a
local file is execution authority.
