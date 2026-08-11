# Exp446 Task 1 implementation report

## Scope and status

Task 1 freezes the Exp446 engineering-only publication-materialization repair before
product code. `Exp446` is `preregistered_implementation_pending`; no official lifecycle
command, Dataset root, TEST label, product module, or test was accessed or changed.

## Immutable predecessor record

Exp445 is recorded as an immutable failed-publication predecessor, not a failed scientific
result: `30/30`, error `0`, labels `0`, CPU/11 with inner thread `1`, aggregate
`af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`, absent publication and
evaluation, five terminal SHA256 values, and exact failure text
`Exp444 original-control replay is invalid`. The only approved future code change is recursive
JSON materialization at the control-row encoding boundary.

## Files

- `docs/EXP446_PREREGISTRATION_antigravity_KO.md`
- `docs/EXP446_RUNBOOK_antigravity_KO.md`
- `docs/EXP446_IMPLEMENTATION_REPORT_antigravity.md`
- `EXPERIMENT_CATALOG.json`, `RESEARCH_STATE.json`, `FAILURE_LEDGER.json`, `CHAMPION_LEDGER.json`
- `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

The prior `codemap_exp445_antigravity.lock` is stale for the current base: it records
`0529afb3c4e6b245604c6fbb508b3e6b4fb4de78`, while this Task 1 base is
`dcd527cafd91dde737a9d84c14a8299a6d85f2dc`. Exp446 therefore has a separate initial versioned
triplet rather than modifying the Exp445 triplet.

## Initial map topology and unknowns

The Exp446 map retains the Exp445 structural baseline at 20 nodes, replaces the obsolete
Exp425 checkpoint node with `exp446_governance`, and has no Exp446 product edge or flow.
`exp446_governance` is a documentation/control boundary with no callers, downstream product
effects, or tests yet; all implementation relationships are explicitly unknown until code exists.

## Validation

Commands and output summaries are recorded after the final map regeneration:

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp446_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp446_antigravity.lock >/dev/null
git diff --check
```

Recorded result: all six JSON parsers exit 0 and both diff checks are clean. The map validator
passed `evidence=1084 nodes=20 edges=58 flows=5 fingerprints=9`; its HTML payload is
byte-identical to JSON, every evidence path/symbol and edge/flow reference resolves, and lock
fingerprints match the staged repository inventory.

## Commit

Commit target: `docs(exp446): freeze publication materialization repair`. The commit contains
only the files listed above; its resolved identifier is supplied by Git history after this report
is committed.

## Remaining concerns

The map deliberately does not assert future product wiring, test coverage, import boundaries, or
runtime flows. Those remain unknown and must be resolved by the product-code tasks while updating
the complete Exp446 map triplet. This governance commit does not authorize official execution.
