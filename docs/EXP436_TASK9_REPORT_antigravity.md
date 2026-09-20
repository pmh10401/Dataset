# Exp436 Task9 pre-readiness verification report

## Scope

This report records the committed pre-readiness implementation state only.
No official readiness, hardware smoke, prediction, resume, evaluation, TEST
label opening, or write beneath the fixed external run parent was performed.

The readiness exit regression was reproduced at clean base
`61398ee7575d24755cca09ea46254ba0928d43c9`: a successful canonical readiness
publication reached public `main(['readiness'])` but returned `1`, because the
strict readiness artifact deliberately has no `state` field. The minimal repair
returns `0` only for the successful `readiness` command; it does not add a
`state` field or change any other command's exit classification.

## TDD record

The new CLI-level test creates a real canonical readiness file in a temporary,
previously absent parent through the private production dependency boundary and
calls public `main(['readiness'])`. It asserts exit `0`, exact canonical
publication, and absence of `state`.

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity.CliAntigravityTests.test_main_readiness_returns_zero_after_canonical_publication_without_state -v
# RED at 61398ee7: AssertionError: (1, ['readiness'], '') != (0, ['readiness'], '')

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity -v
# GREEN: 26 tests, OK
```

## Integrated verification

```text
python3 -m unittest discover -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests -p 'test_*_antigravity.py'
# 280 tests in 146.388s, OK; skips=0

python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v
# 30 tests, OK; skips=0

python3 -m unittest discover -s experiments/exp435_train_only_consensus_recovery_antigravity/tests -p 'test_*_antigravity.py'
# 187 tests, OK; skips=0

python3 -m unittest discover -s experiments/exp426_verified_baseline/tests -p 'test_*_antigravity.py'
# 58 tests, OK; skipped=1
```

The Exp426 skip is intentional: `test_production_source_is_exactly_1117_983_134`
requires `EXP426_RUN_PRODUCTION_SOURCE_TEST=1` and is explicitly a production
source audit. It was not enabled because Task9 excludes external production
source/readiness work.

The required Exp417 full command completed after a CPU-bound JSON serialization
phase. All 270 discovered tests passed with zero skips:

```text
python3 -m unittest discover -s experiments/exp417_causal_multihorizon_shadow/tests -p 'test_*.py'
# 270 tests, OK; skips=0
```

## Static governance

All passed before final control/map regeneration:

```text
rg -n 'TEST.*label|labels_blob|family.*performance|tail replacement' experiments/exp436_missing_safe_strict_consensus_antigravity
rg --files experiments/exp436_missing_safe_strict_consensus_antigravity scratch docs | python3 -c '... filename policy ...'
python3 -m json.tool RESEARCH_STATE.json
python3 -m json.tool EXPERIMENT_CATALOG.json
git diff --check
```

The 20 TEST-label scan matches were inspected. They are either the isolated
post-freeze evaluator (13), explicit no-label/anti-smuggling prohibitions in
contracts or reporting (2), or evaluation/firewall tests and fixtures (5).
None is a prediction input, scoring, runtime, or readiness label route.

## Readiness-pending state

`official_readiness=false`; PID remains null; no checkpoint, output, or
evaluation artifact exists; TEST labels remain unopened. The next external
action, only after the clean committed HEAD review, is exactly:

```text
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py readiness
```
