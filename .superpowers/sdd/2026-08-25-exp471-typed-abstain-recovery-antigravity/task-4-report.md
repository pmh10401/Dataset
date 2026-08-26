# Task 4 Report: Typed Abstain Evaluation Supports

## Scope

Implemented Task 4 in the isolated Exp471 worktree from `6b7b5f5d`.
Only these owned paths changed:

- `evaluation_antigravity.py`
- `test_evaluation_antigravity.py`
- `docs/codemap/codemap_exp471_antigravity.{html,json,lock}`
- this report

## Result

- Reconstructed the strict v2 cache as scored-only raw/micro score maps plus
  complete per-method status/reason and exact TEST-index authority. All shard
  partitions are rebound before the label gate; missing identities, failed
  partitions, malformed abstentions, changed TEST coordinates, and production
  count drift fail closed.
- Published evaluation payload v2 with the exact top-level authorities:
  coverage, full-population decisions, scored support, all-ten common support,
  curve tables, and an ordered SHA-256-bound support authority.
- Abstentions are no-alert outcomes for decisions and remain semantic-null
  continuous results with `insufficient_train_rows`; no continuous score is
  created for them.
- Proposal decisions use raw `score >= 0.5`; baseline conformal decisions use
  the frozen `ALPHAS`; matched budget uses Candidate C's alert count and the
  evaluated method's own raw scores with frozen tie behavior.
- Exp450 continuous metric and curve functions are called unchanged once for
  scored support and once for all-ten support. Production-shaped support is
  ordered and requires exactly 922 datasets.

## TDD Evidence

The Task 4 fixture was first migrated from its obsolete v1-shaped score shard
and manifest to strict v2 schema/status/reason/TEST-index records. The first
new cache gate was observed RED before evaluator changes:

```text
test_typed_failed_cache_never_unlocks_labels ... FAIL
AssertionError: ValueError not raised
```

The failure showed that a nonzero durable `failed_shards` partition still
opened the label loader. After the cache gate implementation, the same test
was GREEN:

```text
test_typed_failed_cache_never_unlocks_labels ... ok
Ran 1 test
OK
```

Additional fixed-array tests cover no-alert abstention confusion and semantic
nulls, exact Exp450 metric reuse, proposal and conformal boundaries,
Candidate-C matched budgets, ordered 922 support/digest, and ROC/PR axes.

## Verification Evidence

```text
Task 4 evaluator suite: 22 tests in 1.507s — OK
Task 2/3 focused regressions: 81 tests in 28.709s — OK
Code-map suite: 5 tests in 9.128s — OK
Full Exp471 suite: 209 tests in 88.270s — OK
```

The following also exited zero:

```sh
python scratch/generate_exp471_codemap_antigravity.py
python scratch/generate_exp471_codemap_antigravity.py --check
python -m compileall -q experiments/exp471_exp467_evaluation_recovery_antigravity \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py \
  scratch/generate_exp471_codemap_antigravity.py
git diff --check
```

No TEST labels, official cache/results, original data, secrets, external
services, or official processes were read or modified. The evaluator is pure
orchestration/metric routing; it does not alter CPU provenance or introduce a
PyTorch, MLX, or MPS workload.

## Fix Round 1: Durable Reopen Scientific Validation

The v2 reopen boundary now reconstructs and validates the complete scientific
payload rather than only its outer schema. Continuous supports verify exact
method/dataset order, finite-or-semantic-null pairing, macro and micro
denominators, paired metric relations, and ordered pairwise scored-intersection
lists/counts/digests. Curve supports verify the frozen FPR/TPR and
recall/precision axes plus exact ROC/PR semantic-null placement. Full-population
decisions verify deterministic method/protocol/alpha/dataset order, typed
status/reason, no-alert abstentions, confusion support, derived metrics, and
all summary support/alert/coverage/count relations. The v2 artifact retains
pairwise support authority while public reconstruction returns the unchanged
Exp450 metric-table schema.

Adversarial RED first confirmed that both test-label coordinates and rehashed
payload mutations were previously accepted. The added tests now reject builtin
`bool`/`float`, negative, duplicate, and reordered label coordinates after
exactly one loader call and before publication. A 17-case rehashed tamper
matrix covers both continuous supports where applicable, semantic-null lists,
pairwise order/count, decision fields, and summaries.

A public synthetic durable v2 fixture has real empty abstained B/C/D shards on
different datasets. Public `evaluate` and `read` prove no-alert full-population
confusion, exact four-metric semantic nulls, scored-row denominators, ordered
all-ten intersection/digest, and direct B--A pairwise support. A matched-budget
tie test confirms deterministic frozen selection.

Verification after the fix:

```text
Targeted adversarial tests: 4 tests — OK
Task 4 evaluator suite: 26 tests in 2.504s — OK
Full Exp471 suite: 213 tests in 112.575s — OK
compileall, codemap generator/check, and git diff --check — OK
```
