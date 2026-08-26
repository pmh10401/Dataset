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
