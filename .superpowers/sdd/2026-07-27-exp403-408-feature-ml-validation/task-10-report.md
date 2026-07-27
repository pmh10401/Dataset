# Task 10 Report: Exp408 Independence and Consensus

## Status

Implemented and self-reviewed. Exp408 only packages review-only candidates; it
copies the verified Hard, Standard, and Priority lanes exactly and never
promotes a candidate to Hard.

## Implementation

- Added `experiments/exp403_408/consensus.py`.
  - Defines the five preregistered `VARIANTS` exactly:
    `tri_view_only`, `synthetic_only`, `tri_view_and_synthetic`,
    `one_new_plus_existing_independent`, and
    `two_new_plus_existing_independent`.
  - Consumes `ConsensusSource` entries containing a lineage identity, source
    kind, aligned TRAIN OOF scores, and frozen structured
    `ShadowSourcePrediction` records.
  - Collapses source names sharing a lineage before computing absolute
    Spearman correlation components across distinct lineages.
  - Treats fewer than 20 aligned rows, constant ranks, or otherwise undefined
    correlation as `independence_unverified`. Those components remain visible
    in the audit but never enter a multi-source consensus vote.
  - Requires a verified `FrozenLaneUniverse`, filters copied Hard indices from
    review candidates, and returns exact immutable Hard/Standard/Priority
    lane copies alongside separate variant candidate packages.
- Added `experiments/exp403_408/tests/test_consensus.py`.
  - Covers lineage-first collapse with low observed correlation, high absolute
    OOF correlation collapse across distinct lineages, the 19-row unverified
    boundary, all five variants, review-only Hard exclusion, full lane parity,
    and deduplicated component provenance.

## TDD Evidence

```text
python3 -m unittest experiments.exp403_408.tests.test_consensus -v
RED: ModuleNotFoundError: No module named
'experiments.exp403_408.consensus'

python3 -m unittest experiments.exp403_408.tests.test_consensus.ConsensusTests.test_five_frozen_variants_use_only_verified_independent_votes -v
RED: supporting_components contained the existing component twice

python3 -m unittest experiments.exp403_408.tests.test_consensus -v
GREEN: Ran 5 tests - OK
```

## Final Verification

```text
python3 -m unittest experiments.exp403_408.tests.test_consensus -v
Ran 5 tests - OK

python3 -m unittest discover -s experiments/exp403_408/tests -v
Ran 96 tests in 17.952s - OK

python3 -m py_compile experiments/exp403_408/consensus.py \
  experiments/exp403_408/tests/test_consensus.py
Passed (no output)

git diff --check
Passed (no output)
```

## MLX Assessment

Not applicable. Exp408 introduces no score computation or learning stage: it
uses already frozen CPU-produced Exp406/407 scores and TRAIN OOF identities.
Using MLX would add no eligible accelerated operation and would not change the
consensus algorithm.

## Self-review

- Same lineage is collapsed before any correlation comparison.
- Correlation is absolute Spearman on aligned OOF identities only.
- Any insufficient alignment makes the affected component unverified and
  excludes it from two- and three-source votes.
- All five outputs are candidate packages only; no result exposes an
  operational routing action.
- Hard, Standard, and Priority are copied from `FrozenLaneUniverse` without
  candidate-driven edits.

## Concerns

- This task verifies the consensus contract on compact fixtures. A future
  pipeline task must supply hash-verified Exp406/407 publications and actual
  feature-lineage metadata across all 1,117 datasets before retrospective
  evaluation/reporting.

---

## Fix Round 1: Immutable Provenance And Correlation Boundaries

### Root Cause

- `ConsensusSource.name` was copied into test Shadow records but was not
  validated against the source marker already frozen by Exp406 or Exp407. A
  caller could therefore label synthetic evidence as self-supervised evidence
  or invent an existing-source marker.
- The correlation fixture discarded its requested value and always supplied
  perfectly correlated scores, so it did not prove the preregistered `.90`
  collapse boundary.

### Changes

- Added `expected_source_marker(kind, name)`, a frozen identity registry:
  `tri_view` maps only to `self_supervised_shadow`, the registered synthetic
  source identities map only to `synthetic_anomaly_shadow`, and existing
  sources must use one of the feature-registry source names as their exact
  marker.
- Validates every `ShadowSourcePrediction.source` against that registry and
  rejects both marker mismatches and unknown kind/name combinations before
  source independence is calculated.
- Retains immutable marker tuples in `IndependenceComponent.source_markers`
  and `ConsensusCandidate.supporting_source_markers` so review-only output
  carries the original evidence provenance.
- Replaced the placeholder correlation fixture with rank permutations having
  observed Spearman values of approximately `.90075` and `.89925`. Tests now
  prove same-lineage collapse below `.90`, distinct-source collapse at or
  above `.90`, distinct-source separation below `.90`, and the 19-row
  unverified/no-vote boundary.

### TDD And Verification

```text
python3 -m unittest experiments.exp403_408.tests.test_consensus -v
RED: ImportError: cannot import name 'expected_source_marker'

python3 -m unittest experiments.exp403_408.tests.test_consensus.ConsensusTests.test_rejects_mismatched_frozen_shadow_source_marker -v
RED: expected mismatch test had no structured record to validate

python3 -m unittest experiments.exp403_408.tests.test_consensus -v
GREEN: Ran 8 tests in 0.058s - OK

python3 -m unittest discover -s experiments/exp403_408/tests -v
GREEN: Ran 99 tests in 18.024s - OK
```

### Remaining Concern

Existing-source markers are intentionally frozen to the current
`FEATURE_SOURCES` registry. Adding an existing evidence source requires an
explicit registry change and a new provenance review rather than a caller
supplied label.
