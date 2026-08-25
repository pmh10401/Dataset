# Exp471 Typed-Abstain Recovery Design

## 1. Objective

Exp471 must finish the paper evaluation without inventing scores for public
baselines that cannot legally train. The frozen baseline manifests contain no
execution failures, but they contain 395 preregistered
`insufficient_train_rows` abstentions:

- classical baselines: 5 method/dataset entries;
- Deep SVDD: 195 entries;
- DROCC: 195 entries.

The current Exp471 implementation incorrectly requires every one of the 11,170
method/dataset entries to contain a finite score vector. The repaired pipeline
will preserve all 11,170 identities while representing each as either scored or
typed-abstained. It will produce full-population and common-support results
without changing the frozen methods, thresholds, labels, or source artifacts.

## 2. Frozen Boundaries

The following remain unchanged:

- the Exp467 proposal source authority and 4,468 A/B/C/D coordinates;
- the three pinned public-baseline roots, manifests, configuration hashes, and
  score-shard bytes;
- the database, TEST coordinate order, method order, alpha values, Numba
  calibration, label gate, metric definitions, and bootstrap seed;
- create-only publication and exact-byte reuse;
- `numba_serial_safe` proposal preprocessing and its result-preserving speed
  gate.

No baseline is rerun. No abstention is converted into a synthetic, neutral,
zero, repeated, interpolated, or imputed continuous score.

## 3. Source Preflight

Baseline verification moves before expensive proposal score composition. The
preflight verifies the pinned roots and classifies every baseline entry as:

- `scored`: source status is `complete`, score vectors are finite, and TEST
  identity is exact;
- `abstained`: source status is `abstain`, reason is exactly
  `insufficient_train_rows`, score vectors are empty, and TEST identity is
  exact;
- invalid: absent entries, unknown status/reason, nonempty abstain scores,
  malformed coordinates, or any source failure.

Invalid input fails before the 4,468-item proposal preprocessing stage. The
preflight records exact expected counts: 10,775 scored and 395 abstained entries
across 11,170 identities.

## 4. Score-Input and Cache Contracts

The verified score-input schema becomes a typed v3 schema. Each
method/dataset record contains public identity, source authority, TEST identity,
and one of two mutually exclusive payloads:

- scored payload: `status="scored"`, finite raw scores, and finite micro scores;
- abstain payload: `status="abstained"`,
  `reason="insufficient_train_rows"`, and no score vectors.

The cache shard schema becomes v2 with the same discriminated union. Scored
entries follow the existing Numba calibration and durable publication path.
Abstained entries skip calibration workers and publish a small create-only
typed shard directly. Both types bind the cache-context digest, source hashes,
method, dataset, TEST row count, and TEST indices.

The terminal manifest still requires exact 11,170-identity coverage, but records
`scored_shards=10775`, `abstained_shards=395`, and `failed_shards=0`. A complete
manifest is impossible when an identity is missing, duplicated, malformed, or
failed. Resume logic reuses either verified shard type and schedules only
missing scored work.

## 5. Evaluation Semantics

Evaluation retains all 1,117 datasets in the full-population tables.

For scored entries:

- existing threshold, matched-budget, AP, AUROC, ROC, and PR calculations are
  unchanged.

For typed abstentions:

- threshold and matched-budget predictions are all-negative, matching the
  preregistered rule that abstention emits no alert;
- TP, FP, FN, TN and derived full-population decision metrics remain defined;
- AP, AUROC, ROC, and PR are `null` because no continuous score exists;
- coverage and the abstention reason remain explicit.

Continuous-score macro summaries use only scored datasets and report their
denominator. Continuous-score comparisons and curves are also produced on the
all-ten-method common support, which is exactly 922 datasets. Paired statistics
use only the explicitly recorded common scored support for the compared pair
and record that ordered support. No result silently drops an abstention.

## 6. Reporting

The report exposes, for every method:

- population count, scored count, abstained count, failed count, and coverage;
- full-population threshold and matched-budget metrics with abstention as no
  alert;
- scored-support macro AP/AUROC with denominators;
- common-support AP/AUROC, ROC, and PR artifacts for fair continuous-score
  comparison;
- abstention reason counts and the fixed all-ten common-support count of 922.

Report wording must not describe abstentions as runtime failures or completed
continuous scores. The artifact map and report manifest bind the new cache and
evaluation schema digests.

## 7. Failure Handling and Resume

- A known typed abstention is durable evidence, not a retryable failure.
- A source `failure`, absent entry, unknown abstention reason, malformed vector,
  or changed authority fails closed.
- Interrupted preparation reopens both scored and abstained shards and never
  recalculates verified entries.
- Existing Exp471 progress logs remain readable. No complete cache or evaluation
  artifact currently exists, so no published artifact migration is required.
- After implementation verification, the failed official `report` process is
  restarted once. Numba preprocessing may be recomputed, but baseline preflight
  prevents another late coverage failure.

## 8. Verification

Tests must independently prove:

1. exact 10,775 scored / 395 abstained / 0 failed accounting;
2. only `insufficient_train_rows` is accepted as a baseline abstention;
3. abstain shards have exact TEST identity and no score vectors;
4. one- and multi-worker runs produce identical terminal bytes;
5. resume skips both scored and abstained durable identities;
6. labels remain unreachable until complete 11,170-identity cache authority;
7. full-population no-alert confusion counts include all 1,117 datasets;
8. continuous metrics are null for abstentions and denominators are explicit;
9. all-ten common support is exactly 922 datasets and all curve axes remain
   correct;
10. report coverage, abstention, common support, artifact hashes, and wording
    are consistent;
11. the complete Exp471 test suite, compilation, diff check, and synchronized
    Exp471 code-map triplet pass before restart.

## 9. Completion Criteria

The repair is complete only when the official run publishes a verified cache
with 11,170 typed identities, a label-gated evaluation with explicit abstention
semantics, and the final paper report with both full-population and 922-dataset
common-support results. A preprocessing completion event alone is not
completion.
