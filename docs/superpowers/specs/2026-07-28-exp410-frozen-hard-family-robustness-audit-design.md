# Exp410 Frozen-Hard Family Robustness Audit Design

Date: 2026-07-28
Status: approved design, not implemented
Audience: research, implementation, and audit reviewers

## 1. Purpose

Exp410 determines whether the current frozen automatic Hard result remains
credible when large one-vs-all families no longer receive one vote per
dataset. It independently reaggregates the existing 2,218 Hard alerts and
their frozen retrospective labels. It does not generate a new prediction,
train a detector, select a threshold, or modify any lane.

The primary question is:

> How much does the mean dataset F1 change when every source family, rather
> than every derived dataset, receives equal aggregate weight?

This is necessary because the 1,117-dataset population contains 128 families
with very unequal sizes. `ShapesAll` contributes 60 datasets,
`PigAirwayPressure`, `PigArtPressure`, and `PigCVP` contribute 52 each, and
`FiftyWords` contributes 50. An ordinary dataset-macro average therefore gives
these families much more influence than singleton or small families.

Exp410 is a retrospective robustness audit. It is not a new performance
experiment and does not repair the absence of genuinely new semiconductor
equipment data.

## 2. Claim Boundary

Exp410 may claim:

- the difference between dataset-macro and family-macro Hard metrics;
- the sensitivity of the result to a preregistered family-weight cap;
- which families contribute the most TP, FP, alerts, and metric influence;
- how the frozen result differs across family size, normal TRAIN size, exact
  normal TRAIN/TEST overlap, and TEST anomaly-count strata;
- whether the frozen result meets the preregistered descriptive robustness
  classification.

Exp410 must not claim:

- performance on an unseen family;
- prospective validation;
- semiconductor field performance;
- deployment validation or eligibility;
- an autonomous detection improvement;
- safe Hard, Standard, or Priority modification;
- a newly selected source, route, alpha, threshold, or review budget;
- model superiority.

The Hard policy has already been studied on the current retrospective
population. Family-balanced reaggregation reduces family-size distortion but
does not create a clean holdout. Any future policy change requires a new
experiment number and preregistration.

## 3. Frozen Audit Contract

| Field | Frozen value |
|---|---|
| Experiment | `Exp410` |
| Mode | read-only retrospective family robustness audit |
| Dataset population | 1,117 |
| Family population | 128 |
| Frozen Hard alerts | 2,218 |
| Frozen Hard TP / FP / FN | 1,915 / 303 / 2,003 |
| Dataset-macro mean F1 exact | `0.6089825086931719` |
| Dataset-macro mean F1 display (6 decimals) | `0.608983` |
| Prediction SHA256 | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` |
| Frozen label SHA256 | `b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4` |
| Database SHA256 | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |
| Exp291 lane SHA256 | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` |
| Exp292 lane SHA256 | `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b` |
| Bootstrap unit | family |
| Bootstrap iterations | 2,000 |
| Bootstrap seed | 410 |
| Family weight caps | 1, 3, 5, 10, unlimited |
| Metric float tolerance | `1e-12` |
| Operational policy changed | false |
| Deployment eligible | false |

All five family-weight settings are published. Exp410 must not select one
after observing the result.

## 4. Inputs and Provenance

The audit consumes only:

- the verified Exp291/292 frozen lane publications and manifests;
- `exp404_hard_confidence.jsonl`;
- `exp404_prediction_manifest.json`;
- `exp403_408_frozen_test_labels.csv`;
- `exp403_408_evaluation_input_manifest.json`;
- `exp403_408_evaluation_publication_manifest.json`;
- the current SQLite database in read-only mode;
- the Exp410 frozen configuration and execution code.

The current publications provide:

- 2,218 Hard prediction rows;
- 195,900 frozen label rows;
- labels for all 1,117 datasets;
- Hard alerts in 888 datasets.

Before parsing values, Exp410 verifies every pinned SHA256 and the artifact
hashes recorded by both evaluation manifests. It also verifies
`PRAGMA quick_check`, records the DB/WAL/SHM snapshot when present, and checks
that the snapshot does not change between preflight and publication.

The audit does not consume:

- Exp406-408 review-only candidates as Hard;
- model checkpoints;
- TEST-derived source rankings;
- historical family-level TEST performance for routing;
- missing per-dataset difficulty CSVs;
- a reconstructed or invented Exp403-408 preregistration document.

## 5. Canonical Family and Dataset Profiles

The canonical family is the exact prefix before `_normal_` in the dataset
identifier. Every one of the 1,117 datasets must contain that marker, have a
nonempty prefix and signed decimal class suffix, and map to exactly one family.
The signed form is required because the pinned universe contains five
`_normal_-1` dataset identifiers. The current pinned population must produce
exactly 128 distinct families.

The implementation reuses the semantic rule already present in the
Exp403-408 `_lineage` helper but implements and tests an Exp410-owned
read-only parser. It does not silently accept `_anomaly_` or arbitrary
fallback names because the frozen Exp410 population is normal-class
one-vs-all data.

The SQLite profile scan reads:

- dataset identity;
- TRAIN rows with label `normal`;
- TEST rows and their instance label;
- each instance `values_blob`.

For each dataset it computes:

- normal TRAIN count;
- TEST total count;
- TEST anomaly count;
- TEST normal count;
- number and fraction of TEST normal instances whose `values_blob` SHA256 is
  present among that dataset's normal TRAIN blobs.

Exact overlap is defined by the stored blob bytes. No interpolation,
normalization, tolerance, or learned embedding is used. `labels_blob` is not
read or interpreted as a timestep anomaly mask.

The scan is streaming and bounded per dataset. It does not retain the full
1.6 GB database as arrays in memory.

## 6. Dataset and Family Metrics

For each dataset Exp410 independently recomputes:

- Hard alert count;
- TP, FP, FN;
- precision;
- recall;
- F1;
- false alerts per dataset.

Every frozen TEST row supplies the anomaly denominator, while the Hard
prediction rows supply the predicted indices. The implementation rejects
duplicate identities, out-of-range indices, labels outside `{0, 1}`, and any
Hard identity absent from the frozen label table.

Dataset precision is recorded as `null` when there is no alert. Dataset F1 is
zero when the dataset contains a positive label and has no true-positive
alert. Every current dataset must contain at least one frozen anomaly; an
anomaly-free dataset fails the pinned-universe contract instead of receiving
an invented score.

For each family Exp410 publishes two views:

1. **Mean-dataset view:** arithmetic mean of the member dataset metrics.
2. **Pooled-count view:** TP, FP, and FN summed within the family before
   computing precision, recall, and F1.

The primary family score is the mean-dataset family F1. The global
family-macro F1 is the arithmetic mean of the 128 primary family scores.

## 7. Weight Sensitivity and Family Influence

Let `m_f` be the mean-dataset F1 for family `f`, `n_f` its dataset count, and
`c` one of the fixed caps. The capped score is:

```text
sum_f min(n_f, c) * m_f / sum_f min(n_f, c)
```

The audit publishes caps `1`, `3`, `5`, `10`, and `unlimited`.

- cap `1` is the family-macro result;
- `unlimited` is algebraically equivalent to dataset-macro F1;
- intermediate caps show how quickly large-family influence returns.

No datasets are sampled or discarded for this sensitivity analysis.

For every family, a leave-one-family-out table reports:

- change in dataset-macro mean F1;
- change in pooled Hard precision;
- removed TP, FP, FN, and alert count;
- family size and share of all datasets;
- family share of all FP.

This table is diagnostic. Exp410 does not remove influential families from the
official result.

## 8. Stratified Analysis

The following buckets are frozen before result generation.

### Family size

- `1`
- `2-5`
- `6-20`
- `21+`

The current 128-family population contains respectively 1, 74, 39, and 14
families in these buckets. These counts are verified before metric
publication.

### Normal TRAIN count

- `<20`
- `20-49`
- `50+`

### Exact normal TRAIN/TEST overlap

- `<50%`
- `50-<90%`
- `>=90%`

### TEST anomaly count

- `1`
- `2-10`
- `11+`

Dataset-level strata are aggregated with equal family weight: calculate the
mean of eligible member datasets inside each family, then average across
families that have at least one eligible member. Every row records both the
family and dataset denominator.

The previously approved difficulty stratification is not executed because
the current restored environment does not contain
`dataset_anomaly_validation_difficulty_20260707.csv` or an equivalent
per-dataset immutable artifact. Exp410 records:

```json
{
  "difficulty_stratification": {
    "status": "not_evaluated",
    "reason": "source_artifact_unavailable"
  }
}
```

It does not infer per-dataset difficulty membership from aggregate report
counts or memory.

## 9. Bootstrap and Robustness Classification

The bootstrap resamples the 128 families with replacement. When a family is
drawn, all of its member dataset rows move together. Resampling individual
datasets is prohibited because it would treat related one-vs-all derivatives
as independent.

Each of 2,000 iterations computes both:

- the resampled family-macro mean F1;
- the resampled dataset-macro mean F1 over the datasets contained in the
  sampled family clusters.

The paired difference is:

```text
delta_f1 = family_macro_f1 - dataset_macro_f1
```

The report publishes the point estimate and percentile 95% interval of the
paired difference.

The descriptive classification is frozen as:

- `robust`: the 95% interval lower bound is at least `-0.05`;
- `sensitive`: the lower bound is below `-0.05`;
- `severe_family_imbalance`: the point estimate is below `-0.10`.

`severe_family_imbalance` takes precedence over `sensitive`. The
classification is not an operational gate and cannot mutate Hard or trigger
automatic threshold selection.

## 10. Component Design

### `Exp410FrozenConfig`

An immutable contract stores the experiment identifier, expected hashes and
counts, bucket boundaries, caps, bootstrap parameters, float tolerance, and
claim flags. Command-line arguments may select input and output locations but
may not override research parameters.

### `FrozenAuditInputs`

A read-only loader verifies manifests and SHA256 values before returning
typed Hard identities and frozen labels. It cross-checks Exp291/292 lane
identity, Exp404 prediction lane parity, dataset population, Hard count, and
label coverage.

### `DatasetProfileScanner`

A SQLite read-only scanner computes TRAIN counts, TEST counts, and exact blob
overlap. It verifies DB integrity and snapshot stability and emits one typed
profile per dataset.

### `FamilyRobustnessAggregator`

A pure calculation component accepts verified dataset metrics and profiles.
It produces family metrics, cap sensitivity, leave-one-family-out influence,
stratified metrics, paired family bootstrap intervals, and the descriptive
classification. It has no filesystem, database, lane, prediction, or model
API.

### `AuditPublisher`

The publisher writes artifacts to a temporary output directory and atomically
renames it only after all invariants and hashes pass. A failed run may write a
failure status outside the publication directory but cannot publish partial
metrics.

## 11. Fail-Closed Rules

Exp410 publishes no audit result when any of the following occurs:

- Git worktree is dirty at publication;
- a pinned SHA256 or manifest artifact hash differs;
- SQLite `PRAGMA quick_check` is not `ok`;
- DB/WAL/SHM identity changes during the run;
- dataset population differs from 1,117;
- family population differs from 128;
- family-size bucket counts differ from `1 / 74 / 39 / 14`;
- Hard count differs from 2,218;
- TP/FP/FN differs from `1,915 / 303 / 2,003`;
- exact dataset-macro mean F1 differs from `0.6089825086931719` by more than
  `1e-12`, or its 6-decimal display differs from `0.608983`;
- Exp291/292 lane parity or Exp404 Hard identity parity fails;
- a dataset identifier is malformed;
- a label or index is invalid, missing, duplicated, or out of range;
- a preregistered cap, bucket, bootstrap parameter, or margin is changed;
- an output cannot be written and hashed atomically.

No fail-closed path changes existing artifacts.

## 12. Outputs

The canonical output set is:

- `exp410_frozen_config.json`
- `exp410_dataset_metrics.csv`
- `exp410_family_metrics.csv`
- `exp410_weight_sensitivity.csv`
- `exp410_family_influence.csv`
- `exp410_stratified_metrics.csv`
- `exp410_summary.json`
- `EXP410_RESULT_REPORT_KO.md`
- `MANIFEST.json`

`exp410_summary.json` explicitly includes:

- `evaluation_mode: "retrospective_family_robustness_audit"`
- `operational_policy_changed: false`
- `deployment_eligible: false`
- `unseen_family_validated: false`
- `prospective_validated: false`
- the difficulty-stratification `not_evaluated` record;
- every robustness classification input, not only the final label.

The manifest records every input, code, configuration, and output SHA256 plus
runtime versions and platform information.

## 13. Runtime and Acceleration Assessment

The current verified environment is:

- Python 3.14.6;
- NumPy 2.3.5;
- PyTorch 2.13.0;
- PyTorch MPS built: true;
- PyTorch MPS available: false;
- MLX installed: false;
- SQLite database size: approximately 1.6 GB.

Exp410 uses CPU SQLite streaming, `hashlib`, and NumPy. Exact blob hashing,
CSV/JSON parsing, grouped counters, and a 128-cluster bootstrap are not
tensor-heavy operations and would gain no justified benefit from PyTorch,
MLX, or MPS. Installing MLX or transferring byte payloads through a tensor
backend would add dependencies and could obscure exact byte semantics.

No new acceleration package is installed for Exp410. CPU is the canonical
path.

For this and every future experiment design, the design review must include
an explicit PyTorch/MLX/MPS assessment covering:

1. whether the dominant workload is eligible for tensor acceleration;
2. current backend availability;
3. expected transfer and compilation overhead;
4. deterministic and numerical parity requirements;
5. the canonical CPU fallback;
6. a parity benchmark before a non-CPU path becomes canonical.

When a future experiment contains substantial tensor training or batched
inference, PyTorch MPS and MLX must be evaluated rather than dismissed by
default.

## 14. Testing and Execution Scope

Unit tests use compact SQLite and artifact fixtures to verify:

- canonical family parsing and rejection of malformed names;
- exact blob overlap including duplicate TEST normals;
- TP/FP/FN, precision, recall, and F1 edge cases;
- mean-dataset and pooled-count family metrics;
- all five cap formulas and unlimited equivalence to dataset-macro;
- leave-one-family-out influence;
- family-unit rather than dataset-unit bootstrap;
- seed-410 determinism and interval schema;
- all fixed bucket boundaries;
- difficulty `not_evaluated` behavior;
- every hash, count, parity, and snapshot fail-closed boundary;
- atomic publication and manifest self-consistency;
- absence of any lane or prediction mutation API.

A representative smoke uses a bounded, deterministic set of families that
covers singleton, medium, and large family sizes plus all TRAIN, overlap, and
anomaly-count strata available in the sample. Smoke validates execution and
schema, not the final robustness classification.

No full 1.6 GB DB scan is part of implementation verification. After unit
tests and representative smoke pass, the full Exp410 audit is a separate
explicit execution step. The full run writes only to a new, nonexistent
Exp410 output directory.

## 15. Acceptance Criteria

Implementation is ready for a full-run decision only when:

1. all unit and regression tests pass;
2. the representative smoke publishes atomically and deterministically;
3. all pinned hashes and counts verify;
4. no existing lane, prediction, checkpoint, or report changes;
5. output schemas include every claim-boundary flag;
6. difficulty stratification is explicitly `not_evaluated`;
7. runtime provenance records the CPU decision and backend availability;
8. the runbook separates implementation smoke from the authorized full run;
9. the full run has not started without explicit user authorization.
