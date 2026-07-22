# Exp201-205 Large-Dataset Zero-F1 Recovery Design

## Goal

Reduce the 57 Exp137 Hard-alert Zero-F1 datasets among the 251 datasets with
`train_normal_count >= 100`, while requiring the newly added candidate set to
reach at least 80% retrospective precision. Existing Exp137 autonomous Hard
alerts remain unchanged. Every added candidate is a
`shadow_hard_diagnostic` until a later prospective validation authorizes an
operational change.

## Frozen Context

- The target population contains 251 datasets, 2,974 true anomalies, and 57
  Exp137 Hard-alert Zero-F1 datasets.
- The size strata are fixed before prediction:
  - `100 <= train_normal_count < 200`: 74 datasets and 16 Zero-F1 datasets;
  - `train_normal_count >= 200`: 177 datasets and 41 Zero-F1 datasets.
- Exp137 Hard alerts are the autonomous baseline and must remain byte-for-byte
  unchanged.
- Exp190 high and standard review evidence is the frozen current-source
  baseline.
- Exp162 provides three full-coverage, family-independent alternative score
  sources: derivative-spectral statistics, normal-shape prototype distance,
  and interval-quantile distance.
- Exp164 provides the TRAIN-normal risk audit used to preregister the
  alternative-source agreement threshold.

## Scientific Contract

- Prediction may use TRAIN-normal references, TEST values, frozen prediction
  indices, and `train_normal_count`.
- Prediction must not use TEST labels, anomaly positions, Zero-F1 membership,
  family names, dataset-name policy branches, TEST length, or historical
  family TEST performance.
- Dataset names are lookup keys only.
- `alpha=0.01` is frozen from the Exp164 TRAIN-normal risk audit. The
  three-source agreement Wilson upper false-candidate rate was below the
  preregistered 0.001 risk ceiling, while `alpha=0.02` was above it.
- No TEST-length top-N budget, tail replacement, Priority-review promotion, or
  post-evaluation rule adjustment is permitted.
- Prediction JSONL is hash-frozen before TEST labels are opened by evaluation.
- Negative and positive results are both retained.
- All experiments are retrospective counterfactual diagnostics, not
  prospective or deployment validation.

## Candidate Sources

### Current-source evidence

The current-source baseline is the frozen Exp190 routing evidence:

- high-confidence evidence: conservative inherited evidence and canonical KNN
  agree;
- standard evidence: conservative inherited evidence without canonical KNN
  confirmation.

These candidates remain separate from Exp137 Hard alerts.

### Alternative-source evidence

For every TEST index, independently convert each Exp162 score source using its
TRAIN-normal empirical upper-tail threshold at frozen `alpha=0.01`. An
alternative candidate requires the same index to be selected by all three
sources. Candidates already present in any Exp137 lane are excluded so the
experiment measures unique recovery.

### Current-alternative cross confirmation

A cross-confirmed candidate must have all-three alternative-source support and
the frozen Exp198 current-source confirmation at the same `alpha=0.01`.
Exp198 already requires at least one inherited source and canonical KNN to
agree, so this avoids reconstructing a historical score rule with subtly
different arithmetic. This is the precision-first rescue rule.

## Experiments

### Exp201: Frozen large-size baseline

Reaggregate Exp137 and Exp190 over the fixed `train_normal_count >= 100`
population. Emit the `100-199` and `200+` strata independently. Exp201 creates
no new candidate rule and establishes the exact baseline hashes and counts.

### Exp202: Alternative three-of-three TRAIN-risk rescue

Create unique shadow candidates from three-of-three Exp162 agreement at
`alpha=0.01`. Emit two preregistered diagnostic variants without selecting one
from TEST performance:

- `no_hard_alert_gate`: apply only when Exp137 emitted no Hard alert;
- `all_large_datasets`: apply to every dataset in the fixed population.

The first gate uses only a frozen model output, not the correctness of that
output. Neither variant may inspect Zero-F1 membership.

### Exp203: Current-alternative cross confirmation

Require the Exp202 three-of-three agreement and at least one current source at
`alpha=0.01`. This isolates whether independent representation agreement can
increase precision enough for future shadow promotion.

### Exp204: Size-stratified frozen shadow router

Apply a preregistered size-only routing policy:

- `100-199`: use Exp203 cross-confirmed candidates only;
- `200+`: use frozen Exp190 high and standard candidates plus Exp203 unique
  rescue candidates.

The resulting indices remain `shadow_hard_diagnostic`; they are never merged
into the autonomous Exp137 Hard lane in stored operational results.

### Exp205: Independent frontier reaggregation

Independently recompute Exp201-204 totals, candidate overlaps, lane transitions,
size-stratum metrics, and Zero-F1 changes from hash-verified artifacts. Exp205
does not choose a best policy.

## Success and Promotion Gates

Retrospective research success is evaluated separately for `100-199`, `200+`,
and the combined `>=100` population:

- incremental candidate precision is at least 80%;
- the number of Exp137 Hard-alert Zero-F1 datasets decreases;
- no new Zero-F1 dataset is hidden from the report;
- added TP, FP, workload distribution, and dataset concentration are reported;
- candidate sets smaller than 30 are labelled exploratory because the point
  estimate is unstable;
- a 95% Wilson interval is reported but is not used to retune the rule.

Operational promotion is out of scope. A later prospective validation should
require at least 100 adjudicated added candidates and a one-sided confidence
lower bound at or above 80%, under a separately approved operating contract.

## Prediction and Evaluation Boundary

Prediction artifacts contain dataset lookup keys, size stratum, candidate
indices, source evidence, frozen alpha, lane, and explicit
`contains_test_labels=false` and `contains_evaluation_metrics=false` flags.
They contain no TP, FP, FN, precision, recall, F1, labels, anomaly positions,
or Zero-F1 flags.

Evaluation verifies the prediction SHA256 before reading TEST labels. Only the
evaluation stage computes TP, FP, precision, dataset F1, current and resulting
Zero-F1 membership, and anomaly recovery.

## Metrics

For every experiment and size stratum report:

- eligible, active, and zero-candidate dataset counts;
- added candidate count, TP, FP, micro precision, and anomaly recall;
- mean candidate-only dataset F1;
- workload mean, median, p90, p95, and maximum;
- Exp137 baseline TP, FP, precision, recall, and mean Hard F1;
- counterfactual union TP, FP, precision, recall, and mean Hard F1;
- Zero-F1 before, after, fixed, and newly introduced counts;
- exact candidate overlap and unique contribution;
- 95% Wilson interval and paired dataset bootstrap interval.

Autonomous Exp137 metrics and shadow counterfactual metrics must remain visibly
separate in CSVs, summaries, and the dashboard.

## Runtime and Dashboard

- Register Exp201 through Exp205 as five individual sequential queue items.
- Run one experiment at a time; use seven CPU workers inside a stage where
  independent dataset reads benefit from parallelism.
- Reuse Exp162 and Exp186-200 frozen caches; do not recompute expensive feature
  transforms unless a hash or coverage gate fails.
- Each stage writes a progress log, hash-frozen prediction, result CSV, summary
  CSV, manifest, and completion marker.
- Missing coverage or a hash mismatch fails the stage visibly; no dataset is
  silently represented by an empty candidate set.
- The dashboard lists all five items before execution, shows the active stage,
  progress, rows, datasets, elapsed time, worker count, and latest log message.
- The queue runner exits when all five stages finish. No agent monitor remains.

## Required Tests

- Prediction entry points do not accept `y_test`.
- Prediction artifacts reject evaluation fields.
- TEST label and anomaly-position mutations leave predictions unchanged.
- Dataset/family rename leaves policy output unchanged after lookup remapping.
- TEST length is not a policy or budget input.
- Population is exactly 251 datasets with complete Exp137, Exp190, Exp162, and
  Exp164 coverage.
- Alpha and source agreement are constants and cannot be selected from TEST
  metrics.
- Candidate indices are unique, in range, and excluded from existing Exp137
  lanes.
- Exp137 autonomous Hard indices remain unchanged.
- Prediction hash tampering prevents evaluation.
- Independent Exp205 arithmetic matches every stage summary.
- Queue order is exactly Exp201, Exp202, Exp203, Exp204, Exp205.

## Interpretation Limits

Passing the 80% retrospective point-precision gate would identify a promising
shadow candidate rule, not authorize automatic alerts. Failure must be reported
without changing alpha or source rules. Historical TEST reuse, inherited
feature-configuration provenance, and absence of prospective equipment data
remain explicit limitations.
