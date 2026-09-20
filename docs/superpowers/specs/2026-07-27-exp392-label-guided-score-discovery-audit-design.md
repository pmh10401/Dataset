# Exp392 Label-Guided Score Discovery Audit Design

## 1. Purpose

Exp392 uses retrospective TP, FP, FN, and TN outcomes to discover hypotheses
about useful evidence sources and source combinations. It is not an automatic
alert optimization experiment.

The audit answers:

1. Which frozen evidence sources reach baseline false negatives?
2. Which sources add the fewest normal candidates while doing so?
3. Which source pairs or triples provide complementary rather than duplicate
   evidence?
4. Are those observations stable across datasets, or concentrated in a small
   number of related datasets?

Exp392 does not change the frozen Hard, Standard review, Priority review, or
exact-normal lanes.

## 2. Non-Negotiable Boundary

TEST labels are used in Exp392. Therefore:

- every output is marked `label_guided_discovery=true`;
- every output is marked `retrospective_counterfactual=true`;
- every discovered rule is marked `eligible_for_deployment=false`;
- no Alpha, source, source combination, threshold, vote count, family rule, or
  dataset-size rule is promoted into the current pipeline;
- the experiment cannot establish end-to-end TRAIN-only compliance;
- the experiment cannot establish prospective equipment performance.

Exp392 may generate a hypothesis for a separately preregistered experiment.
That later experiment must freeze its rule before evaluation and must initially
route results to Shadow or review, not autonomous Hard alert.

## 3. Frozen Inputs

Exp392 reuses the exact inputs verified by Exp391:

- Exp389 frozen routing predictions;
- Exp386-388 frozen Exp357 evidence at Alpha values
  `0.005, 0.010, 0.020, 0.050, 0.100, 0.200`;
- Exp381 TRAIN coverage metadata;
- TEST labels from `univariate_ts.db`, opened only after all frozen file hashes
  and coverage checks pass.

The nine existing evidence sources are:

- frequency;
- ewma;
- normal_distance;
- temporal;
- topology;
- delta;
- delta_order_1;
- delta_order_2;
- structural.

The frozen artifacts contain thresholded candidate indices rather than every
source's continuous score. Exp392 therefore audits evidence-source reachability,
not continuous score calibration.

## 4. Candidate-Level Discovery Matrix

For every dataset and TEST index, create one row containing:

- dataset and index;
- baseline lane;
- true outcome used only in the discovery/evaluation phase;
- one binary indicator for every source and Alpha;
- number of available sources;
- source agreement count;
- TRAIN-normal count and TEST size for diagnostic stratification;
- exact-normal exclusion status.

The discovery universe excludes existing Hard and exact-normal indices when
measuring possible FN recovery. Existing review membership is retained so that
review prioritization and genuinely novel recovery can be reported separately.

Dataset and family names may be used to group retrospective stability reports.
They must not create a routing rule.

## 5. Analyses

### 5.1 Individual source analysis

For every fixed source and Alpha, report:

- candidate count;
- recovered FN;
- added FP;
- candidate precision;
- FN recovery rate;
- overlap with existing review;
- novel recovered FN;
- number of datasets contributing TP and FP.

### 5.2 Combination lattice

Evaluate every fixed combination of:

- all source pairs;
- all source triples;
- intersection;
- union;
- at-least-two agreement for triples.

All combinations are reported. TEST performance does not select a winner.

### 5.3 Complementarity

For every pair, report:

- exact candidate Jaccard;
- TP Jaccard;
- FP Jaccard;
- FN uniquely reached by each source;
- FN reached by both;
- conditional TP and FP contribution after the other source.

### 5.4 Stability

Report observations separately for:

- all datasets;
- TEST size below 100 and at least 100;
- baseline Zero-F1 and non-Zero-F1;
- no-Hard and wrong-Hard Zero-F1;
- TRAIN-normal count bands;
- base dataset lineage groups used only as evaluation groups.

Use dataset-level paired bootstrap intervals and leave-one-lineage-group-out
diagnostics. A source or combination concentrated in one lineage is explicitly
flagged.

### 5.5 Exploratory association

For each frozen source/Alpha indicator, calculate its association with a true
anomaly among non-Hard, non-exact-normal indices. Report effect size,
confidence interval, and a multiple-comparison-adjusted exploratory p-value.

Statistical significance is not a deployment criterion.

## 6. Output Classes

Every observation is assigned one of three research labels:

1. `unsupported`: low FN recovery or excessive FP;
2. `research_signal`: useful retrospective association but insufficient
   stability or coverage;
3. `preregistration_candidate`: stable enough to justify a new frozen
   Shadow/review experiment.

No output class means `automatic_alert_candidate`.

The classification thresholds are descriptive and preregistered in Exp392.
They do not select an operating policy. Raw tables remain the primary output.

## 7. Required Artifacts

Exp392 writes to a new directory and never overwrites Exp391:

`/Users/minho/Documents/Dataset/exp392_label_guided_score_discovery_audit`

Required artifacts:

- candidate-level discovery matrix;
- individual source summary;
- pair and triple combination summary;
- source complementarity matrix;
- stability and concentration report;
- high-FN dataset diagnostic;
- manifest with all input and output SHA256 values;
- Korean result report;
- progress/status JSON for the sequential dashboard.

The candidate-level matrix may be large. A compressed CSV is acceptable if the
dashboard-facing summaries remain ordinary CSV/JSON.

## 8. Verification

Tests must prove:

- frozen inputs are verified before TEST labels are opened;
- prediction artifacts contain no evaluation metrics;
- source and combination enumeration is deterministic;
- all fixed combinations are emitted;
- no TEST-derived selection field is set;
- baseline lanes are byte-for-byte unchanged;
- Hard plus review output is not rewritten;
- same input and seed produce identical sorted output hashes;
- source and combination counts reconcile with candidate-level rows;
- no candidate index is out of TEST range.

## 9. Interpretation

Exp392 is useful if it identifies repeatable failure signatures and suggests
which source family should be investigated next. It is not useful as evidence
that the current automatic alert should be expanded.

Because the 1,117 TEST datasets have already been observed in many experiments,
no subset of them can now be presented as a truly untouched final holdout.
Any operational promotion requires a newly locked benchmark or prospective
Shadow validation.

## 10. Next-Step Gate

After Exp392, at most one new experiment family is proposed from the audit.
Its design must:

- describe the hypothesis without using family performance for routing;
- define source, Alpha, agreement, and workload rules before evaluation;
- use TRAIN-normal calibration wherever applicable;
- leave the current autonomous Hard lane unchanged initially;
- evaluate Shadow/review usefulness first;
- prohibit retuning from the resulting TEST metrics.
