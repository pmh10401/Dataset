# Exp173/174 Balanced TRAIN-only Selector Design

## 1. Purpose

Exp171 proved that one review-only policy can be frozen without using TEST
labels or TEST anomaly positions while keeping the simultaneous upper bound of
normal review requests below `1%`. Exp172 showed that the selected policy was
too narrow: it produced 40 review candidates across 316 datasets, with 75%
retrospective precision and 284 candidate-zero datasets.

Exp173 replaces the Exp171 rule that maximized the number of TRAIN-normal
review requests under the risk cap. It freezes one policy using a source-
diversity and calibration-stability rule. Exp174 then replays exactly that one
frozen policy and evaluates it only after the prediction artifact is hashed.

This work remains review-only. It does not create or modify autonomous Hard
alerts.

## 2. Fixed Operating Balance

The following contract is fixed before Exp174 evaluation:

- Aggregate TRAIN-normal review-risk target: `0.01` per virtual normal run.
- Confidence: one-sided `95%`.
- Multiple-comparison correction: Bonferroni over the exact 60-policy grid.
- Source agreement: `2of3`, meaning any two of frequency-pattern, shape-pattern,
  and interval-statistics scores must agree.
- Calibration stability partition: five deterministic dataset folds.
- Fold assignment: `int(SHA256(dataset)[:16], 16) % 5`.
- Candidate lane: research review candidate only.
- TEST labels, TEST positions, TEST length, family names, dataset names as policy
  conditions, and prior family TEST performance are forbidden selection inputs.
- `mean_combined_f1` is not an autonomous metric and is not used for selection.

The `1%` value is an aggregate retrospective TRAIN OOB statistical guard. It is
not a per-dataset guarantee, a user workload SLA, or a prospective equipment
false-alarm guarantee.

## 3. Considered Approaches

### Approach A: Keep Exp171 unchanged

This has the smallest implementation risk and preserves the existing normal
risk contract. It is rejected as the next research step because maximizing
normal review-count utilization selected a policy with very low anomaly
coverage and only 75% retrospective review precision.

### Approach B: Source-diverse fold-stability selector

This is the selected approach. It keeps the hard `1%` risk gate, fixes broad
`2of3` source agreement, chooses the largest eligible alpha, and then selects
the pool whose normal review rate is most stable across deterministic dataset
folds. It improves the selection objective without using TEST outcomes.

### Approach C: Combine multiple policies under one shared risk budget

This could increase coverage but requires allocation of the `1%` budget across
policies and adds overlap handling. It is deferred until one-policy selection
has been validated because it would make the risk interpretation and user
workload less transparent.

## 4. Exp173 Selection Algorithm

Exp173 consumes only:

- `outputs/exp169_170_oob_pooled_calibration/04_exp169_dataset_audit.csv`
- `outputs/exp169_170_oob_pooled_calibration/05_exp169_global_summary.csv`
- the frozen Exp169 contract and manifest hashes

It must validate the exact 60-row policy grid and all 316 datasets before
selection.

The deterministic selection steps are:

1. Recompute the simultaneous Bonferroni-Wilson upper bound for all 60 rows.
2. Mark a row risk-eligible only when the upper bound is at most `0.01` and the
   number of resolution-blocked datasets is zero.
3. Restrict the selectable set to `source_config == "2of3"`.
4. Choose the largest alpha for which at least one `2of3` row is risk-eligible.
5. For each remaining pool, aggregate `normal_review_count / n_train` in the
   five deterministic folds.
6. Rank by the population standard deviation of the five fold rates, ascending.
7. Break exact ties by maximum fold rate ascending, mean reference size
   descending, then the frozen pool order `similarity_pool`,
   `coarse_strata_pool`, `global_balanced_pool`.
8. Write one frozen policy and its SHA256 before any TEST metric is read.

Normal review count is reported but is not a ranking objective. This removes
the Exp171 behavior that treated use of more false-alarm budget as a proxy for
anomaly coverage.

If no `2of3` policy satisfies the simultaneous `1%` risk guard, Exp173 records
policy abstention. Exp174 then produces no prediction and no evaluation.

## 5. Exp174 Replay and Evaluation

Exp174 first verifies the frozen policy SHA256 and the Exp170 frozen-prediction
manifest SHA256. It extracts exactly one label-free prediction row per dataset
for the frozen policy key and writes a new deterministic prediction artifact.

Only after that artifact is hashed may the evaluation phase read TEST labels.
Evaluation reports:

- review candidate count, TP, FP, and micro precision;
- mean dataset recall and mean dataset F1;
- candidate-zero and zero-F1 dataset counts;
- review candidates per dataset, including median, p90, p95, and maximum;
- exact equality with the same policy row already present in Exp170.

No Hard alert metric is produced. No Exp174 result may change alpha, source
agreement, pool policy, fold count, tie-break order, or the `1%` guard.

## 6. Data Flow and Boundaries

```text
Exp169 TRAIN-normal OOB audit
        |
        v
simultaneous 1% risk eligibility
        |
        v
fixed 2-of-3 source agreement
        |
        v
largest eligible alpha
        |
        v
lowest five-fold normal-rate variation
        |
        v
Exp173 frozen policy + SHA256
        |
        v
label-free Exp174 prediction + SHA256
        |
        v
TEST labels read once for retrospective evaluation
```

Prediction and evaluation remain separate functions. The prediction function
must not accept `y_test`, anomaly positions, family, or a TEST-derived budget.
Dataset names may be used only as lookup keys and for deterministic fold
assignment during the TRAIN audit; they may not trigger named policy branches.

## 7. Required Tests

- Reject an incomplete or duplicated 60-policy grid.
- Reject incomplete 316-dataset coverage.
- Reject a selected policy whose simultaneous risk upper bound exceeds `0.01`.
- Verify deterministic five-fold assignment and row-order invariance.
- Verify the selector only considers `2of3` rows.
- Verify the largest eligible alpha is selected before fold stability ranking.
- Verify fold-rate standard deviation and tie-break ordering.
- Verify policy abstention when no eligible `2of3` row exists.
- Verify the selection path cannot read Exp170 TEST metrics.
- Verify prediction contains no labels or evaluation metrics.
- Verify policy and prediction hash tampering is rejected.
- Verify Exp174 dataset coverage, candidate index bounds, and independent metric
  reaggregation.
- Verify the dashboard and sequential queue expose Exp173 and Exp174 separately.

## 8. Interpretation Limits

The 60 Exp170 TEST outcomes have already been observed during earlier research.
Therefore, even though the Exp173 implementation is structurally TRAIN-only,
Exp174 is not an independent holdout validation of the newly written selection
rule. It is a retrospective counterfactual and reproducibility check.

The following claims remain prohibited:

- the selector was developed without any historical awareness of TEST results;
- the full development lineage is end-to-end strict TRAIN-only;
- the `1%` bound is a prospective equipment false-alarm guarantee;
- Exp174 is deployment validation;
- review candidates are autonomous Hard alerts.

Operational adoption requires a temporally newer locked benchmark or shadow
run in which the selector, all constants, and all prediction hashes are frozen
before labels or user feedback are opened.

## 9. Success Criteria

Exp173/174 is technically complete when:

- the selector is deterministic and uses only the frozen TRAIN-normal audit;
- the selected policy satisfies the simultaneous `1%` risk guard;
- Exp174 evaluates exactly one frozen policy without changing it afterward;
- all hashes, input files, commands, and metric scopes are recorded;
- independent reaggregation matches the official result;
- queue and dashboard status remain visible;
- the final report separates false-alarm risk, anomaly coverage, and user
  workload rather than choosing a policy from mean F1 alone.

An improvement over Exp172 is desirable but is not a completion condition. A
worse result must be reported unchanged.
