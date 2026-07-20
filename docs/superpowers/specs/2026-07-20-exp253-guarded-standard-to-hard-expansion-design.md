# Exp253 Guarded Standard-to-Hard Expansion Design

## 1. Objective

Exp253 evaluates a modest expansion of autonomous Hard alerts while preserving
the project's false-alarm-first operating objective. It does not search the
whole TEST sequence for new Hard alerts. It only considers candidates already
placed in the frozen Exp236 Standard review lane.

The primary retrospective question is whether official MiniROCKET evidence can
promote a limited subset of Standard review candidates without materially
degrading autonomous precision.

## 2. Interpretation Boundary

Exp253 is a retrospective counterfactual experiment on a repeatedly observed
benchmark. The rule below was chosen after reviewing prior TEST results.
Therefore:

- `selected_from_prior_test_review` is true.
- `deployment_eligible` is false.
- Exp253 cannot establish prospective equipment performance.
- No result may be described as an end-to-end TRAIN-only policy validation.
- A successful result may justify a future frozen-rule shadow validation only.

## 3. Frozen Inputs

Exp253 reads but does not modify these artifacts:

1. Exp236 frozen baseline lanes for all 1,117 datasets.
2. Exp249 official MiniROCKET `kernels_1024_seed_intersection` candidates at
   alpha 0.005.
3. Exp249 official MiniROCKET `kernels_4096_seed_intersection` candidates at
   alpha 0.005.
4. Exp240 ordinal-pattern, Exp242 complexity-signature, Exp245 smooth-Fourier,
   and Exp247 masked-context candidates at alpha 0.005 for confidence grading
   only.

Every input file and its SHA256 must be recorded before prediction. Candidate
indices must be in TEST range and each input must cover exactly 1,117 unique
datasets.

## 4. Primary Promotion Rule

For each dataset, define:

```text
baseline_hard = Exp236 Hard indices
baseline_standard = Exp236 Standard review indices

mini1024 = Exp249 MiniROCKET 1024 seed-intersection candidates, alpha 0.005
mini4096 = Exp249 MiniROCKET 4096 seed-intersection candidates, alpha 0.005

promoted = baseline_standard AND (mini1024 OR mini4096)
new_hard = baseline_hard OR promoted
new_standard = baseline_standard MINUS promoted
new_priority = baseline_priority
```

The rule is a set operation on frozen candidate indices. It does not recalculate
features, scores, thresholds, alpha, or candidate budgets.

Candidates outside all Exp236 lanes are never promoted. Priority review remains
review-only. Tail position, family name, dataset name, TEST length, TEST labels,
and actual anomaly positions are prohibited routing inputs.

## 5. Confidence Grades

Confidence grades explain the evidence supporting a promoted alert. They do not
change whether the candidate is promoted.

- `S`: MiniROCKET 1024 and 4096 both agree, and at least one methodologically
  different source among Exp240, Exp242, Exp245, or Exp247 agrees.
- `A`: MiniROCKET 1024 and 4096 both agree without the independent-source flag.
- `B`: exactly one of MiniROCKET 1024 or 4096 agrees.

The independent sources are used only to display confidence and review
priority. Their retrospective TEST performance must not be used to add another
promotion branch after prediction is frozen.

## 6. Prediction and Evaluation Separation

Prediction must have the structure:

```python
prediction = predict(
    baseline_lanes=frozen_exp236_lanes,
    minirocket_candidates=frozen_exp249_candidates,
    confidence_evidence=frozen_exp240_242_245_247_candidates,
)
```

Prediction must not accept or load TEST labels. It writes a deterministic JSONL
artifact containing baseline lanes, promoted indices, final lanes, confidence
grades, and explicit leakage flags. The artifact SHA256 is frozen before
evaluation.

Evaluation loads TEST labels only after verifying the prediction hash. It writes
dataset-level and aggregate autonomous metrics separately from review metrics.

## 7. Required Metrics

The report must include:

- baseline and Exp253 Hard count, TP, FP, FN, micro precision, micro recall, and
  mean dataset F1;
- promoted count, TP, FP, precision, and dataset coverage;
- Standard review count, TP, FP, and precision after promotion;
- Priority review metrics, unchanged and clearly human-assisted;
- S/A/B confidence-grade count, TP, FP, and precision;
- dataset-level lane transitions and exact candidate indices;
- zero-Hard-alert and zero-Hard-F1 dataset counts;
- paired dataset bootstrap 95% intervals for FP, recall, and mean F1 changes.

The user-stated operating adequacy threshold is an incremental promoted-candidate
precision of at least 80%. This is reported as a diagnostic gate, not used to
alter the frozen rule.

## 8. Expected Retrospective Reference

Prior read-only analysis observed the following. These numbers are provenance
for why Exp253 is being tested, not acceptance data and not a tuning target:

```text
promoted candidates: 137
promoted TP: 126
promoted FP: 11
promoted precision: approximately 91.97%

baseline Hard: 2,116 alerts, 1,789 TP, 327 FP
counterfactual Hard: 2,253 alerts, 1,915 TP, 338 FP
counterfactual micro precision: approximately 85.0%
```

Any independently reproduced difference must be reported without changing the
rule.

## 9. Invariants and Failure Handling

The experiment must fail before evaluation when:

- any required dataset or source row is missing or duplicated;
- input SHA256 differs from the frozen audit manifest;
- candidate or lane indices are outside TEST range;
- Hard, Standard, and Priority lanes overlap after routing;
- a promoted candidate was not in the baseline Standard lane;
- prediction contains TEST labels or evaluation metrics;
- prediction output is not deterministic under reordered input rows;
- independent reaggregation does not match the primary summary.

No missing source is treated as an empty candidate set. Existing Exp236 and
Exp249 files are never overwritten.

## 10. Outputs and Dashboard

Exp253 writes to a new output directory and registers as a separate dashboard
experiment. Required artifacts are:

- frozen input audit and hashes;
- frozen prediction JSONL and manifest;
- dataset-level result CSV;
- aggregate summary CSV;
- lane-transition CSV;
- confidence-grade CSV;
- invariant-test log;
- independent reaggregation report;
- retrospective interpretation report;
- reproducibility manifest.

The dashboard must show baseline versus expanded Hard metrics, promoted TP/FP,
confidence-grade counts, current stage, dataset coverage, and errors. It must
label Exp253 as retrospective and not deployment eligible.

## 11. Follow-up Boundary

Exp252 statistical-feature results are not included because Exp252 was still
running when this rule was frozen. Any use of Exp252 evidence requires a new
experiment number and a new preregistration. Prospective or shadow validation
must freeze this rule before observing new labels and must be treated as a
separate phase.
