# Exp211-214 Calibration Transport Result Analysis

## Scope

This is a retrospective research analysis over the 251 datasets with complete
three-source calibration artifacts. It is not comparable as a replacement for
the 1,117-dataset Exp137 operational scorecard. Exp137 autonomous Hard alerts
were not changed by any of these experiments.

The shared baseline mean Hard F1 in this restricted cohort is 0.482322, with
57 datasets having zero Hard F1.

## Verified Findings

1. Spectral and interval score sources are stable under TRAIN-only transport.
   Spectral passed the pre-registered stability rule on 240/251 datasets and
   interval on 223/251. Their median OOF-to-full rank correlations are 0.996753
   and 0.995438 respectively.

2. The shapelet-normal-prototype source is not stable enough for an automatic
   operational rule. It passed on 0/251 datasets. Its median rank correlation
   is 0.517727 and its p10 is 0.159425, so its ordering changes materially when
   moving from folds to the full TRAIN-normal fit.

3. The apparent best mean-F1 gain comes from an unstable source. Shapelet
   pooled cross-conformal at alpha=0.01 raises cohort mean F1 to 0.487303
   (+0.004981), but produces 395 candidates: 88 TP and 307 FP (22.28%
   candidate precision). It is research evidence, not an operating policy.

4. The highest candidate precision is also too narrow to repair the problem.
   Shapelet conservative-max at alpha=0.005 produces 11 TP and 2 FP (84.62%)
   but only 13 candidates, fixes none of the 57 zero-F1 datasets, and changes
   mean F1 by only +0.000318.

5. Relaxing alpha repairs more zero-F1 datasets only by creating excessive
   false alarms. The most permissive spectral pooled variant repairs 24 zeros
   but emits 6,808 candidates with 6,645 FP (2.39% precision), and reduces
   mean F1 to 0.258053. This is not compatible with the false-alarm-first
   operating objective.

6. Exp213 confirms that a strict TRAIN-only stability guard removes the
   only source that showed modest aggregate benefit. The remaining stable
   spectral/interval combinations produce at most 43 TP against 1,146 FP at
   alpha=0.05, and do not create an operational improvement.

7. Exp214 independently reaggregated all 72 Exp212 variants and all 8 Exp213
   variants. The 72 Exp212 aggregates match exactly; it also records that
   Exp137 autonomous results remained unchanged.

## Interpretation

The bottleneck is not simply the p-value aggregation rule. A source can be
stable (spectral/interval) yet weakly selective for true anomalies, while the
shapelet source can be selective on a few datasets but is not calibratable
enough across TRAIN folds. Combining sources cannot compensate when the
underlying sources disagree or when one source is unstable.

The pooled cross-conformal correction was necessary for p-value resolution,
but it did not create a useful automatic-alert policy. It makes low alpha
reachable; it does not make the source more discriminative.

## Decision

Do not promote any Exp212/213 variant to autonomous Hard alert. Keep Exp137
unchanged. Treat the shapelet source as a research-only candidate until its
fold-to-full instability is reduced by a separately preregistered change to
the representation or prototype fitting procedure.

## Evidence

- `/Users/minho/Documents/Dataset/exp211_214_calibration_transport/experiment_211_calibration_transport_audit_summary.csv`
- `/Users/minho/Documents/Dataset/exp211_214_calibration_transport/experiment_212_fold_ensemble_query_scoring_summary.csv`
- `/Users/minho/Documents/Dataset/exp211_214_calibration_transport/experiment_213_train_only_source_stability_guard_summary.csv`
- `/Users/minho/Documents/Dataset/exp211_214_calibration_transport/experiment_214_calibration_transport_reaggregation_summary.csv`
