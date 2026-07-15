# Exp201-205 Independent Review Request

You are reviewing a retrospective time-series anomaly experiment. Do not edit
code or choose a policy from TEST metrics. Read the referenced source and CSV
files, then provide a skeptical root-cause analysis in Korean.

Repository: `/tmp/timeseries-exp177-185-batch`
Data/results: `/Users/minho/Documents/Dataset`

Read first:

- `docs/superpowers/specs/2026-07-15-exp201-205-large-zero-f1-recovery-design.md`
- `experiments/exp201_205/core.py`
- `experiments/exp201_205/artifacts.py`
- `experiments/exp201_205/run_experiment_201_205.py`
- `outputs/exp164_train_normal_agreement_risk_audit/09_n_train_bin_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_201_frozen_large_size_baseline_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_202_alternative_3of3_train_risk_rescue_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_203_current_alternative_cross_confirmation_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_204_size_stratified_shadow_router_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_205_independent_large_zero_f1_frontier_summary.csv`
- the corresponding Exp201-204 detail CSVs and frozen prediction JSONLs

Frozen objective:

- Population: 251 datasets with TRAIN-normal count >=100.
- Baseline Exp137: 1,115 TP, 206 FP, precision 84.4058%, mean F1
  0.482322, 57 Zero-F1 datasets, 2,974 true anomaly instances.
- Added autonomous candidate success criterion: added candidates themselves
  must have retrospective precision >=80%; subject to that, reduce Zero-F1.
- Exp137 Hard alerts are unchanged. New candidates are shadow diagnostics.
- TEST labels, anomaly positions, family names, dataset-name routing, and TEST
  length are prohibited prediction inputs.
- Alpha 0.01 was frozen from TRAIN-normal Exp164 audit for the alternative
  3-of-3 rule. For n_train>=100, Exp164 reported 50 false reviews over 120,575
  pseudo-normal runs, rate 0.000415, Wilson upper 0.000523.

Observed results:

1. Exp201 frozen Exp190 high+standard:
   - 100-199: 6 TP / 2 FP, added precision 75%, Zero-F1 16 -> 15.
   - 200+: 216 TP / 39 FP, added precision 84.7059%, Zero-F1 41 -> 36.
   - all >=100: 222 TP / 41 FP, 84.4106%, Zero-F1 57 -> 51.

2. Exp202 alternative 3-of-3 alpha 0.01:
   - 100-199: no candidates.
   - 200+ all-large: 25 candidates, 0 TP, 25 FP, Zero-F1 unchanged.
   - no-Hard-alert gate: 2 candidates, 0 TP, 2 FP.

3. Exp203 alternative 3-of-3 intersect frozen Exp198 current confirmation:
   - zero candidates in both size strata.

4. Exp204 size router:
   - 100-199: zero candidates.
   - 200+: identical to Exp201 200+ (216 TP / 39 FP); alternative rescue added
     nothing.
   - all >=100: 216 TP / 39 FP, Zero-F1 57 -> 52.

Questions:

1. Verify the result arithmetic and whether implementation follows the frozen
   contract. Identify any correctness defect before interpreting performance.
2. Explain how extremely low TRAIN pseudo-normal 3-of-3 false-review rate can
   coexist with 0/25 TEST anomaly precision. Separate calibration validity,
   covariate shift, source dependence, score direction, and anomaly mismatch.
3. Inspect the 25 Exp202 candidates: concentration by dataset, score margins,
   source thresholds, baseline Hard status, and whether they are near true
   anomalies. Do not use this inspection to propose a tuned rule; use it only
   for diagnosis.
4. Explain why Exp203 has an empty intersection. Is this expected independent
   evidence disagreement or an index/config mismatch?
5. Decide which claims are supported, conditionally supported, or prohibited.
6. Recommend the next preregistered experiment family for reducing Zero-F1 in
   TRAIN>=100 while protecting 80% added precision. Do not recommend choosing
   alpha/threshold from the current TEST results, family TEST routing, or tail
   replacement.
7. Clearly separate 100-199 and 200+ conclusions.

Return concise tables plus evidence paths and line references where possible.
