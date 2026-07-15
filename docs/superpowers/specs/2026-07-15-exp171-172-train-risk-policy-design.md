# Exp171/172 TRAIN-normal Risk Policy Design

## Purpose

Exp169/170 removed the conformal resolution blocker for the 316 datasets with
`5 <= n_train < 20`, but deliberately reported all 60 pool/alpha/source
configurations without choosing one from TEST performance. Exp171 freezes one
review-only policy using TRAIN-normal OOB evidence. Exp172 evaluates that one
frozen policy retrospectively after the prediction artifact is hashed.

This work does not create an autonomous Hard alert policy. It does not claim
end-to-end TRAIN-only provenance or prospective deployment validity.

## Fixed Operational Contract

- Operational unit: one virtual wafer-run trace.
- Normal review-request risk target: at most `0.01` requests per normal run.
- Confidence statement: one-sided `95%` Wilson upper bound.
- Simultaneous comparisons: Bonferroni correction over exactly 60 preregistered
  Exp169 configurations.
- Inputs for policy selection: Exp169 TRAIN-normal OOB audit only.
- TEST labels, TEST positions, TEST metrics, family names, and dataset names are
  forbidden policy-selection inputs.
- The four alphas remain `(0.005, 0.01, 0.02, 0.05)`.
- The three pool policies and five source configurations remain unchanged.
- Candidate lane: `research_review_candidate`; no Hard alert is produced.

## Policy Selection

For each of the 60 fixed configurations, Exp171 independently recomputes a
simultaneous one-sided Wilson upper bound from
`total_normal_review_count / total_normal_instances` using
`z = NormalDist().inv_cdf(1 - 0.05 / 60)`.

A configuration is eligible only when all of the following hold:

1. its key belongs to the exact fixed 3 x 4 x 5 grid;
2. all 316 datasets are represented in the Exp169 dataset audit;
3. `resolution_blocked_datasets == 0`;
4. its simultaneous upper bound is no greater than `0.01`.

Among eligible configurations, choose the configuration with the largest
TRAIN-normal review count. This is a risk-budget-utilization proxy, not a claim
of anomaly sensitivity. Ties are resolved before execution by:

1. larger fixed alpha;
2. source order `2of3`, `spectral+shapelet`, `spectral+interval`,
   `shapelet+interval`, `3of3`;
3. pool order `similarity_pool`, `coarse_strata_pool`,
   `global_balanced_pool`.

If no configuration is eligible, Exp171 records policy abstention and Exp172
must not create predictions.

## Data Flow

1. Exp171 verifies hashes and contracts for Exp169 audit files.
2. Exp171 recomputes all 60 risk bounds and writes a ranked candidate table.
3. Exp171 writes one frozen policy JSON and its SHA256.
4. Exp172 verifies the policy hash and Exp170 frozen-prediction hash.
5. Exp172 filters exactly one label-free row per dataset into a new frozen
   prediction JSONL; it does not read TEST labels.
6. Exp172 hashes the prediction artifact.
7. Only the evaluation phase verifies that hash and then reads TEST labels.
8. An independent reaggregator verifies the selected key, 316-row coverage,
   index bounds, and global metrics.

## Outputs

The output directory is `outputs/exp171_172_train_risk_policy/` and contains:

- `01_contract.json`
- `02_exp171_policy_candidates.csv`
- `03_exp171_frozen_policy.json`
- `04_exp172_frozen_predictions.jsonl`
- `05_prediction_manifest.json`
- `06_exp172_dataset_metrics.csv`
- `07_exp172_global_summary.csv`
- `08_invariance_report.md`
- `09_result_analysis.md`
- `10_completion.json`
- `11_independent_reaggregation.json`
- `MANIFEST.csv`

## Invariants

- Prediction rows contain no `y_test`, TP, FP, precision, recall, or F1.
- Policy selection cannot import or read Exp170 evaluation summaries.
- Prediction cannot read TEST labels.
- Evaluation refuses a modified prediction file or policy file.
- Exactly 316 datasets and 316 selected prediction rows are required.
- Dataset order and input CSV row order do not change the selected policy.
- No TEST-length top-N budget, family branch, tail rule, or Priority-to-Hard
  promotion is introduced.

## Interpretation

Exp171 answers whether a single review policy can be frozen under a TRAIN-only
normal-risk contract. Exp172 answers how that already-frozen policy behaves on
the existing TEST benchmark. Exp172 is a retrospective counterfactual
evaluation and cannot validate the `1%` target prospectively.
