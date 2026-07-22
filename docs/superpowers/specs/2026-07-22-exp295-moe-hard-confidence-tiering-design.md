# Exp295 MoE Hard Alert Confidence Tiering Design

## Purpose

Exp295 measures whether the frozen TRAIN-only sparse MoE evidence can explain the confidence of existing Exp291 automatic Hard alerts. It must not remove an existing Hard alert, promote a review candidate, or select one Gate variant from TEST metrics.

## Frozen Inputs

- Exp291 automatic Hard indices
- Exp293 frozen Gate choices for `risk_first_top2`, `stability_first_top2`, and `balanced_rank_top2`
- Four provenance-verified source-support artifacts: statistical, structural, change-point, and official MultiROCKET

Every input is verified with its existing SHA256 contract before prediction.

## Confidence Tiers

For each existing Hard alert, count how many of the two TRAIN-only selected experts independently support the same index.

- `high`: two selected experts support the alert
- `medium`: one selected expert supports the alert
- `needs_confirmation`: no selected expert supports the alert

All three tiers remain autonomous Hard alerts. The tier is explanatory metadata only.

## Prediction and Evaluation Separation

Prediction creates one row per dataset and Gate variant. It records the unchanged Hard indices, supporting experts, support counts, and confidence tier without reading TEST labels or evaluation metrics. The file is frozen and hashed before evaluation.

Evaluation verifies the prediction hash, then loads TEST labels and reports TP, FP, precision, and recall by tier. Results are retrospective sensitivity diagnostics. No Gate variant is selected as primary.

## Invariants

- 1,117 datasets and three rows per dataset
- Hard candidate set exactly matches Exp291 for every Gate variant
- each Hard index appears in exactly one confidence tier
- no review candidate is promoted or demoted
- TEST labels are loaded only after prediction freeze
- no family name, dataset-name policy, TEST length policy, or TEST metric selection

## Dashboard

Exp295 appears as a completed diagnostic experiment with autonomous scope described as `Exp291 Hard unchanged`. It is excluded from operational model comparison.
