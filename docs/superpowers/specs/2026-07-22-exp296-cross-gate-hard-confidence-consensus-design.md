# Exp296 Cross-Gate Hard Confidence Consensus Design

## Purpose

Exp296 combines the three frozen Exp295 Gate annotations into one explanatory confidence tier for every existing Exp291 automatic Hard alert. It is a retrospective diagnostic and does not change any alert or review lane.

## Frozen Input

The only prediction input is `exp295_frozen_predictions.jsonl`, verified against the Exp295 prediction manifest. Exp296 does not recompute experts, inspect TEST labels during prediction, or select a Gate variant.

## Mutually Exclusive Tiers

For each Hard index, use the selected-expert support count from all three Gate variants.

- `unanimous_high`: all three Gates have support count 2
- `unanimous_supported`: all three Gates have support count at least 1, but the alert is not `unanimous_high`
- `partial_supported`: one or two Gates have support count at least 1
- `no_selected_support`: none of the three Gates has selected-expert support

All four tiers remain autonomous Hard alerts. No tier is suppressed, demoted, promoted, or declared an operational policy.

## Prediction and Evaluation Separation

Prediction produces one row per dataset, freezes and hashes it, and contains no TEST labels or evaluation metrics. Evaluation verifies that hash before reading labels and reports tier-level TP, FP, precision, recall, micro F1, and mean dataset F1.

## Required Invariants

- exactly 1,117 dataset rows
- exactly three Exp295 Gate rows per dataset
- identical Hard indices across all three Gate rows
- exactly 2,218 Hard records in total
- every Hard index appears in exactly one tier
- automatic policy unchanged and no review candidate promotion
- no Gate selected from TEST metrics
- retrospective only; not prospective validation

## Interpretation Boundary

The tier design follows an Exp295 retrospective diagnostic. Therefore Exp296 can show whether confidence stratification exists in the existing benchmark, but it cannot justify automatic suppression or deployment without independently frozen future data.
