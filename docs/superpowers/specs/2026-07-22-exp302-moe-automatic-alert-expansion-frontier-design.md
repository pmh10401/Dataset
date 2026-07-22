# Exp302 MoE Automatic Alert Expansion Frontier Design

## Purpose

Exp302 evaluates how far the frozen Standard review lane can be expanded into a counterfactual automatic Hard lane using only the three frozen TRAIN-only MoE Gate annotations. Precision 80% is not a pass/fail threshold. The experiment reports detection gain, false alerts, and workload for every preregistered rule without selecting a winner from TEST metrics.

## Frozen Input

The only prediction input is the SHA256-verified Exp294 frozen prediction file. Exp291 Hard indices and the 806 Standard candidates are inherited unchanged. Prediction does not read TEST labels.

## Nested Expansion Rules

For each Standard candidate, let each of the three Gates report whether at least one or both selected experts support the same index.

- `all_gates_two_experts`: every Gate has two selected-expert supports
- `all_gates_any_support`: every Gate has at least one selected-expert support
- `majority_gates_any_support`: at least two Gates have at least one support
- `any_gate_any_support`: at least one Gate has at least one support

The rules are nested from narrowest to widest. No global Top-N workload, TEST-length budget, family policy, TEST label, or TEST metric is used during prediction.

## Output and Evaluation

Prediction freezes one row per dataset containing the unchanged baseline Hard indices, unchanged Standard candidate universe, Gate support counts, and promoted indices for all four rules. Evaluation verifies the prediction hash and then reads labels.

For every rule and for the `baseline_no_expansion` reference, report:

- automatic alerts, TP, FP, precision, recall, micro F1, and mean dataset F1
- incremental promoted candidates, TP, FP, and precision
- TP gain, FP gain, recall gain, and zero-F1 changes
- per-dataset alert workload distribution
- all 1,117 datasets and the `n_test >= 100` diagnostic subset

## Interpretation Boundary

All expansion rules are retrospective counterfactuals. Exp302 does not update the deployed automatic policy, declare a primary rule, or establish a precision floor. A future policy choice requires a frozen operating objective and prospective or shadow validation.
