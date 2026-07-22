# Exp304 Reconstruction Fifth Source Common-support Design

## Purpose

Exp304 measures whether masked-reconstruction evidence adds useful information beyond the four verified score families. Because Exp257 exists only for 251 datasets with at least 100 TRAIN-normal runs, the experiment is limited to that frozen common-support population and must not be reported as a full 1,117-dataset result.

## Frozen Inputs

- Exp294 frozen Hard and Standard candidate universe
- Exp252 statistical support
- Exp254 structural support
- Exp255 change-point support
- Exp250 official MultiROCKET support
- Exp257 masked-reconstruction support

All prediction files and manifests are SHA256-verified. Alpha is fixed at `0.005`. Exp257 support requires stable evidence across its preregistered mask/seed variants. No missing dataset is treated as empty reconstruction support.

## Fixed Policies

- `verified_2of4`, `verified_1of4`: four-source common-support references
- `five_source_3of5`, `five_source_2of5`, `five_source_1of5`: direct five-source consensus
- `reconstruction_and_existing`: reconstruction plus at least one existing source
- `reconstruction_only_residual`: reconstruction support with zero existing-source support

Every rule is a retrospective counterfactual expansion over frozen Standard candidates. Precision 80% is not a pass/fail criterion and no rule is selected from TEST metrics.

## Provenance Boundary

Exp257 prediction/evaluation separation, mask variants, MPS device, population rule, and frozen hashes are available. The original dedicated generator package is not present in the current repository, so masked reconstruction remains a research source rather than a fully provenance-verified operational source.

## Evaluation

Prediction is frozen without labels. Evaluation then reports added TP/FP, overall precision/recall/F1, zero-F1 change, active datasets, and workload for the 251 common-support datasets. Operational Hard alerts remain unchanged.
