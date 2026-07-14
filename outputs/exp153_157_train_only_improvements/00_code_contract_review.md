# Exp153-157 code contract review

Reviewed before any Exp153-157 TEST evaluation was executed.

## Frozen decisions

- Cross-dataset pooling: allowed as research sensitivity only.
- Risk targets: report 0.005, 0.01, and 0.02 per normal run; no primary target selected.
- Confidence bound: one-sided 95% Wilson upper bound.
- Grouping: TRAIN count bins x TRAIN target-length quartiles only.
- Minimum pooled reference: 200.
- `n_train < 5`: complete calibration abstention.
- Confirmation variants: legacy, conformal strict, and conformal balanced; no primary variant selected.
- Long-running cache and disk use: approved.
- D2 configuration provenance audit: approved in parallel.
- Locked prospective benchmark: not currently supplied; all results remain retrospective counterfactual.

## Contract checks

| Contract | Status | Evidence |
|---|---|---|
| DB path remains `/Users/minho/Documents/Dataset/univariate_ts.db` | PASS | `experiments/exp153_157/exp153_157_bridge.py:30-32` |
| Shared cache does not load TEST labels | PASS by static call-path review | `build_exp153_157_score_cache.py:26-28` calls `build_dataset_cache`; `exp153_157_bridge.py:134-196` calls `load_dataset_series_only`, source scoring, and confirmation scoring. `load_test_labels` exists separately at `exp153_157_bridge.py:46-47` and is not called by the cache path. |
| Prediction JSONL rejects evaluation fields | PASS by static review | `exp153_157_reporting.py:15-23,62-71,74-104` rejects forbidden keys and freezes SHA256. |
| Evaluation verifies prediction SHA256 before reading TEST labels | PASS by static review | `exp153_157_reporting.py:139-160` verifies the frozen file before `load_test_labels`. |
| Exp154 excludes the target dataset from pooled reference | PASS by static review | `run_experiment_154_pooled_conformal.py:109-126` subtracts the target standardized TRAIN reference; helper contract is in `exp153_157_core.py:161-188`. |
| Exp154 grouping uses only TRAIN count and TRAIN target length | PASS by static review | `run_experiment_154_pooled_conformal.py:45-63`; group key implementation at `exp153_157_core.py:224-254`. |
| Exp155 correlation weights use TRAIN OOF scores only | PASS by static review | `run_experiment_155_source_fusion.py:26-36,55-79`; weights are calculated from `train_matrix`. |
| Exp156 keeps the frozen Exp152 candidate universe and changes confirmation only | PASS by static review | `run_experiment_156_confirmation_calibration.py:26-34,64-114,117-136`; Exp152 hash is verified and `candidate_indices` are retained while Block-B/C evidence is recalculated from the shared cache. |
| Exp157 selects alpha only from Exp153 TRAIN audit | PASS by static review | `run_experiment_157_risk_controlled_alpha_policy.py:35-68,98-120`; audit and frozen Exp152 hashes are inputs, and TEST labels are absent from prediction. |
| All variants/targets are sensitivity-only | PASS by preregistration and code constants | `exp153_157_core.py:20-49`, Exp154 `VARIANTS`, Exp155 `VARIANTS`, and Exp157 `RISK_TARGETS_PER_RUN`. No primary/optimal selector is present. |

## D2 configuration provenance finding

Policy execution is family-independent and TEST-length independent, but end-to-end configuration provenance is not cleanly TRAIN-only.

- `experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py:86-113` freezes ROCKET Exp40, `exp55_best`, `exp56_best`, and Exp84 configurations.
- `run_experiment_60_62_rocket_imaging_selector_variants.py:128-165` names Exp55/Exp56 sources as `best` and binds them to prior result CSV configurations.
- `docs/experiment_51_full_timeseries_imaging_analysis_20260707.md:23-60` explicitly compares and identifies a best full-dataset imaging strategy after evaluation.
- Exp84's fixed configuration is declared in `run_model_hard_research_experiments.py:1260-1297`; the repository currently contains no preregistration proving that this configuration was chosen without earlier TEST-performance inspection.

Therefore Exp153-157 may be described as policy-level TRAIN-only sensitivity experiments over frozen sources. They must not be described as end-to-end strict TRAIN-only or prospectively validated. A separate D2 report will preserve the detailed provenance evidence and unresolved items.

## Pre-execution code changes

No production-code contract correction has been made at this point. Package path transcription was normalized from the prompt's `init.py` to the ZIP's actual `__init__.py`, and the ZIP-provided `independent_reaggregate_exp153_157.py` was included because the runbook requires it.

The first prescribed unittest invocation failed before collecting the statistical tests because `test_exp153_157_train_only_improvements.py` imported its sibling `exp153_157_core.py` as a top-level module while unittest loaded the test through the `experiments.exp153_157` package path. The test now adds its own directory to `sys.path`, matching the package's direct-script execution contract. This changes test discovery only; no experiment policy or statistic was changed.

The first integrated smoke run completed Exp153 but stopped at Exp154 because the package called `exp153_157_core.agreement_indices` without defining it. A failing unit test was added first, then the missing pure helper was restored with the same deterministic 2-of-N p-value agreement semantics used by Exp151/152. No alpha, threshold, source, or routing constant changed.
