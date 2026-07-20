# Task 8 Report: Full Execution And Acceptance Gate

## Scope And Outcome

- Worktree: `/private/tmp/timeseries-exp239-248-score-sources`
- Production output root: `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/`
- Result: **all acceptance criteria passed**.
- This was an execution and verification pass only. No alpha, variant, threshold,
  agreement count, `n_train` boundary, or routing policy was changed from TEST
  results.

## Exact Commands And Durations

Commands were run in this order; production phases used no `--dataset-limit` and
no `--expected-total-anomalies` override.

```bash
python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v
# 55 tests passed; real 9.29 s

python3 -m unittest scratch/test_dashboard_state_contract.py -v
# 12 tests passed; real 0.31 s

python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase audit
# real 2.02 s

python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase predict
# real 2.21 s

# Standalone frozen-prediction assertion: SHA256, 1,117 rows and unique datasets,
# 20 missing overrides, 1,097 nonmissing, exact label-free schema, no lane
# overlap/range error, and Exp253 Hard subset preservation. Passed; 1.49 s.

python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase evaluate
# real 1.93 s

# Standalone evaluation-manifest assertion: prediction SHA match and 3,918/3,918
# anomalies with smoke_contract_override=False. Passed.

python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase report
# real 3.18 s

# Final acceptance assertion: persisted audit, prediction, evaluation,
# independent reaggregation, manifest, report provenance, and status contracts.
# Passed.
```

The required operational sequence was therefore `audit -> predict -> frozen
prediction verification -> evaluate -> report`.

## Contract Evidence

| Gate | Evidence | Result |
|---|---|---|
| Source audit | Core baseline/statistical/structural/change-point coverage `1117/1117`; missing override `20/20`; error count `0` | Pass |
| Frozen prediction | `1117` rows and unique datasets; missing `20`; nonmissing `1097`; label/evaluation flags false | Pass |
| Routing invariants | Lane overlaps `0`; range errors `0`; every Exp253 baseline Hard index preserved | Pass |
| Evaluation boundary | Frozen SHA verified before labels; `3918/3918` expected/actual anomalies | Pass |
| Independent reporting | `summary_agreement.exact=true`; autonomous Hard and review metrics separately reported | Pass |
| Dashboard/process | `status.json=completed`, `phase=report`, `1117/1117`, errors `0`; existing dashboard PID `32416` alive; no active Exp263/264 worker | Pass |

The final prediction SHA256 is
`8c7a2f52027c6a3127c79166bb109b61e348558574d98a42a8ccf5622921d653`.

Selected manifest hashes:

| Artifact | SHA256 |
|---|---|
| DB snapshot | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |
| Frozen prediction | `8c7a2f52027c6a3127c79166bb109b61e348558574d98a42a8ccf5622921d653` |
| `results.csv` | `ebf9cb3084cff71056e5bf6e75afc40c4f8853da2949f7a23e8d17f2a4c3aa28` |
| `summary.json` | `d957210c07502b9bc12eb7e119531f94ea4f98e18f54f074e75e85ca5199cc61` |
| Independent reaggregation | `47df006a114e4382d60b60395409319bfceba29f1ce7d99e046ea406a7ec10dc` |
| Result report | `5f8103e4421b5346b3d8c15145038ac216f0d5b33b9b0e229c6bad44858474c7` |

`00_source_contract_audit.json` records the exact SHA256 values of all six
frozen source artifacts. `MANIFEST.json` records the code, DB, input, output,
and prediction hashes.

## Exp264 Versus Exp253 Autonomous Hard

All comparisons are paired, retrospective `Exp264 autonomous Hard - Exp253
baseline Hard`; no subgroup was used for selection.

| Group | Datasets | Exp253 alerts / TP / FP | Exp264 alerts / TP / FP | Delta alerts / TP / FP | Delta macro Mean F1 |
|---|---:|---:|---:|---:|---:|
| Overall | 1117 | 2253 / 1915 / 338 | 2255 / 1917 / 338 | +2 / +2 / +0 | +0.000052 |
| Missing override | 20 | 34 / 22 / 12 | 34 / 22 / 12 | +0 / +0 / +0 | +0.000000 |
| Nonmissing | 1097 | 2219 / 1893 / 326 | 2221 / 1895 / 326 | +2 / +2 / +0 | +0.000053 |
| `n_train < 100` | 866 | 684 / 576 / 108 | 684 / 576 / 108 | +0 / +0 / +0 | +0.000000 |
| `n_train >= 100` | 251 | 1569 / 1339 / 230 | 1571 / 1341 / 230 | +2 / +2 / +0 | +0.000231 |

Overall autonomous Hard changed from precision `0.849978`, recall `0.488770`,
macro Mean F1 `0.608659` to precision `0.850111`, recall `0.489280`, macro
Mean F1 `0.608711`. Two Standard candidates moved to Hard, and 22 moved to
Priority review. Human-assisted lanes remain separate: Standard review had 806
alerts (365 TP, 441 FP), Priority review 31 (26 TP, 5 FP), and their combined
review lane 837 (391 TP, 446 FP).

## Acceptance Interpretation

The two promotions add two true positives with no false-positive increase over
Exp253. Per the preregistered interpretation, this is a retrospective expansion
signal, not a policy promotion or prospective validation. The missing 20 did
not change at autonomous Hard scope, so this run does not establish a global
missing-aware model improvement.

## Execution Limits And Resource Relevance

- No source regeneration was necessary or performed: this integration consumed
  the audited frozen Exp252/253/254/255/257/260 artifacts.
- MPS/MLX was not invoked. There was no MPS-heavy stage to serialize.
- Seven CPU workers were not relevant because no source generation was needed;
  the production work was deterministic artifact loading, routing, evaluation,
  and reaggregation.
- The existing dashboard server was neither restarted nor signaled. Its process
  remained alive through completion.
- The evidence remains retrospective, not prospective validation and not
  end-to-end strict TRAIN-only operating evidence. Exp253 provenance remains
  `selected_from_prior_test_review=True` and `deployment_eligible=False`.

## Artifacts

- [Production status](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/status.json)
- [Source contract audit](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/00_source_contract_audit.json)
- [Frozen prediction manifest](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/prediction_manifest.json)
- [Evaluation manifest](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/evaluation_manifest.json)
- [Summary](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/summary.json)
- [Independent reaggregation](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/independent_reaggregation.json)
- [Result report](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/RESULT_REPORT.md)
- [Full manifest](/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/MANIFEST.json)

External production result files were not staged or committed. Existing dirty
workspace changes were preserved.
