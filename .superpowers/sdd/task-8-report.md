# Task 8 Report: Full Execution And Acceptance Gate

## Scope And Outcome

- Worktree: `/private/tmp/timeseries-exp239-248-score-sources`
- Production output root: `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/`
- Result: **all acceptance criteria passed** after the final whole-branch review
  hardening and production rerun.
- This was an execution and verification pass only. No alpha, variant, threshold,
  agreement count, `n_train` boundary, or routing policy was changed from TEST
  results.

## Exact Commands And Durations

Commands were run in this order; production phases used no `--dataset-limit` and
no `--expected-total-anomalies` override.

```bash
python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v
# 60 tests passed; real 6.61 s

python3 -m unittest scratch.test_run_exp263_264_full_missing_aware_integration scratch.test_dashboard_state_contract -v
# 23 tests passed; real 0.07 s

/usr/bin/time -p python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase audit
# real 1.68 s

/usr/bin/time -p python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase predict
# real 1.65 s

python3 - <<'PY'
# Standalone frozen-prediction assertion: SHA256 and prediction manifest agreement,
# 1,117 rows and unique datasets, 20 missing overrides, 1,097 nonmissing, exact
# label-free schema, no lane overlap/range error, and Exp253 Hard subset preservation.
PY
# Passed; real 0.10 s

/usr/bin/time -p python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase evaluate
# real 1.61 s

/usr/bin/time -p python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase report
# real 2.71 s

python3 - <<'PY'
# Final acceptance assertion: 3,918 anomalies, summary/reaggregation agreement,
# paired Exp253 comparisons, all MANIFEST code/input/evidence/output hashes,
# self-hash exclusion, completed report status, dashboard PID, and no worker.
PY
# Passed; real 0.20 s
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
| Independent reporting | Frozen prediction hash/preflight then lane-index-plus-independent-truth reaggregation; persisted lane TP/FP/FN/precision/recall/F1 agreement; `summary_agreement.exact=true`; autonomous Hard and review metrics separately reported | Pass |
| End-to-end manifest | Source audit, coverage CSV, evaluation manifest, completed status, prior evidence, pipeline, scratch runner, dashboard integration, and output hashes captured; `MANIFEST.json` excluded solely to avoid recursive self-hashing | Pass |
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
| Source contract audit | `41fad73740be8bc75f722a258f77cdfed3b2a77bed4f19a201f74d558611e492` |
| Source coverage | `fe9d34862a3efe46354f79c08b5c7d8050ac7880789989c6a171670085843797` |
| Evaluation manifest | `886c5f8dacf75490964515822a7a583268ff5cc9ba0f353aa04e25eb76c4f24a` |
| Completed status | `d8cfb1dca9119f84c904f4968d10a8fd7759f1af5de68062f6e2c16a23e0006c` |
| Independent reaggregation | `7607e500db4a7771f47ce38a332fa2490141d3209719a8abfc966abd33b148a5` |
| Result report | `5f8103e4421b5346b3d8c15145038ac216f0d5b33b9b0e229c6bad44858474c7` |

`00_source_contract_audit.json` records the exact SHA256 values of all six
frozen source artifacts. Fixed Exp252/254/255 source provenance additionally
requires the frozen audit and manifest experiment/config/seed fields, alpha
`0.005`, and literal selected variants. Exp257 is bound by its frozen audit
SHA, literal `mask_025__seed_intersection` variant, alpha, 251-dataset MPS
audit configuration, and seed-intersection selection; its frozen audit and
manifest do not expose the numeric seed values separately. `MANIFEST.json`
records code, DB, input, evidence, output, and prediction hashes; it excludes
only itself because hashing the manifest recursively is not deterministic.

## Final Review Hardening

- The label-free audit API now accepts an explicit output root; its unit test
  writes only to a temporary directory, never the production root.
- Any `--dataset-limit` run requires an explicit non-production output root
  before status or log initialization, for every pipeline phase and the scratch
  runner. Regression tests prove rejected production smoke runs leave no
  `status.json` or log behind.
- Reporting preflights frozen prediction SHA before independently loading truth,
  recomputes every lane's TP, FP, FN, precision, recall, and F1 from lane
  indices, and rejects altered persisted metric columns. The adversarial test
  changes a persisted TP and confirms report generation fails.
- Fixed Exp252/254/255/257 provenance is fail-closed against the recorded
  frozen audit/manifest fields, source audit SHA, literal variant, alpha, and
  recorded seed/config evidence. Regression tests mutate a seed and variant.
- Manifest finalization occurs only after `status.json` is `completed` for the
  report phase. It captures end-to-end evidence and documents its sole
  self-referential exclusion.

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
- The production prediction, results, summary, and result-report hashes are
  unchanged from the prior accepted run. The source audit, evaluation manifest,
  completed status, independent reaggregation, and top-level manifest changed
  because this rerun added strengthened verification and fresh terminal-status
  evidence; metrics are unchanged.

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
