# Experiment 137 - Operational Alert Triage

Exp137 converts the validated Exp93/Exp133 anomaly indices into an operational three-tier decision policy. It does not retrain a model or tune a threshold on test labels.

## Operating Policy

- Hard alert: Exp93 index independently supported by ROCKET Block B.
- Standard review: Exp93 index without Block B support.
- Priority review: narrow Exp135 all-Standard candidate confirmed by Block C.
- No alert: every remaining test index.

## Full Evaluation Results

The evaluation covers 1,117 datasets and 128 families.

| Lane | Candidates | TP | FP | Precision |
|---|---:|---:|---:|---:|
| Hard alert | 2,005 | 1,691 | 314 | 84.339% |
| Standard review | 639 | 292 | 347 | 45.696% |
| Priority review | 9 | 8 | 1 | 88.889% |

- Autonomous Hard-alert mean F1: `0.600772`
- Autonomous Hard-alert zero-F1 datasets: `362`
- Human-assisted union mean F1: `0.700776`
- Human-assisted union zero-F1 datasets: `233`
- Human-assisted union mean FP: `0.592659`

The human-assisted union assumes that a person resolves every review candidate correctly. It must not be reported as autonomous model performance.

The Priority review rule was discovered retrospectively. It remains a research-backed review rule until validated on genuinely new equipment, recipe, or time-period data.

## Files

- `experiment_137_operational_triage_summary.csv`: one-row aggregate operating metrics.
- `experiment_137_operational_triage_results.csv`: one row per evaluated dataset, including tier indices and separate Hard/Review/Combined metrics.
