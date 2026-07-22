# Exp226-235 Connected Pipeline Smoke Report

## Scope

- Date: 2026-07-19
- Dataset limit: 10, sorted by dataset key
- External output: `/Users/minho/Documents/Dataset/exp226_235_hard_alert_quality_smoke_v2_20260719`
- Classification: retrospective counterfactual smoke validation
- Policy selection from TEST metrics: no

## Command

```bash
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline \
  --stage all --experiments 226-235 --dataset-limit 10 --workers 2 --reset \
  --output-root /Users/minho/Documents/Dataset/exp226_235_hard_alert_quality_smoke_v2_20260719
```

## Verified Inputs

- Database datasets: 1,119 raw, 1,117 Exp137 evaluation datasets
- Database SHA-256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Frozen full score coverage: 1,117
- Frozen canonical score coverage: 639
- Exp137 replay: 2,005 Hard alerts = 1,691 TP + 314 FP
- Exp137 lane overlap datasets: 0
- Exp137 index-error datasets: 0

## Results

- Exp226-235 prediction, evaluation, and independent reaggregation completed.
- Independent reaggregation passed for every stored variant.
- Prediction rows contain no TEST labels or evaluation metrics.
- Hard, review, and research lanes are mutually disjoint.
- Exp235 exported 144 label-free shadow candidate rows.
- Exp226 first-ten rows were outside the frozen 251-dataset Exp219 population and were explicitly marked unavailable; a real covered dataset replay was separately verified with `CBF_normal_1`.
- Exp230 remains explicitly blocked because no valid frozen multi-seed/kernel repetition cache exists.

## Tests

```text
python3 -m py_compile experiments/exp226_235_hard_alert_quality/*.py serve_rank_dashboard.py
python3 -m unittest discover -s experiments/exp226_235_hard_alert_quality/tests -p 'test_*.py'
41 tests passed
```

The legacy `scratch/test_rank_dashboard_operational.py` script still fails an
existing Exp149 four-variant catalog assertion. The focused Exp226-235
heartbeat and dashboard catalog tests pass; this legacy fixture mismatch is not
treated as evidence that the new pipeline is complete.
