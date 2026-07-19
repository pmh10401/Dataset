# Exp237 Artifact Manifest

- Experiment: `experiment_237_new_score_source_qualification`
- Code commit at execution: `2bbfbaa572b5783f0a8f438504636093b9134944`
- Database SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Seed: `20260719`
- Scope: retrospective research score-source qualification
- Autonomous policy change: none

## Commands

```bash
python3 -m experiments.exp237_new_score_source_qualification.run_experiment_237 --stage all --dataset-limit 10 --workers 2 --reset-cache
python3 -m experiments.exp237_new_score_source_qualification.run_experiment_237 --stage all --workers 7 --reset-cache
python3 -m experiments.exp237_new_score_source_qualification.run_experiment_237 --stage predict
python3 -m experiments.exp237_new_score_source_qualification.run_experiment_237 --stage evaluate
python3 -m experiments.exp237_new_score_source_qualification.independent_reaggregate --output-root /Users/minho/Documents/Dataset/exp237_new_score_source_qualification
```

The full `all` run completed its 1,117 score caches but stopped at prediction serialization because non-empty NumPy candidate indices were not JSON serializable. Commit `2bbfbaa5` added a regression test and converted indices to Python integers. The cached scores were retained; `predict` and `evaluate` were rerun successfully without recalculating or changing any score, alpha, or threshold.

## Key hashes

| File | SHA256 |
| --- | --- |
| `experiment_237_summary.csv` | `b7c1865e01d52a1a36ba17b1c346ea82df4b38a3eeaef7cb1bc49ee01b53a41c` |
| `experiment_237_results.csv` | `5bb59fa765206b291a1ea9813acfce9c9a05d536569887c23d19f7194da87456` |
| `exp237_prediction_manifest.json` | `00713015e120205c6ffd9d70c4be19ee9740fdfca5e9bdda28bde8ae1453d751` |
| `exp237_independent_reaggregation.json` | `f4522aa7efee61035c4106b80691a1ddb4ca28f6c3c77e2afbe70d41024c32f0` |

Per-file hashes for the copied report artifacts are also recorded in `MANIFEST.csv`. Full score caches remain under `/Users/minho/Documents/Dataset/exp237_new_score_source_qualification/cache/` and are not copied into Git.
