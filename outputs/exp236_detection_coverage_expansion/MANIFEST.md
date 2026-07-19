# Exp236 Artifact Manifest

- Code commit: `915ddf956d018697c78d1353af3d341460fefcf3`
- Experiment: `experiment_236_three_source_hard_two_source_review`
- Prediction SHA-256: `a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524`
- Exp137 input SHA-256: `942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d`
- Exp234 input SHA-256: `6a7ebfd2f6a90982ff9d1103e4dcc21ecfb9d07ee3afd67e8c2225ac3fe94d60`
- Dataset coverage: 1,117/1,117
- Independent reaggregation: passed
- Classification: retrospective candidate packaging
- Deployment eligible: no

## Commands

```bash
python3 -m experiments.exp236_detection_coverage_expansion.run_experiment_236 --stage predict
python3 -m experiments.exp236_detection_coverage_expansion.run_experiment_236 --stage evaluate
python3 -m experiments.exp236_detection_coverage_expansion.independent_reaggregate \
  --output-root /Users/minho/Documents/Dataset/exp236_detection_coverage_expansion
```

The machine-readable SHA-256 inventory is in `MANIFEST.csv`. Hard metrics are
autonomous retrospective counterfactuals. Standard and Priority review metrics
remain human-assisted diagnostics. The rule was nominated after prior TEST
review, so a newly locked or prospective validation is still required.
