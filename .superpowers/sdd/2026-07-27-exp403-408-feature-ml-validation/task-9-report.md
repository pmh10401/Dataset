# Task 9 Report: Exp407 Synthetic-anomaly Shadow Evidence

## Status

Implemented, self-reviewed, and verified. The commit SHA is recorded in the
task response.

## Files

- `experiments/exp403_408/synthetic_anomaly.py`
  - Defines immutable `SyntheticAnomalyConfig` and deterministic
    `SyntheticAnomalies` payloads with a SHA256 over ordered perturbation
    types, shapes, and little-endian values.
  - Generates equal TRAIN-derived counts for `spike`, `drift`, `level_shift`,
    `flat_line`, `amplitude_scale`, `frequency_warp`, and `phase_order`.
    Magnitudes use only TRAIN median/MAD (with a finite TRAIN-std fallback),
    and placement/ordering uses a frozen seed and length fractions.
  - Enforces learned-method eligibility (`n_train >= 50`), deterministic
    five-fold TRAIN OOF normal scoring, empirical p-value calibration, and
    separate fixed Logistic Regression, HistGradientBoosting, and SmallMLP
    artifacts. No TEST label, metric, threshold selection, or model selection
    is accepted by the public API.
  - Adds `synthetic_shadow_predictions`, which filters a copied frozen Hard
    set only and returns explicit `source="synthetic_anomaly_shadow"` records.
- `experiments/exp403_408/contracts.py`
  - Adds an optional `model_name` to `LearnedSourceScores`, so all three
    Exp407 artifacts remain separately identifiable without breaking existing
    learned sources.
- `experiments/exp403_408/tests/test_synthetic_anomaly.py`
  - Covers all perturbation types and equal counts, repeatable synthetic
    SHA256, eligibility abstention, all three fixed classifiers, deterministic
    score hashing, TEST-query isolation, Hard-lane immutability/provenance,
    and no leakage of Torch process-global determinism state.

## TDD Evidence

```text
python3 -m unittest experiments.exp403_408.tests.test_synthetic_anomaly -v
RED: ModuleNotFoundError: No module named
'experiments.exp403_408.synthetic_anomaly'

python3 -m unittest experiments.exp403_408.tests.test_synthetic_anomaly -v
GREEN: Ran 7 tests in 8.867s - OK
```

The full regression suite then exposed that an initial global
`torch.use_deterministic_algorithms(True)` call from the SmallMLP affected
Exp406 MPS masking. A focused regression test failed with `True != False` for
the prior process setting; removing that global mutation retains seeded CPU
determinism and restores Exp406 compatibility.

## Final Verification

```text
python3 -m unittest experiments.exp403_408.tests.test_synthetic_anomaly -v
Ran 8 tests in 9.083s - OK

python3 -m unittest discover -s experiments/exp403_408/tests -v
Ran 90 tests - OK

python3 -m py_compile experiments/exp403_408/contracts.py \
  experiments/exp403_408/synthetic_anomaly.py \
  experiments/exp403_408/tests/test_synthetic_anomaly.py
Passed (no output)

git diff --check
Passed (no output)
```

## MLX Assessment

`mlx` 0.32.0 is installed, but Exp407 remains CPU-only. The required
LogisticRegression and HistGradientBoostingClassifier are frozen sklearn
implementations, and replacing either with MLX would change the experiment.
The SmallMLP is also kept on seeded CPU Torch so the three-artifact protocol
has one implementation path. Therefore the exact full source does not meet
the MLX parity gate; no MLX/MPS-heavy full run was launched.

## Self-review

- Synthetic generation, all model fits, OOF references, feature scaling, and
  calibration are TRAIN-only; TEST is queried only after each model is fixed.
- Appending an unrelated TEST query does not change an existing raw score or
  calibrated p-value.
- Shadow output cannot promote or alter Hard, Standard, or Priority lanes.
- The default-only `model_name` contract extension preserves existing Exp406
  callers and MPS parity behavior.

## Concerns

- This task validates the source on bounded synthetic fixtures; no full
  1,117-dataset Shadow publication or retrospective TEST-label analysis was
  run.
- The MLP uses its frozen 25-epoch default (12 in bounded tests); changing it,
  any perturbation parameter, or model list requires a new frozen config and
  reproducibility review.
