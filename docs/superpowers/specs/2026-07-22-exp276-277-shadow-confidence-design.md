# Exp276-277 Recovery Shadow Confidence Design

Exp276 freezes a label-free confidence record for every Exp272 Recovery Shadow
candidate. Exp277 evaluates the frozen records retrospectively. Exp253 Hard is
immutable and every candidate remains review-only.

For each qualified continuous source group, empirical p-values are rebuilt from
the frozen TRAIN OOF reference and TEST score. The normalized tail margin is:

`(alpha - p) / (alpha - 1/(n_reference+1))`, clipped to `[0, 1]`.

The following constants are frozen before evaluation:

- primary alpha: `0.005`;
- minimum continuous groups: `2`;
- deep margin: `0.5`;
- maximum TRAIN-normal Wilson upper risk: `0.005`;
- consistency standard-deviation limit: `0.25`.

Three sensitivity guards are reported without selecting a winner:

1. all groups deep;
2. median group deep;
3. mean deep and cross-group margin standard deviation at most `0.25`.

The continuous confidence score is a diagnostic review ranking only. TEST
labels, family, dataset name, TEST length, and previous TEST performance are not
prediction inputs. No result can activate an autonomous alert.
