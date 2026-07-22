# Exp329-330: Large-dataset FN improvement probes

Both experiments are retrospective research Shadow analyses over the 251 datasets with at least 100 normal TRAIN instances. They do not modify the Exp291 automatic-alert policy.

## Exp329

Exp329 replays the frozen Exp328 evidence without reading TEST labels during prediction. It reports all five preregistered candidate-verifier rules:

- `review_independent_2`
- `review_total_support_3`
- `outside_strict_4of8`
- `combined_strict`
- `combined_balanced`

The evaluation phase reads labels only after the prediction JSONL is frozen and hashed. No rule is selected from TEST performance.

## Exp330

Exp330 creates a new normal-only representation from fixed multiscale features:

- 64 resampled raw values
- 32 first-difference values
- 32 log FFT magnitudes
- 16 block means and 16 block standard deviations

`StandardScaler` and PCA are fit only on normal TRAIN rows. KNN calibration scores exclude the row itself. TEST candidate p-values use the finite-sample upper-tail empirical rule. PCA dimensions 32 and 64 and alpha values 0.005, 0.01, and 0.02 are all reported as frozen sensitivity variants; none is selected from TEST performance.

These results are retrospective and neither prospective nor deployment validation.
