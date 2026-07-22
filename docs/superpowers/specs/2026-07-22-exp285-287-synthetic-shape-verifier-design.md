# Exp285-287 Synthetic Shape Verifier Design

## Goal and boundary

Use only TRAIN normal traces and preregistered synthetic transformations to
train a second-stage anomaly-shape verifier for the frozen 179 Exp272 Recovery
Shadow/Priority candidates. Exp253 Hard remains unchanged; all new candidates
remain retrospective review-only results.

Prediction cannot use TEST labels, anomaly locations, family, TEST order,
TEST-length budgets, or past TEST performance. TEST labels are loaded only
after the Exp286 prediction SHA256 is frozen and verified.

## Frozen synthetic transformations

For each selected TRAIN normal base trace, generate exactly four traces using
seed `20260722` and TRAIN-trace robust scale:

1. `spike`: one signed impulse of `5 * scale` at a seeded interior position;
2. `level_shift`: signed `3 * scale` shift after 50% of the trace;
3. `drift`: signed linear ramp ending at `4 * scale`;
4. `frequency`: signed sinusoid with 7 cycles and amplitude `2 * scale`.

At most 512 TRAIN bases are selected deterministically per fit to bound runtime.
The verifier uses the existing fixed 16-dimensional statistical/dynamic feature
vector, StandardScaler, and LogisticRegression with `C=1`, `max_iter=500`, and
seed `20260722`.

## Calibration and variants

Five deterministic TRAIN folds are used. Each held-out normal is scored by a
classifier trained only on the other folds and their synthetic traces. The
resulting normal scores form the empirical upper-tail calibration reference.
The final classifier fits all TRAIN normals and their synthetic traces, then
scores only frozen candidates.

All three alphas are reported without TEST-based selection:

- `synthetic_shape_alpha_001`
- `synthetic_shape_alpha_002`
- `synthetic_shape_alpha_005`

Datasets with `n_train < 100` abstain because alpha 0.01 is not empirically
resolvable. A result needs at least 80% retrospective precision only to qualify
for a future prospective review experiment; it never activates Hard alerts.
