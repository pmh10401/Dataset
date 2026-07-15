# Exp169/170 OOB Pooled Calibration Design

## 1. Purpose

Exp167/168 removed the score-generation mismatch between cross-fitted TRAIN-normal
calibration and TEST scoring, but the local empirical p-value could not resolve the
fixed alpha grid for almost every dataset with `5 <= n_train < 20`. Exp169/170 tests
whether a deterministic out-of-bag (OOB) score statistic and target-excluded,
TRAIN-only pooled references recover useful resolution without reintroducing TEST
labels, TEST positions, family routing, or TEST-length budgets.

The target population is the same frozen set of 316 datasets used by Exp165-168.
The work is a retrospective research-review sensitivity study. It does not choose
an autonomous alert policy or claim prospective equipment validation.

## 2. Experiment IDs and Scope

- `experiment_169_oob_pooled_calibration_audit`
  - label-free OOB score and TRAIN-normal false-review audit
- `experiment_170_oob_pooled_review_evaluation`
  - frozen prediction followed by retrospective TEST-label evaluation

Existing Exp165-168 code and outputs remain immutable.

## 3. Frozen Inputs

- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`
- Data directory: `/Users/minho/Documents/Dataset`
- Database SHA256: inherited from the verified Exp167/168 contract
- Target rule: `5 <= n_train < 20`
- Expected datasets: 316
- Sources:
  - `spectral`: frequency-change score
  - `shapelet`: normal-pattern distance score
  - `interval`: interval-statistics score
- Alphas: `(0.005, 0.01, 0.02, 0.05)`
- Source configurations:
  - `spectral+shapelet`
  - `spectral+interval`
  - `shapelet+interval`
  - `2of3`
  - `3of3`

No result may select one alpha, source configuration, or pool policy.

## 4. Balanced OOB Score Statistic

### 4.1 Split plan

Each dataset uses 32 deterministic subsample models with seed `20260715`.

- Holdout size: `max(2, ceil(0.35 * n_train))`
- Fit size: `n_train - holdout_size`
- A seeded base order is used with cyclic holdout windows.
- Split generation must guarantee every TRAIN-normal instance is OOB at least once.
- `q_d` is the minimum OOB model count over TRAIN instances in dataset `d`.
- Each TRAIN instance uses its first `q_d` OOB scores in model-index order.
- Each TEST instance uses the first `q_d` model scores.

This gives TRAIN and TEST the same number of component scores and the same fit-size
distribution. The selection is deterministic and does not inspect TEST values,
labels, positions, or length.

### 4.2 Model-level normalization

For every model and source, the model's fit-TRAIN scores define the normalization:

```text
center = median(fit TRAIN scores)
scale = max(1.4826 * MAD, IQR / 1.349, standard deviation, epsilon)
z = (query score - center) / scale
```

`epsilon` is `max(1e-12, 1e-9 * max(1, abs(center)))`. Taking the maximum of the
three scale estimates prevents the tiny-MAD heavy-tail amplification observed in
Exp165. The same model-level center and scale transform its OOB TRAIN and TEST
scores.

### 4.3 Instance aggregation

The final source score for an instance is the median of its selected `q_d`
model-level z-scores. Non-finite component scores cause the dataset to fail; they
are not silently dropped. The audit records split count, fit size, holdout size,
`q_d`, and per-source score distribution diagnostics.

## 5. TRAIN-only Donor References

All references consist only of OOB aggregate scores from TRAIN-normal instances.
The target dataset is always excluded. A donor contributes at most 20 normal
instances per source so that a single dataset cannot dominate a pool. Every pool
must contain at least 200 reference scores per source.

Three policies are preregistered and all are reported:

### 5.1 Similarity pool

Donors are ordered by Euclidean distance between robust-scaled TRAIN-only dataset
descriptors. The descriptor contains only:

- log TRAIN count
- log TRAIN target length
- per-source OOB aggregate score quantiles: 10%, 25%, 50%, 75%, 90%
- per-source OOB aggregate score mean and standard deviation

Descriptor scaling is fitted across the 316 TRAIN-only descriptors. Donors are
added in distance order, including all ties at the cutoff, until every source has
at least 200 scores.

### 5.2 Coarse-strata pool

Donors first match the target's preregistered TRAIN-count bin and TRAIN-length
quartile. If the minimum reference is unavailable, adjacent length quartiles are
added symmetrically, then adjacent TRAIN-count bins. Expansion order is determined
before prediction and cannot use TEST metrics.

TRAIN-count bins are `[5, 8)`, `[8, 12)`, `[12, 16)`, and `[16, 20)`. Length
quartiles are computed once from the 316 TRAIN target lengths.

### 5.3 Global-balanced pool

All other target datasets are donors, ordered by dataset key only for deterministic
serialization. The donor cap of 20 remains in force.

Pool construction failure is an experiment error. A missing pool is not treated as
an empty candidate set.

## 6. Pooled Empirical P-values and Candidates

For target aggregate score `s` and target-excluded reference `R`:

```text
p = (1 + sum(r >= s for r in R)) / (len(R) + 1)
```

Each source is evaluated independently. Pair candidates require both named sources,
`2of3` requires at least two sources, and `3of3` requires all three. No top-N cap or
TEST-length gate is applied.

At least 200 references give a minimum p-value no larger than `1/201`, restoring
resolution at alpha `0.005` up to the finite-sample boundary. Actual reference
counts and minimum p-values are recorded for every dataset, policy, and source.

## 7. Exp169 Label-free Audit

Exp169 evaluates each target TRAIN-normal instance against a pool that excludes its
dataset. It records for every pool policy, alpha, and source configuration:

- normal review count and rate
- Wilson upper bound
- datasets meeting fixed normal-review targets
- candidate-zero and resolution-blocked counts
- donor count and reference size distributions
- donor concentration and pool expansion diagnostics

Exp169 reads no TEST labels and emits no TP, FP, recall, or F1 fields.

## 8. Exp170 Prediction and Evaluation

### 8.1 Prediction phase

Prediction reads TRAIN-normal values and TEST values but does not read TEST labels.
It emits:

```text
316 datasets x 3 pool policies x 4 alphas x 5 source configurations
= 18,960 rows
```

Every row is a research-review candidate row. Hard, Standard, and Priority alert
lanes are not produced. The sorted JSONL is frozen with SHA256 and a config hash.

### 8.2 Evaluation phase

Evaluation verifies the prediction SHA256 and row-grid contract before loading TEST
labels. It reports candidate count, TP, FP, micro precision, mean recall, macro mean
F1, zero-F1 datasets, candidate-zero datasets, and workload distributions.

`mean_f1` is a retrospective human-review metric. It is not autonomous Hard-alert
F1. No policy is selected from TEST results.

## 9. Comparisons

All fixed variants are compared with:

- Exp166 local cross-fit reference
- Exp166 similarity pool
- Exp168 fold-matched local p-value

Comparisons include candidate transitions, TP/FP changes, exact-index Jaccard,
tolerance matching at +/-1, +/-3, and +/-5, and dataset-level paired bootstrap
intervals. Precision or F1 alone cannot determine adoption.

## 10. Invariance and Safety Tests

Tests must verify:

- deterministic balanced OOB split generation
- every TRAIN instance has at least `q_d` OOB scores
- TRAIN and TEST aggregate exactly `q_d` component scores
- target dataset exclusion from every donor pool
- donor cap and minimum reference enforcement
- family and dataset names are not policy features
- changing TEST labels or anomaly positions leaves predictions unchanged
- changing TEST length does not change split, normalization, pool, or alpha
- prediction function has no `y_test` argument
- evaluation rejects a modified prediction file
- input order does not alter sorted output hash
- all 18,960 rows are present and candidate indices are unique and in range

## 11. Stop Conditions

Full prediction must not start if:

- the frozen target set is not exactly 316 datasets
- DB or inherited input hashes differ
- any dataset cannot produce deterministic OOB coverage
- any source contains non-finite aggregate scores
- any pool includes the target dataset
- any policy has fewer than 200 references for a source
- prediction code accesses TEST labels, positions, family, or TEST-length budgets
- smoke output violates the fixed row-grid or label-free contract

The experiment is not repaired by changing alpha, split count, holdout fraction,
minimum reference, donor cap, or pool policy after viewing TEST metrics. Any such
change requires a new experiment number and preregistration.

## 12. Interpretation Boundary

Success means the experiment accurately measures whether OOB pooled calibration
recovers candidate resolution and useful anomaly coverage while exposing its normal
false-review cost. It does not mean the numerically strongest retrospective variant
is deployment-ready.

Even a favorable result remains:

- retrospective
- metadata-free cross-dataset pooling research
- human-assisted review sensitivity
- not end-to-end configuration-provenance validation
- not prospective semiconductor-equipment validation
