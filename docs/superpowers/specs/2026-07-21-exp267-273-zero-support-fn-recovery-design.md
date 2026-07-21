# Exp267-273 Zero-Support FN Recovery Design

## 1. Purpose

Exp267-273 investigates how to recover false negatives that none of the current
evidence sources identify. The work extends evidence in stages: diagnose the
blind spot, qualify independent score sources, add review-only recovery lanes,
and only then audit whether any narrow subset could become a future autonomous
Hard-alert candidate.

The current frozen reference is Exp266:

- 1,117 datasets and 3,918 true TEST anomalies,
- autonomous Hard: 1,915 TP, 338 FP, 2,003 FN, precision 0.849978,
- human-assisted combined: 2,365 TP, 338 FP, 1,553 FN,
- 1,405 of the remaining combined FN have zero support from all five Exp266
  evidence groups,
- 148 remaining combined FN have exactly one source of support.

The main bottleneck is therefore representation and candidate generation, not
the existing two-source agreement threshold. Relaxing agreement alone cannot
reach the 1,405 zero-support misses.

## 2. Operational Boundary

- A dataset is one virtual equipment/recipe/sensor/step detection environment.
- A TRAIN or TEST instance is one virtual wafer-run trace.
- A candidate index is a dataset instance index, not a time point inside a
  trace.
- TRAIN contains normal instances only.
- Feature fitting, score fitting, calibration, thresholding, eligibility, and
  evidence grouping use TRAIN-normal information only.
- TEST labels and actual anomaly positions are read only after prediction
  artifacts and hashes are frozen.
- Family and dataset names are lookup and retrospective reporting fields only.
- TEST instance count cannot choose a source, threshold, eligibility rule,
  candidate count, or budget.
- Tail replacement and TEST-length top-N budgets are forbidden.
- Existing Exp253 Hard and Exp266 review lanes remain immutable during
  Exp267-272.

All experiments are retrospective counterfactual studies on a repeatedly
observed benchmark. They are not prospective validation or deployment proof.

## 3. Why TRAIN Size May Control Learner Eligibility

Learner eligibility is based only on the number of TRAIN-normal instances. It
never uses TEST size or TEST performance.

The frozen tiers are:

| TRAIN-normal count | Allowed evidence |
|---|---|
| `n_train < 5` | calibration abstention |
| `5 <= n_train < 20` | deterministic statistical and structural sources |
| `20 <= n_train < 100` | deterministic sources plus lightweight nonlinear ridge forecasting |
| `n_train >= 100` | all above plus normal-only Causal TCN forecasting |

The `n_train >= 100` learner threshold reuses the prior eligibility boundary
used by the masked-reconstruction source. It is frozen before the new TEST
evaluation. It is not selected by comparing Exp270 TEST metrics.

Current retrospective workload context, reported only to explain potential
impact, is:

- 251 datasets have `n_train >= 100`,
- these datasets contain 1,283 of the 1,553 remaining combined FN,
- learner ineligibility is always recorded as `learner_abstained`, never as a
  negative vote or empty successful result.

## 4. Design Alternatives And Decision

### 4.1 Threshold or agreement relaxation

This is computationally cheap but can directly reach at most the 148 remaining
single-source FN. It risks a large FP increase and is retained only as a
sensitivity diagnostic.

### 4.2 Independent source expansion

This is the selected primary direction. New universal score sources must expose
different information from random convolution, statistical, structural,
change-point, and masked-reconstruction evidence.

### 4.3 Learned model replacement

A full learned replacement is rejected for this cycle. TRAIN size varies from
1 to 4,261, so one learner cannot be assumed valid everywhere. Learned
forecasting is introduced only as an additional evidence source for datasets
with at least 100 TRAIN-normal instances.

## 5. Experiment Map

### Exp267: Zero-Support FN Diagnostic Harness

Exp267 freezes the diagnostic population and describes the blind spot without
changing predictions. It reports:

- support count 0/1/2/3+ among remaining FN,
- TRAIN-normal count and TRAIN-derived target-length distributions,
- raw-trace variability, periodicity, spectral concentration, local dynamics,
  and shape complexity,
- dataset-level and family-level summaries for retrospective explanation only,
- primary diagnostic slices `n_test >= 500` and `n_test >= 100`, while explicitly
  forbidding those slices from model routing.

Exp267 does not select a source configuration. Its labels are diagnostic only.

### Exp268: Multiscale Scattering Deviation

Exp268 adds a fixed wavelet-scattering representation rather than repeating the
Exp239 wavelet-energy summary.

- First- and second-order scattering coefficients capture interactions across
  scales and time.
- Scattering configuration is derived from TRAIN target length and frozen
  constants.
- A robust TRAIN-normal scaler and shrinkage distance produce the anomaly
  score.
- Query-excluded TRAIN OOF scores calibrate empirical upper-tail p-values.

This source targets transient oscillation, modulation, and multiscale shape
changes that energy summaries may hide.

### Exp269: Ordered Path-Signature Deviation

Exp269 represents the ordered geometry of each trace.

- Channels are value, first difference, and TRAIN-normalized time.
- Fixed-depth truncated path signatures are computed over the full trace and
  preregistered multiscale windows.
- Robust shrinkage distance to TRAIN-normal signatures produces the score.
- OOF calibration follows the same query-exclusion contract.

This source distinguishes traces with similar global statistics but different
change order, direction, or lead-lag structure.

### Exp270-A: Lightweight Nonlinear Normal Forecasting

Eligibility: `n_train >= 20`.

- Fixed multiscale lag windows are extracted from TRAIN-normal traces.
- Random Fourier features plus ridge regression predict the next fixed segment.
- Ridge strength is selected only by deterministic TRAIN cross-fitting over a
  frozen grid.
- The score combines robust median, p95, and maximum forecast residuals.
- Every pseudo-test TRAIN instance is excluded from its fitted predictor and
  calibration reference.

### Exp270-B: Causal TCN Normal Forecasting

Eligibility: `n_train >= 100`.

- A small causal dilated convolutional network predicts future patches from
  past context.
- It does not reconstruct the visible input and therefore remains distinct from
  Exp247/257 masked reconstruction.
- Architecture, patch size, dilation schedule, loss, epochs, optimizer, and two
  seeds are frozen before full TEST evaluation.
- TRAIN-normal validation controls early stopping with a frozen maximum epoch;
  no TEST metric controls training.
- Score is a preregistered robust aggregate of future-patch residuals.
- Seed median and seed intersection are both reported; neither is selected from
  TEST performance.

MPS is used only if a smoke benchmark shows lower runtime than CPU while
preserving the numerical and reproducibility contract. Only one MPS job may run
at a time.

### Exp270-C: Forecasting Risk And Stability Audit

Exp270-C audits both forecast sources on TRAIN-normal pseudo-runs:

- normal candidate rate and one-sided 95% Wilson upper bound,
- fold and seed stability,
- score distribution drift across TRAIN-count tiers,
- abstention and failure coverage,
- correlation with all existing certified evidence groups.

### Exp271: New-Source Qualification

Each new source is assessed independently before it can join a vote. Required
views are:

1. standalone candidates,
2. confirmation inside frozen Hard,
3. prioritization inside frozen review lanes,
4. incremental candidates outside all frozen lanes,
5. recovered zero-support FN after prediction freeze,
6. aligned TRAIN OOF source correlation.

Highly correlated variants are collapsed into one evidence group. Variants of
the same algorithm never receive multiple independent votes.

### Exp272: Recovery Review Selector

Exp272 combines qualified new evidence with the existing certified groups.
Existing lanes remain byte-for-byte unchanged.

Routing order:

1. preserve existing Hard, Shadow, Priority, and Standard lanes,
2. route a data-quality violation to Data Quality Review,
3. route an outside-lane candidate supported by at least three independent
   groups to Recovery Shadow,
4. route an outside-lane candidate supported by exactly two independent groups
   to Recovery Priority,
5. retain zero- or one-source candidates as No Alert; report one-source
   sensitivity separately.

All Exp272 additions are human-assisted review candidates.

### Exp273: Autonomous Promotion Audit

Exp273 does not automatically change Hard. It asks whether a narrow, frozen
subset is eligible for a future Hard-policy experiment.

A promotion candidate must satisfy all of the following preregistered gates:

- at least two genuinely independent evidence groups,
- TRAIN-normal autonomous false-alert Wilson upper bound at most 0.005 per
  virtual run,
- retrospective incremental precision at least 0.80,
- overall autonomous precision at least 0.80 after hypothetical addition,
- no family or dataset-name policy branch,
- complete reproducibility and invariance checks.

Passing these gates creates a future-policy candidate only. It does not prove
deployment safety, and it cannot be activated without a new experiment number
and prospective or locked-data validation.

## 6. Frozen Calibration Contract

- Primary alpha remains `0.005` for parity with current certified evidence.
- Sensitivity alphas `0.01`, `0.02`, and `0.05` may be reported but cannot
  replace the primary rule based on TEST performance.
- Empirical upper-tail p-values use finite-sample correction.
- OOF query instances are excluded from source fitting and calibration
  reference construction.
- Every source persists raw score vectors, OOF scores, candidate indices,
  eligibility, config, seed, code hash, and DB hash.
- Missing or invalid source outputs cause explicit abstention or failure, never
  silent empty candidates.

## 7. Success Criteria

These gates describe research usefulness, not deployment approval.

### New evidence usefulness

- Minimum useful signal: recover at least 5% of the 1,405 current zero-support
  FN, approximately 70 instances, in review lanes.
- Strong signal: recover at least 10%, approximately 141 instances.
- TRAIN-normal recovery-review Wilson upper bound: at most 0.01 per virtual run.
- Full 1,117-dataset accounting with no silently missing dataset.

### Autonomous candidate safety

- Exp253 Hard remains unchanged through Exp272.
- Exp273 applies the 0.80 precision and 0.005 TRAIN-risk gates without changing
  them after TEST evaluation.
- Autonomous and human-assisted metrics remain separate.

Failing a gate is a valid experimental result. It cannot trigger threshold,
epoch, source, or eligibility modification under the same experiment number.

## 8. Evaluation Views

Every experiment reports:

- all 1,117 datasets,
- `n_train` tiers `<5`, `5-19`, `20-99`, `100-199`, `200-499`, and `>=500`,
- retrospective TEST-size slices `<100`, `100-249`, `250-499`, and `>=500`,
- Hard alerts, TP, FP, FN, micro precision and recall, and mean dataset F1,
- review candidates, TP, FP, precision, workload per TEST run, and mean
  human-assisted F1,
- zero-F1 dataset count and zero-support FN count,
- candidate overlap, exact Jaccard, and source correlation,
- abstention, resolution blocking, errors, and coverage.

TEST-size and family views are post-evaluation diagnostics only. They never
control prediction.

## 9. Prediction And Evaluation Separation

Prediction functions accept only TRAIN-normal values, TEST values, frozen
configuration, and source artifacts. They do not accept or read:

- `y_test`, actual anomaly positions, or evaluation metrics,
- family names for policy decisions,
- dataset names for policy decisions,
- TEST length for threshold, budget, eligibility, or source selection.

Evaluation verifies the sorted prediction SHA-256 before reading TEST labels.
Prediction JSONL must not contain TP, FP, FN, precision, recall, F1, labels, or
actual anomaly positions.

## 10. Integrity And Invariance Tests

- TEST-label permutation leaves prediction hashes unchanged.
- Actual anomaly-position mutation leaves predictions unchanged.
- Dataset and family rename leave policy results unchanged.
- TEST append leaves TRAIN-derived model, calibration, threshold, and
  eligibility unchanged; prefix score behavior is reported with receptive-field
  boundaries.
- Input order changes do not change canonically sorted dataset predictions.
- Same input, config, and seed reproduce the same sorted output hash.
- TCN prediction does not read TEST labels and never trains on TEST values.
- `n_train < 100` cannot invoke the TCN path.
- Source variants from the same family count as one vote.
- Existing Exp253/266 lane hashes remain unchanged through Exp272.
- Evaluation refuses modified or incomplete prediction artifacts.

## 11. Execution And Dashboard Contract

Execution is sequential:

1. Exp267 diagnostic freeze,
2. Exp268 scattering source,
3. Exp269 path-signature source,
4. Exp270-A lightweight forecast,
5. Exp270-B Causal TCN,
6. Exp270-C risk audit,
7. Exp271 qualification,
8. Exp272 recovery review selector,
9. Exp273 promotion audit.

Only one full experiment runs at a time. CPU experiments use up to seven
workers. MPS is exclusive. The dashboard records queued/running/completed/
blocked state, dataset progress, current dataset, recent throughput, ETA,
errors, source coverage, and output hashes. Dashboard failure must not stop or
erase the experiment; atomic local status files remain the source of truth.

## 12. Outputs

Large artifacts are stored outside Git under:

```text
/Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery/
```

Repository evidence is stored under:

```text
outputs/exp267_273_zero_support_fn_recovery/
```

Required outputs include the frozen config manifest, source cache manifest,
TRAIN OOF risk audit, prediction JSONL and hash, dataset and global result CSVs,
source-correlation report, zero-support FN diagnostics, independent
reaggregation, invariance logs, result report, and MANIFEST.

## 13. Interpretation Limits

- The same benchmark TEST labels have been observed repeatedly during prior
  research. Exp267-273 are retrospective even when prediction and evaluation
  are structurally separated.
- The `n_train >= 100` learner rule is an eligibility contract, not evidence
  that a learner is superior.
- Passing Exp273 does not establish end-to-end strict TRAIN-only development
  provenance for all historical feature choices.
- No result establishes real-equipment performance, operating K, or prospective
  user trust.
- Automatic-alert activation requires a separate preregistered experiment and
  new locked or prospective data.
