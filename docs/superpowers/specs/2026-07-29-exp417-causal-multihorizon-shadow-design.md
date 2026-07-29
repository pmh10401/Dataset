# Exp417 Causal Multi-Horizon Shadow Design

Date: 2026-07-29

## 1. Purpose

Exp417 creates a new review-only Shadow score source from normal TRAIN series.
It does not modify or replace the frozen Hard, Standard, Priority, or
automatic-normal lanes.

The experiment asks whether a small causal model can learn normal waveform
dynamics and identify series whose future evolution is persistently
unpredictable at multiple time horizons. Exp417 is false-alarm-first: a
dataset may abstain, and the full prediction may fail publication, rather than
relax a threshold or trim candidates after seeing unlabeled TEST workload.

Exp417 never opens TEST labels. Any retrospective label audit is a separately
preregistered Exp418.

## 2. Non-Negotiable Governance

1. Model fitting, normalization, calibration, eligibility, thresholds,
   architecture, persistence rules, seed agreement, and workload limits use no
   TEST labels, anomaly locations, historical TEST performance, family
   performance, or human-reviewed outcomes.
2. TEST values may be scored only after the normal TRAIN contract and frozen
   configuration have been verified.
3. Existing Hard, Standard, Priority, and automatic-normal coordinates remain
   byte-for-byte unchanged.
4. New coordinates are `causal_multihorizon_shadow` only. They are not
   automatic alerts and are not promoted by Exp417.
5. No tail replacement, TEST-layout rule, top-N trimming, padding, truncation,
   or family-specific routing is allowed.
6. If a qualification or workload gate fails, Exp417 abstains or refuses
   official publication. It does not repair the failure by changing a
   threshold, horizon, seed, source, or budget.
7. Generated predictions, checkpoints, metrics, and dataset-derived results
   remain local and are not published to GitHub. GitHub may contain code,
   preregistration, runbooks, and synthetic test evidence only.
8. Any rule change after Exp418 requires a new experiment number and a new
   preregistration.

## 3. Scope and Frozen Lane Relationship

The Exp417 population is the same frozen 1,117-dataset research benchmark used
by Exp403-416. The benchmark is retrospective research evidence and is not
prospective semiconductor-equipment validation.

Before fitting or scoring, Exp417 verifies and freezes:

- source database identity;
- the 1,117-dataset population identity;
- frozen Hard, Standard, Priority, and automatic-normal coordinate artifacts;
- Exp417 configuration and code commit;
- Python, NumPy, Numba, PyTorch, and platform versions.

The primary Shadow output excludes every coordinate already present in a
frozen lane. Overlaps are counted in dataset qualification diagnostics but are
not duplicated into the new Shadow prediction.

The existing byte-exact normal TRAIN guard remains mandatory. Exp414 did not
qualify a near-repeat rule, so Exp417 records the near-repeat guard as
`not_available` and does not apply an unqualified near-distance rule.

## 4. Data Flow and Label Isolation

The required sequence is:

1. verify Git state, database identity, population, frozen lane artifacts, and
   frozen Exp417 configuration;
2. load normal TRAIN series and construct TRAIN-only length groups;
3. fit preprocessing, grouped folds, causal models, OOF calibration, and
   TRAIN-only safety gates;
4. freeze all eligible dataset contracts and model identities;
5. open unlabeled TEST values and calculate scores;
6. apply the frozen multi-horizon, persistence, seed-consensus, exact-normal,
   lane-exclusion, and workload gates;
7. revalidate every input snapshot;
8. atomically publish the exact Exp417 artifact set.

There is no TEST-label loader in the Exp417 package. A static import and
filesystem guard must prove that the package cannot reach the frozen label
artifact or database label columns.

## 5. TRAIN-Only Preprocessing

### 5.1 Length handling

Exp417 never resamples, pads, or truncates a series. Within each dataset,
series are grouped by exact observed length. Each group is trained and scored
independently with the same frozen architecture.

A length group is ineligible if:

- the length is too short to produce three distinct frozen horizons;
- fewer than 50 normal TRAIN series remain;
- any non-finite value remains after the preregistered Exp414 deterministic
  repair;
- the robust scale of either input channel is non-finite or at most `1e-12`.

### 5.2 Input channels

For each length group, preprocessing is fitted only on normal TRAIN values.
The two channels are:

1. the raw waveform standardized by the group-level normal TRAIN median and
   MAD-derived robust scale;
2. the first difference standardized by its own group-level normal TRAIN
   median and MAD-derived robust scale.

Scaling is group-level, not per-series, so persistent level and amplitude
changes are not normalized away. A zero robust scale makes the group abstain;
no epsilon chosen from TEST behavior is introduced.

### 5.3 Repeated-normal fold isolation

Normal TRAIN series with identical canonical waveform bytes belong to one
identity group. All members of an identity group remain in the same blocked
fold. Five OOF folds are assigned deterministically by identity hash and
balanced by series count. This preserves repeated-normal operation while
preventing a duplicate waveform from appearing on both sides of an OOF fit.

## 6. Causal Multi-Horizon Model

### 6.1 Frozen horizons

For a series of length `L`, the three horizons are:

- short: `1`;
- medium: `ceil(0.02 * L)`;
- long: `ceil(0.10 * L)`.

The group abstains unless these are three distinct positive integers and each
horizon leaves a non-empty causal prediction region.

### 6.2 Architecture

The model is a small causal dilated 1D convolutional network:

- input channels: 2;
- causal convolution blocks: 3;
- hidden channels: 32;
- kernel size: 5;
- dilations: 1, 2, and 4;
- output heads: one per frozen horizon;
- loss: equal-weight mean of per-head Huber losses;
- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- batch size: `min(32, training-series count)`;
- maximum epochs: 100;
- early-stopping patience: 10 epochs;
- fixed seeds: `41701`, `41702`, and `41703`;
- no architecture, learning-rate, epoch, or loss search against TEST
  outcomes.

Early stopping uses a deterministic 10% identity-group validation split drawn
only from the training side of each OOF fold. The split is assigned by
identity hash; no identical waveform can appear in both early-stopping train
and validation subsets. If the split cannot contain at least five identity
groups on both sides, the length group abstains.

### 6.3 Dataset eligibility

- length group with fewer than 50 normal TRAIN series: no model,
  `insufficient_train_abstain`;
- length group with 50 to 99: diagnostic OOF score only, no official Shadow
  candidate;
- length group with at least 100: candidate eligibility is possible after all
  safety gates.

The 100-series length-group boundary provides enough series-level empirical
resolution for the frozen `p <= 0.01` candidate condition. Dataset status is
the deterministic aggregation of all its length-group statuses; every group
must be accounted for even when another group qualifies.

## 7. Series-Level Score and Candidate Rule

For each horizon:

1. calculate the absolute causal prediction residual at every valid position;
2. normalize residuals with the horizon-specific normal OOF median and robust
   scale;
3. let `Q_h` be the 95th percentile of normalized residuals for the series;
4. define the point exceedance threshold as the normal OOF 99th percentile;
5. let `R_h` be the longest consecutive exceedance run divided by the number
   of valid prediction positions;
6. define the horizon score `S_h = Q_h + 2 * R_h`;
7. convert `S_h` to a series-level empirical p-value using only normal OOF
   series:

   `p_h = (1 + count(S_oof >= S_test)) / (1 + n_oof)`.

A TEST series becomes a preliminary seed-level candidate only if:

- at least two of three horizons have `p_h <= 0.01`; and
- every agreeing horizon has `R_h >= 0.05`.

An official dataset candidate additionally requires:

- identical candidate classification from all three fixed seeds;
- passage of the exact-normal TRAIN guard;
- absence from Hard, Standard, Priority, and automatic-normal coordinates;
- passage of the dataset qualification and workload gates.

One horizon, one seed, or one isolated spike cannot create an official Shadow
candidate.

## 8. TRAIN-Only Qualification

Each candidate-eligible dataset or length group must satisfy all of:

1. observed normal OOF candidate rate at most 1%;
2. 95% Wilson upper bound for the normal OOF candidate rate at most 5%;
3. three-seed candidate decision agreement of 100%;
4. no non-finite score, parameter, loss, or metric;
5. no backend fallback within a published model record;
6. pooled fixed-synthetic sensitivity at least 70%;
7. sensitivity at least 50% for each primary perturbation class.

The fixed primary perturbations are:

- persistent level shift;
- gradual drift;
- persistent amplitude scaling;
- persistent period change.

A short isolated spike is reported as a secondary diagnostic and is not a
qualification requirement. Perturbation magnitude, onset, duration, and
sampling seed are fixed as follows:

- onset: 40% of the valid series length;
- duration: the remaining 60%;
- level shift: add `3.0` normal TRAIN robust scales;
- gradual drift: linearly increase from zero to `3.0` robust scales;
- amplitude scaling: multiply the centered waveform by `1.5`;
- period change: deterministic time warp by factor `1.25`, evaluated only
  where the causal target remains defined;
- secondary spike: add `5.0` robust scales for 1% of the series length;
- perturbation seeds: `41711`, `41712`, and `41713`.

Every eligible normal OOF series receives every primary perturbation once per
perturbation seed. Actual anomalies do not define perturbation type,
magnitude, onset, duration, or sampling.

Failure of any qualification gate produces an abstention, not a looser rule.

## 9. Workload Gates

Workload gates use unlabeled candidate counts only and never select among
candidates.

### 9.1 Dataset gate

If more than 5% of a dataset's unlabeled TEST series would become new Shadow
candidates, the entire dataset is marked `excessive_dataset_workload_abstain`.
No top-N subset is retained.

### 9.2 Global gate

After dataset abstentions and frozen-lane exclusions, the new Shadow count
must be at most 1% of the full unlabeled TEST population. If this fails:

- no official eight-file Exp417 bundle is published;
- no threshold, horizon, seed, or budget is changed;
- the verified checkpoint workspace records
  `global_workload_gate_failed`;
- rerunning with changed settings requires a new experiment number.

Every dataset must end in exactly one state: `qualified`, `abstained`, or
`failed`. An unexplained missing dataset prevents publication.

## 10. PyTorch, MPS, Numba, and MLX

### 10.1 Canonical path

The canonical reference is deterministic PyTorch CPU. NumPy and Numba implement
score aggregation, persistence runs, empirical p-values, workload accounting,
and deterministic artifact checks.

MLX is explicitly excluded by project decision.

### 10.2 MPS adoption

MPS is optional and may become canonical only when a frozen TRAIN-only
benchmark passes all of:

- identical seeds and inputs;
- 100% identical candidate coordinates;
- maximum normalized score difference at most `1e-4`;
- Spearman score rank correlation at least `0.999`;
- no non-finite values or fallback;
- end-to-end speedup at least `1.1x`, including transfers and synchronization.

Benchmark datasets are selected deterministically from TRAIN-only normal-count
and series-length strata, with lexicographic tie-breaking. No TEST outcome
influences benchmark selection. If any gate fails, CPU remains canonical.

Only one MPS-heavy process runs at a time. Outer dataset multiprocessing is not
combined with concurrent MPS jobs without a separate deterministic benchmark.

## 11. Checkpoints and Fail-Closed Resume

Dataset records are written atomically. A checkpoint record is reusable only
when all of these match:

- Git commit;
- database and population identity;
- frozen-lane artifact hashes;
- frozen configuration hash;
- preprocessing contract;
- length group and identity-fold contract;
- model architecture and seed;
- PyTorch/NumPy/Numba versions;
- backend, device, and parity record;
- synthetic perturbation contract.

Mismatch, truncation, duplicate dataset records, invalid types, non-finite
values, failed parity, or unexplained fallback rejects resume. Source
checkpoints and input artifacts remain read-only.

## 12. Official Exp417 Artifacts

On successful global qualification and workload gating, Exp417 atomically
publishes exactly:

1. `exp417_shadow_predictions.jsonl`;
2. `exp417_dataset_qualification.csv`;
3. `exp417_oof_normal_metrics.csv`;
4. `exp417_synthetic_safety.csv`;
5. `exp417_backend_benchmark.json`;
6. `exp417_frozen_config.json`;
7. `EXP417_RESULT_REPORT_KO.md`;
8. `MANIFEST.json`.

The prediction rows contain only:

- dataset and candidate index;
- length-group identity;
- the three TRAIN-calibrated horizon scores and p-values;
- persistence evidence;
- seed-consensus evidence;
- frozen-lane exclusion status;
- explicit `contains_test_labels=false`;
- explicit `contains_evaluation_metrics=false`.

No TP, FP, FN, TN, precision, recall, F1, anomaly label, anomaly location, or
human-reviewed result may appear in the Exp417 bundle.

Publication is new-directory, exact-file, no-clobber, provenance-bound, and
descriptor-verified. Any failed verification leaves no official output.

## 13. Required Tests

Tests must prove:

- the Exp417 package has no TEST-label loading path;
- labels cannot be opened before or during prediction;
- repeated normal identities never cross OOF folds;
- padding, truncation, resampling, and tail replacement are absent;
- horizon construction and short-length abstention are exact;
- robust preprocessing is fitted only on normal TRAIN;
- model causality prevents future positions from entering earlier
  predictions;
- 50/100 TRAIN boundaries are exact and reject bool-as-int;
- empirical p-values, persistence runs, and two-of-three agreement match
  hand calculations;
- all three seeds must agree;
- exact-normal and all frozen lanes remain unchanged;
- qualification failures abstain rather than relax a threshold;
- dataset and global workload failures never perform top-N trimming;
- CPU/MPS benchmark gates and CPU fallback are fail-closed;
- checkpoint provenance rejects every independently mutated identity;
- prediction payloads reject labels and evaluation metrics;
- exact eight-file atomic publication rejects symlinks, races, mutation,
  extra files, missing files, and provenance mismatch;
- the same clean commit, input, config, seed, and backend produce identical
  sorted candidate coordinates and artifact hashes.

Exp414 and Exp416 regression suites remain green.

## 14. Exp418 Boundary

Exp418 is a separate retrospective audit created only after the Exp417
prediction and manifest are frozen.

Exp418 may report:

- Shadow TP and FP;
- precision and dataset-balanced diagnostics;
- dataset and family concentration;
- overlap diagnostics and review workload.

Exp418 cannot:

- change Exp417 horizons, architecture, p-value, persistence, seeds,
  perturbations, eligibility, or budgets;
- select a dataset or family routing rule;
- promote Shadow coordinates to Hard, Standard, or Priority;
- rewrite the Exp417 prediction.

An unfavorable Exp418 result leaves Exp417 frozen as research evidence. Any
new rule is Exp419 or later with a new preregistration.

## 15. Success Definition

Exp417 succeeds as an experiment when:

- all label-isolation, frozen-input, causality, determinism, qualification,
  workload, resume, and publication tests pass;
- the full population has complete qualified/abstained/failed accounting;
- an official prediction is published only if every global gate passes;
- existing lanes and operational policy remain unchanged;
- the output is interpretable as label-free Shadow evidence only.

Exp417 does not need to generate a nonzero candidate count to be valid. A
clean, reproducible abstention is preferable to an unqualified expansion.
