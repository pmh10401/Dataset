# Exp438 TRAIN-Frozen Frequency-Complexity Evidence Design

## 1. Purpose

Exp438 asks one narrow question:

> Does a TRAIN-frozen high/low-frequency decomposition plus amplitude-sensitive
> complexity expose useful anomaly evidence that Exp437's raw, gradient, and ACF
> channels do not expose?

Exp438 is inspired by the decomposition and amplitude-sensitive permutation
entropy ideas in PhysDiff, but it is not a reproduction of the full PhysDiff
diffusion model. The paper uses a multivariate MAFD implementation,
dataset-specific component counts, dataset-specific score weights, and
dataset-specific thresholds. Those choices do not transfer safely to this
project's 1,117 independent univariate datasets and strict label-free firewall.

The paper source is:

- Long Li et al., "PhysDiff: A Physically-Guided Diffusion Model for
  Multivariate Time Series Anomaly Detection," NeurIPS 2025.
- <https://papers.nips.cc/paper_files/paper/2025/file/980ea04d23d1f6908964eba2a74afe45-Paper-Conference.pdf>

### 1.1 Third-party provenance and licensing

Exp438 pins the following public source before implementation:

- ASPE reference semantics: `Brioal/ASPE` `entropy/aspe.m::aspe`, commit
  `f17cf8d796c4d25bc1b5ace0e9a7ef7be2944a89`, Apache-2.0,
  <https://github.com/Brioal/ASPE>.
- Adaptive Fourier Decomposition reference: `pikipity/Toolbox-for-Adaptive-
  Fourier-Decomposition`, commit
  `2b11bcc0eacd30f979ef4d6b24715c7fec327b02`, documented as
  CC BY-NC-ND 4.0,
  <https://github.com/pikipity/Toolbox-for-Adaptive-Fourier-Decomposition>.

The ASPE source permits a Python port with attribution. The implementation
must preserve the pinned algorithm's semantics, include the upstream URL,
commit, and license in its source header and manifest, and be checked against
independently calculated golden vectors. The original MATLAB source is not
copied line-for-line.

The AFD toolbox is not copied, modified, imported, vendored, or used as a
runtime dependency because its no-derivatives/non-commercial terms are not a
safe fit for this project. It is a read-only research reference only. Exp438's
FFT split below is a project-owned transparent approximation and must not be
called AFD or MAFD. The PhysDiff code link in the paper was not publicly
retrievable during design, so Exp438 makes no code-compatibility claim with
PhysDiff.

The similarly named public repository `VUT-HFUT/PhysDiff`, inspected at commit
`c4a17679cc3c81d0eea7e5ec07cccada1272bad7`, is explicitly excluded. Its
repository description identifies it as the AAAI 2025 "Physiology-based
Dynamicity Disentangled Diffusion Model for Remote Physiological Measurement."
Its `model/physdiff.py` consumes camera-derived spatial-temporal maps and
fixed-length blood-volume-pulse signals, and hard-codes CUDA in its training and
sampling paths. It is not the NeurIPS 2025 multivariate time-series anomaly
detection implementation used as Exp438's research inspiration. The repository
also exposes no detected license, so no code is copied from it.

## 2. Isolation From Exp437

Exp438 is developed on branch
`codex/exp438-frequency-complexity-evidence` in worktree
`/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence`.

The Exp438 run parent is:

`/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807`

Exp438 must not write to, import from, resume from, or clean any Exp437 runtime
root, checkpoint, progress file, prediction, or publication artifact. Exp437
artifacts are read-only comparison evidence only after both experiments have
independently frozen their predictions.

While the Exp437 PID is active, Exp438 may perform documentation work, code
implementation, and bounded single-process unit tests. Exp438 must not run an
official readiness benchmark, an MLX/MPS-heavy benchmark, or the 1,117-dataset
prediction. Official Exp438 readiness must prove that the Exp437 PID is
inactive before it can create the Exp438 run parent.

## 3. Scientific Firewall

Prediction may read only:

- the frozen database
  `/Users/mac/Documents/Dataset/univariate_ts.db`, SHA256
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`;
- normal TRAIN raw values and identity;
- TEST raw values, dataset/instance identity, and raw length;
- the frozen 1,117-dataset population and the four frozen lane coordinate sets.

Before prediction, manifest, and artifact hashes are frozen, prediction must
not select or open TEST labels, anomaly positions, TEST metrics, family-level
past TEST performance, or human-assisted metrics. TEST outcomes must not
reselect decomposition depth, feature definitions, source, threshold, alpha,
budget, routing, backend, or score fusion inside Exp438.

The prediction-phase TEST-label access count must be exactly zero. Exp438 is
`review_only=true` and `operational_alert_eligible=false`.

## 4. Relationship To Earlier Sources

Exp438 must remain distinguishable from these already evaluated sources:

- Exp242 computed one fixed complexity signature containing ordinary
  permutation entropy and spectral entropy.
- Exp245 selected TRAIN-dominant Fourier harmonics and combined reconstruction
  residual with coefficient distance.
- Exp437 compares raw, gradient, and ACF waveforms to exact-length normal TRAIN.

Exp438 instead freezes a normal-TRAIN-selected decomposition depth, retains
time-aligned low- and high-frequency components, and computes an
amplitude-sensitive ordinal signature on the original and both components.
It reports all three resulting score channels separately. It never counts the
three channels as independent votes in Exp438.

## 5. Exact-Length Input Contract

Rows are grouped by `(dataset, raw_length)`. No padding, truncation,
interpolation, resampling, tail replacement, or sibling value substitution is
allowed.

- Input values are decoded as little-endian float32 and retain original row
  identity and length.
- Minimum supported raw length is 16.
- At least four finite normal TRAIN rows of the exact raw length are required.
- A nonfinite normal TRAIN row is excluded in full and written to the TRAIN
  exclusion audit with dataset, instance, length, nonfinite count, positions,
  and original values-blob SHA256.
- A nonfinite TEST row produces only `nonfinite_test` abstention evidence. Its
  finite siblings continue.
- A group with fewer than four remaining finite normal TRAIN rows produces
  `insufficient_finite_train_rows` abstention evidence.
- Missing exact length, unsupported length, zero-scale channel, corrupt input,
  or nonfinite output fails closed or abstains exactly as preregistered; values
  are never repaired or synthesized.

## 6. Frozen Project-Owned Decomposition

### 6.1 Candidate depths

For a row of length `n`, define:

```text
max_depth(n) = min(5, max(1, floor(log2(n)) - 3))
depths = 1, ..., max_depth(n)
```

For depth `d`, compute a real FFT. Let `nyquist_bin = floor(n / 2)` and
`cutoff_bin = max(1, floor(nyquist_bin / 2**d))`. The low-frequency spectrum
retains bins `0..cutoff_bin` inclusive and sets every higher bin to zero. The
low-frequency component is the length-`n` inverse real FFT. The high-frequency
component is `original - low`. Both are canonicalized to contiguous float32.

Every row must satisfy finite output and
`max(abs(original - (low + high))) <= 1e-5`. Otherwise the row is unavailable.

### 6.2 Amplitude-sensitive permutation entropy

Exp438 uses embedding dimension `3` and delays `(1, 2, 4)`. Length 16 makes all
three delays available. The definition follows the pinned `Brioal/ASPE`
reference semantics:

1. Min-max normalize the complete input vector to `[0, 1]`.
2. Form every dimension-3 embedded window at the requested delay.
3. Use a stable ascending sort to assign one of the `3!` ordinal patterns;
   equal values retain original-position order.
4. For that individual window, calculate the coefficient-of-variation weight
   `sample_std(window) / mean(window)`. An undefined `0 / 0` weight contributes
   zero.
5. Sum the window weights by ordinal pattern and divide by the total finite
   weight to obtain the six weighted pattern probabilities.
6. Return normalized Shannon entropy
   `-sum(p * log2(p)) / log2(3!)`, with zero-probability terms omitted.

Exp438 adds one explicit industrial-safety extension: if input peak-to-peak is
`<=1e-12`, or if the final finite weight total is `<=1e-12`, return exactly
`0.0` and record `flat_or_zero_weight=true`. This prevents the upstream
constant-signal `NaN` from entering scoring while preserving the meaning of no
ordinal complexity. Any other nonfinite intermediate fails the row closed.

All arithmetic is float64 until the final feature vector is canonicalized to
contiguous float32. The manifest names this a source-compatible Python port
with the documented constant-signal extension; it must not claim bitwise
identity with MATLAB or full PhysDiff compatibility.

### 6.3 TRAIN-only stopping rule

For each row and each candidate depth, calculate ASPE on the high-frequency
residual at all three delays and take their median, producing `A[d]`.

The row's preferred depth is the depth before the first non-decrease:

```text
first d >= 2 where A[d] >= A[d - 1] - 1e-6  -> choose d - 1
no such d                                    -> choose max_depth
max_depth == 1                               -> choose 1
```

The final TEST model uses the lower median of the finite TRAIN row preferred
depths. TRAIN out-of-fold calibration repeats depth selection without the
pseudo-test TRAIN row. A pseudo-test row must not influence its own depth,
transform, references, or threshold.

## 7. Three Separate Score Channels

For the frozen depth, each finite row produces:

1. `high_frequency`: the high-frequency component waveform;
2. `low_frequency`: the low-frequency component waveform;
3. `amplitude_complexity`: nine ASPE values formed by
   `(original, low, high) x delays (1, 2, 4)`.

High- and low-frequency reference matrices use one scalar TRAIN median and
`1.4826 * MAD` per channel. A scale `<=1e-12` makes that channel unavailable.
The nine numeric ASPE columns use per-column TRAIN median and
`max(1.4826 * MAD, 1e-6)`. Flat/zero-weight flags are audit metadata only and
are not added as distance dimensions.

For every channel, the score is the median of the three smallest RMS distances
to transformed normal TRAIN references. Each TRAIN calibration score is
leave-one-row-out. The threshold is the `q=0.99`, `method='higher'` quantile of
finite TRAIN out-of-fold scores. A TEST row is abnormal only under strict
`score > threshold`.

The three abnormal sets are preserved separately. Exp438 does not use OR, AND,
weighted fusion, majority voting, best-channel selection, or channel-specific
alpha selection. Each outside-lane channel may contain at most 10,000
coordinates; the total across the three separately named sets may contain at
most 30,000 records. A budget breach fails prediction rather than cutting a
top-N list or changing a threshold.

Every score record also preserves whether its coordinate lies in each frozen
lane, but an Exp438 recovery candidate is defined as abnormal and outside all
four lanes. Frozen lanes are never modified.

## 8. Backend And Resource Policy

The canonical implementation is NumPy CPU. The official CPU worker count is:

```text
max(1, logical_cpu_count - 1)
```

On the current 12-logical-CPU Mac this is 11 dataset workers, with one
numerical-library thread per worker.

Numba may accelerate the ASPE loop only if it produces finite outputs,
bit-identical candidate coordinates, score error within `1e-5`, and at least
`1.1x` stage speedup including compilation amortized over the preregistered
smoke workload. MLX and PyTorch MPS may accelerate batched FFT or dense distance
stages only under the repository acceleration policy. The readiness seed is
`43801`; batch sizes are `(16, 32, 64, 128, 256)`; each candidate backend uses
two untimed warm-ups and five paired timed runs including transfer and device
synchronization. One accelerator-heavy process is allowed at a time.

The manifest records versions, device, timings, parity, candidate identity,
selection, and fallback reason. CPU fallback is a valid result.

## 9. Runtime And Publication

The CLI exposes explicit `readiness`, `new`, `resume`, `inspect`, and `evaluate`
modes. `new` requires a clean reviewed HEAD and an exact verified readiness
artifact. `resume` is legal only for an interrupted Exp438 root whose frozen
identity, config, database descriptor, manifest, checkpoint prefix, and
progress binding all match. A failed root is immutable.

The process itself atomically writes:

- `output_antigravity/progress_antigravity.json`;
- `output_antigravity/events_antigravity.jsonl`;
- `output_antigravity/parent_antigravity.log`;
- `output_antigravity/run_antigravity.pid`;
- ordered checkpoint records under `checkpoint_antigravity`.

Progress includes PID, heartbeat, current dataset, TRAIN/TEST row counts and
lengths, byte/value sizes, stage, backend, completed/total, success/abstention/
error counts, throughput, elapsed time, ETA, checkpoint, per-channel candidate
counts, and latest stable error. Codex does not repeatedly poll an unchanged
run.

Publication requires exactly 1,117 unique completed datasets, no missing or
extra population member, zero failed datasets, inactive PID,
`current_dataset=null`, budget compliance, zero prediction label access, and
recomputed agreement among checkpoint, event, summary, prediction, manifest,
and artifact hashes.

## 10. Retrospective Evaluation

Only after publication hashes are frozen may evaluation open TEST labels. All
three channels are reported without selecting a winner inside Exp438.

For each channel, evaluation reports:

- full and outside-lane candidate counts;
- TP, FP, FN, TN, precision, recall, specificity, FPR, and F1;
- newly detected datasets and datasets still missed;
- regressions, especially on previously strong datasets;
- counterfactual union with the frozen Exp404 Hard comparator;
- paired per-dataset micro and dataset-balanced macro deltas;
- 20,000 dataset-cluster bootstrap samples, seed `43820260807`, percentile
  2.5%/97.5% intervals;
- TRAIN OOF Spearman correlation against Exp437 raw, gradient, and ACF scores;
- label-free candidate overlap with Exp437 after both predictions are frozen.

Two decisions remain separate:

1. `direct_improvement_pass`: candidate count at most 100, at least 10 rescued
   datasets, at most 30 added false positives, positive paired micro and macro
   F1 point deltas, and both 95% CI lower bounds above zero.
2. `integration_evidence_pass`: at least 25 outside-lane true positives, at
   least 10 newly detected datasets, outside-lane precision at least 0.35, and
   absolute TRAIN OOF Spearman below 0.90 against every available Exp437
   channel with at least 20 aligned scores.

An unavailable or underpowered correlation cannot pass the independence gate.
Passing the second rule authorizes only discussion and preregistration of a new
experiment. It does not integrate a channel or change any operational lane.

## 11. Verification Strategy

Implementation follows strict TDD. Required tests include:

- exact FFT cutoff and `original == low + high` reconstruction;
- pinned ASPE golden vectors cover normalization, per-window sample
  coefficient-of-variation weighting, weighted pattern aggregation, and
  normalized base-2 entropy;
- ASPE distinguishes equal ordinal counts with different amplitude variation;
- stable tie handling, exact `0.0` flat-signal behavior, and the recorded
  `flat_or_zero_weight` extension;
- source headers and manifests pin the ASPE URL, commit, and Apache-2.0 license;
- no AFD toolbox file, import, or runtime dependency is present;
- pseudo-test TRAIN rows cannot influence their own depth, transforms,
  references, or threshold;
- exact-length grouping and row-local nonfinite abstention;
- deterministic output under TRAIN row reordering and worker-count changes;
- all three score channels remain separate and no fusion path exists;
- TEST label/value access counters enforce the frozen boundary;
- budget overflow fails without top-N cutting;
- checkpoint crash/restart, ordered-prefix recovery, and failed-root
  immutability;
- CPU/Numba/MLX/MPS readiness parity and truthful fallback evidence;
- prediction hash verification before the first TEST-label query;
- population conservation and independent evaluation reaggregation.

The existing code map is stale in `docs` and `root` relative to the current
HEAD. Before product-code implementation begins, regenerate
`docs/codemap/codemap.html`, `codemap.json`, and `codemap.lock` together and use
the map to identify callers, downstream effects, and tests for every reused
module.

## 12. Explicit Non-Goals

Exp438 does not:

- implement or claim full PhysDiff diffusion reconstruction;
- use dataset names or past family TEST performance for routing;
- choose a best score channel from Exp438 TEST results;
- fuse Exp438 with Exp437 or any frozen lane;
- modify Exp437 code, process, checkpoint, prediction, or result;
- start its official 1,117-dataset run while Exp437 is active;
- promote review-only evidence to an automatic alert.
