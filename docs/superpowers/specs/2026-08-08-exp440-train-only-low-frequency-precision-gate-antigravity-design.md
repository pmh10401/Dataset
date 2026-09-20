# Exp440 TRAIN-Only Low-Frequency Precision Gate Design

## Research question

Can Exp439's distinct low-frequency reconstruction evidence be narrowed with
TRAIN-only stability and multi-depth agreement so that it recovers missed
anomalies without the large false-alarm increase observed in Exp439?

Exp440 is a new experiment. Exp439 remains immutable and unfavorable as a direct
union. Exp440 may reuse verified implementation components, but it must not
retune Exp439 or read Exp439 retrospective evaluation during prediction.

## Evidence motivating the new experiment

After Exp439 prediction SHA256
`500f6956b8640ea329cd47c8cdb504debaeff6ec67c44d0c56b96c1ecd77f3ef`
was frozen, retrospective evaluation found that its low-frequency channel added
233 true positives across 116 rescued datasets but also 996 false positives.
The counterfactual Hard union reduced micro F1 from `0.6241851368970013` to
`0.5832993890020367`. The low-frequency evidence is therefore useful but not
precise enough for direct promotion.

This result chooses the next research question only. No Exp440 prediction rule
may use TEST labels, anomaly positions, dataset-family TEST history, or the
Exp439 evaluation directory.

## Approaches considered

### A. TRAIN-only multi-depth consensus with a hard global budget — chosen

Require low-frequency nearest-normal evidence to exceed independently calibrated
TRAIN OOF thresholds at multiple adjacent FFT depths. Apply a TRAIN-only
calibration-stability gate, then retain at most the 100 strongest consensus
candidates using a fully frozen cross-dataset ordering.

This directly targets Exp439's main problem: isolated single-depth excursions
that create too many candidates.

### B. Intersect Exp439 low-frequency candidates with Exp437 candidates

This would be cheap and likely precise, but Exp439/Exp437 candidate overlap is
only about 5.5%. It would discard most of the genuinely distinct evidence that
motivates Exp440 and would make the new method dependent on a separate broad
score source.

### C. Keep one depth and take the global top 100

This satisfies the budget but does not test whether the score is structurally
stable. Cross-dataset scale differences could dominate the ranking and preserve
the same false-alarm mechanism in a smaller list.

## Frozen scientific design

### Identity and population

```text
Experiment: Exp440
Run identity: exp440-train-only-low-frequency-precision-gate-antigravity-20260808
Dataset population: 1,117 sorted unique datasets
Parent evidence: frozen Exp439 publication and implementation components
Comparator: frozen Exp404 Hard coordinates
Review only: true
Operational alert eligible: false
Prediction TEST-label access: exactly zero
CPU workers: max(1, logical CPUs - 1), expected 11 on this Mac
```

### Allowed inputs

Prediction may read only:

- normal TRAIN raw values;
- label-free TEST raw values and exact identity/length metadata;
- frozen Hard lane coordinates for exclusion;
- frozen source manifests and code/config identities;
- the Exp439 prediction publication only for provenance and population binding,
  never its evaluation directory.

TRAIN rows containing any NaN or infinity are excluded whole and audited. A
TEST row containing any nonfinite value abstains. No imputation is allowed.

### Low-frequency views

For each exact-length TRAIN group with at least four finite rows:

1. Compute every row's Exp439 preferred depth and take the lower median `d`.
2. Define the available adjacent depth set as the sorted unique intersection of
   `{d - 1, d, d + 1}` with `[1, maximum_depth(raw_length)]`.
3. If fewer than two depths are available, mark the group unavailable.
4. At each available depth, use the unchanged Exp439 FFT split, low-frequency
   vector, scalar robust transform, and nearest-normal `k=3` distance.
5. Build a completely fold-local leave-one-out OOF score vector at each depth.
6. Set the depth threshold to OOF quantile `q=0.99`, method `higher`; TEST uses
   strict `score > threshold`.

### TRAIN-only stability gate

A depth is stable only when all of the following hold:

- every OOF score and threshold is finite;
- the robust scale is greater than `1e-12`;
- at least four OOF scores exist;
- deterministic delete-one threshold replay produces finite thresholds; and
- `max(delete_one_thresholds) / max(min(delete_one_thresholds), 1e-12) <= 2.0`.

A group is eligible only when at least two adjacent depths are stable.

### Candidate consensus and ordering

For an eligible TEST row, define at each stable depth:

```text
exceeds = score > threshold
excess_ratio = score / max(threshold, 1e-12)
```

The row becomes a provisional candidate only when at least two stable depths
exceed their thresholds. Its frozen ranking key is:

1. number of exceeding depths, descending;
2. second-largest excess ratio, descending;
3. second-smallest empirical upper-tail OOF rank among the agreeing depths,
   ascending;
4. dataset name, ascending;
5. TEST instance index, ascending.

The empirical upper-tail rank is
`(1 + count(OOF score >= TEST score)) / (n_oof + 1)`.

Remove every coordinate already present in the frozen Hard lane, then retain the
first 100 coordinates globally. The total outside-Hard candidate budget is a
hard maximum of 100. There is no per-family route, manual override, tail
replacement, or post-TEST threshold change.

### Explanation evidence

For every retained candidate, publish the group depth, stable depth set,
per-depth score, threshold, excess ratio, empirical rank, agreement count, and
the final deterministic ranking position. This evidence is review-only.

## Acceleration and runtime

The NumPy CPU path is canonical. Before the full run, benchmark MLX for the exact
batched low-frequency dense-distance stage using the same inputs and seed.
Adopt MLX only if outputs are finite, all candidate coordinates match at the
frozen rules, absolute numerical differences are within `1e-5`, and measured
stage speed including transfer/synchronization is at least `1.1x` CPU. Record
MLX and mlx-metal versions, device, batch sizes `(16, 32, 64, 128, 256)`, timing,
parity, and fallback reason.

If MLX fails any gate, use CPU with 11 outer dataset workers and one numerical
thread per worker. Run only one MLX/MPS-heavy process at a time. Numba may be
benchmarked only as an exact optional implementation of a measured CPU hotspot;
it must pass the same candidate-parity and `1.1x` speed gate.

## Architecture

Exp440 receives its own package, tests, wrapper, documents, run identity, and
artifact roots. It may import pure Exp439 FFT and database primitives only
through explicit adapters. It must not alias Exp439 checkpoint, publication, or
evaluation roots.

Focused units:

- frozen configuration and contracts;
- exact-length/nonfinite TRAIN preparation;
- multi-depth low-frequency calibration and stability;
- deterministic global candidate budget;
- optional MLX/Numba readiness;
- ordered checkpoint/runtime/progress logging;
- no-clobber prediction publication;
- lazy post-freeze evaluation.

The long-running process writes atomic progress JSON, append-only JSONL, and a
text log containing PID, current dataset, TRAIN/TEST size and length, stage,
backend, completed/total, import/recompute/success/error counts, throughput,
elapsed time, ETA, checkpoint, heartbeat, and latest error. Resume is permitted
only from an exact interrupted ordered checkpoint.

## Evaluation and success rule

Only after the complete prediction is published and SHA256-frozen may the
explicit evaluation command open TEST labels. It compares `Hard union Exp440`
against frozen Hard over all 1,117 datasets.

Exp440 passes only if every condition holds:

- outside-Hard candidate count is at most 100;
- at least 10 previously missed datasets gain a true positive;
- added false positives are at most 30;
- outside-Hard candidate precision is at least 35%;
- micro F1 delta is positive;
- dataset-balanced macro F1 delta is positive;
- the 20,000-sample dataset bootstrap 95% lower bound is greater than zero for
  both micro and macro F1 delta;
- no frozen strong-dataset severe-regression guard is violated.

The strong-dataset guard fails when a dataset with baseline F1 at least `0.8`
loses more than `0.2` absolute F1. Some smaller regressions are allowed only when
all global and rescue/false-alarm rules above pass.

Failure selects no winner and leaves Exp440 review-only. TEST results cannot be
used to change Exp440 thresholds, depth set, stability ratio, ranking, or budget.
Any change requires Exp441 and a new preregistration.

## Testing and verification

Implementation uses TDD and must cover:

- exact adjacent-depth selection at minimum and maximum boundaries;
- fold-local OOF calibration and delete-one stability replay;
- strict two-depth consensus;
- deterministic cross-dataset ranking and exact 100-candidate truncation;
- Hard-coordinate exclusion before budgeting;
- whole-row nonfinite TRAIN exclusion and per-row TEST abstention;
- forbidden label/evaluation-path access before prediction freeze;
- CPU/MLX candidate parity and truthful CPU fallback;
- 11-worker ordered checkpoint determinism, crash recovery, and progress fields;
- no-clobber publication and post-freeze-only evaluation;
- Exp439 pure-component compatibility without reading Exp439 evaluation data.

## Code-map impact

Exp440 adds a new isolated research pipeline but should remain grouped under the
existing frequency-research primary node to preserve the 20-node limit. The map
must show the runner calling Exp440 runtime, runtime reading the frozen database
and Hard exclusion source, runtime writing its own checkpoint/logs, publication
reading the completed checkpoint, and evaluation reading frozen predictions and
TEST labels only after publication. All relationships require source-symbol
evidence.
