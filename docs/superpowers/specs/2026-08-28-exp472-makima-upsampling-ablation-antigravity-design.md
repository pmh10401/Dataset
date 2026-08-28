# Exp472 Makima Upsampling Ablation Design

## Status

Approved by the user on 2026-08-28. Exp472 is an isolated supplementary
ablation. It does not modify or supersede the completed Exp471 artifacts.

## Purpose

Exp472 tests whether replacing PCHIP with modified Akima (Makima) during
time-series upsampling changes the proposed detector's ranking quality or
shape behavior. The experiment adds exactly one score condition and reuses
the frozen Exp471 evidence wherever identities match.

## Scientific Question

The primary question is whether Makima improves AP or AUROC relative to the
otherwise identical Exp471 condition D. The secondary question is whether
Makima changes overshoot and extrema behavior on rows that actually require
upsampling.

## Frozen Conditions

| ID | Downsampling | Upsampling | Image construction | Authority |
|---|---|---|---|---|
| D | Polyphase | PCHIP | Area/local-mean pooling | Frozen Exp471 control |
| E | Polyphase | Makima | Area/local-mean pooling | New Exp472 condition |

Condition E differs from D only when a row is shorter than its frozen target
length. Equal-length rows, singleton rows, downsampling, missing-value repair,
target-length selection, image features, estimator, seeds, thresholds, routes,
score orientation, dataset population, and TEST-row identities remain frozen.

The existing Exp471 A--D results and eligible public baselines may be shown as
context, but the preregistered primary contrast is E minus D. Exp472 does not
select a new production method automatically.

## Exact Makima Transform

- Equal source and target lengths return the source unchanged.
- A singleton finite source value is repeated to the target length.
- Downsampling uses the existing `scipy.signal.resample_poly` definition with
  the reduced integer ratio and exact-length validation.
- Upsampling uses `scipy.interpolate.Akima1DInterpolator` with
  `method="makima"` over normalized source positions `[0, 1]` and
  `extrapolate=False`, evaluated only on normalized target positions `[0, 1]`.
- Inputs with two or more finite points use Makima; the installed SciPy 1.17.1
  supports finite endpoint-preserving output for source lengths 2 and above.
- No clipping, smoothing, fallback to another interpolator, or label-dependent
  choice is permitted.
- Output must have the exact target length, be finite, and preserve the source
  endpoints within the preregistered numeric tolerance.

## Scope

### Included

- Pure Makima transform and routing tests.
- Synthetic shape diagnostics and the frozen 23-dataset values-only smoke.
- One new E score checkpoint for each of the 1,117 frozen datasets.
- Reuse of independently verified Exp471 D and contextual A--C/public-baseline
  artifacts without recomputation.
- Full-population and actual-upsampling-subset AP/AUROC analysis.
- Paired E-minus-D bootstrap intervals and win/tie/loss summaries.
- Paper-ready tables, ROC/precision-recall curves, and a Korean interpretation.
- Durable checkpoints, progress logs, resume, and remaining-time estimates.

### Excluded

- Akima, quadratic spline, conventional cubic spline, or global polynomial
  candidates.
- Modification of Exp471 code, score roots, reports, database, or provenance.
- TEST-label access before the complete E score root is frozen and verified.
- Hyperparameter search, per-dataset interpolation selection, result-based
  fallback, or automatic champion promotion.
- New security or operational machinery unrelated to result correctness,
  resumability, or artifact authority.

## Smoke and Promotion Gate

The smoke is operational and shape-based; it never reads TEST labels or metric
values.

1. Fixed synthetic fixtures cover constant, monotone, peaked, oscillatory,
   short, singleton, equal-length, upsampling, and downsampling cases.
2. Record determinism, exact length, finite output, endpoint preservation,
   source-range overshoot, and added extrema. Overshoot and extrema are
   diagnostics rather than success-based selection criteria.
3. Run E on the same frozen 23-dataset values-only smoke inventory used by
   Exp450/Exp471.
4. Promote only if every expected score is finite, every identity and shape
   matches, the selected backend remains fixed, and resume verification passes.

If the smoke fails, preserve the exact failure evidence and stop. Do not
silently substitute PCHIP, linear interpolation, or CPU/MPS backends.

## Full Execution and Reuse

Exp472 computes only condition E. It does not recompute condition D or the
public baselines. Before evaluation, the verifier must prove that reused
artifacts match the frozen database digest, dataset keys, TEST-row identities,
score orientation, and applicable source/evaluation digests.

Each successful dataset E result is checkpointed immediately. A transient
failure is recorded and the remaining datasets continue. Failed or unattempted
datasets may be retried in one bounded recovery wave under the identical
frozen context. Evaluation starts only after all 1,117 E datasets are complete
and verified with zero unresolved failures.

## Metrics and Analysis

### Primary analysis

- Macro AP and macro AUROC across eligible datasets.
- Paired per-dataset E-minus-D changes with 20,000 bootstrap resamples and a
  fixed Exp472 seed recorded in the implementation contract.
- Win/tie/loss counts with absolute metric difference `<= 1e-12` treated as a
  tie.

### Secondary analysis

- Micro AP and micro AUROC from pooled eligible TEST rows.
- The same metrics restricted to a values-only, label-independent inventory of
  datasets whose preprocessing actually invokes upsampling for at least one
  scored row.
- Overshoot magnitude and added-extrema diagnostics on upsampled rows.
- Context tables containing A--E and every eligible frozen public baseline.

Datasets lacking the required class support are reported in denominator audits
and are never silently discarded.

### Figures

- ROC x-axis: false-positive rate; y-axis: true-positive rate.
- Precision-recall x-axis: recall; y-axis: precision.
- Canonical curve-coordinate tables are written before PNG, SVG, and HTML
  figures.
- D and E must be visually distinguishable; A--C and public baselines are
  contextual rather than part of the primary claim.

## Backend and Parallelism Assessment

- Makima and polyphase are canonical SciPy CPU operations. They are not moved
  to PyTorch, MLX, or MPS because reimplementing the exact SciPy algorithms
  would change the frozen method and require a separate parity benchmark.
- The unchanged area/local-mean image path uses the existing frozen PyTorch MPS
  implementation when MPS is available at readiness time; otherwise it uses
  the existing recorded CPU fallback. The backend is frozen for the score root.
- MLX is assessed and recorded as unsuitable for the exact irregular 1-D SciPy
  interpolation path. It becomes eligible only in a separate implementation
  after exact index parity, finite output, numerical tolerance, and at least
  1.1x end-to-end speed are demonstrated.
- Run only one MPS-heavy process at a time.
- CPU-parallel stages use `max(1, logical_cpu_count - 1)` workers and record the
  detected logical CPU count and selected worker count. Inner numerical threads
  remain one where required by the existing pipeline.

## Isolation, Code Map, and Artifacts

Implementation occurs on branch `codex/exp472-makima-ablation-antigravity` in
the isolated worktree
`/Users/mac/Documents/Time Series/.worktrees/exp472-makima-ablation-antigravity`.

Before the first scientific source edit, create and verify the complete Exp472
triplet:

- `docs/codemap/codemap_exp472_antigravity.html`
- `docs/codemap/codemap_exp472_antigravity.json`
- `docs/codemap/codemap_exp472_antigravity.lock`

The map must identify the transform's callers, affected report/cache paths, and
covering tests. All new repository and runtime filenames contain
`antigravity`. Runtime outputs use a new Exp472 root and never overwrite an
Exp471 artifact.

Required runtime artifacts include the frozen config and values-only catalog,
backend decision, per-dataset E checkpoints, progress JSONL, E score manifest,
reuse-authority manifest, denominator audit, paired metric tables, canonical
curve tables, figures, paper report, and a leaf-hash/evaluation-digest binding.

## Verification

Before smoke:

- transform unit tests cover every routing and boundary case;
- E is proven identical to D on equal-length and downsampling-only fixtures;
- Makima is proven to be invoked only for true upsampling;
- configuration, source authority, resume, logging, filename, and Exp472 code
  map tests pass;
- MPS readiness is checked when the frozen area path selects MPS.

Before evaluation:

- all 1,117 E checkpoints verify with zero unresolved failures;
- the E manifest and every reused artifact pass independent identity checks;
- no TEST-label module was imported before E score freeze;
- score orientation and finite row counts match the frozen contract.

Before reporting:

- AP/AUROC are independently recomputed from canonical score/label tables;
- ROC and precision-recall axes are checked from canonical curve tables;
- report leaves are hash-bound to the verified evaluation digest;
- the official Exp472 process has exited cleanly.

## Interpretation Rule

Exp472 may conclude that Makima improves, degrades, or is statistically
indistinguishable from PCHIP. Algorithmically valid numerical differences are
accepted and reported. A confidence interval crossing zero is described as no
decisive paired difference, not as proof of equivalence. Any adoption of
Makima as the proposed default requires a separate explicit user decision
after the frozen Exp472 results are reviewed.
