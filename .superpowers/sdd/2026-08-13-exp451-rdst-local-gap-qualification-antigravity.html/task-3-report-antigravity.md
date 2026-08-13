# Exp451 Task 3 report — official RDST + LocalGap qualification

## Status

DONE. This task implements label-free, small-fixture-only score construction. It
does not open labels, read a database, create official roots, or run the 48-dataset
official lifecycle.

## RED / GREEN

RED was observed with the three requested modules absent: the focused unittest
command failed importing `rdst_antigravity`, `scoring_antigravity`, and
`backend_gate_antigravity`. GREEN passed after the minimal interfaces were added.

## Interfaces and scientific boundary

- `RDSTRepresentationAntigravity` constructs the audited public aeon
  `RandomDilatedShapeletTransform(max_shapelets=1024, random_state=seed, n_jobs=1)`
  and passes `y=None` only. It accepts exact finite `(rows, 1, length)` input,
  exposes fitted `n_shapelets_`, and requires width `3 * retained_shapelets`.
- `score_candidate_antigravity` validates label-free TRAIN/TEST rows, builds five
  fresh fold RDSTs plus one fresh full RDST, and invokes the fresh LocalGap backend
  factory exactly six times. Fold references use
  `seed + 1009 * (fold + 1) + 1024`; full reference uses `seed + 7919 + 1024`.
  The reference cap is 1000 and query chunks are at most 128.
- CPU LocalGap delegates to the existing Exp449 `LocalGapModelAntigravity`; the
  fixture parity test checks scores and empirical candidate indices directly.
- Aggregation accepts the frozen ordered seed pair only and uses elementwise median.
  The 100-row alpha=0.005 empirical calibration test reports attainable `1/101`,
  resolution-blocked, and no indices.

## Evidence

- Focused plus Exp449 scoring/MLX-gate regression: **33 tests passed** with the
  fixed `detection-coverage-antigravity-aeon140-py314` interpreter.
- Bounded real official RDST smoke: TRAIN `(10,)`, TEST `(3,)`, feature width
  `3072`, retained shapelets `1024`, all scores finite.
- Lazy backend probe: MLX is not installed; PyTorch `2.13.0` reports MPS device
  available. The MPS availability-only transfer/synchronization smoke recorded
  CPU `0.000089291s` and MPS `0.052576333s`; MLX absence probe recorded
  `0.020343083s`. Both decisions selected CPU. MLX fallback is `unavailable`; MPS is
  fail-closed because an exact robust-scaling/Euclidean LocalGap accelerator path
  has not been implemented and therefore cannot establish finite parity, exact
  candidate identity, normalized error <= `1e-5`, or inclusive speedup >= `1.1`.
  Those timings are explicitly availability smoke evidence, not LocalGap speed-gate
  evidence, so no speed gate claim is made.
- `py_compile`, codemap validation, and `git diff --check` passed. The map contains
  18 source-backed nodes (<=30) and the final map-only child has source parent
  `f1637aa5`; HTML/JSON/lock fingerprints validate.

## Commits

- `9760d3456185664e80063d25a32905ff384bec5b` — `feat(exp451): score official RDST with LocalGap`
- `325727e1` — `docs(codemap): refresh exp451 map`
- `f1637aa5` — `fix(exp451): time backend availability smoke`
- `450d8d21` — `docs(codemap): refresh exp451 map`

## Concerns

The official aeon transform emitted its own documented UserWarning about converting
internal invalid transform values to zero on tiny fixtures; the wrapper performs no
missing-value repair and output scores/features were finite. Accelerator selection
is intentionally CPU-only until a separately parity-proven exact LocalGap
implementation can pass the frozen gate.

## Fix round 1/5 — closed backend selection authority

RED: focused tests first showed `BackendProbeAntigravity` had no explicit
`implementation_qualified` authority (15 contract errors), accepted an
`np.int64` aggregate seed, and the previous LocalGap empirical assertion was
trivially resolution-blocked. The corrected hand calibration has p-values
`4/7, 3/7` and a nonempty `(1,)` candidate result at alpha `0.5`.

GREEN: a probe now records an exact boolean `implementation_qualified`; an
accelerator can only be selected when the probe passes finite/index/error/speed
gates, is implementation-qualified, and has a factory-supported exact
implementation. The supported-factory authority currently contains CPU only,
so synthetic numerically passing MLX/MPS probes still select CPU with an exact
fallback reason, and selection-to-factory consistency is tested. Probe and
decision schemas reject malformed names (including `np.str_`), bool-as-number,
negative/nonfinite error, empty version/device, bad timings, unqualified or
factory-unsupported accelerator choices, and direct invalid accelerator
decisions. Aggregate metadata and seed records now require exact positive
`int` values and the frozen ordered seed tuple.

Observable scoring tests now verify exact five fold and full reference seeds,
six distinct official RDST transformer identities with random_state equal to
the requested seed, fold-only TRAIN fit rows excluding held-out/TEST rows,
chunk transforms at most 128, and unchanged OOF alignment when fold iteration
order is reversed. Optional package/import/device/allocation/sync errors are
caught as closed CPU evidence rather than escaping the probe.

- Command evidence: focused Exp451 plus Exp449 scoring/MLX-gate regression:
  **38 tests passed**; map validation, `py_compile`, and `git diff --check`
  passed. No additional real smoke, DB, labels, official roots, or full run
  was performed.
- Commits: `f459cd90` (`fix(exp451): close backend selection authority`) and
  clean map-only child `ec50a6a9` (`docs(codemap): refresh exp451 map`), whose
  source parent is `f459cd90c1308edd9967f27cde50fea3e9602686`.

## Fix round 2/5 — exact probe retention and complete boundary evidence

RED: the inherited focused backend test rejected empty and duplicate probe
collections and arbitrary reasons, but exposed that an `mps` probe could replace
the actual `mlx` probe when both happened to derive identical reason text.

GREEN: canonical CPU fallback evidence now includes the exact probed backend name,
so the immutable decision retains exactly one validated probe and rejects empty,
multiple, substituted, or arbitrarily reasoned constructions. Aggregation evidence
now covers wrong seed, reversed order, duplicate seed, cross-seed TRAIN/TEST length
differences, coupled feature-dimension/retained-shapelet incompatibility, and exact
`bool`/`np.integer` rejection. The fit-boundary fixture gives every TRAIN and TEST
row a unique value signature and proves all five fold fits exclude every TEST row
and their exact held-out TRAIN rows, while the full fit contains exact TRAIN only
and excludes every TEST row.

- Focused Exp451 plus Exp449 scoring/MLX-gate regressions: **52 tests passed**.
- `py_compile`, codemap validation, and `git diff --check` passed. No real smoke,
  DB, labels, official roots, or full run was performed.
