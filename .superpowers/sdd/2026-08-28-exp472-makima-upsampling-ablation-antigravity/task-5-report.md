# Exp472 Task 5 report

## Scope

Created the label-gated Exp471 reuse authority and Exp472 condition-E evaluator.
The implementation accepts fixture evidence for isolated tests and exposes the
production authority boundary without starting an official run.

## Authority controls

- Requires the frozen Exp471 terminal partition: `11,170 = 10,775 + 395 + 0`.
- Requires `1,117` datasets, `922` common-support datasets, the fixed Exp471
  method order, SHA-256-shaped source/cache/evaluation/report/database digests,
  and every ordered TEST coordinate before labels may be requested.
- Compares TEST counts, indices, blob digests, and row digests exactly between E
  and D. Coordinate or database disagreement fails closed.

## Evaluation controls

- Loads labels once, after authority binding; public-baseline score absence stays
  a typed abstention rather than an imputed score.
- Emits per-dataset, macro, and micro AP/AUROC with explicit denominators and a
  label-independent `upsampled_rows > 0` subset.
- Uses E-minus-D paired percentile bootstrap (20,000 draws, seed `47220260828`),
  `1e-12` win/tie/loss tolerance, and bounded batches to avoid a full draw matrix.
- Emits canonical ROC and PR coordinates and checks bounds, ROC monotonicity and
  endpoints, and decreasing PR recall direction.

## Verification

The Task 5 test module is fixture-only; it did not access frozen TEST labels,
the original database, Exp471/Exp472 runtime roots, or live official processes.

## Review round 1 RED/GREEN

- **RED:** the review rejected the initial mapping-based authority boundary. The
  replacement test imports the new Path-only verified-bundle contracts, builds
  all 1,117 ordered datasets and all 11,170 typed Exp471 coordinates in memory,
  and patches only the local read-only Path adapters.
- **GREEN:** `verify_exp471_reuse_exp472_antigravity(Path)` now accepts no
  mapping. It reconstructs a typed bundle from Exp471's public source, cache,
  and shard readers; checks the evaluation/report leaves locally because Exp471
  has no public label-free reader for them; recomputes the exact terminal
  counts, ordered 1,117-coordinate inventory, and 922 all-method support.
- `evaluate_exp472_antigravity(e_root: Path, exp471_root: Path, label_loader)`
  verifies both Path authorities and every database/coordinate binding before
  the sole label-loader invocation. Tests prove failed Exp471 verification,
  failed E verification, coordinate drift, and nonfinite E rejection all leave
  the label call count at zero; the happy path calls it exactly once.
- The result supplies a canonical schema/digest record for Task 6 report
  binding, all-population and upsampled-row subset denominator audits, and the
  fixed paired bootstrap seed/draw/percentile/tie metadata.

## Review round 2

The E adapter now requires a typed `Exp472RunContextAntigravity` and exact
ordered `Exp472CatalogEntryAntigravity` catalog. It calls Task 4's public
`verify_manifest_exp472_antigravity` and `verify_score_shard_exp472_antigravity`
for every E descriptor before comparing the database and every TEST coordinate
with the verified Exp471 bundle. The evaluation digest now includes every
per-dataset/macro/micro/subset/denominator/paired/curve output.

## Review round 3 RED/GREEN

- **RED:** production-path fixtures rejected the old four-key coordinate shape
  and the former private E-path adapter seam. The new tests require a real
  typed `Exp472RunContextAntigravity`, all 1,117 ordered Task4 catalog entries,
  public Task4 manifest/shard verifier calls, and zero label reads on manifest,
  shard, or blob-coordinate failure.
- **GREEN:** the Exp471 adapter opens the public verified Exp467 source
  session and uses its verified catalog TEST identity. It projects the
  catalog's `dataset_id`-bearing identity to the exact Exp472 schema:
  `dataset`, `series_length`, `split`, ordered indices, ordered per-row blob
  digests, then a freshly computed Exp472 row digest. It does not traverse the
  database, and it validates every D shard's count and ordered indices against
  that source inventory.
- The E adapter validates the typed context/catalog before authority work,
  requires the manifest catalog digest, re-verifies every shard through Task4's
  public API, freezes E coordinates/scores/upsample counts, and binds all E
  coordinates to D before its only label callback. Same-count/index blob drift
  is covered by a pre-label failure test.
- The evaluator now exposes full and `upsampled_rows > 0` per-dataset, macro,
  micro, support/denominator, and E-minus-D paired outputs. The canonical
  artifact digest has mutation coverage for full/subset metric tables,
  denominators, both paired populations, ROC, and PR coordinates.

## Round 3 verification

- Focused authority/evaluation: 8 tests passed (split as 7 tests in 15.989s,
  plus digest mutation 1 test in 14.986s; the immutable bundle test is included
  in the 7-test run).
- Task1-5 regression: configuration/artifact runner 27 tests passed in 6.097s;
  scoring/transforms 14 tests passed in 22.279s with the mandated Numba cache.
  The only warning was an external Joblib/Loky macOS physical-core detection
  warning; no Task5 failure occurred.

## Review round 4 test hardening RED/GREEN

- **RED:** the reviewer identified that the Exp471 authority happy-path test
  replaced the private `_canonical_leaf` function. That meant the test could
  not detect a bypass of the production `lstat`, no-symlink, canonical-JSON,
  digest-field, or evaluation-to-report chain checks. The existing composed
  label firewall coverage also did not make the real Exp471 verifier fail
  through a public Exp471 reader.
- **GREEN:** the authority test now creates a temporary Exp471 root with the
  exact expected canonical `evaluation_antigravity/evaluation_antigravity.json`
  and `report_antigravity/exp471_typed_report_manifest_antigravity.json`
  leaves. It patches only the public Exp471 source/cache/shard readers and
  executes the real leaf validator and digest chain. A focused symlink leaf
  fixture is rejected before shard work.
- The composed public evaluator test leaves
  `verify_exp471_reuse_exp472_antigravity` unpatched, makes it fail through
  the public Exp471 source-authority reader, and proves both the Task 4 E
  manifest verifier and label loader remain uncalled. The happy path now also
  directly proves higher anomaly scores yield AP/AUROC of `1.0` for
  `(labels, scores) = ((0, 1), (0.2, 0.8))`, and checks paired bootstrap seed,
  draw count, and confidence level against frozen configuration constants.
- No Task 5 production change was required: the focused tests passed against
  the existing production authority and evaluation paths.

## Round 4 verification

- Focused Task 5 suite: 10 tests passed in 30.796 seconds using the prescribed
  Numba cache and bytecode settings.
- Task 1--5 regression: 51 tests passed in 61.241 seconds using the same
  settings. The pre-existing Joblib/Loky macOS physical-core detection warning
  appeared in the scoring test only; no test failed or errored.
- Regenerated the Exp472 code-map triplet and ran its `--check`; JSON and
  compilation/diff verification completed before commit. The map regeneration
  updated only the Task 5 test fingerprint in the lock file.
