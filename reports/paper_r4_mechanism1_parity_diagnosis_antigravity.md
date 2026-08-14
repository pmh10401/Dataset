# Paper r4 Mechanism 1 parity diagnosis

## Status

Task 4 establishes one reproducible, values-only **input-boundary mismatch**
for `ChlorineConcentration_normal_3`.  It does not establish the lower-level
score or selector primitive that introduced the mismatch, and it makes no
production-route repair.

The bounded replay found that current source/paper Exp93 contains coordinate
`2332`, while the SHA-pinned frozen Exp137 CSV records an eleven-coordinate
Exp93 input that excludes it.  The frozen Exp137 source verifies that exact
input before it invokes `route_tiers`.  Therefore the first persisted
selection boundary available in frozen evidence is **Exp93 -> Exp133**.

## Frozen evidence and isolation rules

| Item | Bound evidence |
| --- | --- |
| Frozen Exp137 result source | Git `3b118bf7a65543c7ccc23889d51e5765e5eff645`, `results/exp137/experiment_137_operational_triage_results.csv` |
| Frozen Exp137 program | `run_experiment_137_operational_triage.py`, SHA256 `5e389de7ee26063028b893d4856c9f1e24897f8fae32780ed2437bb90dd3b04c` |
| Live Exp137 authority | `experiment_137_operational_triage_results_antigravity.csv`, SHA256 `942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d` |
| Report-only cache manifest | `01_score_cache_manifest_antigravity.csv`, SHA256 `bf73796c046a6102fce53d3b6fc606afa1ed0a5161ebb6e92d94698ab56a68a0` |
| Report-only Chlorine cache | SHA256 `0e8b89549180492e9b53248a966aae3879e09de9b0addf667521932944c0a43b` |

`scratch/diagnose_paper_r4_mechanism1_parity_antigravity.py` first replays
current values-only source functions from a frozen-origin AST, then reads the
historical CSV and cache only for post-score comparison.  It reads no TEST
labels, does not use historical coordinates or cache values as a scorer or
selector input, and does not run the 1,117-dataset population or the 33-dataset
smoke.

Every authority and committed diagnostic claim is snapshot-bound: the file is
read once into immutable bytes, those exact bytes are SHA-verified, and CSV,
JSONL, or NPZ parsing uses `io.StringIO` or `io.BytesIO` from that snapshot.
No verified authority, frozen Exp137 CSV, score-cache manifest, or score-cache
NPZ path is reopened after hashing.  Symlink-replacement regressions cover the
authority loader, frozen CSV claim, and report-only cache claim.

## Exact boundary evidence

For the bounded `ChlorineConcentration_normal_3` replay (`n_train=1845`,
`n_test=2350`):

| Stage / artifact | Result relevant to `2332` |
| --- | --- |
| Current paper Exp74 | excluded; hard input has 11 coordinates |
| Current frozen-origin AST Exp89, Exp90, Exp93 | included |
| Current paper Exp89, Exp90, Exp93 | included |
| Frozen Exp137 CSV `source_exp93_hard_count` | `11` |
| Frozen Exp137 `route_tiers` input | `{2303, 2305, 2311, 2323, 2324, 2326, 2327, 2336, 2338, 2341, 2346}`; excludes `2332` |
| Frozen Exp137 output lanes | `high={2303,2305,2311,2323,2324,2327,2336,2338,2341,2346}`, `standard={2326}` |

The historical `run_one` AST was SHA-verified and confirms the invariant
`high | standard == source_hard` before the `route_tiers` call.  Thus the
frozen CSV's `source_exp93_hard_count` and lane union are a direct stored
assertion of the Exp93 input to Exp133.  Current Exp93 is that same set plus
`2332`; it is not acceptable to feed the stored set back into production
scoring merely to force parity.

## What the bounded checks rule out

The current paper and frozen-origin replay agree at Exp40, Exp84, Block A, and
Block B for this representative.  Exp55 has only a tiny floating-point score
difference and preserves candidates.  Exp56 has a real float32/float64 grid
difference and different intermediate candidates, but both current source and
paper still carry `2332` through Exp89/90/93; it is therefore not the causal
explanation for this frozen input-boundary mismatch.  The SHA-verified
report-only cache also contains `2332` in legacy Block B and Block C, which is
consistent with the current replay but was not used as replay input.

## Deliberately unresolved lower-level cause

The frozen Exp137 commit contains the result CSV and the Exp137 program, but
not the historical Exp89/90/93 result rows or their contemporaneous source
revisions.  Evidence therefore identifies the first **stored boundary** and
the frozen `run_one` / `route_tiers` invariant, not an exact lower-level
primitive that originally added or removed `2332`.  Claiming an Exp56, Block,
or route implementation repair from this evidence would be speculation.

### Bounded contemporaneous-evidence search

The repository and artifact index were searched once before retaining that
conclusion:

- `git show -s --format='%H%x09%cs%x09%s'
  3b118bf7a65543c7ccc23889d51e5765e5eff645` identifies frozen Exp137 as
  `2026-07-14` (`Add Exp137 operational triage results`).
- `git ls-tree -r --name-only 3b118bf7a65543c7ccc23889d51e5765e5eff645 |
  rg -i '(run_experiment_(89|90|93)|exp(eriment)?[_-]?(89|90|93))'` returned
  no path.
- `git log --all --reverse --format='%H%x09%cs%x09%s' --
  run_experiment_89_74d_with_exp84_candidate.py
  run_experiment_90_zero_f1_repair_selector.py
  run_experiment_93_nonpos_candidate_reranker.py` returned only
  `b9b4a49054e9a16b7552771904baa14d90bc8f03` on `2026-07-15`; its diff marks
  all three files `A`dded.
- Exact-name and structured-text scans of `/Users/mac/Documents/Dataset/experiments`
  returned no Exp89/90/93 artifact, CSV, JSON, JSONL, or manifest reference.
  Repository documentation describes the selectors, but it is not a
  contemporaneous frozen per-dataset result artifact.

This search supports the unresolved status; it does not justify inventing a
primitive or treating later source files as the source revision that generated
the frozen Exp137 row.

No `hard_route_paper.py`, `legacy_route_sources_paper.py`, Exp56, Block, or
production selector code is changed in Task 4.  A later repair requires a
separately frozen upstream Exp89/90/93 artifact or equivalent contemporaneous
source evidence and a new RED/GREEN cycle.

## T6 RED evidence

The recovery test preserves the boundary as an intentional current-head RED:

```text
/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python \
  -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_paper_r4_full_route_parity_recovery_antigravity.FullRouteParityRecoveryAntigravityTests.test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence

FAIL: test_mechanism1_diagnostic_isolation_reproduces_and_fixes_divergence
AssertionError: Items in the first set but not the second:
2332
Ran 1 test in 51.170s
FAILED (failures=1)
```

The test computes the present values-only replay before reading the frozen CSV,
then asserts the frozen `source_exp93_hard_count` and `route_tiers` lane union.
Its failure is expected until a separately justified future repair is made.

Separate T6b–T6d integrity regressions prove that an authority CSV, diagnostic
CSV claim, or report-only NPZ cache remains parsed from its already SHA-verified
bytes when a symlink is replaced immediately after the read.

## Execution and acceleration assessment

The diagnostic ran one bounded representative serially.  This host has 12
logical CPUs; the general CPU-parallel policy would allow 11 workers, but no
parallel experiment stage was started.  The frozen source functions are
NumPy/scikit-learn/aeon-style numerical and selector code, not a tensor-eligible
new scoring design.  PyTorch, MLX, and MPS are not canonical paths here:
transfer/compilation would change neither the required frozen algorithm nor its
deterministic parity boundary, and no CPU-versus-accelerated parity/speed
benchmark was authorized.  CPU remains the canonical diagnostic path.
