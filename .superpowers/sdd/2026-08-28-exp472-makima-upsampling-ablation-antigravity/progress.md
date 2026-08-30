# SDD ledger — plan: docs/superpowers/plans/2026-08-28-exp472-makima-upsampling-ablation-antigravity.md

Spec authority: docs/superpowers/specs/2026-08-28-exp472-makima-upsampling-ablation-antigravity-design.md
Branch start: 60db771104bc32f44bbf3686af8ecdb145e471f6
Design commit: 9e0d3f00
Plan commits: 0073f7fd, 5cb3236f
Agent policy: Grok 4.6 / high, Nemotron 3 Ultra / ultra, Laguna S 2.1 / ultra. Implementers never spawn subagents. Reviews are read-only.

## Pre-flight interface scan

| Row | Producer / consumer | Shared file or interface | Finding |
|---|---|---|---|
| Self-1 | Task 1 | code-map generator, config, contracts, research state | Internally consistent after initial map is generated before scientific files and regenerated after them. |
| Self-2 | Task 2 | Makima transform and diagnostics | Tests cover singleton/equal/down/up, length 2, endpoints, determinism, finite output, and synthetic shape families. |
| Self-3 | Task 3 | Exp450 injection boundary and E adapter | Original vendoring instruction conflicted with the review rubric's duplication rule; corrected in plan commit 5cb3236f. |
| Self-4 | Task 4 | shard lifecycle and scheduler | MPS is one worker; CPU fallback is logical CPUs minus one; two waves and terminal manifest rules agree. |
| Self-5 | Task 5 | Exp471 authority and evaluation | Label access is gated by both E and Exp471 authorities; full and upsampling-subset metrics are distinct. |
| Self-6 | Task 6 | report and CLI | Recursive manifest self-hash was removed in plan commit 5cb3236f; 15 content leaves are bound by a separate manifest and marker. |
| Self-7 | Task 7 | full implementation verification | Test cache path is writable; historical root-presence tests are deliberately excluded because their roots exist. |
| Self-8 | Task 8 | 23-dataset smoke | Operational gates never read labels or metrics and record diagnostics without selection. |
| Self-9 | Task 9 | 1,117-dataset E run | Reuses durable shards, records failures, and does not auto-restart an unexplained process exit. |
| Self-10 | Task 10 | evaluation/report/final state | Evaluation occurs once after both authorities verify; completion requires digest reopen and process exit. |
| 1→2 | Task 1 / Task 2 | config, contracts, code map | Backend and condition types are available before transform implementation; map is refreshed with new callers/tests. |
| 1→3 | Task 1 / Task 3 | condition/backend contracts and code map | E adapter consumes the frozen condition and backend; no Exp471 file is modified. |
| 1→4 | Task 1 / Task 4 | run context, shard types, roots | Artifact paths and identities are frozen before durable writes. |
| 1→5 | Task 1 / Task 5 | Exp471 reuse root and bootstrap constants | Exact root, seed, draws, and method order remain centralized. |
| 1→6 | Task 1 / Task 6 | report roots and authority types | CLI/report consumes only frozen roots and typed authority records. |
| 1→7 | Task 1 / Task 7 | code-map triplet | Every scientific/module-boundary task refreshes all three files together. |
| 1→10 | Task 1 / Task 10 | RESEARCH_STATE.json, EXPERIMENT_CATALOG.json | Task 1 appends planned state; Task 10 only changes the Exp472 record to completed evidence. |
| 2→3 | Task 2 / Task 3 | align_rows_exp472_antigravity | Exact two-argument override matches the new Exp450 injection callable. |
| 2→4 | Task 2 / Task 4 | diagnostics and upsampled row count | Final shards receive label-free diagnostics from the same transform path. |
| 2→5 | Task 2 / Task 5 | upsampled_rows | Evaluation subset is fixed by values-only preprocessing metadata, not metric results. |
| 3→4 | Task 3 / Task 4 | score_dataset_condition_e and compose_final_score | Dataset-local typed result becomes exactly one hard_plus_stable float64 shard. |
| 4→5 | Task 4 / Task 5 | verified E manifest | Evaluation refuses incomplete/failing inventory before any label import. |
| 4→6 | Task 4 / Task 6 | status/progress/manifest | CLI exposes read-only counters and never starts science for status/verify. |
| 4→8 | Task 4 / Task 8 | smoke runner and artifacts | Smoke uses the same artifact schema under a different frozen root. |
| 4→9 | Task 4 / Task 9 | full runner and resume | Official run reopens verified shards and schedules only missing/failed identities. |
| 5→6 | Task 5 / Task 6 | Exp472EvaluationAntigravity and reuse authority | Report binds exact evaluation and reuse digests. |
| 5→10 | Task 5 / Task 10 | evaluation command | Labels open once only after terminal E and Exp471 authority verification. |
| 6→7 | Task 6 / Task 7 | CLI/report tests and code map | Whole implementation verification covers closed commands and filename policy. |
| 6→8 | Task 6 / Task 8 | readiness/smoke/status/verify commands | Smoke procedure uses only declared commands and mode option. |
| 6→9 | Task 6 / Task 9 | run/status/verify commands | Full execution and monitoring interfaces are fixed and non-parameterized scientifically. |
| 6→10 | Task 6 / Task 10 | evaluate/report/status commands | Report publication resumes missing post-score stages but never recomputes E. |
| 7→8 | Task 7 / Task 8 | implementation verification gate | Smoke cannot start until focused and retained boundary suites pass. |
| 8→9 | Task 8 / Task 9 | smoke authority | Full run is enabled only by 23 completed, zero failed, zero remaining. |
| 8→10 | Task 8 / Task 10 | RESEARCH_STATE.json | Smoke authority is retained when final results are appended; no prior field is overwritten silently. |
| 9→10 | Task 9 / Task 10 | terminal E manifest | Evaluation requires 1,117 verified shards and zero unresolved failures. |

## Rulings

Ruling: replace the planned 600-line Exp450 scorer vendoring with a keyword-only alignment-function injection boundary — the spec requires scientific isolation but does not require duplication — cost if wrong: the shared Exp450 scorer source changes, so A--D canonical-byte parity tests must block the task on any behavioral drift.

Ruling: treat the report manifest as authority over 15 content leaves and let a separate completion marker bind the manifest — a manifest cannot include its own cryptographic hash without recursion — cost if wrong: downstream tooling expecting the manifest itself in the 15-leaf map must be updated to read the completion marker.

Ruling: Tasks 8–10 official runtime commands are executed by the Codex controller after agent-reviewed preflight, not by OpenRouter subagents — free endpoints must not receive private raw data or mutate official roots — cost if wrong: less agent parallelism during the long run, but no scientific computation or review is omitted.

Ruling: derive each Exp471 D TEST blob/row identity from the verified Exp467 source-session catalog and recompute the Exp472 identity schema, rather than inventing hashes or traversing the raw database — cost if wrong: E–D coordinates could bind to different values, so exact cross-test identity blocks evaluation before labels.

Ruling: verify the frozen Exp471 report through its real completion-marker → artifact-map → nested typed-manifest chain, not a synthetic self-digested report leaf — cost if wrong: official evaluation fails closed on the actual Exp471 publication tree.

Ruling: report completion is bound by independently reopened Exp471→E typed verifier paths, not mutable same-root completion flags — cost if wrong: a partial or foreign E root could reach labels or publication.

Ruling: default post-score commands replay the backend and worker decision already bound into scored E shards, while only pre-score smoke/run assess the live machine — cost if wrong: paper provenance could describe hardware or workers that did not produce the reported scores.

Ruling: shape diagnostics are accepted only in the authentic nested Task3/Task4 schema, self-digest checked, and deterministically flattened with train/test prefixes — cost if wrong: official 1,117-shard reporting either fails or silently publishes invented diagnostic values.

Ruling (actual-trace v3 superseding): retire v2/raw-replay evidence.  A sealed
Exp472 trace binds the exact ordered live `RawDataset` boundary consumed by
Exp450: train/test indices and blobs plus every row's dtype, shape, memory
order, and C-order bytes digest, without labels.  It also binds canonical bytes
of the exact Exp450 result returned by the scorer.  A same-name input with
stale blobs but changed live rows, or a substituted same-input result, fails
before composition.  Plan callbacks must precede real passive observations;
all omissions, observer/scorer errors, malformed primitive IPC diagnostics,
and non-`<f4`/C-order output evidence poison or reject the trace.  This retains
the v2 history above but makes it non-authoritative.

Task 1 dispatch: Nemotron 3 Ultra / ultra failed before work because the endpoint was at capacity; reassigned to Laguna S 2.1 / ultra with the same brief and base 5cb3236f.

Task 1 dispatch retry: Laguna searched the wrong default worktree, then made no edits after receiving the absolute path; interrupted. Local Grok 4.6 / high read the brief but exited after a read_file tool-output error with no edits. Reassigned to GPT-5.6 Sol / ultra fallback at the same base 5cb3236f.

Task 1 environment correction: Sol also made no edits because collaboration agents can write only inside the current Codex workspace. The clean Exp472 worktree was moved without branch/history changes to `/Users/mac/.codex/worktrees/e1be/Time Series/.worktrees/exp472-makima-ablation-antigravity`; the design path was updated. Task 1 will be re-dispatched from base after the path-only commit.

Task 1 implementation: GPT-5.6 Sol / ultra completed the scoped identity, contracts, research records, and synchronized code-map work from base `cc61b742` in commits `0a008e3b` and `e6ddf4be`; official unittest passed 12/12 and code-map/JSON checks passed.

Task 1 review round 1: Grok 4.6 / high found one Important omission: bootstrap percentile 95% was not frozen. The original implementer fixed it and the directly related smoke-order, root-source, literal-identity-test, and code-map VERSION gaps in `e6ddf4be`.

Task 1 final review attempts: Laguna S 2.1 / ultra failed with provider 429 and Nemotron 3 Ultra / ultra failed with provider capacity before returning a verdict. Fresh Grok 4.6 / high re-review approved the complete `cc61b742..e6ddf4be` range with no Critical or Important findings; two non-blocking Minor nits remain (shared threshold constant and stricter interval-method mutation test).

Task 2 dispatch: local Grok 4.6 / high was tried twice in the isolated worktree but its `read_file` tool failed and the recovery session exited before edits. Terra / high completed Task 2 from base `e6ddf4be` in commit `a7c0249e`; Task 2 unittest passed 9/9, Task 1+2 regression passed 21/21, and code-map/JSON checks passed.

Task 2 review: fresh Grok 4.6 / high approved `e6ddf4be..a7c0249e` with no production Critical or Important findings. Non-blocking test-hardening notes: assert overshoot retention more directly, assert TEST identity `series_length`/`count`, add a downsampled aggregate fixture, and share the Task 1 canonical JSON encoder later.

Task 3 implementation: Terra / high completed the narrow Exp450 alignment injection and Exp472 E scorer from base `a7c0249e` in `70fdcd4f`. Initial Task 3 tests passed 5/5 and Task 1--3 regression passed 26/26, but the first Grok 4.6 / high review blocked on missing frozen pre-refactor A--D canonical-byte evidence.

Task 3 review fix: the original implementer captured public-API A--D canonical SHA-256 values from a detached `a7c0249e` base, added a real frozen digest regression plus stronger E/composition assertions, and committed `571c132a`. The Codex controller ran the unmocked ACSF1 B--D retained test once with official Python; it passed 1/1 in 69.614 seconds. Fresh Grok 4.6 / high re-review approved `a7c0249e..571c132a` with no Critical or Important findings.

Task 4 implementation: Terra / high completed durable shards, progress, resume, two-wave scheduling, and manifest publication in `6bc122dc`. The first Grok 4.6 / high review found Important gaps in mandatory catalog/TEST identity binding, CPU callback pickle safety, lifecycle/terminal runner coverage, and monotonic ETA semantics.

Task 4 review fix: the original implementer added digest-bound typed catalog entries, pickle-safe CPU packets/error preservation, torn-shard/resume/verifier-drift/23-entry manifest tests, monotonic ETA, completed-over-failed scheduling, and symlink-safe status in `5d9a391a`. Focused Task 4 passed 15/15 and Task 1--4 regression passed 41/41. Fresh Grok 4.6 / high re-review approved `571c132a..5d9a391a` with no Critical or Important findings.

Task 5 implementation and review rounds 1–3: Terra / high implemented Path-only Exp471 reuse authority, E authority verification, label-gated evaluation, metrics/bootstrap/curves, and digest binding across `470c8066`, `afdbc235`, `eac1dc83`, and `63cea709`. Grok 4.6 / high successively found and the implementer closed caller-Mapping trust, missing production E verification/digest coverage, Task 4 identity incompatibility, and synthetic D TEST hashes by projecting the verified Exp467 source catalog into Exp472's exact identity schema.

Task 5 review round 4: fresh Terra / high added production-path tests using real temporary canonical evaluation/report leaves, symlink rejection, and a public Exp471 reader failure proving E verification and labels remain uncalled in `740c4b6b`; `33c0bc8d` removed the ignored internal report from tracking and resynchronized the code map. Focused Task 5 passed 10/10 and Task 1--5 regression passed 51/51.

Task 5 independent review: Nemotron 3 Ultra / ultra and Laguna S 2.1 / ultra approved the minimized evidence summary for Task 6 entry, while Grok 4.6 / high inspected the actual six-commit code and found one remaining Important production mismatch: the report verifier assumed an invented root-level self-digested typed manifest instead of the frozen Exp471 publication chain.

Task 5 final round 5: fresh Terra / high replaced the invented leaf with the real canonical evaluation → `.exp471_report_complete_antigravity.json` → `artifact_sha256_antigravity.json` → nested typed-manifest authority chain in `1f989a69`, with no-symlink/regular-file/digest binding and production-shaped canonical-byte, digest-tamper, and symlink tests. Focused Task 5 passed 12/12, Task 1--5 regression passed 41/41, and compile/code-map/JSON/diff checks passed. Fresh Grok 4.6 / high approved `33c0bc8d..1f989a69` with no Critical, Important, or Minor findings affecting the contract.

Task 5 environment note: a read-only probe of the actual frozen root stops earlier at the existing Exp467 closure guard because the active feature worktree differs from fixed Git bytes. This is fail-closed before cache/report/E/labels and is an execution-environment prerequisite for Tasks 6–8, not a remaining Task 5 report-chain defect.

Task 6 implementation: Terra / high created deterministic 15-leaf reporting, non-recursive manifest/completion authority, closed CLI, exact launcher, runbook, and synchronized codemap in `3df2a2ab`. Initial focused tests passed 5/5 and full Exp472 regression passed 58/58.

Task 6 independent preflight: Nemotron 3 Ultra / ultra and Laguna S 2.1 / ultra reviewed the minimized contract and highlighted exact leaf inventory, canonical digest chain, label/scoring firewalls, full/subset denominators, axes, atomic publication, and backend provenance. Grok 4.6 / high inspected actual interfaces and supplied the production integration checklist.

Task 6 review round 1: Grok 4.6 / high found Important gaps in typed/default controller wiring, real shard diagnostics/backend provenance, production firewall tests, actual PNG curves, and subset denominator audit. The original Terra implementer added an executable typed controller and default runtime, Task4/5 authority wiring, real chart rendering, denominator preservation, and 12 production tests in `751dda0f`; full regression passed 65/65.

Task 6 review round 2: Grok approved four of five prior issues but found backend provenance was reassessed at report time. The original implementer reconstructed an untrusted candidate context from canonical scored-E shard evidence and required the Task4 manifest plus all 1,117 shard verifiers to authenticate it before post-score commands in `a4298f65`; only pre-score smoke/run assess live. Focused tests passed 15/15 and full regression passed 68/68.

Task 6 review round 3: Grok confirmed backend replay but found the reporter expected an invented flat diagnostic rather than Task3's authentic nested schema. The original implementer added exact nested schema and self-digest validation, fixed train/test-prefixed CSV flattening, authentic nested fixtures, tamper/schema/duplicate/missing failures, and canonical shuffled curve tests in `97e52dd8`. Focused tests passed 18/18, full Exp472 regression passed 71/71, and code-map/compile/JSON/diff checks passed. Fresh Grok 4.6 / high approved `a4298f65..97e52dd8` with no Critical, Important, or Minor findings.

Task 7 verification: fresh Terra / high ran the full Exp472 discovery in the writable Numba cache (71/71 passed in 56.125s) and retained Exp450/Exp471 boundary suites (53/53 passed, 6 planned skips, 7.972s). Codemap check, JSON, diff check, Exp471 namespace isolation, added-filename policy, and clean tracked status all passed. No code repair or empty commit was needed; no science, labels, official roots, database, or process was touched.

Task 7 independent review: Grok 4.6 / high was attempted three times but its CLI sessions exited after reading intent without returning a verdict. Nemotron 3 Ultra / ultra and Laguna S 2.1 / ultra independently approved entry to the values-only Task 8 smoke from the exact verification evidence, while explicitly withholding approval for the later full science run.

Ruling: Exp472 converts a loaded value object only at the verified Exp471-to-current-scorer boundary, requiring the captured `RawDataset` class, the expected dataset identity, contiguous value-only indices, exact bytes, and exact float32 row/blob correspondence before rebuilding the current `RawDataset` with the same tuples and arrays — cost if wrong: loosening the current scorer's type guard would accept an unverified or cross-dataset object, while retaining the captured class would fail every current low-frequency score before a shard is written.

Task 8 preflight boundary repair: the terminated 23-dataset smoke exposed a class-identity mismatch between Exp471's captured `RawDataset` and the current Exp472 scorer. Added the fail-closed boundary adapter and exact `status/verify --mode smoke|full` parser, including rejection cases. RED: 5/5 failed before implementation (adapter absent, mode grammar absent, captured class reached current scorer). GREEN: 6/6 focused and 25/25 CLI/report tests passed; full Exp472 discovery passed 79/79 in 63.607s. The retained Exp471 discovery is not a clean gate in this Exp472 worktree: it reported 230 tests with 1 failure/2 errors because the pre-existing Exp471 codemap expected-edge inventory excludes Exp472→Exp471 imports, the Exp471 codemap JSON is stale, and its production-coordinate test intentionally fails closed when this feature worktree differs from fixed Exp467 Git bytes. The generated Exp472 code-map check, Python compile, JSON parse, and diff check passed. No official root, source worktree, database, label route, or experiment process was changed.

Task 8 review round 1 repair: closure now includes `cli_antigravity.py`, so any current-contract adapter or closed-command boundary edit changes the durable E context and old shards cannot authenticate under the new code. The adapter now rejects an empty TRAIN or TEST value group while retaining legitimate variable row lengths. `contracts_paper.py` is an explicit retained code-map boundary with the CLI import edge in HTML/JSON/lock. RED 3 test cases reproduced all omissions; GREEN focused 3/3, CLI/report 28/28, and full Exp472 82/82 in 71.296s passed. No official runtime, DB, labels, or process was touched.

Ruling: a dataset-wide Exp450 `nonfinite_input` abstention records zero Makima invocation explicitly as `not_invoked/nonfinite_input`, rather than attempting a diagnostic or looking like a completed zero-upsample dataset — cost if wrong: the values-only score stage either fails after a valid abstention or the paper report makes absent interpolation evidence indistinguishable from completed evidence.

Task 8 second smoke repair: two identical failure waves established that 20 nonfinite datasets reached Exp472 diagnostics despite Exp450 abstaining before alignment, while 3 persisted shards failed only because JSON list identities became frozen tuples after reconstruction. The repair adopts shape-diagnostic schema v2 with raw/aligned cardinalities and explicit completed versus not-invoked state, preserves the finite input guard on actual resampling, and compares reconstructed/persisted shard records through canonical JSON bytes. RED reproduced both root causes; GREEN new targeted tests 3/3, transforms/artifacts/reporting 55/55, contracts 13/13, scoring with writable Numba cache 5/5, and authority/evaluation 12/12. The six Exp472 discovery modules therefore pass 85/85, split into bounded commands solely because the tool collection limit is 30 seconds. The Exp472 code-map generator and `--check`, compile, JSON, and diff checks passed. No official runtime, DB, labels, Exp471 root, or process was changed.

Ruling (actual-trace v3 superseding): the v2 raw-replay and
`not_invoked` rulings are historical only and are non-authoritative. The sole
authoritative Exp472 evidence is the actual Exp450 aligner-call trace v3: it
binds exact ordered live train/test rows and the exact returned Exp450 result,
requires observed phase/source calls to equal the gate-derived expected plan,
and performs strict complete diagnostic validation before CPU IPC typed-shard
reconstruction. Reporting/evaluation parser migration to consume this v3
evidence remains the next sequential task.

Task 8 v3 reporting/evaluation migration: evaluation independently reopens the
terminal E manifest and every shard, then invokes the public v3 diagnostic
validator again before the label loader. Its digest binds per-dataset unique
`upsampled_rows`, repeated `makima_row_invocation_count`, and trace digest.
Reporting accepts the same strict v3 trace only, writes one diagnostic CSV row
per actual alignment invocation with call-specific target/evidence, and
publishes a denominator row plus Korean/runbook explanation that distinguishes
unique slots from repeated exposures. Retired v2/raw-replay inputs are rejected
before labels or report completion.

Migration verification: reporting/CLI 31/31; authority/evaluation 13/13
(split because the two 1,117-dataset / 20,000-bootstrap tests each complete
within the command watchdog on their own); transforms/artifacts/config 47/47;
scoring 12/12 split around the expensive composition fixture. The retained Exp450
actual-observer plan and public A--D canonical-byte tests both pass. Code-map
check, JSON parsing, compileall, and diff whitespace checks pass. No smoke,
official result root, database, labels outside temporary fixtures, or process
was started or changed by this migration.

CLI synthetic E2E repair: the production `main_antigravity` path now has a
1,117-dataset synthetic authority test that performs first `evaluate`, second
`evaluate`, and `report` through the real persistence reader and report
publisher. The label loader is the only science-input seam: it returns
in-memory synthetic labels and records one opening. The six Exp471/E authority
verification functions are deliberately mocked to canonical synthetic
authorities, so the E2E does not read private source trees or result roots. The
test proves the first evaluate persists,
the second strictly reopens without another label opening, report consumes the
same evaluation without labels, and the actual 15-leaf report manifest/marker
chain reopens with the report verifier. RED initially exposed that canonical
JSON sorts manifest leaf keys while the report authority constructor requires
the frozen leaf order; the test reconstructs that contractual order before
reopening. GREEN focused command: 1/1 passed in 19.601s. No production path,
official data, TEST label, database, result root, or process was accessed.

Persisted evaluation authority repair: evaluation publication now writes a
separate create-or-identical canonical completion marker that binds the
evaluation digest, Exp471 authority, terminal E-manifest digest, and context
digest. The reader requires that marker as well as the canonical artifact.
It also verifies that `subset_per_dataset` is the exact full-table projection,
checks ROC thresholds/coordinates and PR coordinates against producer
invariants, and cross-checks micro AP/AUROC against their persisted curves.
Raw TEST labels and score vectors are not copied into the durable artifact;
the completion marker is the immutable generated-science commitment, while
reopening performs the strongest available no-label structural and numerical
checks. A faulty computed result now fails its expected authority/order checks
before any publication write. The score-runner status no longer reports
noncanonical evaluation/report paths; closed CLI status reports only canonical
presence markers.

Final publication-boundary ruling (a0ee23e2, a7943330, final capability
repair): the frozen Exp471/E terminal score authority and the create-only,
0600 private `exp472_private_label_evidence_antigravity.json` are replayed to
rebuild every public evaluation section before a persisted evaluation is
accepted.  The public evaluation writer was removed.  Only
`read_or_evaluate_exp472_antigravity` may invoke the underscore-private
no-clobber publisher after trusted replay, and the trusted reader returns an
opaque module-private verified capability only after canonical equality.
Reporting rejects raw v3, v2, and look-alike capabilities.  This is a Python
internal API boundary, not a defense against a privileged process that imports
private symbols or deletes/replaces every anchor; that stronger threat requires
external signing/WORM storage.  Private evidence remains outside the 15 public
leaves, report manifest, and Git.  Focused boundary tests cover raw/fake report
rejection, public writer absence, trusted input requirement, and the synthetic
production CLI evaluate→reopen→report flow; complete Exp472 and retained Exp450
evidence is recorded with the final commit.

Final verification: official Python full Exp472 suite passed 132/132 in
905.242 seconds with `NUMBA_CACHE_DIR=/private/tmp/exp472_numba_cache_antigravity`
and `PYTHONDONTWRITEBYTECODE=1`. Retained Exp450 boundary suite passed 17/17
(six MPS-unavailable skips). Code-map regeneration/check, compileall, and
`git diff --check` passed. The only environment notice was joblib's macOS
physical-core probe fallback; it did not change the deterministic test result.

Capability immutability ruling: the report capability retains only canonical
evaluation-record bytes and bound digests, never an evaluation object or its
nested mappings. Report publication revalidates the byte snapshot and builds a
fresh local report view; a mismatched E-manifest digest stops before any leaf.

Immutable-capability verification (80b176dc follow-up): full official Python
Exp472 passed 132/132 in 860.511 seconds after the capability stopped exposing
evaluation fields; the affected denominator assertion now reads its canonical
snapshot via `artifact_record()`. No full-suite errors remained.

Independent snapshot-hash ruling: capability now binds SHA-256 of its complete
canonical record bytes separately from the evaluation self-digest. Report
checks that hash before parsing, then checks the self-digest and all issuance
digests; E-manifest mismatch and snapshot replacement fail before root/leaf
creation.

Independent-hash verification: official Python Exp472 passed 133/133 in
853.236 seconds; retained Exp450 passed 17/17 (six MPS-unavailable skips).
Code-map regeneration/check, compileall, and diff whitespace checks passed.

Task 8 operational completion: the mixed-context incomplete smoke root was
preserved whole as `smoke_recovery_evidence_20260830_antigravity`, then a clean
canonical values-only Smoke ran outside the sandbox with MPS and one effective
worker. It completed 23/23 with 0 failed and 0 remaining. Independent read-only
verification exited 0 and bound terminal manifest field SHA-256
`261835f92c208178f63f92bef7677587da63299a8d4d42891f44f2fb9713691b`;
the manifest file SHA-256 is
`81bf89ad33f4ba54560c835615cce046ed92e09e906a1554044400949918da42`.
Commit `a03bf5d0` records the authority and `full_run_authorized=true` only in
`RESEARCH_STATE.json`; Terra implementation checks and a fresh Sol review both
passed. No TEST labels were opened.
