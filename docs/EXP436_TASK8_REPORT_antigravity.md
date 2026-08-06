# Exp436 Task8 Phase C Report

## Scope and preserved work

Task8 adds the fixed five-command boundary, readiness publication, post-freeze
evaluation firewall, review-only publication integration, outside-CWD wrapper,
and the final code-map triplet.  Phase A checkpoint hardening and Phase B
progress hardening remain unchanged except that the runtime's existing exact
`exp436.readiness-binding.v1` schema now validates Phase C evidence before any
input-session factory is invoked.

The retained runtime public calls are exactly:

`run_exp436_new(*, readiness_path, run_root, checkpoint_root, reviewed_commit,
input_session_factory)`, `resume_exp436(...)`, and `inspect_exp436(...)`.

## Phase C RED/GREEN record

The initial RED command was:

```text
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity -v
```

It ran 60 tests and failed as intended: no atomic readiness staging helper,
no exclusive parent-rename path, no resume-state guard, the old readiness
schema rejected the new evidence binding, the old CLI schema was incompatible
with the Task7 runtime, and the direct wrapper could not import outside the
repository.  The test fixture's initial obsolete `cpu_fallback` argument was
corrected before implementation; the subsequent RED failures were all missing
or incompatible required behavior.

The GREEN rerun of that same command passed all 60 tests.  The replacement
CLI now stages `readiness_antigravity.json` in a same-parent antigravity
directory, fsyncs and byte-verifies it, and only then exclusively renames the
directory into an absent fixed `RUN_PARENT`.  Failure cleanup removes staging
and cannot leave an empty parent; an existing or symlinked parent is rejected.

## Readiness and command evidence

- Readiness is canonical `exp436.readiness-binding.v1`, binds fixed
  output/checkpoint/evaluation sibling roots, reviewed clean commit, source and
  lane hashes, frozen population/config, full backend decision, focused-test
  result, candidate fixture hash, CPU topology, runtime versions, and zero
  TEST-label access.
- The runtime validates all those fields before creating the input session and
  continues to bind the complete artifact SHA256 into the stored run context.
- `new` validates byte-canonical readiness and clean HEAD twice; the final
  checked commit is passed only through the exact runtime API and input-session
  factory.
- `resume` accepts only `interrupted` or genuine `checkpoint_complete` state;
  failed, running, and fake complete states are rejected.  Runtime completion
  retains its no-session/no-engine/no-pool replay path.
- `inspect` calls only the read-only runtime path and never publishes.  The
  evaluator is imported only by `evaluate`; an evaluated run exits zero even
  when `success`/champion is false.

## Final repair review

- The production backend smoke now constructs the Task6 scalar fixtures
  correctly: one one-dimensional float32 query, 32 references, and the frozen
  finite threshold `0.0`.  A deterministic test observes both MLX/MPS calls;
  it does not run a hardware benchmark.
- Readiness publication uses Darwin `renamex_np(..., RENAME_EXCL)` after
  staged-byte verification.  A public `run_readiness()` integration test uses
  no CLI dependency injection and exercises the fixed-path, source-binding,
  canonical-publish route with only git/database/hardware boundaries faked.
- Publication now checks every staged byte and replays checkpoint/runtime
  source authorities again before rename.  Empty-destination and source-change
  races are rejection-tested.  Published-tree `inspect` selects immutable
  publication replay rather than treating `publication_antigravity` as a
  progress-inventory violation.
- Evaluator pre-label validation now consumes the publisher's canonical
  `reviewed_commit` and `reviewed_commit_sha256`.  A real 1,117-record
  publisher-to-evaluator handoff test covers this contract without patching
  inspection or validation.  An existing evaluation root is replayed only
  after full publication/source validation and must bind the publication
  prediction/manifest hashes, frozen database hash, and reviewed commit; this
  path opens no labels and creates no artifacts.
- Public `publish_exp436` and `main` no longer accept test override
  parameters.  Their private test seams remain internal; public evaluation
  continues to require its exact fixed sibling paths.

## Second repair review

- Publication/inspect replay now validates the exact four progress artifacts
  from immutable in-memory bytes.  It never creates the former sibling
  `.runtime-replay-*` directory or writes a transient copy; the runtime
  inventory itself remains strict.
- Existing evaluation replay now requires the frozen 1,117-row population,
  publication candidate count, label accounting and identity digest, exact
  aggregate confusion/F1/report projections, frozen 20,000-repetition
  bootstrap seed `43520260806`, frozen database hash, and reviewed commit.
  It remains a no-label/no-write route.  The production-shaped test fixture
  computes the genuine 20,000-repetition bootstrap once; the inspector checks
  frozen metadata and derived consistency rather than recomputing all 20,000
  samples on every immutable replay.
- After labels are loaded, source revalidation now completes before evaluation
  staging allocation and again immediately before exclusive rename.  A test
  makes any staging write fail, proving a post-label source mutation leaves no
  evaluation root or staged artifact.
- Readiness derives candidate evidence from the same scalar CPU score as the
  MLX/MPS fixture (`threshold=0.0`, candidate index `[0]`), rather than
  emitting a contradictory hard-coded empty list.  The public readiness test
  invokes the real helper and fakes only backend availability/benchmark calls.

## Third repair review

- Public publication inspection is now exactly `(output, completed_store)`;
  population expectations are private fixture-only machinery and are not
  exported.  Production callers therefore cannot redirect frozen replay.
- Existing public evaluation replay recomputes the exact production paired
  bootstrap from all parsed 1,117 dataset rows (20,000 repetitions, seed
  `43520260806`, NumPy linear percentiles), and requires equality with both
  stored bootstrap and top-level confidence intervals.  It also binds every
  ordered `(dataset, test_size)` row to publication summaries and derives
  predicted-positive accounting from per-dataset confusion deltas.  Forged CI,
  summary identity, and candidate-accounting artifacts fail before any label
  or write route.

## Verification evidence

Fresh commands completed with exit status zero:

```text
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity -v
# 60 tests

python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_evaluation_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity -q
# 76 tests: reporting 25, evaluation 26, CLI 25

python3 -m unittest discover -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests -p 'test_*_antigravity.py' -q
# 279 discovered tests

python3 -m unittest discover -s experiments/exp435_train_only_consensus_recovery_antigravity/tests -p 'test_*_antigravity.py' -q
# 187 tests

python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -q
# 30 tests

python3 -m compileall -q experiments/exp436_missing_safe_strict_consensus_antigravity scratch/run_exp436_missing_safe_strict_consensus_antigravity.py
```

Static checks also passed: the required TEST-label scan had matches only in
the isolated evaluator, explicit anti-smuggling guard, and evaluator tests;
the filename policy scan, both control-file JSON parses, `git diff --check`,
and code-map HTML/JSON identity check passed.  The map contains 20 nodes, 50
edges, and 10 flows; its lock contains refreshed fingerprints.

## Self-review and remaining concerns

No official readiness, database prediction, TEST-label evaluation, or hardware
benchmark was run.  This is intentional: focused tests use temporary roots
and fakes only below external git/database/hardware boundaries.  The eventual
operator readiness run remains gated on a reviewed clean commit and the
previously absent production `RUN_PARENT`; it will fail closed if any fixed
root already exists or source evidence changes.

The map uses the existing bounded Exp436 boundary node rather than adding a
twenty-first node; its evidence and five new flows explicitly cover runtime,
CLI, wrapper, publication, and evaluation.
