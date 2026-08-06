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

It ran 57 tests and failed as intended: no atomic readiness staging helper,
no exclusive parent-rename path, no resume-state guard, the old readiness
schema rejected the new evidence binding, the old CLI schema was incompatible
with the Task7 runtime, and the direct wrapper could not import outside the
repository.  The test fixture's initial obsolete `cpu_fallback` argument was
corrected before implementation; the subsequent RED failures were all missing
or incompatible required behavior.

The GREEN rerun of that same command passed all 57 tests.  The replacement
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

## Verification evidence

Fresh commands completed with exit status zero:

```text
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity -v
# 57 tests

python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_evaluation_antigravity experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity -q
# 65 tests: reporting 18, evaluation 25, CLI 22

python3 -m unittest discover -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests -p 'test_*_antigravity.py' -q
# 268 discovered tests

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
