# Task 5 parity-serialization retry3 addendum — 2026-08-05

The second authorized retry root was
`/Users/mac/Documents/Dataset/exp425_task5_mps_parity_20260804_architecture_a_retry2`.
Its PID was 80796. It completed the one-fixture TRAIN-only MPS native computation
but failed before parity comparison, checkpoint record commit, result publication,
or success publication because the harness passed nested immutable
`mappingproxy` containers to canonical JSON serialization. During exception
unwinding, the outer cleanup path also attempted to close the already-closed
database snapshot a second time.

The retry2 loader counts were exactly TRAIN 1, TEST values 0, and TEST labels 0.
No TEST values, TEST labels, evaluation metric, candidate-selection feedback, or
shared-parity outcome was observed. The frozen model hashes, score source,
threshold, alpha, seeds, fixture, comparison tolerance, and routing policy are
unchanged. This is an execution-harness correction, not an adaptive scientific
choice, so it remains Task 5 of Exp425 rather than allocating a new experiment.

The preserved retry2 artifacts are:

- `progress.json` SHA256
  `861794bd14c833664bf2d1a02682e0a6e57e36661472a367744d16a37c65097c`;
- `events.jsonl` SHA256
  `79d492ba5c8d897cb7d532334e5a400e128a19e6c5a50781fcfe41c229918eba`;
- `parent.log` SHA256
  `38437c56a1c47cd99f0e4337430bb9895f4085a4c7be953341f9bcc12e6446fd`;
- `run.pid` SHA256
  `1e80561d68c826148b0bcec3e40198e7976a840f799eeeed6b93269bf319f558`;
- `checkpoint/checkpoint_provenance.json` SHA256
  `98846bfc255e7a4c4f5f035abbadfa1d21697b02aa0cab4a2d344e9a40a4ecda`;
- `checkpoint/run_context.json` SHA256
  `339c532d19b7fbc5202ffd99cf6ff48d6b82faaad1828d8b970546390716c696`;
- `checkpoint/progress.json` SHA256
  `b97303d898e743cc7189189ebfde160e9eb5d7b2cf14089b8dca486b948e92e3`.

These files and their root remain immutable. PID 80796 has exited. The retry2
checkpoint has zero committed dataset records and must not be resumed or reused.

## Frozen engineering correction

Before retry3, tests must prove both properties:

1. the allowlisted shared scientific payload recursively treats immutable
   mappings/tuples as their JSON object/array equivalents for both digest and
   value comparison, without dropping, renaming, or changing any value; and
2. after a runtime session owns the database snapshot, every success or failure
   path closes the session and runtime writer inside the snapshot context exactly
   once before publishing a terminal result or failure.

The correction must pass the focused parity suite, the complete Exp425 suite,
relevant frozen Exp417 regression tests, clean-source checks, and a fresh
Sol/High review. It must not alter any frozen scientific or label-free contract.

## Authorized retry3 root

After those gates pass, exactly one invocation may use:

`/Users/mac/Documents/Dataset/exp425_task5_mps_parity_20260804_architecture_a_retry3`

The original root, retry, retry2, and every arbitrary sibling are forbidden.
Retry3 remains a one-fixture TRAIN-only MPS parity smoke. It authorizes no TEST
value access, TEST-label access, evaluation, readiness command, or official
1,117-dataset execution.
