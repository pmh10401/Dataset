STATUS: complete
OBJECTIVE: Complete Exp444 Task 10 with a safe fixed five-command operator boundary and pre-mutation values-only source verification.

CHANGES:
- `readiness_antigravity.py`: adds immutable clean-HEAD/source/environment/control/backend readiness evidence while retaining an absent fixed run root.
- `cli_antigravity.py`: adds exactly `readiness`, `new`, `resume`, `inspect`, and `evaluate`; rejects unreviewed overrides; inspects publication before lazy evaluator import.
- `runtime_antigravity.py`: materializes and closes source inputs before new-root allocation and before interrupted writer/checkpoint owner mutation; complete finalization remains source-free.
- `test_readiness_antigravity.py`, `test_cli_antigravity.py`, and `test_runtime_antigravity.py`: cover missing boundary, unsafe override/basename rejection, evaluator import ordering, canonical wrapper errors, source-failure byte preservation, and pre-mutation source ordering.
- `run_exp444_train_only_stability_precision_gate_antigravity.py`: pins all seven thread variables before importing the CLI.
- `EXP444_RUNBOOK_antigravity_KO.md`: replaces pre-implementation warnings with the reviewed five-command boundary and no-poll/final-review gates.
- `codemap_exp444_antigravity.{html,json,lock}`: records the Task 10 boundary, evidence paths, five-flow cap, embedded-JSON identity, and functional-commit fingerprints.

TDD RED:
- `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_cli_antigravity` initially produced eight expected errors: missing `readiness_antigravity`, `cli_antigravity`, and wrapper modules.
- `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity.Exp444RuntimeAntigravityTests.test_new_source_failure_precedes_root_allocation_antigravity` initially failed because the absent new root existed after the injected source failure.

VERIFIED:
- Focused readiness/CLI command passed `Ran 8 tests ... OK`.
- Amended runtime suite passed `Ran 19 tests ... OK`, including new-root absence, interrupted byte identity/no-rebind, source-before-mutation ordering, and source-free complete finalization coverage.
- Exact discovery command `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*_antigravity.py'` exited zero with 181 discovered test methods and no failure/error output (about 27 seconds wall-clock).
- `scratch/run_exp444_train_only_stability_precision_gate_antigravity.py --help` exposed only `{readiness,new,resume,inspect,evaluate}`.
- `git diff --check` passed before both commits.

CODEMAP:
- Callers: the new scratch wrapper calls `cli_antigravity.main_antigravity`; the CLI calls readiness/runtime/reporting and imports evaluation only after `inspect_publication_antigravity` succeeds.
- Effects: runtime owns new-root/checkpoint/progress mutation; source materialization now completes before either new-root or interrupted ownership mutation; reporting remains inspection/publication; evaluation retains the label boundary.
- Tests: `test_runtime_antigravity.py`, `test_readiness_antigravity.py`, and `test_cli_antigravity.py` cover the changed boundaries.
- Validation parsed JSON and lock; checked all node/edge evidence paths and symbols, allowed edge types, 20 nodes/65 edges/5 flows, HTML embedded JSON equality, nine deterministic module fingerprints, functional source commit `303c5b1fb756178a7b7c81b89609ac4057fda486`, versioned triplet presence, and absence of an unversioned triplet.

JUDGMENT CALLS: Readiness returns detached evidence instead of persisting beneath the official root, preserving the required root absence for `new`. The map uses a minimal second commit so its lock can bind the functional commit without self-reference.

GAPS: None. No official run, publication, evaluation, or TEST-label access was performed.

---

## Fix Round 1 — reviewed commit, reviewed-document, and Task7 parity authority correction

STATUS: functional verification complete; code-map synchronization pending its separate map commit.

CHANGES:
- `config_antigravity.py` freezes the actual preregistration digest and a hardcoded Task7 worker-parity authority filename/SHA256 binding.
- `worker_parity_authority_antigravity.json` is a tracked canonical authority with the measured Task7 fields: `candidate_count=30`, `canonical_byte_equal=true`, `workers=11`, `label_access_count=0`, and equal `sequential_sha256`/`parallel_sha256` values of `729b31c0698fda6b60c17012a3fd2cae3f4b0899e01e5a3bf4d569902e814857`.
- `provenance_antigravity.py` and `readiness_antigravity.py` independently bind that file’s bytes to the frozen SHA and validate its exact schema and values. Readiness opens and hashes the actual approved design and preregistration files, then exposes both hashes and the parity-authority SHA in detached evidence.
- `CheckpointContextAntigravity` now carries an exact lowercase 40-hex `reviewed_commit`. Runtime creates the context from the supplied reviewed commit and resume requires an exact stored-context match before owner rebinding. Publication replay now parses and validates that stored commit-bearing context, rather than reconstructing a source-free context without a commit.
- CLI `new` and `resume` independently gate the exact current clean HEAD before runtime or publication. Direct runtime new/resume applies the same gate before provenance, source access, root allocation, writer, or checkpoint mutation. Complete-finalization still avoids provenance and values-only source reopening after that commit/context gate.
- `EXP444_RUNBOOK_antigravity_KO.md` removes the nonexistent `$RUN_ROOT/readiness_antigravity.json`; readiness evidence is documented as detached stdout JSON.
- `test_reporting_antigravity.py` was mechanically updated because its prior direct runtime fixture used the invalid literal `reviewed`; it now supplies the exact 40-hex test commit through the controlled clean-HEAD seam. No publication behavior changed.

TDD RED:
- Recovery RED command: `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity` ran 23 tests and failed with 18 expected errors: missing document/parity readiness APIs and payload fields, plus the missing `CheckpointContextAntigravity.reviewed_commit` parameter.
- Extended RED command: `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_readiness_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_cli_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_contracts_antigravity experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_checkpoint_antigravity` ran 24 tests and failed with 2 expected failures and 23 expected errors. The CLI failures proved it invoked `run_resume_antigravity` for dirty/mismatched HEAD; runtime lacked the injected clean-HEAD gate; readiness lacked exact document/parity implementations; and context lacked the commit field.

VERIFIED:
- Focused GREEN command over readiness, CLI, runtime, contracts, checkpoint, and reporting suites passed `Ran 79 tests ... OK`.
- Full Exp444 discovery command `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python -m unittest discover -s experiments/exp444_train_only_stability_precision_gate_antigravity/tests -p 'test_*_antigravity.py'` passed `Ran 188 tests ... OK` in 70.978 seconds.
- `scratch/run_exp444_train_only_stability_precision_gate_antigravity.py --help` exposed exactly `{readiness,new,resume,inspect,evaluate}`.
- `git diff --check` passed before the functional commit.

JUDGMENT CALLS:
- The Task7 payload retains the measured `sequential_sha256` and `parallel_sha256` names rather than renaming those evidence fields. The external file SHA is a hardcoded frozen config value, so it is not self-derived from the bytes being validated.
- Reviewed-document checks occur in readiness and runtime’s pre-mutation commit gate. The complete-finalization branch remains values-source-free: it does not call provenance verification or values materialization after commit/context validation.

GAPS: No official readiness/new/resume/inspect/evaluate command was run, no publication was created, and no TEST label was opened.

## Closeout correction

- Regenerated the HTML embedded base64 mechanically from `docs/codemap/codemap_exp444_antigravity.json`; decoding `const encoded` is byte-identical to that JSON. The temporary runtime overlay was deleted while preserving the existing interactive renderer.
- Removed this ignored scratch report from Git tracking with index-only removal. The local report remains available for audit, but the corrective commit does not add it to repository history.
- Verified embedded equality, 20 nodes/65 edges/5 flows, documented evidence symbols and paths, lock fingerprints, no unversioned codemap triplet, no tracked Exp444 Task 10 report, diff whitespace, and clean status after commit.
