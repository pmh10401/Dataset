# Exp441 Task 7 implementation report

## Scope completed

- Added atomic, label-free readiness ownership with private sibling staging, canonical `0600` evidence, directory fsync, Darwin no-clobber rename, and preserved failed staging.
- Added the fixed `readiness`, `new`, `inspect`, and lazy `evaluate` CLI plus an outside-CWD wrapper that passes only the integer exit code to `SystemExit`.
- Added readiness/CLI behavior tests and regenerated the code-map JSON, HTML embedded payload, and lock triplet.

## TDD and verification evidence

- RED: the initial five focused public-boundary tests failed because both new modules were absent (`ModuleNotFoundError`).
- GREEN focused matrix: 11 tests passed. It covers exact command/flag surface, canonical stdout/integer exit, governance/source/selection tampering, existing/symlink parents, source failure preservation, clean-head mismatch, byte replay, lazy evaluator boundary, and outside-CWD help.
- Full Exp441 discovery: 129 tests passed, including the existing real 1,117-source replay and frozen selection identity tests.
- Fresh-interpreter firewall: mocked `readiness`, `new`, and `inspect` completed with `sqlite3.connect` exploding and import sentinels rejecting the Exp441 evaluator, Exp440 evaluator, and Exp417 inputs. Static AST checked config/contracts/artifacts/sources/selector/progress/runtime/reporting/readiness/wrapper for those forbidden imports.
- `git diff --check` and JSON code-map validation passed. No official readiness, new, inspect, or evaluate command was run; no SQLite or TEST-label route was opened.

## Judgment calls and concerns

- The selector has no tensor stage, so readiness records CPU/11 workers and an explicitly inapplicable acceleration benchmark rather than attempting MLX.
- The real official readiness is intentionally still pending: it requires a clean committed HEAD and must be run only after the requested review gate.

## Review fix round 1

- Readiness now treats the successful no-clobber rename as provisional until its parent directory fsync succeeds. If that fsync fails, only the captured owned inode is renamed no-clobber into a unique failed sibling, where canonical failure evidence is written; it never deletes or adopts a concurrent winner.
- Pre-rename failures, including a concurrent winner, retain private staging as a named failed sibling. Any evidence-write, rename, or fsync secondary error is attached to and chained from the original readiness failure.
- Validation now retains the exact `0700` root and `0600` readiness file descriptor pin across the entire source/selector/environment replay and final binding construction, then revalidates the root identity, namespace, mode, and bytes immediately before return.
- Review-fix matrix: 5/5 new tests passed; focused readiness/CLI set: 16/16; full Exp441 discovery: 134/134 passed, including the real 1,117-source replay. No official route was run.

## Review fix round 2

- The readiness staging directory and its exact readiness child are now pinned before `RENAME_EXCL`. Post-rename checks bind that retained inode and child descriptors to the destination namespace; they never first-capture ownership from the destination path.
- A deterministic move-aside plus empty replacement probe now fails closed without moving or adopting the replacement.
- Route/failure matrix now adds staging write/fsync, rollback rename/failed-parent-fsync chaining, CLI moving-HEAD, malformed root/file namespace, governance/backend/source mutation, mocked success stdout, fresh-interpreter firewall, and outside-CWD help coverage. Full Exp441 discovery passed with 142 tests.
- Round 3 establishes a full post-rename root metadata baseline after the retained descriptor rebind, rechecks the destination root after child checks, and rejects a chmod-away-and-back ctime-only mutation. Focused matrix passed 19/19 and full discovery passed 143/143, including the real 1,117-source replay.
- Separate actual-module firewall plus fixed-publication inspection/replay/tamper cases passed 5/5. The firewall imports the production CLI, readiness, runtime, and reporting modules after installing label-path and SQLite sentinels; no official route was run.

## Review fix round 4

- Added exactly four production-shaped regression families. The readiness family starts from the complete locally observed Python 3.14.6 / NumPy 2.4.6 / PyTorch 2.13.0 / MPS / MLX 0.32.0 / mlx-metal 0.32.0 payload, recursively discovers exactly 30 environment, backend-core, and nested-source leaf paths, proves the discovered path sets exhaustively, and rejects missing, wrong-type, and wrong-value substitutions for all 90 cases. Source hashes are recomputed after source mutations so the nested value checks do not pass merely because the enclosing digest is stale.
- Added post-rollback failure injection for the canonical failure-evidence write and failed-directory fsync. In both cases the original parent-fsync error object remains primary, the preservation error is the chained cause and note, the fixed parent is absent, the same owned inode is retained under one private `0700` failed sibling, and the recoverable `0600` readiness bytes remain present. Where the evidence write completed, its bytes are canonical.
- Added four genuinely fresh interpreter cases for `--help`, `readiness`, `new`, and `inspect`. Each installs the SQLite connection bomb and evaluator/Exp417-input import sentinel before importing the real production CLI, readiness, runtime, and reporting modules, verifies their filesystem-backed module paths, uses only the route-result patches allowed by the Task 7 plan, and never preloads fake modules into `sys.modules`.
- Added actual CLI `inspect` over an isolated valid six-file publication. The test captures the exact 16-entry run tree, entry types, modes, bytes, and stable device/inode identities before and after inspection; the tree is unchanged, stdout is one canonical JSON document, and the return value is integer zero.
- RED exposed one real production defect: a real `FrozenPublicationAntigravity` contains `SelectionResultAntigravity.hard_coordinates` as a `frozenset`, which the CLI status encoder could not serialize. The minimal fix converts frozen coordinate sets to sorted JSON-safe lists at the CLI serialization boundary. No readiness, source, selector, threshold, routing, runtime, or publication algorithm changed.
- GREEN route matrix: 11/11 passed. Focused CLI/readiness/review-fix/route matrix: 28/28 passed. Full Exp441 discovery: 147/147 passed in 60.617 seconds, including both existing real 1,117-source replay tests. The later-plan standalone `scratch/verify_exp441_production_replay_antigravity.py` is not present in this Task 7 checkout, so no separate-script result is claimed.
- No official readiness, `new`, `inspect`, or `evaluate` route was run. All route invocations used isolated fixtures or permitted patched results; SQLite and TEST-label access remained blocked.

## Review fix round 5

- RED used a real `ReadinessBindingAntigravity` containing the production `BackendAssessmentAntigravity` and its recursively frozen `MappingProxyType` environment. Direct serialization and the patched readiness CLI route both failed in `dataclasses.asdict()` because its deep copy raised `TypeError: cannot pickle 'mappingproxy' object`. A heterogeneous frozen-set probe separately failed under Python comparison.
- The minimal CLI-only fix removes `dataclasses.asdict()` entirely. It walks dataclass fields with `dataclasses.fields()` plus `getattr()`, recursively accepts only finite JSON primitives, string-keyed mappings including `MappingProxyType`, tuples/lists, and sets/frozensets, orders mappings by key and sets by each converted item's canonical JSON bytes, and explicitly rejects unsupported values and non-string keys.
- Regression coverage uses the real local readiness environment and a real `FrozenPublicationAntigravity` with nonempty Hard coordinates. Repeated serialization and reversed mapping/set insertion order produce byte-identical canonical stdout; `main_antigravity(["readiness"])` returns integer zero and emits exactly that one document.
- GREEN serializer regressions: 3/3 passed. Focused Task 7 readiness/CLI/review-fix/route matrix: 31/31 passed. Full Exp441 discovery: 150/150 passed in 62.182 seconds, including both real 1,117-source replay tests.
- No official readiness, `new`, `inspect`, or `evaluate` route was run. No readiness root, attempt, publication, evaluation, SQLite session, or TEST-label access was created.
