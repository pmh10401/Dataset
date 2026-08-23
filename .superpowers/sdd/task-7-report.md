# Task 7 Report: Exp451 Code Map and Full Verification

## Source and scope

- Starting implementation commit: `a04930fb4f69f5123c906f353c83df24ab0ba72b`.
- Task-owned implementation:
  - `scratch/generate_exp451_codemap_antigravity.py`
  - `docs/codemap/codemap_exp451_antigravity.{html,json,lock}`
  - `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_codemap_antigravity.py`
  - this report
- Launch-blocker review fixes additionally cover Exp451 backend/contracts,
  migration/verifier, reporting, their focused tests, and the Korean provenance
  document.
- No Dataset root, TEST label, official command, running process, remote, or
  external publication was accessed or changed.

## TDD evidence

- The first focused run failed with `ModuleNotFoundError` because the Exp451
  generator did not exist.
- After the generator was staged, the next run had five passing contracts and
  failed only because the Exp451 triplet did not exist.
- A later synchronization test failed because the first lock did not bind the
  standalone JSON and HTML hashes. The lock now records both artifact SHA-256
  values and is published last.
- Final focused result: 7 tests passed.

## Code-map closure

- Exact node count: 42.
- Exact AST-import edge count at this commit: 344.
- Composition:
  - 12 Exp451 production modules;
  - 7 Exp451 tests, including the code-map contract test;
  - the Exp451 CLI and generator;
  - 21 direct external dependencies.
- Direct external dependencies:
  - Exp449 frozen config;
  - Exp450 artifacts, contracts, evaluation, reporting, and runner;
  - paper calibration, catalog, contracts, data, hard-route, and routing;
  - paper hard-sources, imaging, KNN, legacy-route, MiniRocket,
    MultiRocket/Hydra, preprocessing, ROCKET, and stability sources.
- Every node has exactly `path`, `module`, `callers`, `effects`, and `tests`.
- All 12 production nodes have non-empty caller, effect, and test answers.
- Critical assertions cover the Exp450 evaluator/reporter delegates, verifier
  checkpoint delegates, score-source dependencies, CLI lifecycle imports,
  catalog/data access, and static test links to both the CLI and generator.
- Caller discovery scans all index-tracked Python files. The selected closure
  itself admits no untracked files.

## Triplet and frozen Exp450 policy

- JSON and HTML carry the same parsed payload.
- The lock contains the exact sorted 42-path scope, current SHA-256 for every
  selected source, and SHA-256 bindings for the standalone JSON and HTML.
- Generation stages and fsyncs all three files before replacement, publishes
  the lock last, fsyncs the directory, and removes only temporary paths created
  by that invocation.
- Only no arguments (generate) and exact `--check` (read-only verification) are
  accepted. Any other argument fails before generation.
- The frozen Exp450 lock remains byte-identical with SHA-256
  `026ee148b0b9db0e7be5cdfe7e311815896eb4eb8b9a468e7fb5d2a0decd0c3a`.
  Its 36 recorded paths and all 36 current source hashes were reverified; the
  canonical fingerprint-map SHA-256 remains
  `29660a700804ac75bfa825e4ea49d8fd514d47526b7f752bc438d385cf0db402`.
- The Exp450 HTML, JSON, and lock were not modified.
- The legacy Exp450 `--check` is expected to report stale JSON because its
  repository-wide caller scan now sees later Exp451 files: stored edges are
  238 and current derived edges are 275. An independent comparison removed
  caller/test relationships and found the remaining payload identical, while
  the generated Exp450 lock remained byte-identical. This is caller-edge drift,
  not a frozen code-fingerprint failure, so the Exp450 triplet was not
  regenerated under the controller ruling.

## Launch-blocker fix round

- Imported checkpoint verification now obtains its expected backend only from
  migration's canonical frozen Exp450 backend record. Migration writes that
  same record into every imported target envelope. Native verification remains
  bound to the current Exp451 backend artifact. Production-path regressions
  proved that migration output previously failed imported verification while a
  wrongly substituted current backend passed; both directions are now closed.
- Report publication no longer creates the destination before its completion
  marker. It builds a complete marker-bearing sibling, fsyncs it, and publishes
  it through Darwin `RENAME_EXCL` while holding the kernel-released `fcntl`
  lock. Faults immediately before rename leave no destination; faults
  immediately after rename leave a complete reusable destination. Markerless
  foreign destinations, including the former progress-marker shape, are never
  deleted or adopted.
- The backend artifact now contains a complete `mlx_assessment`. The reviewed
  environment reports MLX 0.32.0, mlx-metal 0.32.0, metal available, and
  `Device(gpu, 0)`. Only tensor sub-stages are eligible. The frozen SciPy,
  scikit-learn, aeon, and PyTorch score path has no validated exact MLX
  expression, so gate one fails and transfer/compile measurement, parity, and
  the 1.1x speed benchmark are truthfully recorded as not run. The unchanged
  CPU/MPS path is the explicit canonical fallback.
- Focused RED evidence covered the imported/current backend mismatch, unsafe
  legacy directory reclamation, both atomic-publication crash windows, absent
  MLX assessment fields, and acceptance of an incomplete assessment. Nine
  focused review tests now pass.

## Verification

- `python3 scratch/generate_exp451_codemap_antigravity.py --check`
  - passed.
- `python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_codemap_antigravity`
  - passed: 7 tests.
- `python3 -m unittest discover -s experiments/exp451_verified_accelerated_resume_antigravity/tests -p 'test_*_antigravity.py'`
  - passed: 105 tests after the launch-blocker regressions were added.
- `python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_transforms_backend_antigravity experiments.exp450_resampling_image_ablation_antigravity.tests.test_imaging_antigravity`
  - passed: 17 tests.
- Independent lock/source fingerprint validation:
  - passed: 42 Exp451 source hashes and 36 frozen Exp450 source hashes.
- `python3 -m py_compile` for the Exp451 generator, CLI, all 12 production
  modules, and the code-map test:
  - passed.
- `git diff --check` and `git diff --cached --check`:
  - passed.

## Residual caveats

- AST import discovery intentionally cannot infer runtime-only dynamic imports;
  the code-map test statically imports the CLI and generator to expose their
  factual test relationship.
- Repository-wide caller discovery means a future tracked Python caller can
  make Exp451 `--check` stale even when the selected 42 source bytes do not
  change. That drift must be reviewed and the Exp451 triplet regenerated
  together; it must not be used to rewrite the frozen Exp450 triplet.
- The final commit SHA is reported in the completion response because a commit
  cannot contain its own SHA.
