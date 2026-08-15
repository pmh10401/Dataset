# Exp449 Smoke A Task 4 report

## Scope delivered

- Replaced legacy operations progress and terminal filenames with the pinned
  Smoke A metadata names.
- Added lease-gated, result-neutral progress append with short-write looping,
  `EINTR` retry, and fd `fsync`.
- Added atomic-replace derived status cache with directory `fsync`; it is not
  used as recovery authority.
- Made public read-only status inspect only existing claim/receipt/progress
  tail/checkpoint-count/manifest/terminal metadata, without scorer import or
  filesystem healing.
- Retained temporary-fixture best-effort composition logging; configured Smoke
  A recovery requires the active lease for durable progress publication.

## TDD evidence

The new execution-progress test was written first. Its RED run failed on the
legacy `operations_progress_antigravity.jsonl` name and the absent authorized
writer/root helper. GREEN added the minimal durable writer and status cache,
then focused tests covered short writes, `EINTR`, append/fsync/cache failures,
crash-after-append status visibility, terminal idempotence, and forbidden
payload fields.

## Verification

Pinned Python 3.14 focused suite:

```text
Ran 73 tests
OK
```

The fresh-import status firewall confirmed `proposal_score_paper` and the
Exp449 runner remain absent from `sys.modules`; it returned `not_started` and
all six fixed roots were absent before and after the probe. `git diff --check`
was clean and the Exp449 JSON/HTML codemap payloads were byte-equivalent after
JSON parsing.

## Judgment call

The immutable paper collector's internal rolling queue is outside Task 4's
owned file set. The configured Smoke A parent boundary now propagates any
progress append/fsync failure before it invokes composition or processes a
subsequent parent progress callback; the temporary composition seam retains
best-effort logging solely for scorer-free fixture coverage.

## Fix round 1

Review found that the paper best-effort progress helper swallowed the
authorized callback and that the collector had no successful intermediate
checkpoint callback. The fix adds an explicit opt-in strict operational hook:
after every create-only success checkpoint or append-only attempt record, and
before the ProcessPool rolling queue is replenished, it validates and invokes
the callback without swallowing its exception. Default paper/Exp448 progress
remains best-effort. Exp449 passes this hook only for an active authorized
Smoke A lease. The red test first failed because the collector did not accept
`strict_progress`; green verifies a simulated fsync failure leaves the initial
window submitted and never submits the next dataset.

Derived cache replace/fsync errors are now caught at the cache-only boundary,
so authoritative terminal completion can still publish. The fsync regression
now creates the temporary operations root and asserts the patched fsync is
actually reached.

Fix-round verification: the full Exp449 test discovery suite passed 126 tests;
the affected paper artifact/data/lineage suites passed 70 tests; paper and
Exp449 codemap validators passed; and all four changed production modules
compiled with the pinned Python interpreter. The affected paper suite emitted
pre-existing SQLite `ResourceWarning` notices while still passing.

## Fix round 2

RED first exposed two independent defects: authorized strict mode also supplied
the durable callback through legacy `progress`, and the Exp449 fingerprint lock
omitted the already changed `paper` module. GREEN makes the runner branches
exclusive: authorized mode passes only `strict_progress`; non-strict fixture
and paper callers retain legacy best-effort `progress`. The new integration
regression makes legacy start/error/end invoke the callback if it is wired,
then raises a simulated fsync failure from strict progress. It proves no legacy
channel is present and the simulated next rolling submission is unreachable.

The Exp449 JSON/HTML/lock triplet was regenerated together. Its lock now
computes the changed modules against the pinned pre-Task-4 lock and records
`.superpowers`, `experiments`, and `paper`; the map describes the exclusive
strict channel and the persisted-checkpoint/attempt hook. Focused Exp449
progress/runner/codemap coverage passed 43 tests.

Final pinned-interpreter verification used
`/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`
with `pycatch22==0.5.0`: full Exp449 discovery passed 128 tests, and the
affected paper artifact/data/lineage suites passed 70 tests. The four affected
production modules compiled; JSON/HTML codemap payload and lock validation,
`git diff --check`, and a fresh scorer-free read-only status/root-absence probe
all passed. The unrelated long full paper discovery was intentionally stopped
after it entered hard-route raw-smoke coverage; it is outside this minimal
operations-only change.
