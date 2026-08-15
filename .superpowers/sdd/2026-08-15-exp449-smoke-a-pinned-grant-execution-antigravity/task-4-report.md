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
