# Task 2 implementation report

STATUS: complete

OBJECTIVE: Add create-only, context-bound canonical Task4+5 per-dataset success records that reconstruct exact typed source and route results.

CHANGES:

- `paper/equipment_anomaly_detection/artifacts_paper.py`
  - Added `Task4Task5SuccessAuthority`, `Task4Task5SuccessRecord`, and typed `write_task4_task5_success_record` / `read_task4_task5_success_record` interfaces.
  - Records are canonical JSON only at `stage_artifacts_paper/proposal_work/task4_task5_success/<sha256(dataset)>_antigravity/task4_task5_success_antigravity.json`; write uses existing `O_EXCL` create-only semantics.
  - Envelope binds schema, context/catalog/config/closure SHA256s, complete raw catalog identity, verified backend record, dataset, typed payload, and payload SHA256.
  - Payload retains every source's allowlisted immutable lineage origin, TRAIN OOF score, TEST score, TEST p-value, threshold, native pass, eligibility, and reason, plus every `HardRouteResult` field. Reader creates only allowlisted `LineageNode`, `HardSourceResult`, `RouteLanes`, `SourceExpansion`, and `HardRouteResult` values; reconstructed arrays are read-only and route sets are frozensets.
  - Writer performs a typed encode/decode/re-encode canonical-byte equality check before the create-only write.
- `paper/equipment_anomaly_detection/tests_paper/test_artifacts_runner_paper.py`
  - Added RED→GREEN tests for eligible and abstained source records, all route stages, continuous arrays, immutable reconstruction, authority/raw/backend/config/payload tampering, unknown files, symlink attack, create-only collision, and byte-identical Task8 shard projection after reread.
- `paper/equipment_anomaly_detection/codemap_paper.lock`
  - Refreshed the strict closure fingerprints for the changed artifact and test modules so runner closure verification remains fail-closed.

VERIFIED:

- RED: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper.Task4Task5SuccessRecordTests -v` initially failed all three tests with missing `read_task4_task5_success_record` / `write_task4_task5_success_record` APIs.
- GREEN: the same focused command passed `Ran 3 tests ... OK` after implementation.
- `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -v` passed `Ran 24 tests ... OK`.
- A combined quiet hard-source/route/config command was terminated by the parent after exceeding the Task 2 cost budget because it included a capable long test; its result is inconclusive and is not claimed as passing. Task 2 completion evidence relies on the focused artifact suite and strict validators below.
- `validate_codemap_triplet`, `validate_lineage_lock`, and `validate_source_provenance_paper` printed `paper-codemap-lineage-ok`.
- `python3 -m json.tool` validated both paper codemap JSON files; `git diff --check` exited 0.
- Process/root audit found no proposal process (only the pre-existing r3 log `tail`), confirmed `paper_proposal_full_r4_antigravity_20260813` absent, and only observed the r3 root without writes.

JUDGMENT CALLS:

- Kept Task8 final shard schema unchanged. The work record is intentionally a separate Task4+5 envelope because Task8 shards omit TRAIN OOF scores and source origin/threshold fields.
- Used a SHA256 dataset directory rather than a dataset spelling in the filesystem path, making path traversal impossible even if a future dataset name contains path characters. The mandated work namespace and an `antigravity` record filename are preserved.
- Did not alter `proposal_score_paper.py` or executor/retry behavior: Task 3 is responsible for passing parent context/catalog/backend and publishing/reusing records.
- Deferred the Exp448 governance-triplet semantic refresh to the dedicated Task 5; this task refreshed the paper closure lock that current strict tests require.

GAPS: no Task 2 implementation gap. The broad capable hard-source/route command is intentionally inconclusive; it is out of scope for this serialization-only task.

## Fix round 1/5

CHANGES:

- `Task4Task5SuccessAuthority` is now constructible only through `from_verified_context`. The factory recomputes the runner context SHA, binds the exact catalog SHA and canonical code-closure SHA, verifies the acceleration manifest against that context, derives the frozen Task4/Task5 source/route configuration SHA, and derives the backend from the verified assessment.
- `_collect_task4_task5` now accepts an optional parent-only `publish_success` callback. `task4_task5_success_publisher` closes over root/catalog/derived authority in the parent; workers still receive only `RawDataset`.
- Eligible records require float64 TRAIN OOF/TEST/p-value arrays, exact `upper_tail_pvalues(TRAIN OOF, TEST)`, and the frozen native-pass relation. Task8 projection now represents an abstained source's unavailable TEST score as its existing empty JSON array while preserving the unchanged shard schema.
- The work-record inventory rejects unrecognized success-token directories or ambiguous files. Both paper and Exp448 codemap triplets describe the authority/publication flow.

VERIFIED:

- RED: authority factory/publication/dtype/semantic-projection tests failed before implementation (missing factory/direct authority construction and absent float64/semantic checks).
- GREEN: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -v` passed 27 tests after the fix.

LIMITATION:

- A record whose TEST score, p-values, native-pass flags, and payload hash are all forged consistently can satisfy serialization-local checks. The cache is authenticated computation output, not an independent model recomputation oracle; the context-bound, create-only parent publication after worker return is the trust anchor. Rerunning the frozen source models is required to independently disprove such a fully self-consistent forgery.

## Fix round 2/5

CHANGES:

- Production `runner_paper._write_complete` now passes the verified root, context, catalog, and acceleration manifest into `compose_proposal_score`; composition derives one authority and always supplies the parent-only publisher. In-memory composition remains available only when those production inputs are omitted.
- Empty abstained TRAIN/TEST arrays must be float64 before canonical serialization and reconstruct as float64.
- Task8 final shard schema is now `paper-task8-proposal-shard.v3`: eligible source scores remain finite TEST-length arrays; ineligible sources carry their honest empty TEST-score array. The v3 reader rejects old schema or malformed shapes.
- The byte-equivalence fixture now combines an abstained source, nonempty Task5 route stages, consistent Task6/7 policy evidence, full v3 freeze/verify, and manifest-byte equality.

VERIFIED:

- Parent independently ran `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -q`: `Ran 28 tests ... OK` in 1.753 seconds.
- `git diff --check` exited 0 immediately before the final commit.

JUDGMENT CALL:

- The synchronized codemap triplets remain structurally current; semantic finalization of their descriptions is deferred to the dedicated Task 5 ownership, per parent direction.

## Fix round 3/5

CHANGES:

- `verify_journal` now accepts `proposal_work` only after re-deriving the current Task4+5 authority and verifying the complete present success-record inventory against current catalog/raw/backend evidence. Empty or absent work is valid; it never completes proposal scoring.
- Added a bounded recursive-journal regression covering a published record and rejecting an extra work file.

VERIFIED:

- `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper -q`: `Ran 29 tests in 1.619s` — `OK`.
