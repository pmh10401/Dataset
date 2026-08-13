# Task 4 report — Exp448 predecessor and source firewall

## TDD evidence

- RED: the first six focused predecessor/source tests failed because both Task 4 modules were absent. The map test then failed with missing `exp448_immutable_predecessor`.
- GREEN: focused Task 4 tests pass 6/6; the Task 4 map topology test passes after adding source-backed nodes and edges.
- Full Exp451 suite: 45 tests pass. Required Exp449 predecessor/source regressions: 26 tests pass.

## Actual read-only evidence

- Exp448 hashes: artifact map `ea9856c7de039697ed4f352864def0fde1c0d336714e68c9b0cd586d2c796f05`, evaluation `a58a74f69dcc13a64efb2a679ee7e93eca40ac8c5db87b29b12ed11ea068b21c`, decision `54ac58d89367ec5396dafdb180777f787384ffbd21ead1a96764bfe627bbddd5`.
- Authority: population 1,117; 48 unique ordered targets; stable-26 confusion `(1930,314,1988,191668)`; family FN `(246,200,88,75,63,56,56,53)`; TEST rows 72,300; label access 0.
- Database: SHA256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`; inode `1048138`; nlink `1`; size `1,753,284,608`.
- One values-only sample: `ChlorineConcentration_normal_1`, normalized TRAIN `(800,166)`, TEST `(1000,166)`. No label SQL or public label wrapper was invoked.

## Boundary and attack coverage

- Descriptor-pinned Exp448 verification is inherited read-only from the frozen Exp449 utility and converted into an equal-but-distinct frozen Exp451 authority. Exp449 regressions cover root/leaf symlink, hardlink, inventory, canonical bytes, digest, schema, confusion, ordering, replacement/drift and descriptor cleanup attacks.
- Exp451 focused rows reject bool/gapped/reordered indices, malformed float32 bytes, nonfinite values, zero variance and unequal lengths. Production SQLite uses separate values-only and label-only authorizers; row-wise z-normalization preserves the approved LocalGap input contract and never imputes.
- The private synthetic label seam validates exact authority type/canonical round trip/order/count before opening, queries exactly 48 datasets and 72,300 contiguous exact-text binary labels, freezes returned vectors, consumes the guard terminally, and preserves primary plus cleanup failures.
- Synthetic tests patch `PRODUCTION_LABEL_CONNECTION_FACTORY_PATH`, prove it is never called, and never invoke `open_ordered_label_session_antigravity` or any official database label path.

## Commands and resource checks

- Fixed interpreter `python -m py_compile` passed for both Task 4 modules and the map builder.
- `python -m unittest discover -s experiments/exp451_rdst_local_gap_qualification_antigravity/tests -p 'test_*_antigravity.py'`: 45 passed.
- Exp449 predecessor/source unittest modules: 26 passed.
- Production database descriptor/content is checked before and after each production phase and resources close once. Actual nlink remains 1.
- All four official Exp451 paths remain absent and `RESEARCH_STATE.json` remains label access/session 0.

## Code map and commits

- Functional commit message: `feat(exp451): pin predecessor and source boundaries`.
- The clean map-only child contains only the Exp451 HTML/JSON/lock triplet, with a functional-parent source commit, `working_tree_dirty=false`, source-backed evidence, HTML/JSON parity, and no unknown relationships.

## Concerns

- None. Exp451 remains planned; no official lifecycle or actual TEST-label read occurred.

## Review fix round 1

- RED proved the original production validator could not dynamically delegate to the absent future Task 5 artifact validator, did not return a complete authenticated authority, and skipped post-read database verification on a query failure. GREEN now delegates dynamically and fails before opener when Task 5 is absent or rejects the artifact.
- `ValidatedPredictionAuthorityAntigravity` binds reviewed commit, canonical SHA256, ordered `(order,family,name)` identities, exact per-dataset lengths and candidate indices. The private seam requires equality with the exact fixture authority; production independently requires the exact Exp448 48 identities/order and frozen TEST lengths `(1000,1000,2350,...,1550,1800)` totaling 72,300.
- Direct Task 4 tests cover production-opener count 0, reviewed-commit/hash/family/length/candidate attacks, label/schema/function/write/pragma/attach authorizer denial, DB symlink/hardlink/inode/SHA attacks, value postverification after read failure, label under/over/bool/gap/non-text attacks, iterator/query plus close failure distinction, terminal guard, and single close attempt.
- Values production now runs postverification exactly once after every attempted read even when SQL/parsing/normalization/bundle construction fails. Combined primary and cleanup/postverification errors use an inspectable `ExceptionGroup`; connection and descriptor cleanup remain one-attempt.
- Label close is marked attempted before invocation, preventing a second close after a raising close. Query/iterator and close errors remain separate members of an `ExceptionGroup` and the guard remains terminal.
- The speculative firewall-to-scoring edge was removed. The map now records the concrete `sources_antigravity.py` import of Exp451 contracts.
