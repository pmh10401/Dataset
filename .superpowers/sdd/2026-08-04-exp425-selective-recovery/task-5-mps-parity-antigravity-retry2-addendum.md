# Task 5 sparse-lane preflight retry2 addendum — 2026-08-05

The first authorized retry root was
`/Users/mac/Documents/Dataset/exp425_task5_mps_parity_20260804_architecture_a_retry`.
Its PID was 79638. It failed closed during `Exp425DatabaseSnapshot.freeze_context`
with `frozen lane coordinates do not exactly cover TEST identities`.

This was an engineering preflight failure before TRAIN loading, TEST-value loading,
TEST-label loading, MPS model construction, candidate generation, checkpoint creation,
or prediction. The preflight opened the immutable database snapshot and queried only
the already-authorized dataset and TEST identity metadata needed to freeze the run
context. It produced no adaptive/model result and therefore does not allocate a new
research experiment number.

The preserved retry failure artifacts are:

- `progress.json` SHA256
  `053a1553a8facaa75a8033c952968e7f905bec379ae55dab1bedf094e2cd17f5`;
- `events.jsonl` SHA256
  `f6f42551970d255f0733191346320a1a494b4b209453df9f7355011db36facbc`;
- `parent.log` SHA256
  `1212d8b6f6ea3f6b47a11d2dec15409d7bad740dbb6343014a31fcca5be48d7d`;
- `run.pid` SHA256
  `8b2aa31b7f96b2c561ca99e3994a94bda4b1b4385ede74b19aa59911cfd14770`.

They remain immutable. The stale-looking `run.pid` is preserved as evidence and is
not treated as a live lease because PID 79638 has exited.

## Root cause and frozen correction

The Exp291/292 lane artifacts are sparse routing exclusions, not a partition of the
entire TEST population. Across the frozen production inputs they contain 3,090 routed
coordinates while the database contains 195,900 TEST identities. Every one of the
1,117 datasets was rejected by the erroneous equality check; there were 192,810
unrouted TEST identities and zero lane coordinates outside the database identities.

The corrected Task 4 integrity rule is frozen as:

1. the lane population must remain exactly the selected 1,117 datasets;
2. `FrozenLaneSnapshot` continues to enforce sorted unique non-overlapping lanes and
   the exact Hard/Standard/Priority/automatic-normal totals and artifact hashes;
3. every routed lane coordinate must be a member of that dataset's descriptor-bound
   database TEST identity set;
4. sparse lane coverage is valid and must not equal the full TEST identity set;
5. the full TEST identity manifest and per-dataset TEST sizes continue to come only
   from the immutable database snapshot;
6. candidate coordinates remain forbidden from every frozen lane.

A production read-only context smoke after the correction must report exactly 1,117
datasets, 195,900 TEST identities, 3,090 lane coordinates, one full DB hash, and clean
owned-link cleanup. No TEST values or labels may be opened by that smoke.

## Authorized retry2 root

After TDD, full Exp425 verification, clean Git, immutable failure-hash checks, and a
fresh Sol/High `ship` review, exactly one invocation may use:

`/Users/mac/Documents/Dataset/exp425_task5_mps_parity_20260804_architecture_a_retry2`

The old failed root, the first retry root, and every arbitrary sibling are forbidden.
This addendum authorizes no readiness command, official 1,117-dataset execution,
TEST-value evaluation, TEST-label access, or post-hoc metric computation.
