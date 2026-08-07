# Task 5 MPS parity smoke preregistration — 2026-08-04

This is a bounded engineering parity smoke, not an Exp425 official run,
population selection, readiness decision, or TEST-informed cycle.

## Frozen scope

- Fixture: `ECG200_normal_1`; exactly one normal TRAIN value load.
- TEST values and labels, anomaly/label/evaluation/readiness calls, official
  population calls, and `run_selective_population` are forbidden. Fixed TEST
  length metadata used by the descriptor-safe Task 4 context is not a value load.
- Native call: actual Exp425 private `_compute_native`, after a Task 4
  descriptor-safe `Exp425DatabaseSnapshot` freezes the existing full lane context.
- Device: `torch.device("mps")`, with built and available checks; CPU fallback
  is forbidden. Batch=32, query chunk=256, model seeds=41701/41702/41703,
  perturbation seeds=41711/41712/41713, alpha=0.01, max error=0.0001.
- Explicit `torch.mps.synchronize()` calls: zero. Timing begins immediately
  before `_compute_native` and ends after its return, after production CPU/
  NumPy/serialization/calibration implicit synchronization boundaries.

## Immutable reference gates

- DB `/Users/mac/Documents/Dataset/univariate_ts.db`, SHA256
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`.
- Golden record SHA256 `dc4babf53ab20653322fe0d4f931d20ef68c8c28979244fb40b1367f5c5f0bb7`;
  payload SHA256 `61140e63d63886ac59968c83959dbbed0afbdf42bf2b1a5abe434712480987c4`.
- Source root/progress SHA256 `f7526bf638f4c1051a5970271bdbbccd753986f7d067ec1260817da884c1b757` /
  `5e9aa4a4b7240ac89c078bdc1dce4b41bda6b6dbc505dd9ad49782556da7bd84`,
  source commit `4a398374d880f859f285a271ad63de4a5e55b028`.
- Current byte-identical runner/model/scoring SHA256
  `f784fb77805e353b7652582626672686f52a2fdc06d27f9c1ca58730d27b772c`,
  `422e096613be84ba77350b068e6bd837eebbb46169bb6e3e275f3f03ab2fcedd`,
  `06a92422ea7909efbb7354f1f0499329bd19c04642bbbd5a136fa3f2f5a39941`.
- Raw/repaired/length identities: `d2af39881663b5d1e736bf26847d36a280309989b5e8891944d71537e756cfb1`,
  `7cb08f3fb219dcb11b7472ac839c984d9351fcaef938a24a89ea4d861212ed35`,
  `53595555508db40912a4fcfe3063a358127f52ef6c3b98a3e4bd2f5623fb51be`.
  One length-96 group with 106 TRAIN series is required.
- Final model hashes: `1db68f931ef0c6fc98cccc2ea46b95ba94684641eb6a1d6846ad9a0162f61f37`,
  `36b9bdd6a4ffd854837642889e6e8ea30d2e3daa20b8ecd2b6b58f74cb83c019`,
  `e6bb3e8a51722c4719035cc5043dafe8080c5a53e5683375ac5dc457ee66d9b2`.
  Legacy OOF digest: `5eaa66a03e1fb60ea4163b2bb89fb57bcb8ab8ac9d23e9b3a768abe033310357`.

## Comparison and limitation

The normalized digest contains only dataset; DB/raw/repaired/length identities;
config; counts/lengths; final hashes; legacy OOF digest; qualification;
status/candidates/finiteness/fallback. It excludes only wrappers, paths,
timing, process/run/trace data, commits, backend-specific train identity and
frozen-result digests, and full backend audit. Qualification floats compare at
0.0001. Raw audits remain outside it. Exp424 retained no per-synthetic-query
scores, so direct historical per-query deltas are unavailable.

The review-only authenticated audit (not newly measured) is mps, error
`4.0948700075205124e-07`, Spearman `0.9999999999845686`, speedup
`0.6709759390840965`, identity true, fallback false. It is below 1.1x, hence
does not adopt acceleration. MLX is not adopted absent exact parity and >=1.1x.

## Correction and fail-closed architecture gate

The earlier 61-character OOF value in the delegated packet was corrected by the
task owner before this preregistration; the authenticated 64-hex value above is
governing.

`BackendAudit` rejects MPS with the permitted review-only 0.6709759390840965
speedup, while `Exp425RuntimeSession` requires that MPS type. The harness must
fail closed unless an authorized review-only adapter is supplied that preserves
the frozen numerical path and does not misstate speed. No MPS execution occurs
until that contradiction is resolved.
