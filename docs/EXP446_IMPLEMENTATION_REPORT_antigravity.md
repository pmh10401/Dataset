# Exp446 implementation verification report

## Status and authority

Exp446 is `implementation_verified_awaiting_official_readiness`. This is an
engineering-only verification of the approved recursive JSON materialization repair; it
does not authorize official readiness, `new`, `inspect`, publication, evaluation, TEST-label
access, promotion, or a scientific claim. All official prediction/publication/evaluation/champion/
promotion hashes remain `null`, label access is `0`, and promotion is `false`.

The approved design SHA256 is
`3fce139171efbfc0a85fffeb44ea201116053efbe76e5a190e7198a3bdf3d281`; the preregistration
SHA256 is `440d4e735a2cdd4447c35a44a3431fce09788c7181c8bd35c6e3f2a5262d5838`.
Exp445 remains the immutable failed-publication predecessor. It was authenticated read-only
with its 30 record wrappers and aggregate
`af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`.

## Bounded deterministic parity

The first bounded harness attempt failed closed while checking the actual frozen wrapper schema:
the records are `{sequence,result}`, not `{result}`. That captured failure is retained in the
Task 7 evidence. The corrected bounded run did not recompute scores. It strict-deserialized the
same 30 immutable record wrappers and sent their result values through the production
`sequential_reference_antigravity` and `ordered_parallel_map_antigravity` paths.

- CPU: Apple M2 Max; logical CPUs: `12`; workers: `11`; inner threads: `1`.
- label access: `0`; scientific score recomputation: `false`.
- sequential: `0.012196` seconds; 11-worker: `0.302225` seconds.
- all 30 canonical wrapper bytes and SHA256 values matched Exp445 exactly.
- record-prefix aggregate: `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`.
- serialized result SHA256: `729b31c0698fda6b60c17012a3fd2cae3f4b0899e01e5a3bf4d569902e814857` on both paths.

## Regression and root-cause repair

The first combined discovery boundary found two stale test seams, not product regressions:

- Exp445's readiness matrix required the live official root to be absent even though the approved
  immutable failed-publication predecessor now exists. The test now snapshots and preserves that
  external root's identity while retaining its test-local absent-root checks.
- Exp446's lifecycle seam patched only its facade singleton. After the Task 6
  authorization-first change, reporting correctly rejected that copied profile before terminal
  capture. The test now patches the reporting singleton to the same isolated profile, then proves
  terminal-parity rejection without allocating its test root.

Both original failures were the RED evidence; the focused tests passed after test-only fixes.
The corrected regression boundary passed once:

- Exp444: `206` tests, `327.683` seconds, exit `0` (unchanged product/shared code; first boundary evidence reused).
- Exp445: `39` tests, `1.317` seconds, exit `0`.
- Exp446: `47` tests, `10.132` seconds, exit `0`.

Read-only checks also confirmed source DB inode `1048138` with `nlink=1`, a valid immutable
Exp445 predecessor, and an absent Exp446 root. Full command, log hashes, test timing, and commit
references are retained in the ignored Task 7 report.

## Remaining gate

Only a fresh Sol High read-only final review of the clean implementation candidate may permit
official readiness. This report does not claim that final review or a `ship` verdict.
