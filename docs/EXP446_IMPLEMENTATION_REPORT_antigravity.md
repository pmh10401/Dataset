# Exp446 implementation verification report

## Status and authority

Exp446 is `official_publication_inspected_complete_awaiting_discussion`.
The official CPU/11 runtime and create-only publication completed, but the original CLI wrapper
failed only after return while `dataclasses.asdict` deep-copied a nested `MappingProxyType` during
stdout serialization. The immutable attempt is neither retried nor mutated; source-free official
the authorized source-free inspection pair is complete. Prediction SHA256 is
`602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100`, the publication artifact-map
SHA256 is `d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d`, and manifest SHA256 is
`bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11`. Label access is `0` and
promotion is `false`.

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
- Exp446: `48` tests, `10.625` seconds, exit `0`.

Read-only checks confirmed source DB inode `1048138` with `nlink=1`, a valid immutable Exp445
predecessor, and the immutable completed Exp446 root. The official runtime used PID `74950` and
completed/success/abstention/error `30/30/4/0` in `471.76320550008677` seconds. The CLI repair
replaces only the post-publication dataclass deep-copy route with field-wise recursive serialization;
focused `new` and source-free `inspect` seams pass without touching official artifacts.
Before that mock was installed, one focused CLI test unintentionally performed an official inspect.
It was source-free, label-free, mutation-free captured-output evidence; it is not counted as either
pending authorized inspection, and no additional official inspect was run.

## Authorized inspection closeout

After authorization, the two official source-free inspections both exited `0` and produced
byte-identical output with SHA256
`937b46960ed0b611fc9d3c74bffa7b494ff306af6d78a92129217b0da691fdf0`.
They confirm the immutable publication's `1117` datasets, `30` candidates, `26` stable and `4`
unstable review rows, and label access `0`. The execution commit is
`c5278ba600a2d7ee259f5f5f987e1f9cf20c04cb`; the post-publication CLI serialization repair is
`2da7ab234f138589ac72b8b6179f4051f701bf04`; the reviewed inspection tool commit is
`827d7dedc00c678ca4309c928341fb6ee2946d53`. The original `new` wrapper error remains disclosed:
it occurred only after valid publication while serializing a nested `MappingProxyType`.

## Remaining gate

The authorized inspection pair is complete. Evaluation, champion selection, promotion, and a
`ship` verdict remain unstarted or false; the next state is discussion, with no automatic action.
