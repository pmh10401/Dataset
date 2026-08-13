# Exp448 구현 보고

```text
approved_design_sha256 = ed2b6e5e8b8ee63ab41a15e54f679147f7e14aeca601f4d62f694ca104ed3ce4
approved_plan_sha256 = b79b3b7fc43aae7b15e3c7c52e648dd4404d50930f5bd71ce106c244d21fe934
```

```text
implementation_status = implementation_verified_awaiting_official_readiness
official_readiness = false
official_evaluate = false
official_inspect = false
test_labels_opened = false
official_label_access_count = 0
evaluation_sha256 = null
champion_sha256 = null
promotion_sha256 = null
```

Exp448은 Exp447의 immutable TEXT-label boundary failure를 수리하기 위해 승인된
engineering-only controlled fork다. Task2--Task5 implementation과 synthetic differential
proof는 검증되었지만, official lifecycle은 아직 전혀 실행하지 않았다. 다음 행동은 clean
candidate에 대한 fresh Sol High read-only `ship` verdict이며, 그 verdict 전에는 readiness,
evaluate, inspect 또는 root allocation이 허용되지 않는다.

## Task5 closed command evidence

- Functional CLI commit: `cb46a6bcc233d0cfefe226e1c3054dfaca5081a8`
  (`feat(exp448): add closed evaluation CLI`). The public surface is only
  `readiness --reviewed-commit LOWER40HEX`, `evaluate --reviewed-commit LOWER40HEX`,
  and bare `inspect`; parser abbreviation and lifecycle/override commands are rejected.
- The focused CLI suite passed `6/6`. It exercises exact commit parsing, forbidden override
  matrix, clean-HEAD-before-evaluator ordering, source-free fresh-process inspect, explicit
  JSON serialization, and the outside-CWD launcher’s seven `=1` numerical-thread pins.
- Full Exp448 discovery completed twice: `62` tests in `81.123s` and `62` tests in
  `81.402s`; both are `OK`. The first captured log has a post-observation `exit=0` sentinel
  because the harness detached its wrapper before its trailing write. The second recovery run
  retained its unified session and exact `exit=0` sentinel. This is engineering verification
  only, never an official command.
- Sequential inherited discovery was `66` Exp447 tests in `76.974s`, `50` Exp446 tests in
  `9.456s`, `39` Exp445 tests in `1.241s`, and `206` Exp444 tests in `316.121s`, all `OK`.
  The exact Exp447 `7e85d5c...` package-and-launcher byte guard and `git diff --check` both
  exited zero.

## Read-only authority facts

- Python/NumPy are `3.14.6` / `2.4.6`; design, plan, preregistration, and runbook SHA256 values
  remain respectively `ed2b6e5e...3ce4`, `b79b3b7f...fe934`, `af2b8b10...c612f`, and
  `3a6fa326...c595`.
- The DB SHA256 is `8458e678...ef3c`, inode `1048138`, nlink `1`. The immutable Exp447 audit
  SHA256 is `9f8f7d3d...2b97`, inode `19812443`, nlink `1`.
- Descriptor-pinned Exp446 identity remains `1117` datasets with manifest
  `bbaf9506...c11` and prediction `602ed1b...6100`; Exp404 remains `2218` Hard coordinates.
  Exp447 and Exp448 official roots are absent/non-symlink.
- No real TEST label value was queried, no official command was invoked, no dashboard/network
  module was imported by the no-label command paths, and no root/result/audit was created.
  Exp447 audit truth and champion `pending_candidate` remain unchanged.

## Paper r4 operations implementation

The approved paper-r4 operational launch plan is implemented in this worktree
only. It does not authorize smoke-root or official r4-root creation.

- Public paper CLI now prints canonical JSON for `run`, `status`, and `verify`.
  `completed` exits 0, `proposal_score_incomplete` exits 2, and semantic
  exceptions still propagate instead of being converted to success.
- `scratch/manage_paper_r4_proposal_antigravity.py` exposes `check`, `smoke`,
  `launch`, `status`, `stop`, `resume`, and internal `_worker`. Official paths,
  worker policy (`logical_cpu_count - 1`, inner threads = 1), and scoring
  closure hashes are immutable. `check` is read-only.
- The production interpreter
  `/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python`
  is accepted only as that exact lexical venv symlink. `check` requires
  `sys.executable` to equal that path, `sys.prefix` to equal the venv root, a
  regular resolved executable, and a hash-bound python identity. Authority
  records `executable`, `executable_realpath`, `python_prefix`,
  `python_base_prefix`, and `python_identity_sha256`. Worker and pipeline
  commands invoke the pinned venv path and never resolve it to base Python.
  The exp443 venv is not mutated. The failed 20260813 smoke/ops pair remains
  immutable evidence; official smoke is the v2 path and must stay absent until
  a later authorized smoke. Official r4 remains unauthorized.
- `_command_is_this_worker` matches only the exact 13-token full-worker argv
  grammar, including the stored authority SHA. `stop` rereads OS argv before
  SIGTERM and returns `stop_refused` if identity changed.
- `_publish_smoke_result` is create-only. An existing result is reused only
  when its bytes are identical to the newly verified encoding; otherwise it is
  rejected. Existing smoke evidence is never atomic-replaced.
- Smoke authorization is a two-invocation gate. The first completed wave may
  compute and semantically verify artifacts, but it only writes a create-only
  preauthorization marker and returns `smoke_confirmation_required`. A second
  same-authority invocation verifies the pre-existing completed root without
  recomputing Task4+5 science and only then publishes `smoke_result`. An
  incomplete first wave stays non-authorizing; the later resume may compute
  pending work, and a completed resume still requires one additional
  verification-only invocation. The marker binds current authority, smoke-root
  semantic hashes, and completed status.
- Authority records a frozen scientific distribution inventory for every
  runtime dependency imported by Tasks 1–9 and score sources (numpy, scipy,
  scikit-learn, aeon, pandas, numba, joblib, scikit-image, pycatch22, torch,
  mlx, mlx-metal) as exact version or `absent`, plus a canonical inventory
  SHA included in python identity. check/launch/resume/stop/status rebuild and
  compare that inventory.
- Create-only publication writes an exclusive no-follow temp, fsyncs, installs
  with a no-clobber hard link, fsyncs the directory, and deletes the temp.
  `_publish_authority` recovers an exact one-file partial pair and rejects
  mismatched or unsafe partials.
- `_worker` authenticates the inherited lock fd (regular-file `fstat`,
  canonical non-symlink `lstat`, `st_dev`/`st_ino` equality, and
  `LOCK_EX|LOCK_NB` acquire/confirm) and requires a pre-published exact
  authority rather than creating one. Until both canonical flock ownership and
  the exact stored authority are validated, `_worker` does not mutate state,
  pid, progress log, or events; pre-auth diagnostics go only to stderr and a
  rejected inherited fd is closed. After authentication, ordinary failure
  state/log behavior remains. Resume accepts only an empty r4 root or a root
  whose only child is an empty canonical journal directory, then lets the
  runner initialize; every other partial or polluted shape is rejected.
- CLI `smoke` exits 0 only for top-level `smoke_verified`.
  `smoke_confirmation_required` and `smoke_incomplete` are nonzero.
- Paper `write_create_only` remains a complete write loop, fsync, same-directory
  exclusive no-follow temp, atomic no-clobber install, and directory fsync.
  Journals, stages, shards, and attempts are strict create-only and raise
  `FileExistsError` even when existing bytes match. Only
  `write_task4_task5_success_record` passes `reuse_identical=True`. That path
  recovers one matching same-directory `.tmp_antigravity` leftover after a
  successful `os.link` before temp unlink, then fsyncs the directory; symlink,
  nonregular, mismatched, and ambiguous temps fail closed. Scientific payloads
  and schemas are unchanged.
- Ops confinement is shared: `check` rejects a symlinked smoke-ops root;
  every existing ops/smoke-ops path component must be a real directory; every
  present authority/hash/PID/state/progress/lock/result/marker artifact must
  be a regular non-symlink file. `_acquire_lock` uses no-follow exclusive
  regular-file validation.
- Focused operations tests cover authority/check, a pinned-venv check seam,
  exclusive flock, stale PID recovery, exact argv-grammar stop identity,
  SIGTERM revalidation, the two-invocation smoke gate, inventory mutation,
  atomic no-clobber publication, ops/smoke-ops symlink confinement, status
  fixtures, inherited lock-fd authentication, pre-auth non-mutation of seeded
  sentinel state/log/events, rejected-fd close, pre-genesis resume windows,
  nonzero CLI smoke statuses, Task4+5 post-link leftover recovery, strict
  journal/stage/shard/attempt create-only, and a small typed Task4+Task5 smoke
  resume-equivalence fixture. They write only temporary Git/DB/path fixtures.
- Implementation verification does not create
  `/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813`,
  the smoke root, or the ops root, and it starts no official or detached
  experiment process.
