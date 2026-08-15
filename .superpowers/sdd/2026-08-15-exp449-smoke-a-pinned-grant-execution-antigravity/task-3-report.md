# Exp449 Smoke A Task 3 Report

Base: `67c756474975122646885329e10ab27cc97793a5`.

Implemented the requested one-shot mutation boundary without creating authority
JSON, a fixed production root, a production claim, a database score, or label
access. The authority module now contains a private non-copyable,
non-picklable process-local capability. It binds the verified grant SHA, A/I,
context SHA, target, scientific and operations roots, owner PID, claim device
and inode, locked descriptor, and consumption-receipt digest.

Operations now owns the fresh Smoke A lease: readonly readiness is completed
before mutation; scientific root, operations root, and claim components are
`lstat` checked; the claim is created with `O_EXCL|O_NOFOLLOW`, fsynced and
nonblocking exclusively locked; its FD/path inode is checked; only then is the
operations root created and its create-only consumption receipt written. The
configured-root wave, terminal-incomplete publication, and authorized freeze
are capability-gated. Fixture-only pure composition remains ungated.

Evidence:

- RED: the new claim suite initially reported absent capability/lease symbols
  and a direct configured-root call reached the old recovery validation rather
  than a capability check.
- GREEN: 52 focused tests passed across claim, recovery, authority, and config
  contracts; `py_compile` passed for both modified modules.
- Codemap embedded JSON, graph references, and lock fingerprints validated
  (`54` nodes, `83` edges, `6` flows).
- The nine configured scientific/operations roots, claim, and authority paths
  were `lstat`-absent after tests.
- Static bypass search found no boolean execution authorization path; the
  focused attack tests cover copying, pickle, forked-PID reuse, receipt
  tampering, existing/symlinked paths, second acquisition, and direct
  unauthenticated configured-root mutation.

Judgment: the capability primitive belongs in the authority module while the
lease belongs in operations, so the authority verifier stays free of an
operations import cycle and operations remains the owner of all mutation.

## Fix round 1/5

RED: a direct `flock(LOCK_EX|LOCK_NB)` validation silently reacquired a
released claim, and an unregistered `object.__new__` capability reached mutable
slot lookup. GREEN replaces capability slots with an opaque identity-keyed
closure registry of frozen facts. A child-process contention probe observes the
claim lock without modifying the owner descriptor; it rejects a released,
closed, reopened, or replaced claim path instead of reacquiring it. The registry
also rejects direct construction and copied visible fields because no issued
object identity has matching private facts.

Added minimal regressions for direct construction, released/closed/reopened FD,
claim inode replacement, distinct context, traversal, and a two-thread race.
`python3 -m unittest ...test_execution_claim_antigravity -v` passed 9 tests;
the full focused Exp449 verification is rerun after this report/map update.

## Final synchronization evidence

After the wrapper-symbol and opaque-FD-accessor correction, the complete
Task 3 focused command ran the claim, recovery, authority, and config modules:
`python3 -m unittest experiments.exp449_reproducible_rebaseline_antigravity.tests.test_execution_claim_antigravity experiments.exp449_reproducible_rebaseline_antigravity.tests.test_runner_recovery_antigravity experiments.exp449_reproducible_rebaseline_antigravity.tests.test_execution_authority_antigravity experiments.exp449_reproducible_rebaseline_antigravity.tests.test_config_contracts_antigravity -v` — `56` tests passed in `9.802s`.
`python3 -m py_compile` passed for both changed production modules. The final
triplet regeneration, embedded-payload/graph/fingerprint validation,
`git diff --cached --check`, and lstat fixed-path-absence check are recorded
from the final staged state below.

Final staged-state output: map validation reported `nodes=54 edges=83 flows=6
fingerprints=10`; fixed-path validation reported `fixed-path-absence-ok
count=9`; `git diff --cached --check` was silent; and the final full focused
command passed all `56` tests in `8.371s`. No authority file, configured root,
claim, database, scoring output, or label artifact was created.

The six-module complete Exp449 run was also attempted after the commit. It ran
`116` tests in `18.761s`; `115` passed and one existing readiness preflight
test failed because this worker process is `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14`,
not the frozen capable interpreter required by
`_require_capable_interpreter_antigravity`. The failing assertion is the
environment-sensitive first call in
`test_literal_pinned_interpreter_spelling_rejects_aliases_and_base_python`.
No Task 3 behavior or readiness source was changed to mask that intended gate.

The same six-module command was then run with the frozen capable interpreter.
Its first pass reached `115/116` and exposed a stale lock metadata delta:
`changed_modules_since_prior_lock` had lost `experiments` during a report-only
regeneration. The final regeneration retains the Task 3 `experiments` delta
and the suite is rerun from that corrected triplet state.

With that metadata corrected, the full frozen-interpreter aggregate still ran
`116` tests in `18.789s` but had three readiness-route failures only when the
manifest-verifier and readiness modules share the aggregate process:
`test_authorized_readiness_integrates_real_I_fact_collection_and_grant_comparison`,
`test_authorized_readiness_route_freshly_reverifies_and_recomputes_implementation`,
and `test_authorized_readiness_route_rejects_wrong_type_and_changed_reverification`.
These are outside Task 3 ownership and exercise the pre-existing readonly
authorization-readiness route; no Task 3 source was changed in response.

Root cause and resolution: the concurrent-claim test had two overlapping
thread-local `patch.object` lifetimes for the same module-global readiness
function. Their restore race could leave the `{\"fixture\": true}` mock installed
for later readiness tests. The test now installs one enclosing readiness seam
around the executor and asserts the original function identity is restored.
The exact pinned six-module aggregate command then passed `116` tests in
`18.384s`.

## Fix round 2/5

The Darwin PID-record-lock exploration was removed: Darwin makes that record
lock conflict with the existing exclusive `flock` on the same claim inode.
The settled process-local threat model therefore retains the original
non-reacquiring child flock probe and explicitly excludes hostile same-account
third-party lock substitution. The probe now verifies the exact expected claim
device/inode in the child and counts only `EACCES`/`EAGAIN` (or identical
`EWOULDBLOCK`) as nonblocking contention; open, fstat, flock, wait, abnormal
exit, `EINTR`, and `EIO` fail closed. Temporary-path regressions cover released
or closed/reopened FD, inode replacement, cross-PID rejection, one concurrent
owner, and unexpected contention-probe errno. Claim suite: `10` tests passed.

## Fix round 3/5

The contention probe now splits setup from lock acquisition: open/fstat/lstat
and exact device/inode validation execute outside the only errno classifier.
Only the actual nonblocking `flock` treats `EACCES`/`EAGAIN` (or an identical
`EWOULDBLOCK`) as contention. Regressions prove fstat `EACCES` and lstat
`EAGAIN` reject while flock `EAGAIN` passes; the claim suite is `11` passing
tests and the pinned complete Exp449 suite is `118` passing tests in `18.526s`.
