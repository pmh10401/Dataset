# Exp449 Smoke A Pinned-Grant Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task. Use `sol_advisor_terra_implementer` for implementation and a fresh `sol_advisor_sol_reviewer` at every review boundary.

**Goal:** Exp449의 과학 알고리즘과 scientific artifact bytes를 바꾸지 않고, 사용자 승인 1회용 digest grant로 Smoke A를 시작·중단 후 동일 root에서 재개·동결·검증할 수 있게 한다.

**Architecture:** 기존 Exp449 composition, Task4/5 checkpoint, Task6/7, Candidate C, typed shard 및 outer manifest 구현은 그대로 둔다. 새 authority 모듈은 Git authorization delta와 canonical grant를 검증하고 process-local capability를 발급하며, operations 모듈은 이 capability가 있을 때만 고정 Smoke A root를 변경한다. 모든 grant·claim·PID·진행·terminal 정보는 operations root에만 기록하고 scientific root schema는 변경하지 않는다.

**Tech Stack:** Python 3.14.6, stdlib (`argparse`, `dataclasses`, `fcntl`, `hashlib`, `json`, `os`, `pathlib`, `secrets`, `signal`, `subprocess`, `time`), NumPy, 기존 Exp449/paper pipeline API, `unittest`.

## Global Constraints

- 실제 Smoke A, Smoke B, official scoring, TEST label 접근은 이 구현 계획에서 실행하지 않는다.
- 구현 테스트는 임시 fixture root만 사용하며, 고정된 6개 production root와 claim path는 모든 테스트 전후 absent여야 한다.
- 새 과학 알고리즘, threshold, alpha, seed, preprocessing, score, route, Candidate C 규칙을 추가·수정하지 않는다.
- `ProposalScore`, typed shard, outer manifest scientific schema를 변경하지 않는다.
- 새 보안 범위는 승인 commit/digest, 한 번의 claim, 동시 실행 방지, direct mutation 방지에만 한정한다.
- CPU는 감지 logical CPU 12, dataset workers 11을 readiness/grant/consumption에 기록한다. 새로운 MLX/MPS/PyTorch 구현은 만들지 않고 기존 CPU canonical 결정을 유지한다.
- 모든 새 파일명에는 `antigravity`가 포함되어야 한다.
- 구현 시작 시 Exp449 codemap HTML/JSON/lock에서 대상 모듈의 callers/effects/tests를 확인한다. 불완전하면 세 파일을 먼저 함께 재생성한다.
- 구현 커밋 `I` 전까지 authority JSON을 생성하지 않는다. `I`는 grant가 없을 때 계속 `execution_authority_not_configured`여야 한다.
- authorization commit `A`는 `I`의 direct child이며 정확히 두 canonical authority JSON만 추가한다.
- 실제 실행은 `I`와 `A`의 독립 Sol SHIP, exact `A` SHA에 대한 사용자의 `Smoke A 실행 승인` 이후 별도 단계다.

---

### Task 1: Grant contracts, fixed paths, and codemap exclusion

**Files:**

- Create: `experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/contracts_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_execution_authority_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_config_contracts_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.html`
- Modify together: `docs/codemap/codemap_exp449_antigravity.json`
- Modify together: `docs/codemap/codemap_exp449_antigravity.lock`

**Step 1: Preflight the active map**

Resolve `RESEARCH_STATE.json` as Exp449. From the current map, record:

- callers of `config_antigravity.py`, `contracts_antigravity.py`, and `operations_antigravity.py`;
- effects on readiness, context hashes, proposal freeze, and the CLI;
- covering tests.

Regenerate all three map files before source edits if the new authority boundary cannot be represented. Add an explicit fingerprint exclusion for `docs/authorities/` as authorization data, with a structural assertion that this directory may contain only the two approved `.json` filenames and never Python/source files.

**Step 2: Write RED contract tests**

Add tests requiring these exact frozen values:

```python
SMOKE_A_GRANT_PATH_ANTIGRAVITY = REPOSITORY_ROOT / "docs/authorities/smoke_a_execution_grant_antigravity.json"
SMOKE_A_DIGEST_PATH_ANTIGRAVITY = REPOSITORY_ROOT / "docs/authorities/smoke_a_execution_grant_digest_antigravity.json"
SMOKE_A_CLAIM_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "paper_proposal_exp449_smoke_a_antigravity_20260814_claim_antigravity.lock"
)
```

Define frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class Exp449SmokeAGrantAntigravity: ...

@dataclass(frozen=True, slots=True)
class Exp449SmokeAGrantDigestAntigravity: ...
```

The tests must reject unknown/missing fields, wrong schema/experiment/target/policy/root, noncanonical 32-hex nonce, non-2 waves, label access other than `False`, wrong CPU count/worker count, invalid SHA, mutable nested mappings, and noncanonical dataset order.

Run:

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_execution_authority_antigravity \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_config_contracts_antigravity -v
```

Expected: RED because the new module/classes/constants do not exist.

**Step 3: Implement pure data contracts**

Implement only canonical parsing/serialization and frozen validation. Public pure functions:

```python
def canonical_grant_bytes_antigravity(grant: Exp449SmokeAGrantAntigravity) -> bytes: ...

def parse_grant_bytes_antigravity(data: bytes) -> Exp449SmokeAGrantAntigravity: ...

def canonical_grant_digest_bytes_antigravity(
    digest: Exp449SmokeAGrantDigestAntigravity,
) -> bytes: ...

def parse_grant_digest_bytes_antigravity(
    data: bytes,
) -> Exp449SmokeAGrantDigestAntigravity: ...
```

Use the existing canonical JSON rules (`sort_keys=True`, compact separators, UTF-8, `allow_nan=False`, trailing newline only if the existing authority convention explicitly requires it). Require byte-for-byte canonical reserialization on parse.

The grant must bind all approved design fields, including separate maps for complete scientific closure and explicit authority-code closure. Do not put authorization commit `A` in the grant.

**Step 4: Make codemap exclusion fail closed**

Extend the codemap semantic test to assert:

- `docs/authorities/` is excluded only from source fingerprinting;
- exactly the two approved JSON basenames are permitted;
- any `.py`, symlink, directory, or third file is rejected by the authorization verifier;
- the authority module itself remains inside the normal Exp449 code closure and codemap.

**Step 5: Run GREEN and commit**

Run the focused tests plus `py_compile`, JSON/HTML embedded payload equality, symbol/path validation, lock fingerprint recomputation, and `git diff --check`.

Commit:

```bash
git add experiments/exp449_reproducible_rebaseline_antigravity \
  docs/codemap/codemap_exp449_antigravity.html \
  docs/codemap/codemap_exp449_antigravity.json \
  docs/codemap/codemap_exp449_antigravity.lock
git commit -m "feat(exp449): define pinned Smoke A grant authority"
```

---

### Task 2: Verify authorization commit delta and immutable grant snapshots

**Files:**

- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_execution_authority_antigravity.py`
- Modify together: Exp449 codemap triplet

**Step 1: Write RED adversarial tests**

Create a temporary Git repository fixture with commits `I` and `A`. Require failure for:

- `HEAD != approved_authority_commit`;
- dirty worktree;
- `A` is not a direct child of `I`;
- extra, deleted, modified, renamed, mode-changed, symlinked, or untracked authority path;
- grant/digest read twice with a path swap between hash and parse;
- noncanonical bytes, wrong size/SHA/path/implementation commit;
- scientific or authority closure blob differs between `I` and `A`;
- lineage or codemap lock differs;
- approved SHA is not lowercase 40-hex.

Also prove a valid exact two-file direct-child delta returns one immutable verified object.

**Step 2: Implement a single-snapshot verifier**

Add:

```python
@dataclass(frozen=True, slots=True)
class VerifiedExp449SmokeAAuthorityAntigravity:
    approved_authority_commit: str
    implementation_commit: str
    grant: Exp449SmokeAGrantAntigravity
    grant_sha256: str
    grant_bytes: bytes
    digest_bytes: bytes
    grant_stat_identity: tuple[int, int, int, int]
    digest_stat_identity: tuple[int, int, int, int]

def verify_smoke_a_authorization_commit_antigravity(
    approved_authority_commit: str,
    *,
    repository_root: Path = REPOSITORY_ROOT_ANTIGRAVITY,
) -> VerifiedExp449SmokeAAuthorityAntigravity: ...
```

Open each authority file once with `O_NOFOLLOW`, read all bytes from that file descriptor, capture `fstat`, and parse/hash only retained bytes. Verify tracked regular-file mode and exact Git delta using non-interactive `git diff-tree`/`git ls-tree`; do not trust a working-tree path read to represent committed bytes.

`VerifiedExp449SmokeAAuthorityAntigravity` is read-only verification evidence only. It cannot authorize a mutation.

**Step 3: Bind readiness to implementation commit I narrowly**

Add a private exact-type token issued only after the delta verifier succeeds:

```python
class _AuthorizationDeltaReadCapabilityAntigravity: ...

def readiness_for_authorized_implementation_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
) -> dict[str, object]: ...
```

Refactor `operations_antigravity._require_reviewed_head_antigravity` minimally so the standard public readiness still requires `HEAD == reviewed_commit`, while the private path may recompute non-root readiness facts for `I` only when `HEAD == A`, `parent(A) == I`, and the exact verified capability is presented. Do not add a boolean bypass or caller-supplied callback.

Compare all recomputed grant facts: database path/SHA/size, smoke catalog identity and 23 datasets, context SHA, closures, locks, package inventory, constants, acceleration, 12→11 workers, CPU backend, roots, maximum waves, and label access false.

**Step 4: Run GREEN and commit**

Run focused authority/readiness tests in the pinned Python 3.14.6 interpreter, including a fresh subprocess that has not cached authority modules. Confirm all six roots and claim remain absent.

Commit:

```bash
git commit -am "fix(exp449): verify exact Smoke A authorization delta"
```

---

### Task 3: Process-local capability, claim lock, and operations-root consumption

**Files:**

- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_execution_claim_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py`
- Modify together: Exp449 codemap triplet

**Step 1: Write RED capability/claim tests**

Using only temporary fixed-path seams, require rejection of:

- direct `run_exp449_wave_antigravity(config)` without capability when the config points at an official destination;
- constructed/copied/pickled/serialized capability;
- capability used from a different PID, target, context, root, grant, commit, or claim inode;
- existing/symlinked/traversal claim or operations/scientific root;
- two concurrent claim attempts;
- consumption receipt mismatch or second grant reuse;
- unauthenticated failure path modifying state/logs.

Prove pure `compose_exp449_proposal_score_antigravity` remains callable in fixture tests without capability because it has no official publication authority.

**Step 2: Implement exact process-local capability**

Add a private non-copyable/non-picklable exact class:

```python
class _Exp449ExecutionCapabilityAntigravity:
    __slots__ = (...)

def _require_execution_capability_antigravity(
    capability: object,
    *, target: str, context: Exp449Context, claim_fd: int,
) -> None: ...
```

Bind grant SHA, `A`, `I`, context SHA, target, scientific root, operations root, owner PID, claim path, claim device/inode, and an unexported in-process sentinel. Reject `copy`, `deepcopy`, pickle, cross-PID use, and closed/unlocked/replaced claim FD.

**Step 3: Implement atomic claim and consumption**

Add an owner object/context manager:

```python
@dataclass(slots=True)
class Exp449SmokeAExecutionLeaseAntigravity:
    verified_authority: VerifiedExp449SmokeAAuthorityAntigravity
    context: Exp449Context
    claim_fd: int
    capability: object

def acquire_smoke_a_execution_lease_antigravity(...) -> Exp449SmokeAExecutionLeaseAntigravity: ...
```

Required order:

1. verify authority/readiness;
2. lstat exact scientific root, operations root, and claim components;
3. create claim with `O_CREAT|O_EXCL|O_NOFOLLOW`, fsync, then nonblocking exclusive `flock`;
4. verify FD inode equals path inode;
5. create operations root without symlink traversal;
6. create-only `execution_authority_consumed_antigravity.json` from verified authority and claim facts;
7. mint capability.

Fresh launch requires absent roots and claim. Resume is not handled here.

**Step 4: Gate mutating APIs only**

Change signatures:

```python
def run_exp449_wave_antigravity(
    config: Exp449WaveConfig,
    *,
    _execution_capability: object = None,
) -> Exp449TerminalStatus: ...

def freeze_authorized_exp449_proposal_antigravity(
    score: object,
    *, config: Exp449WaveConfig,
    _execution_capability: object,
) -> Exp449ProposalManifest: ...
```

The public scientific `freeze_exp449_proposal_antigravity` remains pure/testable but the official orchestration must call only the gated wrapper. Terminal publication must also require the capability. Keep test-only seams private and module-owned.

**Step 5: Run GREEN and commit**

Run claim/capability/recovery tests, static search proving no public boolean bypass, fresh-subprocess pickle/cross-PID tests, and fixed-root absence checks.

Commit:

```bash
git commit -am "feat(exp449): require one-shot execution capability"
```

---

### Task 4: Minimal durable progress, status, and failure-safe publication

**Files:**

- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_execution_progress_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py`
- Modify together: Exp449 codemap triplet

**Step 1: Write RED logging tests**

Test exact operations filenames:

- `execution_authority_consumed_antigravity.json`
- `progress_antigravity.jsonl`
- `status_antigravity.json`
- `worker_stdout_antigravity.log`
- `worker_stderr_antigravity.log`
- `terminal_complete_antigravity.json` or `terminal_incomplete_antigravity.json`

Require every progress event to contain schema, timestamp, event, wave, dataset/null, completed, failed, pending, elapsed, throughput, ETA, latest error/null, and worker PID. Reject labels, raw values, scores, p-values, and full exception trace payloads.

Inject short writes, `EINTR`, append failure, fsync failure, cache replace failure, crash after append, and duplicate terminal publication. Progress append/fsync failure must stop before scheduling another dataset. Status cache failure must not change authoritative recovery state.

**Step 2: Replace best-effort progress at the official boundary**

The existing `_emit_progress_antigravity` may remain best-effort for pure test composition, but the official authorized wrapper must use:

```python
def append_authorized_progress_antigravity(
    lease: Exp449SmokeAExecutionLeaseAntigravity,
    event: Mapping[str, object],
) -> None: ...

def refresh_derived_status_antigravity(
    lease: Exp449SmokeAExecutionLeaseAntigravity,
) -> None: ...
```

Loop on short `os.write`, retry `EINTR`, fsync the progress FD, and fail before replenishing the rolling work queue. Use atomic temp+replace+directory fsync only for derived status cache. Keep authoritative state in grant consumption, paper checkpoints/attempts, manifest, and terminal records.

**Step 3: Implement read-only status**

Extend `status_exp449_read_only_antigravity()` to inspect absent/present claim, operations receipt, progress tail, checkpoint counts, proposal manifest, and terminal without importing scoring or reading DB/labels. It must never create or heal files.

**Step 4: Run GREEN and commit**

Run focused progress/recovery tests, fresh import firewall, and verify no official root is created.

Commit:

```bash
git commit -am "feat(exp449): persist fail-safe Smoke A progress"
```

---

### Task 5: Authorized fresh/resume orchestration and scientific equivalence

**Files:**

- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/artifacts_antigravity.py` only if a capability-bound wrapper cannot live in operations without an import cycle
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_execution_lifecycle_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_runner_recovery_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_manifest_verifier_antigravity.py`
- Modify together: Exp449 codemap triplet

**Step 1: Write RED end-to-end fixture tests**

Use genuine typed two-dataset and frozen 23-dataset in-memory fixtures. Patch only dependency preflight and heavy dataset scorer where the environment requires it; use real recovery, Task6/7, policy construction, typed writer, and scorer-free verifier.

Cover:

- fresh Wave 1 complete;
- Wave 1 partial failure while later datasets continue;
- Wave 2 queue equals failed plus never-attempted only;
- Wave 2 incomplete publishes terminal and never runs Task6/7/freeze;
- interruption after success checkpoint then resume without recompute;
- interruption after freeze but before terminal then verifier-only terminal recovery;
- context/code/DB/package/backend/worker drift rejects resume;
- verified success/attempt namespace tampering rejects resume;
- direct composition, authorized fresh, and authorized interrupted/resumed results deeply equal for all scientific fields;
- typed shard bytes exactly equal;
- outer manifest bytes equal after only the already-preregistered root-bound context normalization;
- label firewall remains empty.

**Step 2: Return the completed score from the recovery boundary**

Do not change the scientific composer. Introduce an operations result:

```python
@dataclass(frozen=True, slots=True)
class Exp449AuthorizedWaveResultAntigravity:
    status: Exp449TerminalStatus
    score: object | None
```

The final successful wave must retain the genuine `ProposalScore` returned by `compose_exp449_proposal_score_antigravity`. An incomplete wave returns `score=None`. A resumed root that already has exact Task4/5 completion but no freeze may recompute only Task6/7 from verified reconstructed Task4/5 typed checkpoints if and only if the existing paper collector already supplies this deterministic no-dataset-rescore path; otherwise resume by calling composition and assert scorer call count zero.

**Step 3: Implement authorized lifecycle**

Add:

```python
def execute_smoke_a_fresh_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
) -> Exp449TerminalStatus: ...

def execute_smoke_a_resume_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
) -> Exp449TerminalStatus: ...
```

Both functions must construct `Exp449WaveConfig` internally from the verified readiness result: smoke catalog, values-only raw datasets, exact `Exp449Context`, and acceleration manifest. No caller-supplied root/catalog/context/backend/worker is accepted.

Fresh flow:

1. acquire lease;
2. execute Wave 1;
3. if incomplete but recoverable, execute Wave 2 in the same parent process;
4. if dataset-local complete, run Task6/7/Candidate C through the existing composer result;
5. capability-bound freeze;
6. scorer-free `verify_exp449_proposal_antigravity` using exact retained context/catalog;
7. create-only complete terminal.

Resume flow:

1. verify A/grant/readiness;
2. require scientific/operations root and consumed receipt already present and exact;
3. acquire the same claim path’s inactive flock, verifying inode against receipt;
4. verify every checkpoint/attempt/terminal/freeze artifact before use;
5. run only the remaining preregistered wave or verifier-only terminal recovery.

Do not delete, overwrite, migrate, or repair a mismatched root.

**Step 4: Run GREEN and commit**

Run the lifecycle suite repeatedly with forced interruption points. Confirm that fixture scientific artifacts are byte-identical and all production paths remain absent.

Commit:

```bash
git commit -am "feat(exp449): connect authorized Smoke A lifecycle"
```

---

### Task 6: CLI worker handoff, lock reauthentication, and blocked non-A targets

**Files:**

- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py`
- Modify: `scratch/run_exp449_reproducible_rebaseline_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_execution_cli_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_readiness_antigravity.py`
- Modify together: Exp449 codemap triplet

**Step 1: Write RED CLI/worker tests**

Require these commands:

```text
prepare-smoke-a-grant --reviewed-implementation-commit I
run-smoke-a --approved-authority-commit A
resume-smoke-a --approved-authority-commit A
status
verify-proposal --target smoke-a
```

`run-smoke-b` and `run-official` must still fail `execution_authority_not_configured` before root/DB/authority reads. `prepare-smoke-a-grant` must fail unless both authority files and all production roots/claim are absent.

Test worker re-entry with unrelated, unlocked, replaced, or contended FD; wrong stored authority; wrong PID metadata; signal relay; parent death; and a direct hidden worker invocation. Rejected workers may write stderr only and must not mutate operations artifacts.

**Step 2: Implement grant preparation without execution**

Add:

```python
def prepare_smoke_a_grant_antigravity(
    reviewed_implementation_commit: str,
) -> tuple[Path, Path]: ...
```

It runs existing read-only readiness at clean `I`, creates one nonce with `secrets.token_hex(16)`, builds the exact grant/digest, and publishes the two files create-only. It must not create claim, scientific root, operations root, checkpoints, or logs. After publication, require `git status --porcelain` to contain exactly the two untracked authority JSON paths.

**Step 3: Implement authenticated parent/worker launch**

The public CLI parent verifies `A`, acquires the claim/lease, redirects worker stdout/stderr to fixed operations logs, and passes only an inherited locked FD plus a canonical verified authority digest to the private worker entry. The worker must:

- verify FD is the exact claim inode and holds the expected flock;
- rebuild and compare the authority snapshot/readiness;
- mint its own PID-bound capability only after successful reauthentication;
- run the fresh or resume lifecycle;
- relay `SIGINT`/`SIGTERM` safely;
- publish no operations metadata before authentication.

Do not expose a generic `--root`, `--catalog`, `--context`, `--workers`, or arbitrary log path.

**Step 4: Run GREEN and commit**

Run CLI tests with subprocesses, fixed-path monkeypatch seams, and no production roots. Confirm `status` and `verify-proposal` remain scorer-free/import-safe.

Commit:

```bash
git commit -am "feat(exp449): add pinned Smoke A execution commands"
```

---

### Task 7: Implementation commit I verification and independent review

**Files:**

- Modify: `docs/EXP449_RUNBOOK_antigravity_KO.md`
- Modify: `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `RESEARCH_STATE.json` only to state `implementation_ready_awaiting_grant_authorization`; execution flags remain false/zero
- Modify together: Exp449 codemap triplet
- Append ignored report: `.superpowers/sdd/2026-08-15-exp449-smoke-a-pinned-grant-execution-antigravity/task-implementation-report-antigravity.md`

**Step 1: Document exact non-execution procedure**

Document:

- trust limitation of pinned digest;
- exact `I -> A -> user approval -> run` sequence;
- fixed roots and claim;
- 12→11 worker rule and CPU canonical backend;
- progress/status commands;
- fresh vs resume behavior;
- incomplete terminal behavior;
- Smoke B/official/labels still blocked;
- no official root or grant exists at implementation-review time.

**Step 2: Run bounded complete verification**

Use pinned Python 3.14.6:

```bash
python3 -m unittest discover \
  -s experiments/exp449_reproducible_rebaseline_antigravity/tests \
  -p 'test_*_antigravity.py' -v

python3 -m unittest discover \
  -s experiments/paper_public_baselines/tests \
  -p 'test_*.py' -v
```

Run the curated paper suite excluding exactly the known production DB raw-smoke test. Also run:

- fresh import label/scorer firewall tests;
- 25 repeated subprocess lifecycle/authority tests;
- `py_compile` for changed Python files;
- Exp449 HTML embedded JSON equality, paths/symbols/callers/effects/tests, graph refs, and lock fingerprints;
- `git diff --check`;
- lstat proof that six roots, claim, and both authority files are absent.

Do not invoke `readiness` if it would inspect the real DB during ordinary implementation tests; use fixture seams until the separately reviewed grant-preparation step.

**Step 3: Commit I**

The final implementation commit must contain no authority JSON. Create it, then
record its exact SHA as `I`:

```bash
git commit -m "feat(exp449): implement pinned Smoke A execution lifecycle"
I=$(git rev-parse HEAD)
git status --short
git ls-files docs/authorities
```

Expected: clean tree and no two Smoke A authority files.

If prior task commits exist, use the final reviewed HEAD as `I`; do not squash after grant generation.

**Step 4: Fresh independent Sol review**

Reviewer must independently verify:

- scientific code and artifact schemas unchanged;
- direct mutating API bypass closed;
- exact authorization delta and single-snapshot verification;
- claim/flock/consumption/replay behavior;
- fresh/resume equivalence and no-recompute;
- progress append failure stops scheduling;
- all non-A targets and labels blocked;
- codemap and closure accuracy;
- authority files and production roots absent.

Verdict must be `SHIP`. A `FIX-FIRST` result requires a new final `I`; do not prepare grant yet.

---

### Task 8: Prepare and review authorization commit A without launching Smoke A

**Files:**

- Create only: `docs/authorities/smoke_a_execution_grant_antigravity.json`
- Create only: `docs/authorities/smoke_a_execution_grant_digest_antigravity.json`

**Step 1: Reconfirm clean reviewed I**

Require:

- exact clean `HEAD == I`;
- independent Sol verdict `SHIP` recorded outside runtime authority;
- six production roots and claim absent;
- two authority files absent;
- no label modules loaded.

**Step 2: Prepare grant read-only**

Run only:

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py \
  prepare-smoke-a-grant --reviewed-implementation-commit "$I"
```

This step may read the frozen DB/catalog for readiness identity. It must perform no scoring, create no production root/claim/log/checkpoint, and import no labels.

Verify canonical bytes, grant SHA/size, 23 datasets, context, closure, locks, inventory, constants, acceleration, 12→11 workers, CPU backend, max waves 2, label access false, and one 32-hex nonce.

**Step 3: Create direct-child A**

Before commit, require `git status --porcelain` to show exactly the two new authority JSON files. Then:

```bash
git add \
  docs/authorities/smoke_a_execution_grant_antigravity.json \
  docs/authorities/smoke_a_execution_grant_digest_antigravity.json
git commit -m "chore(exp449): authorize one Smoke A execution"
A=$(git rev-parse HEAD)
test "$(git rev-parse HEAD^)" = "$I"
```

Verify `git diff-tree --no-commit-id --name-status -r "$A"` contains exactly two `A` records for those paths.

Do not update code, docs, state, or codemap in `A`.

**Step 4: Fresh independent Sol review of A**

Reviewer independently recomputes:

- `parent(A) == I`;
- exact two-file delta and regular tracked modes;
- canonical grant/digest and exact byte SHA/size;
- unchanged scientific/authority closure blobs and locks;
- live read-only readiness facts equal grant;
- roots and claim absent;
- no scoring/labels/launch occurred.

Verdict must be `SHIP`. Any issue invalidates `A`; return to a new implementation `I` and generate a new direct-child `A` rather than stacking a fix.

**Step 5: Stop for explicit user approval**

Present only:

- authorization commit `A`;
- implementation parent `I`;
- grant SHA-256;
- Smoke A scientific and operations roots;
- claim path;
- logical CPUs `12`, workers `11`;
- canonical backend `cpu`;
- review verdicts;
- exact command that would launch.

Do **not** execute it. Wait for the user to say exactly:

```text
Smoke A 실행 승인
```

Only a subsequent execution turn may launch Smoke A.

---

## Completion Criteria

This plan is complete only when:

- implementation commit `I` is independently reviewed `SHIP`;
- authorization commit `A` is its exact direct child and independently reviewed `SHIP`;
- no scientific algorithm/schema changed;
- all lifecycle/equivalence/failure tests pass;
- six production roots and claim remain absent;
- Smoke B, official, and label/evaluation remain blocked;
- actual Smoke A has not started;
- the user has been shown exact `A`, `I`, grant, roots, 12→11 worker provenance, and the pending approval phrase.
