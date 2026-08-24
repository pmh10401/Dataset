# Exp471 Exp467 Evaluation Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provenance-preserving, resumable Exp471 evaluation and paper-report pipeline from the completed Exp467 artifacts without rerunning the 4,468 source experiments.

**Architecture:** Exp471 reconstructs and verifies the fixed Exp467 source authority from Git bytes, then persists create-only raw/micro score shards before any label access. A complete cache authority opens the frozen Exp450 metric/curve boundary once, stores a durable normalized evaluation artifact, and renders the report without repeating scoring or metric calculation.

**Tech Stack:** Python 3.14, stdlib `unittest`, dataclasses, SHA-256 canonical JSON, `concurrent.futures`, NumPy, Numba, frozen Exp450/Exp467 evaluation code, Git historical bytes, Matplotlib report generation.

**Spec:** `docs/superpowers/specs/2026-08-24-exp471-exp467-evaluation-recovery-antigravity-design.md`

## Global Constraints

- Do not modify any file under `experiments/exp467_numba_calibration_resume_antigravity` or any existing Exp467 result root.
- Fixed Exp467 commit: `1e616731a88da9bc0268823e1d2cef2a53c1054f`.
- Fixed Exp467 source context: `31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af`.
- Official interpreter: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`.
- Exp471 root: `/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_antigravity_20260824`.
- CPU workers equal detected logical CPU count minus one, minimum one; inner numerical threads equal one.
- Preserve the frozen ten-method order: `A`, `B`, `C`, `D`, `knn_distance`, `lof_novelty`, `ocsvm_rbf`, `isolation_forest`, `deep_svdd_1d`, `drocc_1d`.
- Preserve raw score, micro score, AP, AUROC, paired-statistic, ROC-coordinate, and PR-coordinate definitions.
- Keep raw in-memory ROC `+inf`; map non-finite floats to JSON `null` only in the Exp471 artifact view.
- Never use `allow_nan=True`, drop the threshold field, stringify infinity, or replace it with a finite sentinel.
- All new filenames contain `antigravity`.
- Every task begins with a failing behavior test, captures the expected failure, implements only enough to pass, regenerates the complete Exp471 code-map triplet, and commits.
- Ox Alpha workers receive public code and synthetic fixtures only. They receive no official Dataset path contents, TEST labels, secrets, keys, or private raw data.
- Ox Alpha writes only in `/Users/mac/Documents/Time Series/.worktrees/exp471-exp467-evaluation-recovery-antigravity`; Codex inspects every diff and reruns verification.
- PR #25 remains unmerged and open unless the user separately authorizes closing or retargeting it.

## Cross-Cutting Acceleration Record

- Record PyTorch, MLX, and MPS availability and versions in Exp471 provenance.
- Record that Exp471 does not rerun tensor feature generation and launches no
  MPS-heavy process.
- Reject MLX/MPS promotion for upper-tail calibration because no exact validated
  expression preserves the frozen Numba float64 order-statistic semantics.
- Use Numba/CPU as the canonical path. Parallelize only independent cache shards,
  after literal byte parity and measured wall-time improvement.

## Cross-Cutting Failure Handling and Recovery

- Every persisted file is create-only, hash-bound, reopened before downstream
  use, and accepted on retry only when canonical bytes are exact.
- A malformed existing source, cache, evaluation, or report artifact fails closed
  with its contained path; no worker deletes or overwrites it.
- Durable progress is emitted only after artifact publication. A resumed process
  verifies and skips completed work, completes the first wave, and retries each
  unique failure once in the second wave.
- TEST labels remain unreachable until the complete score-cache authority reopens.
- A durable evaluation artifact prevents report-rendering failures from repeating
  score preparation or metric calculation.

---

## File Map

Create these production files:

- `experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py` — fixed identities, roots, order, and backend policy.
- `experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py` — strict immutable schemas and canonical JSON.
- `experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py` — historical Git closure and complete source verification.
- `experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py` — create-only shard and manifest persistence.
- `experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py` — score preparation, scheduling, retry, and progress.
- `experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py` — frozen metrics, artifact-view normalization, and durable evaluation.
- `experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py` — report rendering and atomic publication.
- `experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py` — command orchestration.
- `scratch/run_exp471_exp467_evaluation_recovery_antigravity.py` — closed public launcher.
- `scratch/generate_exp471_codemap_antigravity.py` — synchronized code-map generator/checker.
- `docs/EXP471_EVALUATION_RECOVERY_RUNBOOK_antigravity_KO.md` — operator commands and log locations.

Create these test files:

- `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_config_contracts_antigravity.py`
- `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_source_authority_antigravity.py`
- `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_cache_artifacts_antigravity.py`
- `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py`
- `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py`
- `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py`
- `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_codemap_antigravity.py`

Create and continually regenerate:

- `docs/codemap/codemap_exp471_antigravity.html`
- `docs/codemap/codemap_exp471_antigravity.json`
- `docs/codemap/codemap_exp471_antigravity.lock`

---

### Task 1: Establish the Exp471 Code Map Before Module Changes

**Files:**
- Create: `scratch/generate_exp471_codemap_antigravity.py`
- Create: `docs/codemap/codemap_exp471_antigravity.html`
- Create: `docs/codemap/codemap_exp471_antigravity.json`
- Create: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: repository root and the approved design/plan paths.
- Produces: `main(argv: Sequence[str] | None = None) -> int`, accepting only no arguments or `--check`.

- [ ] **Step 1: Create the generator with a deliberately missing Exp471 triplet check**

Adapt `scratch/generate_exp467_codemap_antigravity.py`. Freeze:

```python
VERSION_ANTIGRAVITY = "exp471"
PACKAGE_ANTIGRAVITY = "experiments/exp471_exp467_evaluation_recovery_antigravity"
CLI_ANTIGRAVITY = "scratch/run_exp471_exp467_evaluation_recovery_antigravity.py"
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp471_antigravity.html",
    "docs/codemap/codemap_exp471_antigravity.json",
    "docs/codemap/codemap_exp471_antigravity.lock",
)
```

The initial graph must state that Exp471 production modules do not yet exist and
must bind the approved design and plan hashes as governance inputs.

- [ ] **Step 2: Run check mode and capture RED**

Run:

```sh
python3 scratch/generate_exp471_codemap_antigravity.py --check
```

Expected: exit 2 with an explicit message that the Exp471 triplet is missing.

- [ ] **Step 3: Generate the initial complete triplet**

Run:

```sh
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
```

Expected: generation succeeds; check mode exits 0; JSON and the JSON embedded in
HTML are byte-equivalent; the lock binds all three artifacts and governance files.

- [ ] **Step 4: Inspect scope and commit**

Run:

```sh
git diff --check
git status --short
git diff --stat
```

Expected: only the generator and Exp471 triplet are new.

```sh
git add scratch/generate_exp471_codemap_antigravity.py docs/codemap/codemap_exp471_antigravity.*
git commit -m "docs: establish Exp471 code map"
```

---

### Task 2: Freeze Configuration and Strict Contracts

**Files:**
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/config_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_config_contracts_antigravity.py`
- Modify: `docs/codemap/codemap_exp471_antigravity.html`
- Modify: `docs/codemap/codemap_exp471_antigravity.json`
- Modify: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: fixed identities from the spec.
- Produces:
  - `canonical_json_bytes_exp471_antigravity(value: object) -> bytes`
  - `Exp471SourceAuthorityAntigravity`
  - `Exp471ScoreShardAntigravity`
  - `Exp471ScoreCacheManifestAntigravity`
  - `Exp471EvaluationArtifactAntigravity`
  - `Exp471ReportAuthorityAntigravity`

- [ ] **Step 1: Write failing constant and schema tests**

Create literal tests that require:

```python
self.assertEqual(EXP467_SOURCE_COMMIT_ANTIGRAVITY, "1e616731a88da9bc0268823e1d2cef2a53c1054f")
self.assertEqual(EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY, "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af")
self.assertEqual(EXP471_ROOT_ANTIGRAVITY.name, "exp471_exp467_evaluation_recovery_antigravity_20260824")
self.assertEqual(LOGICAL_CPU_COUNT_ANTIGRAVITY, os.cpu_count())
self.assertEqual(CPU_WORKERS_ANTIGRAVITY, max(1, os.cpu_count() - 1))
```

Also require strict finite canonical JSON, exact 64-character lowercase digests,
fixed method order, absolute contained paths, immutable mappings/tuples, and
rejection of duplicate dataset/method shard identities.

- [ ] **Step 2: Run tests and capture RED**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_config_contracts_antigravity
```

Expected: import failure because the Exp471 modules do not exist.

- [ ] **Step 3: Implement minimal configuration and contracts**

Use exact dataclass fields:

```python
@dataclass(frozen=True, slots=True)
class Exp471SourceAuthorityAntigravity:
    source_commit: str
    source_context_sha256: str
    inventory_sha256: str
    catalog_sha256: str
    database_sha256: str
    source_file_sha256: Mapping[str, str]
    source_tree_sha256: str
    condition_manifest_sha256: Mapping[str, str]
    total_successes: int
    imported_total: int
    native_total: int
    authority_sha256: str

@dataclass(frozen=True, slots=True)
class Exp471ScoreShardAntigravity:
    source_authority_sha256: str
    cache_context_sha256: str
    method: str
    dataset: str
    test_count: int
    raw_scores: tuple[float, ...]
    micro_scores: tuple[float, ...]
    source_manifest_sha256: str
    payload_sha256: str
    envelope_sha256: str

@dataclass(frozen=True, slots=True)
class Exp471ScoreCacheManifestAntigravity:
    source_authority_sha256: str
    cache_context_sha256: str
    method_order: tuple[str, ...]
    dataset_order: tuple[str, ...]
    shard_sha256: Mapping[str, str]
    total_shards: int
    manifest_sha256: str

@dataclass(frozen=True, slots=True)
class Exp471EvaluationArtifactAntigravity:
    source_authority_sha256: str
    cache_manifest_sha256: str
    code_closure_sha256: str
    method_order: tuple[str, ...]
    dataset_order: tuple[str, ...]
    payload: Mapping[str, object]
    evaluation_sha256: str

@dataclass(frozen=True, slots=True)
class Exp471ReportAuthorityAntigravity:
    source_authority_sha256: str
    cache_manifest_sha256: str
    evaluation_sha256: str
    report_file_sha256: Mapping[str, str]
    report_manifest_sha256: str
```

The source-file mapping contains every regular source-root file as a contained
relative path and SHA-256. Its canonical digest is `source_tree_sha256`.
Evaluation and report contracts bind source authority, cache manifest, Exp471
code closure, and terminal digests. Implement recursive plain conversion without
allowing non-finite values in generic contracts.

- [ ] **Step 4: Run focused tests, regenerate code map, and commit**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_config_contracts_antigravity
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
git diff --check
git add experiments/exp471_exp467_evaluation_recovery_antigravity \
  docs/codemap/codemap_exp471_antigravity.*
git commit -m "feat: define Exp471 recovery contracts"
```

Expected: focused tests pass and the map answers callers, effects, and tests for
both new modules.

---

### Task 3: Reconstruct and Verify the Fixed Exp467 Source Authority

**Files:**
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_source_authority_antigravity.py`
- Modify: `docs/codemap/codemap_exp471_antigravity.html`
- Modify: `docs/codemap/codemap_exp471_antigravity.json`
- Modify: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: fixed commit/context constants and unchanged Exp467 modules whose
  worktree hashes equal fixed Git bytes.
- Produces:
  - `reconstruct_exp467_source_context_antigravity(repository: Path) -> object`
  - `verify_exp467_source_antigravity(repository: Path, source_root: Path) -> Exp471SourceAuthorityAntigravity`
  - `open_exp467_source_session_antigravity(repository: Path, source_root: Path) -> Exp471SourceSessionAntigravity`
  - `publish_source_authority_antigravity(root: Path, authority: Exp471SourceAuthorityAntigravity) -> Path`
  - `read_source_authority_antigravity(root: Path, repository: Path, source_root: Path) -> Exp471SourceAuthorityAntigravity`
  - `_source_session_is_issued_antigravity(value: object) -> bool`

- [ ] **Step 1: Write failing historical-authority tests**

Tests must use a synthetic repository/fixture and literal expected hashes. Cover:

```python
authority = verify_exp467_source_antigravity(repository, fixture_root)
self.assertEqual(authority.source_context_sha256, FIXED_CONTEXT)
self.assertEqual(authority.total_successes, 4468)
self.assertEqual((authority.imported_total, authority.native_total), (4234, 234))
```

Mutation tests change one fixed Git byte, one worktree byte, catalog hash, database
hash, source envelope, condition manifest, or origin count and require rejection
before a supplied score-reader or label callback runs. Assert Exp467 modules are
loaded lazily only after the historical-byte gate succeeds.

- [ ] **Step 2: Run tests and capture RED**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_source_authority_antigravity
```

Expected: import failure for `source_authority_antigravity`.

- [ ] **Step 3: Implement historical Git reconstruction**

Implement fail-closed primitives:

```python
def _git_bytes_antigravity(repository: Path, path: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(repository), "show", f"{EXP467_SOURCE_COMMIT_ANTIGRAVITY}:{path}"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Exp467 historical closure path is unavailable")
    return result.stdout

def _require_matching_worktree_closure_antigravity(repository: Path, paths: Sequence[str]) -> Mapping[str, str]:
    fixed = {path: sha256(_git_bytes_antigravity(repository, path)).hexdigest() for path in paths}
    if any(sha256((repository / path).read_bytes()).hexdigest() != digest for path, digest in fixed.items()):
        raise ValueError("Exp467 worktree closure differs from fixed Git bytes")
    return MappingProxyType(fixed)
```

Compile the fixed source launcher under a non-`__main__` module name to obtain its
deterministic runtime path list. Only after every path matches fixed Git bytes,
import the unchanged Exp467 verifier and reconstruct its exact run context using
the official interpreter/backend record. Require the fixed context digest, verify
the migration manifest and all source results, reopen the four condition manifests,
capture the complete regular-file relative-path/SHA map, and issue a weak-set-backed
Exp471 source session. Publish the source authority create-only under the Exp471
root; reopening recomputes the source tree and all authority hashes.

- [ ] **Step 4: Run tests and verify the real source read-only**

Focused tests:

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_source_authority_antigravity
```

Codex-only read-only real check, never delegated to Ox:

```sh
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python - <<'PY'
from pathlib import Path
from experiments.exp471_exp467_evaluation_recovery_antigravity.source_authority_antigravity import verify_exp467_source_antigravity
authority = verify_exp467_source_antigravity(
    Path.cwd(),
    Path('/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_official_antigravity_20260823'),
)
print(authority.source_context_sha256, authority.total_successes, authority.imported_total, authority.native_total)
PY
```

Expected real summary: source context `31341d8e...`, 4,468 successes, 4,234
imported, 234 native, zero failures. Snapshot the Exp467 tree before and after and
require identical path/hash maps.

- [ ] **Step 5: Regenerate map and commit**

```sh
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
git diff --check
git add experiments/exp471_exp467_evaluation_recovery_antigravity \
  docs/codemap/codemap_exp471_antigravity.*
git commit -m "feat: verify frozen Exp467 source authority"
```

---

### Task 4: Implement Durable Score Cache Artifacts

**Files:**
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_cache_artifacts_antigravity.py`
- Modify: `docs/codemap/codemap_exp471_antigravity.html`
- Modify: `docs/codemap/codemap_exp471_antigravity.json`
- Modify: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: Task 2 shard/cache contracts.
- Produces:
  - `score_shard_relative_path_antigravity(method: str, dataset: str) -> Path`
  - `publish_score_shard_antigravity(root: Path, shard: Exp471ScoreShardAntigravity) -> Path`
  - `read_score_shard_antigravity(root: Path, method: str, dataset: str) -> Exp471ScoreShardAntigravity`
  - `publish_cache_manifest_antigravity(root: Path, manifest: Exp471ScoreCacheManifestAntigravity) -> Path`
  - `read_cache_manifest_antigravity(root: Path) -> Exp471ScoreCacheManifestAntigravity`

- [ ] **Step 1: Write failing artifact-lifecycle tests**

Require deterministic digest-derived contained paths, regular-file and no-symlink
checks, same-directory temporary files, fsync, exclusive no-clobber publication,
exact-byte reuse, and rejection of changed existing bytes. Test a crash before
publish, concurrent writers, truncated JSON, non-finite vector, wrong row count,
wrong method/dataset order, and a manifest published before complete coverage.

- [ ] **Step 2: Run tests and capture RED**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_cache_artifacts_antigravity
```

Expected: import failure for `cache_artifacts_antigravity`.

- [ ] **Step 3: Implement atomic create-only persistence**

Use the established repository sequence:

```python
descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
try:
    os.write(descriptor, raw)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.link(temporary, destination)
os.unlink(temporary)
```

If `destination` exists, read it without following links and accept only exact
canonical bytes. Fsync the containing directory after publication. Never unlink,
overwrite, or reclaim a foreign destination.

- [ ] **Step 4: Verify, regenerate map, and commit**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_cache_artifacts_antigravity
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
git diff --check
git add experiments/exp471_exp467_evaluation_recovery_antigravity \
  docs/codemap/codemap_exp471_antigravity.*
git commit -m "feat: add Exp471 create-only score cache"
```

---

### Task 5: Build Resumable Parallel Score Preparation

**Files:**
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py`
- Modify: `docs/codemap/codemap_exp471_antigravity.html`
- Modify: `docs/codemap/codemap_exp471_antigravity.json`
- Modify: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: issued source session, Task 4 artifact API, frozen Exp467 score builder,
  frozen baseline manifests, and existing Numba calibration.
- Produces:
  - `prepare_score_cache_antigravity(session: Exp471SourceSessionAntigravity, *, workers: int | None = None, progress: Callable[[Mapping[str, object]], None] | None = None) -> Exp471ScoreCacheManifestAntigravity`
  - `status_score_cache_antigravity(root: Path) -> Mapping[str, object]`
  - `open_complete_score_cache_antigravity(session: Exp471SourceSessionAntigravity) -> Exp471ScoreCacheManifestAntigravity`
  - `benchmark_score_preparation_antigravity(tasks: Sequence[object], *, parallel_workers: int) -> Mapping[str, object]`

- [ ] **Step 1: Write failing preparation/resume tests**

Use synthetic literal vectors for all ten methods and at least three datasets.
Require that:

```python
manifest = prepare_score_cache_antigravity(session, workers=2, progress=events.append)
self.assertEqual(manifest.total_shards, 30)
self.assertEqual(tuple(manifest.method_order), METHOD_ORDER_ANTIGRAVITY)
self.assertEqual(events[-1]["durable_completed"], 30)
```

Interrupt after a durable shard, rerun, and assert the verified shard loader is
not called again for that identity. Test unique failure rows, full first wave
before retry, deterministic order, exact raw/micro separation, finite values,
wrong authority rejection, and label callback non-reachability.

Add a one-worker versus two-worker test that compares complete shard bytes and
requires equality. Benchmark assertions use measured elapsed values supplied by
a fake clock rather than machine timing.

- [ ] **Step 2: Run tests and capture RED**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_score_cache_antigravity
```

Expected: import failure for `score_cache_antigravity`.

- [ ] **Step 3: Implement the label-free task builder**

The parent process verifies the issued source session and constructs immutable
tasks in exact method/dataset order. A task contains only public identity, finite
train/test score vectors, hashes, and row counts; it carries no label capability.

For A/B/C/D, set `raw_scores == micro_scores` from the verified Exp467 score
manifest. For each baseline, preserve raw test scores and compute:

```python
micro_scores = 1.0 - upper_tail_pvalues_numba_antigravity(train_oof_scores, test_scores)
```

with float64, `fastmath=False`, `parallel=False`, and `nogil=True`.

- [ ] **Step 4: Implement deterministic waves and progress**

Use `ProcessPoolExecutor(max_workers=workers)` only after a byte-parity/speed
smoke accepts parallel mode. Default `workers` is `CPU_WORKERS_ANTIGRAVITY`.
Completed verified shards are excluded before scheduling. Persist a success shard
before emitting its progress event. Run all first-wave identities before a unique
retry wave. Publish the terminal manifest only at exact full coverage.

- [ ] **Step 5: Run focused tests and a public synthetic benchmark**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_score_cache_antigravity
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_score_cache_antigravity.ScoreCacheTestsAntigravity.test_public_synthetic_worker_benchmark
```

Expected: tests pass; benchmark records one-worker and 11-worker wall time,
byte-parity result, detected CPU count, selected workers, and fallback reason.
The CLI may select 11 workers only when bytes are identical and wall time improves.

- [ ] **Step 6: Regenerate map and commit**

```sh
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
git diff --check
git add experiments/exp471_exp467_evaluation_recovery_antigravity \
  docs/codemap/codemap_exp471_antigravity.*
git commit -m "feat: prepare resumable Exp471 score cache"
```

---

### Task 6: Persist the Frozen Evaluation with JSON-Safe ROC Metadata

**Files:**
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py`
- Modify: `docs/codemap/codemap_exp471_antigravity.html`
- Modify: `docs/codemap/codemap_exp471_antigravity.json`
- Modify: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: issued source session and complete cache manifest.
- Produces:
  - `artifact_view_exp471_antigravity(value: object) -> object`
  - `evaluate_exp471_antigravity(session: Exp471SourceSessionAntigravity, cache: Exp471ScoreCacheManifestAntigravity, *, label_loader: Callable | None = None) -> Exp471EvaluationArtifactAntigravity`
  - `read_evaluation_exp471_antigravity(root: Path, session: Exp471SourceSessionAntigravity, cache: Exp471ScoreCacheManifestAntigravity) -> Exp471EvaluationArtifactAntigravity`
  - `reconstruct_frozen_evaluation_exp471_antigravity(artifact: Exp471EvaluationArtifactAntigravity) -> Exp450EvaluationAntigravity`

- [ ] **Step 1: Write the real-curve RED regression test**

Use one normal and one anomaly with literal scores for all ten methods. Call the
real frozen curve builder and assert the first ROC threshold for every method is
`math.inf`. Then call `evaluate_exp471_antigravity`; before implementation it must
fail because the module/function does not exist.

The expected green assertions are:

```python
self.assertTrue(all(math.isinf(first[method]["threshold"]) for method in METHOD_ORDER))
self.assertEqual(artifact.payload["curve_tables"]["roc"]["rows"][0]["threshold"], None)
self.assertEqual(raw_metric_values, reconstructed_metric_values)
self.assertEqual(raw_roc_coordinates, reconstructed_roc_coordinates)
self.assertEqual(raw_pr_coordinates, reconstructed_pr_coordinates)
self.assertRegex(artifact.evaluation_sha256, r"^[0-9a-f]{64}$")
```

Also require label-loader non-reachability for missing/tampered/incomplete cache,
one label-loader call after a complete cache, deterministic digest, and durable
artifact exact-byte reuse.

- [ ] **Step 2: Run tests and capture RED**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_evaluation_antigravity
```

Expected: import failure for `evaluation_antigravity`.

- [ ] **Step 3: Implement minimal artifact-view normalization**

```python
def artifact_view_exp471_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): artifact_view_exp471_antigravity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [artifact_view_exp471_antigravity(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
```

Do not call this helper on score shards or generic contracts. Apply it only to the
complete evaluation artifact immediately before strict hashing/publication.

- [ ] **Step 4: Implement exact metric call and durable reconstruction**

Load raw and micro maps from every verified shard and call:

```python
metrics = metric_tables_exp450_antigravity(labels, raw_scores, micro_scores=micro_scores)
curves = curve_tables_exp450_antigravity(labels, micro_scores)
```

Persist normalized JSON. On reopen, permit `null` only in the first ROC threshold
row of each method, reconstruct it as `math.inf`, and reject null/non-finite values
anywhere else. Recompute all hashes and coordinate invariants before issuing the
evaluation authority.

- [ ] **Step 5: Verify, regenerate map, and commit**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_evaluation_antigravity
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
git diff --check
git add experiments/exp471_exp467_evaluation_recovery_antigravity \
  docs/codemap/codemap_exp471_antigravity.*
git commit -m "feat: persist Exp471 continuous-score evaluation"
```

---

### Task 7: Render Reports and Close the CLI Pipeline

**Files:**
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py`
- Create: `scratch/run_exp471_exp467_evaluation_recovery_antigravity.py`
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py`
- Modify: `docs/codemap/codemap_exp471_antigravity.html`
- Modify: `docs/codemap/codemap_exp471_antigravity.json`
- Modify: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: source verification, cache preparation/status, durable evaluation.
- Produces:
  - `write_exp471_report_antigravity(artifact: Exp471EvaluationArtifactAntigravity, root: Path) -> Exp471ReportAuthorityAntigravity`
  - `main_antigravity(argv: Sequence[str] | None = None) -> int`
  - commands: `verify-source`, `benchmark-prepare`, `prepare`, `status`, `report`.

- [ ] **Step 1: Write failing CLI/report lifecycle tests**

Test literal argument routing and JSON output. Require `report` to:

1. verify source;
2. reopen or prepare the complete cache;
3. reopen or compute the durable evaluation once;
4. render from that artifact;
5. publish report atomically.

Inject counters and assert a second `report` call invokes neither shard preparation
nor metric calculation. Test active lock rejection, dead-owner recovery, fault
before and after atomic rename, markerless destination refusal, and final report
artifact-map verification.

- [ ] **Step 2: Run tests and capture RED**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_reporting_cli_antigravity
```

Expected: import failure for reporting/CLI modules.

- [ ] **Step 3: Implement report rendering**

Delegate table/curve semantics to frozen Exp450 reporting where its interface
accepts the reconstructed frozen evaluation. Add an Exp471 provenance JSON that
binds source context, source inventory, cache manifest, evaluation digest,
Exp471 closure, interpreter, dependency versions, worker benchmark, and every
report file SHA-256.

The final report must contain all ten methods in:

- per-dataset AP/AUROC tables;
- macro and micro AP/AUROC tables;
- ROC curves with x = false-positive rate and y = true-positive rate;
- PR curves with x = recall and y = precision;
- machine-readable JSON/CSV and publication figures.

- [ ] **Step 4: Implement CLI orchestration and logs**

The launcher adds its repository root to `sys.path`, accepts only fixed commands,
prints canonical finite JSON, exits 2 on a concise error, and never accepts an
arbitrary source/cache/report root. `status` reports durable counts and paths.
`report` resumes prerequisites instead of recomputing completed stages.

- [ ] **Step 5: Verify, regenerate map, and commit**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_reporting_cli_antigravity
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
git diff --check
git add experiments/exp471_exp467_evaluation_recovery_antigravity \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py \
  docs/codemap/codemap_exp471_antigravity.*
git commit -m "feat: add Exp471 report pipeline"
```

---

### Task 8: Full Verification, Runbook, and Publication Readiness

**Files:**
- Create: `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_codemap_antigravity.py`
- Create: `docs/EXP471_EVALUATION_RECOVERY_RUNBOOK_antigravity_KO.md`
- Modify: `docs/codemap/codemap_exp471_antigravity.html`
- Modify: `docs/codemap/codemap_exp471_antigravity.json`
- Modify: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**
- Consumes: complete Exp471 CLI and artifact graph.
- Produces: operator-ready documentation and synchronized final code map.

- [ ] **Step 1: Write failing closure/code-map tests**

Require exact Exp471 production closure, direct import edges, callers/effects/tests
answers, matching embedded JSON, complete lock hashes, fixed Exp467 source hashes,
and absence of any Exp467 modification in `git diff 1e616731...HEAD`.

- [ ] **Step 2: Run the code-map test and capture RED**

```sh
python3 -m unittest -v \
  experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_codemap_antigravity
```

Expected: failure until the final triplet and source-hash assertions are current.

- [ ] **Step 3: Write the Korean runbook**

Document exact official-interpreter commands, expected JSON fields, progress log,
cache/evaluation/report roots, status interpretation, safe interruption, resume,
failure-ledger handling, and the rule that operators never delete or overwrite a
foreign artifact. Include one example status record with synthetic names only.

- [ ] **Step 4: Run the complete fresh verification suite**

```sh
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  -m unittest discover -v \
  -s experiments/exp471_exp467_evaluation_recovery_antigravity/tests \
  -p 'test_*_antigravity.py'
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  -m unittest discover -q \
  -s experiments/exp467_numba_calibration_resume_antigravity/tests \
  -p 'test_*_antigravity.py'
python3 scratch/generate_exp471_codemap_antigravity.py
python3 scratch/generate_exp471_codemap_antigravity.py --check
python3 -m compileall -q \
  experiments/exp471_exp467_evaluation_recovery_antigravity \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py \
  scratch/generate_exp471_codemap_antigravity.py
git diff --check
git status --short
```

Expected: all Exp471 tests pass; all 112 unchanged Exp467 tests pass; code map is
current; compilation exits 0; only intended Exp471/docs/scratch files differ.

- [ ] **Step 5: Verify source non-mutation and commit**

```sh
test -z "$(git diff --name-only 1e616731a88da9bc0268823e1d2cef2a53c1054f -- \
  experiments/exp467_numba_calibration_resume_antigravity)"
git add experiments/exp471_exp467_evaluation_recovery_antigravity \
  docs/EXP471_EVALUATION_RECOVERY_RUNBOOK_antigravity_KO.md \
  docs/codemap/codemap_exp471_antigravity.*
git commit -m "docs: finalize Exp471 recovery operations"
```

---

### Task 9: Codex-Only Official Preflight and Resumable Report Run

**Files:**
- Write only under: `/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_antigravity_20260824`
- Write logs only under: `/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_evidence_antigravity_20260824`
- Read only under: `/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_official_antigravity_20260823`
- Do not give either root to Ox Alpha.

**Interfaces:**
- Consumes: reviewed, committed Exp471 implementation and official interpreter.
- Produces: complete cache manifest, durable evaluation artifact, final paper report,
  live console/progress logs, and before/after Exp467 byte snapshots.

- [ ] **Step 1: Capture immutable source snapshot**

Run `verify-source`. It publishes
`source_authority_antigravity.json` under the Exp471 root with the complete
contained relative-path/SHA map and `source_tree_sha256`. Save the command output
as
`/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_evidence_antigravity_20260824/verify_source_before_antigravity.log`.
Do not print score payloads or labels.

- [ ] **Step 2: Run source verification and public benchmark**

```sh
set -o pipefail
mkdir -p /Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_evidence_antigravity_20260824
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py verify-source \
  2>&1 | tee /Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_evidence_antigravity_20260824/verify_source_before_antigravity.log
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py benchmark-prepare \
  2>&1 | tee /Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_evidence_antigravity_20260824/benchmark_prepare_antigravity.log
```

Expected: 4,468/4,468, zero failures, fixed source context; benchmark selects the
byte-identical faster worker count or records one-worker fallback.

- [ ] **Step 3: Start one resumable report command with durable logs**

```sh
set -o pipefail
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py report \
  2>&1 | tee /Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_evidence_antigravity_20260824/official_report_console_antigravity.log
```

Redirect stdout/stderr through the established launcher/logging mechanism to an
`antigravity` console log. Do not start a separate `evaluate` process. Monitor
with the `status` command and `progress_antigravity.jsonl`.

- [ ] **Step 4: Verify terminal artifacts and source immutability**

Require cache complete, evaluation digest valid, report artifact map complete,
all ten methods present, ROC/PR axes correct, and no non-finite JSON. Run
`verify-source` again, save output as `verify_source_after_antigravity.log`, and
require the exact same `source_tree_sha256`, `authority_sha256`, file count, and
file map as the create-only source authority from Step 1.

- [ ] **Step 5: Prepare the final PR evidence**

Open a new Exp471 PR only after all Task 8 checks and Task 9 terminal verification
pass. The PR body records commits, test counts, fixed source context, cache/evaluation
digests, report path, worker benchmark, no-source-mutation evidence, and notes that
PR #25 is superseded but remains open pending explicit user authorization.

---

## Plan Completion Gate

Before declaring Exp471 complete, Codex must inspect every task diff and rerun its
verification commands. A task is incomplete if any expected RED was not observed,
any production edit preceded its failing test, an Exp467 file changed, a code-map
triplet is stale, an official root was exposed to Ox, or an artifact claim lacks a
fresh command result.
