# Exp471 Typed-Abstain Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to execute this plan task by task. Track every
> checkbox and do not modify the official experiment worktree until the isolated
> implementation passes the complete verification gate.

**Goal:** Finish Exp471 without rerunning or fabricating public-baseline scores by
preserving all 11,170 method/dataset identities as 10,775 scored shards and 395
typed `insufficient_train_rows` abstention shards, then publish label-gated paper
metrics and figures on both the full population and the fixed 922-dataset
all-ten-method common support.

**Architecture:** A cheap baseline preflight runs before the 4,468-item proposal
preprocessing stage and converts each frozen baseline entry into a strict scored
or abstained record. The verified score-input, cache shard, and cache-manifest
schemas become typed versions that retain exact TEST indices while prohibiting
synthetic abstention scores. Exp471 then reuses the unchanged Exp450 continuous
metric functions twice—once on each method's scored support and once on the
all-ten common support—and adds a small decision-metric layer where abstention
means no alert. An Exp471-native renderer reports coverage, denominators, curves,
and provenance without changing Exp450 or any frozen baseline artifact.

**Tech Stack:** Python 3.14, stdlib `unittest`, frozen dataclasses, canonical
SHA-256 JSON, NumPy, scikit-learn through the unchanged Exp450 metric/curve
functions, existing Numba float64 upper-tail calibration, descriptor-bound
create-only files, and the existing Exp471 atomic report publisher.

**Spec:**
`docs/superpowers/specs/2026-08-25-exp471-typed-abstain-recovery-antigravity-design.md`

## Global Constraints

- Work only in the isolated branch/worktree
  `/Users/mac/Documents/Time Series/.worktrees/exp471-numba-preprocessing-antigravity`
  until all tests pass. Do not write to the official worktree, official result
  root, frozen baseline roots, database, or TEST-label source during implementation.
- Do not rerun a public baseline and do not replace an abstention with zero,
  neutral, repeated, interpolated, imputed, or otherwise synthetic scores.
- Accept only source status `complete`, or source status `abstain` with reason
  exactly `insufficient_train_rows`. Reject failures, missing entries, unknown
  reasons, score-bearing abstentions, and malformed TEST coordinates.
- Preserve the method order `A`, `B`, `C`, `D`, `knn_distance`, `lof_novelty`,
  `ocsvm_rbf`, `isolation_forest`, `deep_svdd_1d`, `drocc_1d` and the catalog's
  1,117-dataset order.
- Preserve Exp467 A/B/C/D score bytes, frozen baseline bytes, the Numba
  `fastmath=False`, `parallel=False`, `nogil=True` calibration, AP/AUROC
  definitions, ROC/PR axes, alpha values, bootstrap seed, and label boundary.
- Candidate C's frozen `score >= 0.5` alert count is the matched-budget reference.
  Matched-budget selection uses only that count and each method's own raw scores;
  it never copies Candidate C coordinates. Abstention always emits no alert.
- Baseline conformal decisions use the existing frozen alpha set and exact
  `1 - upper_tail_p_value` micro scores; proposal threshold decisions use
  `raw_score >= 0.5`.
- Keep existing progress files readable. Publish new cache/evaluation/report
  schemas create-only; there is no completed Exp471 cache or evaluation to migrate.
- Every code task follows RED → GREEN → focused regression → code-map regeneration
  → `git diff --check` → one local commit. Do not push, merge, delete official
  artifacts, or restart the official process during Tasks 1–6.
- All newly created filenames include `antigravity`.

## Code-Map Impact Before Editing

The synchronized `codemap_exp471_antigravity.json` currently answers the required
questions as follows:

- `cli_antigravity.py` is called by the closed launcher and CLI lifecycle tests;
  it affects complete source-to-report orchestration and is covered by
  `test_reporting_cli_antigravity.py`.
- `contracts_antigravity.py` and `cache_artifacts_antigravity.py` are called by
  CLI, cache, evaluation, reporting, and their artifact tests; they affect every
  durable cache authority.
- `score_cache_antigravity.py` is called by CLI, evaluation, reporting, and cache
  tests; it affects label-free preparation, resume, retry, and progress.
- `evaluation_antigravity.py` is called by CLI and reporting and is covered by
  evaluation and reporting tests; it affects the single label gate and paper
  metric authority.
- `reporting_antigravity.py` is called by CLI and reporting lifecycle tests; it
  affects final create-only publication.

Adding `baseline_preflight_antigravity.py` and
`typed_abstain_reporting_antigravity.py` changes module boundaries, so the full
Exp471 HTML/JSON/lock code-map triplet must be regenerated in the same commits
that introduce them.

---

### Task 1: Add a Baseline Preflight Before Proposal Preprocessing

**Files:**

- Create:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/baseline_preflight_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py`
- Modify: `scratch/generate_exp471_codemap_antigravity.py`
- Regenerate: `docs/codemap/codemap_exp471_antigravity.html`
- Regenerate: `docs/codemap/codemap_exp471_antigravity.json`
- Regenerate: `docs/codemap/codemap_exp471_antigravity.lock`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class Exp471BaselinePreflightEntryAntigravity:
    method: str
    dataset: str
    status: Literal["scored", "abstained"]
    test_indices: tuple[int, ...]
    train_oof_scores: tuple[float, ...]
    test_scores: tuple[float, ...]
    abstention_reason: str | None

@dataclass(frozen=True, slots=True)
class Exp471BaselinePreflightAntigravity:
    entries: Mapping[tuple[str, str], Exp471BaselinePreflightEntryAntigravity]
    scored_count: int
    abstained_count: int
    failed_count: int

def preflight_frozen_baselines_antigravity(
    *, datasets: tuple[str, ...], coordinates: Mapping[str, object]
) -> Exp471BaselinePreflightAntigravity:
    raise NotImplementedError
```

- [ ] **Step 1: Write failing preflight tests.** Build synthetic complete,
  abstain, failure, missing, unknown-reason, nonempty-abstain-score, and
  coordinate-mismatch shards. Require exact ordered coverage and verify that the
  production-shaped fixture reports 10,775 scored, 395 abstained, zero failed,
  and 11,170 total identities.

- [ ] **Step 2: Prove the ordering bug with RED.** Patch proposal composition
  with a callback that raises if invoked and feed an invalid baseline root.
  Run:

  ```sh
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m unittest -v \
    experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_reporting_cli_antigravity
  ```

  Expected RED: the proposal callback is reached before the invalid baseline is
  rejected, or the current `production baseline coverage differs` error rejects
  a legal abstention.

- [ ] **Step 3: Implement the strict preflight.** Reuse the pinned-root checks and
  frozen baseline verifier currently embedded in `_production_score_bundle_antigravity`.
  Classify all six methods and all 1,117 datasets before loading proposal raw
  datasets or calling `build_exp467_score_manifests_antigravity`. Keep source
  statuses internal and expose only `scored`/`abstained` to Exp471.

- [ ] **Step 4: Emit score-input v3 records.** Replace the unconditional
  `status == "complete"` rejection with a record shaped as:

  ```python
  {"status": "scored", "test_indices": [0, 1],
   "train_oof_scores": [0.1, 0.2], "test_scores": [0.3, 0.4]}
  {"status": "abstained", "test_indices": [0, 1],
   "reason": "insufficient_train_rows"}
  ```

  Set the bundle schema to `exp471-verified-score-input-antigravity.v3` and add
  the exact preflight counts to its hash-bound metadata.

- [ ] **Step 5: Run focused tests, regenerate the triplet, and commit.**

  ```sh
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m unittest -v \
    experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_reporting_cli_antigravity
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    scratch/generate_exp471_codemap_antigravity.py
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    scratch/generate_exp471_codemap_antigravity.py --check
  git diff --check
  git add experiments/exp471_exp467_evaluation_recovery_antigravity \
    scratch/generate_exp471_codemap_antigravity.py docs/codemap/codemap_exp471_antigravity.*
  git commit -m "fix: preflight Exp471 baseline abstentions"
  ```

---

### Task 2: Make Cache Contracts a Strict Scored/Abstained Union

**Files:**

- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/contracts_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_config_contracts_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_cache_artifacts_antigravity.py`
- Regenerate the three `docs/codemap/codemap_exp471_antigravity.*` files.

**Contract change:**

```python
@dataclass(frozen=True, slots=True)
class Exp471ScoreShardAntigravity:
    source_authority_sha256: str
    cache_context_sha256: str
    method: str
    dataset: str
    status: Literal["scored", "abstained"]
    test_count: int
    test_indices: tuple[int, ...]
    raw_scores: tuple[float, ...]
    micro_scores: tuple[float, ...]
    abstention_reason: str | None
    source_manifest_sha256: str
    payload_sha256: str
    envelope_sha256: str
```

`scored` requires finite vectors of `test_count`, exact TEST indices, and no
reason. `abstained` requires empty vectors, exact TEST indices, and reason exactly
`insufficient_train_rows`. The cache manifest adds `scored_shards`,
`abstained_shards`, and `failed_shards`; their sum must equal `total_shards`, and
the only complete production counts are 10,775, 395, and 0.

- [ ] **Step 1: Write RED contract tests.** Require union exclusivity, immutable
  TEST indices, canonical bytes for each arm, changed status/reason/index digest
  rejection, and manifest arithmetic. Explicitly reject `status="failed"`, a
  scored shard with a reason, and an abstain shard with any score value.

- [ ] **Step 2: Run RED.**

  ```sh
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m unittest -v \
    experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_config_contracts_antigravity \
    experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_cache_artifacts_antigravity
  ```

- [ ] **Step 3: Implement v2 shard and manifest serialization.** Change the
  canonical schema identifiers, strict JSON readers, create-only publishers,
  reopen validators, manifest builder, and status scanner together. Old v1
  shards must not be silently treated as typed v2 shards.

- [ ] **Step 4: Verify exact-byte reuse for both arms.** Publish and reopen one
  scored shard and one abstained shard twice. The second operation must reuse
  exact bytes; conflicting bytes at the same identity must fail closed.

- [ ] **Step 5: Run focused tests, regenerate/check the triplet, run
  `git diff --check`, and commit.**

  ```sh
  git commit -m "feat: type Exp471 score cache abstentions"
  ```

---

### Task 3: Skip Abstentions in Workers and Resume Both Shard Types

**Files:**

- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_score_cache_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py`
- Regenerate the three `docs/codemap/codemap_exp471_antigravity.*` files.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class _ScoreTaskAntigravity:
    source_authority_sha256: str
    cache_context_sha256: str
    method: str
    dataset: str
    test_count: int
    test_indices: tuple[int, ...]
    raw_scores: tuple[float, ...]
    micro_scores: tuple[float, ...]
    source_manifest_sha256: str

def _abstained_shard_antigravity(
    metadata: _VerifiedScoreMetadataAntigravity,
    method: str,
    dataset: str,
) -> Exp471ScoreShardAntigravity:
    raise NotImplementedError
```

- [ ] **Step 1: Write RED preparation tests.** Require that a typed abstention
  never calls `upper_tail_pvalues_numba_antigravity`, never enters
  `ProcessPoolExecutor`, is published durably with progress status `abstained`,
  and is not retried. Require a malformed source failure to remain terminal.

- [ ] **Step 2: Add resume tests.** Prepublish one scored and one abstained shard,
  rerun preparation, and assert that neither identity is recalibrated or
  republished. Interrupt after a mixed partial wave and require byte-identical
  completion on restart.

- [ ] **Step 3: Run RED.**

  ```sh
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m unittest -v \
    experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_score_cache_antigravity
  ```

- [ ] **Step 4: Implement typed metadata and scheduling.** Parse v3 baseline
  records into scored vectors or typed abstentions. Publish missing abstention
  shards in the parent process first; construct worker tasks only for missing
  scored identities. Benchmark and worker selection operate only on scored
  tasks. Preserve the current one-retry rule only for actual scored-task errors.

- [ ] **Step 5: Finalize only complete typed coverage.** Reopen all 11,170
  shards, compute counts from durable status—not expected constants alone—and
  publish the manifest only when counts are 10,775/395/0 and every identity is
  unique. Extend status/progress with scored, abstained, failed, remaining,
  elapsed, and ETA values.

- [ ] **Step 6: Prove worker parity.** A synthetic mixed cache prepared with one
  worker and the policy worker count must produce identical shard and terminal
  manifest bytes. The acceleration record remains Numba/CPU; PyTorch, MLX, and
  MPS are assessed as not tensor-eligible for this schema/routing change, so no
  new device benchmark is required.

- [ ] **Step 7: Run focused regressions, regenerate/check the triplet, run
  `git diff --check`, and commit.**

  ```sh
  git commit -m "fix: resume Exp471 typed score coverage"
  ```

---

### Task 4: Evaluate Full Population, Scored Support, and Common Support

**Files:**

- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_evaluation_antigravity.py`
- Regenerate the three `docs/codemap/codemap_exp471_antigravity.*` files.

**Evaluation payload v2:** The exact top-level keys are `schema`,
`source_authority`, `score_cache`, `implementation`, `coverage`,
`decision_tables`, `continuous_tables`, `curve_tables`, and
`support_authority`. `coverage` contains one row per frozen method with integer
`population`, `scored`, `abstained`, and `failed` fields, a finite `coverage`
ratio, and an `abstention_reasons` mapping. `decision_tables` contains the
`full_population` dataset and summary rows. `continuous_tables` and
`curve_tables` each contain `scored_support` and `all_ten_common_support`.
`support_authority` contains `all_ten_common_support_count=922`, the ordered
dataset list, and its lowercase SHA-256 digest.

- [ ] **Step 1: Write RED cache-gate tests.** A cache with 11,169 identities, a
  nonzero failed count, or an invalid abstention must not call the label loader.
  A complete 11,170-identity typed cache may call it exactly once.

- [ ] **Step 2: Write RED pure-metric tests.** Use fixed arrays to require:

  - abstention produces an empty alert set and valid TP/FP/FN/TN;
  - abstention AP, AUROC, ROC, and PR are semantic null with reason
    `insufficient_train_rows`;
  - scored AP/AUROC equal the unchanged Exp450 functions exactly;
  - proposal threshold is `raw_score >= 0.5`;
  - baseline alpha decisions use the frozen alpha set;
  - matched budget uses Candidate C's alert count and the evaluated method's own
    raw scores;
  - pairwise statistics record their ordered scored intersection;
  - all-ten common support contains exactly 922 production datasets;
  - ROC axes are false-positive rate / true-positive rate and PR axes are recall /
    precision.

- [ ] **Step 3: Run RED.**

  ```sh
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m unittest -v \
    experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_evaluation_antigravity
  ```

- [ ] **Step 4: Implement typed cache reconstruction.** Return score maps only
  for scored identities plus a complete status/reason map. Validate each shard's
  TEST indices against the cache authority. Compute the all-ten support as the
  ordered dataset intersection and bind both its list and digest.

- [ ] **Step 5: Reuse Exp450 continuous functions without modifying Exp450.**
  Call `metric_tables_exp450_antigravity` and
  `curve_tables_exp450_antigravity` on the scored maps. Call them again after
  filtering every method and label map to the 922 all-ten datasets. Retain raw
  in-memory ROC infinity and the existing JSON-null artifact-view rule.

- [ ] **Step 6: Add the decision layer.** Compute threshold and matched-budget
  confusion on all 1,117 datasets. For abstained rows, pass no selected indices.
  Store dataset rows, summed micro counts, derived precision/recall/F1/FPR,
  support count, alert count, and coverage. Do not add pooled continuous ranking
  as a confirmatory claim.

- [ ] **Step 7: Strengthen artifact reopen validation.** Validate the v2 payload,
  exact denominators, support lists/digests, null placement, method order, curve
  method blocks, and cache scored/abstained/failed counts before reporting.

- [ ] **Step 8: Run focused regressions, regenerate/check the triplet, run
  `git diff --check`, and commit.**

  ```sh
  git commit -m "feat: evaluate Exp471 typed abstain supports"
  ```

---

### Task 5: Render Exp471-Native Paper Tables and Both Curve Sets

**Files:**

- Create:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/typed_abstain_reporting_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py`
- Modify: `scratch/generate_exp471_codemap_antigravity.py`
- Regenerate the three `docs/codemap/codemap_exp471_antigravity.*` files.

**Renderer interface:**

```python
@dataclass(frozen=True, slots=True)
class Exp471TypedReportArtifactAntigravity:
    root: Path
    leaves: Mapping[str, str]  # relative path -> sha256

def write_typed_exp471_report_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    root: Path,
) -> Exp471TypedReportArtifactAntigravity:
    raise NotImplementedError
```

- [ ] **Step 1: Write RED report tests.** Require canonical CSV/JSON/HTML plus
  SVG and PNG ROC/PR figures for both scored support and all-ten common support.
  Require every method's population/scored/abstained/failed/coverage values,
  explicit AP/AUROC denominators, reason counts, full-population decision
  metrics, and the common-support count 922.

- [ ] **Step 2: Add wording and axis tests.** The report must state that
  abstention is a preregistered no-alert outcome, not a runtime failure or a
  continuous score. Reject wording that claims score completion, imputation,
  automatic promotion, or historical parity. Inspect serialized figure metadata
  and SVG labels to require ROC x=`false_positive_rate`, y=`true_positive_rate`,
  and PR x=`recall`, y=`precision` for all ten methods.

- [ ] **Step 3: Run RED.**

  ```sh
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m unittest -v \
    experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_reporting_cli_antigravity
  ```

- [ ] **Step 4: Implement the renderer.** Render directly from the fully
  validated Exp471 v2 artifact. Do not reconstruct an Exp450 report object.
  Reuse only pure formatting/curve helpers whose output semantics remain exact.
  Produce these create-only leaves under
  `exp471_typed_report_artifacts_antigravity/`:

  - `exp471_coverage_antigravity.csv` and `.json`;
  - `exp471_full_population_decisions_antigravity.csv` and `.json`;
  - `exp471_scored_support_metrics_antigravity.csv` and `.json`;
  - `exp471_common_support_metrics_antigravity.csv` and `.json`;
  - `exp471_scored_support_curves_antigravity.svg` and `.png`;
  - `exp471_common_support_curves_antigravity.svg` and `.png`;
  - `exp471_paper_report_antigravity.html`;
  - `exp471_typed_report_manifest_antigravity.json`.

- [ ] **Step 5: Keep atomic outer publication.** Update
  `write_exp471_report_antigravity` to accept only the typed renderer result,
  verify every leaf digest, add provenance/artifact map/complete marker, fsync the
  stage, and retain the existing no-replace rename and retry behavior.

- [ ] **Step 6: Run focused regressions, regenerate/check the triplet, run
  `git diff --check`, and commit.**

  ```sh
  git commit -m "feat: report Exp471 typed abstain results"
  ```

---

### Task 6: Close the CLI, Status, and Operator Workflow

**Files:**

- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/cli_antigravity.py`
- Modify:
  `experiments/exp471_exp467_evaluation_recovery_antigravity/tests/test_reporting_cli_antigravity.py`
- Modify: `docs/EXP471_EVALUATION_RECOVERY_RUNBOOK_antigravity_KO.md`
- Regenerate the three `docs/codemap/codemap_exp471_antigravity.*` files.

- [ ] **Step 1: Write RED end-to-end synthetic tests.** Require the call order
  source verification → baseline preflight → proposal preprocessing → typed cache
  → single label read → typed evaluation → typed report. An invalid baseline must
  stop before proposal preprocessing; an existing durable typed cache must skip
  both preflight score preparation and proposal preprocessing where authority can
  be reopened safely.

- [ ] **Step 2: Extend `status`.** Report preprocessing progress, scored shards,
  abstained shards, failed shards, remaining identities, cache completeness,
  evaluation completeness, report completeness, elapsed time, rate, and ETA.
  Ensure an abstention increments durable completion and never the failure count.

- [ ] **Step 3: Update the Korean runbook.** Include the official command, live
  log command, status command, expected 10,775/395/0/11,170 counts, expected
  922 common support, artifact locations, safe resume behavior, and the exact
  meaning of a typed abstention.

- [ ] **Step 4: Run the CLI/report suite, regenerate/check the triplet, run
  `git diff --check`, and commit.**

  ```sh
  git commit -m "docs: close Exp471 typed recovery workflow"
  ```

---

### Task 7: Verify, Review, Integrate, and Restart Once

**Files:** No new production file. Integration changes only the isolated and
official Git branches; official result creation occurs only after every gate
below passes.

- [ ] **Step 1: Run the complete isolated verification suite.**

  ```sh
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m unittest discover -v \
    -s experiments/exp471_exp467_evaluation_recovery_antigravity/tests \
    -p 'test_*_antigravity.py'
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    -m compileall -q \
    experiments/exp471_exp467_evaluation_recovery_antigravity \
    scratch/run_exp471_exp467_evaluation_recovery_antigravity.py \
    scratch/generate_exp471_codemap_antigravity.py
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    scratch/generate_exp471_codemap_antigravity.py --check
  git diff --check
  git status --short
  ```

  Expected: all tests pass; compilation and code-map check exit zero; no
  unexpected generated or official-result files appear in Git status.

- [ ] **Step 2: Run a production-root read-only preflight.** Use the official
  interpreter to verify the three pinned baseline roots and print only aggregate
  counts and authority digests. Do not load TEST labels and do not create cache,
  evaluation, or report files. Require 10,775 scored, 395 abstained, zero failed,
  11,170 total, and 922 all-ten common support.

- [ ] **Step 3: Perform fresh code review.** Review the complete diff against the
  approved design, with special attention to score fabrication, label timing,
  TEST identity, denominator drift, ROC/PR axes, create-only behavior, and the
  official restart boundary. Resolve every high/medium finding and rerun Step 1.

- [ ] **Step 4: Integrate the verified commits into the official Exp471 branch.**
  Confirm no official process is running, confirm the official branch still
  points at the expected Numba commit lineage, then integrate without touching
  frozen roots. Rerun the complete suite in the official worktree.

- [ ] **Step 5: Restart the official `report` command once.** Preserve existing
  failed logs as evidence and write to the existing create-only Exp471 root.
  Monitor the preflight and preprocessing/cache progress logs. Do not restart a
  second time automatically if the typed authority, label gate, or report
  publication fails; capture the exact failure and stop.

- [ ] **Step 6: Verify final acceptance.** Reopen and independently validate:

  - cache manifest: 11,170 total, 10,775 scored, 395 abstained, 0 failed;
  - evaluation: 1,117 full-population datasets and 922 all-ten common-support
    datasets, explicit denominators, correct semantic nulls and curve axes;
  - report: all declared leaves exist, every SHA-256 matches, the completion
    marker binds the evaluation digest, and the Korean wording is accurate;
  - runtime: no official process remains after successful completion.

- [ ] **Step 7: Commit only any verification documentation that is intentionally
  tracked.** Do not commit Dataset-root outputs. Do not push or merge externally
  without separate user approval.

## Plan Self-Review

- Every requirement in the approved typed-abstain design is mapped to a task and
  an executable verification gate.
- The plan never reruns baselines, never invents abstention scores, and keeps TEST
  labels behind complete 11,170-identity cache authority.
- Continuous calculations reuse unchanged Exp450 functions; new code is limited
  to status routing, support selection, decision metrics, and reporting.
- Full-population, per-method scored support, pairwise support, and exact 922
  all-ten support are named separately, preventing silent denominator changes.
- Baseline preflight occurs before the known 10-minute proposal preprocessing
  stage, preventing the same late failure.
- PyTorch/MLX/MPS are explicitly assessed and not used because this repair is not
  tensor-eligible; the verified Numba/CPU path remains canonical.
- The official experiment is restarted only after isolated tests, read-only
  production preflight, fresh review, integration, and official-worktree tests.
