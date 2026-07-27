# Exp400-402 Exp397 Shadow Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze Exp397 review rankings into a label-free retrospective Shadow artifact, evaluate the frozen artifact separately, and expose the comparison through the existing protected `/shadow` dashboard without changing automatic alerts or existing Exp265 data.

**Architecture:** A new Python package builds one immutable Shadow replay row per dataset from verified Exp292 and Exp397 artifacts. A separate evaluator loads TEST labels only after checking the prediction SHA256. A separate Supabase table and protected API expose ranking evidence beside, rather than inside, the existing Exp265 ledger so backward compatibility and policy boundaries remain explicit.

**Tech Stack:** Python 3 standard library, existing NumPy/SQLite dataset readers, unittest, Next.js 16, TypeScript 5.9, Supabase PostgreSQL/RLS, Vitest.

## Global Constraints

- Exp291 automatic Hard alerts remain exactly 2,218.
- Exp292 Standard review candidates remain exactly 806.
- Existing Priority review candidates remain exactly 31 and are not ranking targets.
- Existing Exp265 data, policy versions, tables, APIs, and dashboard modes are not overwritten.
- Prediction artifacts contain no TEST labels, anomaly positions, TP, FP, FN, precision, recall, F1, or combined F1.
- `ALPHAS=(0.005, 0.01, 0.02, 0.05)` are all recorded; TEST metrics select none of them.
- The visible Shadow comparison includes `exp292_baseline` and `tier_preserving_grammar`.
- `additive_grammar_family` and `grammar_only_control` remain research-only controls.
- Priority review is never promoted to autonomous Hard alert.
- Tail replacement is prohibited.
- Every result is marked `retrospective=true`, `prospective=false`, and `deployment_validated=false`.
- Dataset names are lookup keys only and do not change ranking policy.
- Full coverage is 1,117 datasets, including datasets with zero Standard candidates.

---

## File Structure

### Python experiment package

- `experiments/exp400_402/__init__.py`: package marker and experiment IDs.
- `experiments/exp400_402/contracts.py`: constants, forbidden fields, typed row validation, canonical SHA256.
- `experiments/exp400_402/artifact_io.py`: verified Exp292/Exp397 readers and atomic JSONL/manifest writers.
- `experiments/exp400_402/prediction.py`: build 1,117 label-free dataset rows and 806 candidate ranking records.
- `experiments/exp400_402/evaluation.py`: read labels only after prediction verification and calculate fixed workload metrics.
- `experiments/exp400_402/independent_reaggregate.py`: independent consistency calculation.
- `experiments/exp400_402/reporting.py`: Exp401 report and Exp402 readiness report.
- `experiments/exp400_402/run_pipeline.py`: `predict`, `evaluate`, `audit`, and `all` CLI stages.
- `experiments/exp400_402/test_*.py`: focused unit, boundary, pipeline, and invariance tests.

### Dashboard and database

- `vercel-dashboard/supabase/migrations/202607270004_exp397_shadow_ranking.sql`: isolated ranking run/candidate tables, RLS, and immutable ingest RPC.
- `vercel-dashboard/src/lib/shadow-ranking.ts`: TypeScript ranking payload contract and hash validation.
- `vercel-dashboard/src/lib/shadow-ranking-data.ts`: paginated Supabase query and view model construction.
- `vercel-dashboard/app/api/shadow/rankings/ingest/route.ts`: authenticated machine ingestion.
- `vercel-dashboard/app/api/shadow/rankings/route.ts`: authenticated ranking query.
- `vercel-dashboard/app/ui/shadow-ranking-comparison.tsx`: Exp292 versus Exp397 comparison UI.
- `vercel-dashboard/app/ui/shadow-validation.tsx`: add a ranking comparison subview without changing Exp265 modes.
- `vercel-dashboard/app/shadow/shadow.css`: compact ranking table and evidence styles.
- `vercel-dashboard/tests/shadow-ranking-*.test.ts`: contract, migration, API, data, and view tests.

### Operations and registration

- `scratch/run_exp400_402_shadow_validation.py`: repository-root-safe CLI wrapper.
- `scratch/upload_exp400_shadow_replay.py`: enqueue and idempotently upload the frozen artifact.
- `run_rank_experiments_sequential.py`: register one sequential Exp400-402 queue item.
- `serve_rank_dashboard.py`: register output paths and completed summaries.
- `docs/EXP400_402_RUNBOOK_KO.md`: replay, upload, dashboard, and prospective-blocker instructions.

---

### Task 1: Freeze the Python Shadow Contract

**Files:**
- Create: `experiments/exp400_402/__init__.py`
- Create: `experiments/exp400_402/contracts.py`
- Create: `experiments/exp400_402/test_contracts.py`

**Interfaces:**
- Consumes: Exp397 row fields `dataset`, `candidate_index`, `variant`, `alpha`, `rank`, `exp292_tier`, `grammar_support_count`, `p_values`.
- Produces: `validate_shadow_row(row: Mapping[str, object]) -> dict[str, object]`, `prediction_sha256(rows: Sequence[Mapping[str, object]]) -> str`, and constants `ALPHAS`, `VISIBLE_VARIANTS`, `CONTROL_VARIANTS`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_prediction_rejects_test_metrics():
    row = valid_dataset_row()
    row["f1"] = 0.9
    with self.assertRaisesRegex(ValueError, "forbidden"):
        validate_shadow_row(row)

def test_dataset_without_candidates_is_valid():
    row = valid_dataset_row()
    row["candidates"] = []
    self.assertEqual(validate_shadow_row(row)["candidates"], [])
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
python3 -m unittest experiments.exp400_402.test_contracts -v
```

Expected: import failure because `contracts.py` does not exist.

- [ ] **Step 3: Implement exact field validation and canonical hashing**

The top-level prediction row must contain:

```python
{
    "schema_version": 1,
    "collection_mode": "retrospective_replay",
    "policy_version": "exp397-review-ranking-replay-v1",
    "run_id": str,
    "dataset": str,
    "n_test": int,
    "automatic_hard_count": int,
    "priority_review_count": int,
    "standard_review_count": int,
    "candidates": list,
    "contains_test_labels": False,
    "contains_evaluation_metrics": False,
    "automatic_policy_changed": False,
    "retrospective": True,
    "prospective": False,
}
```

Each candidate must contain the candidate index, Exp292 tier/global rank, and
all four alpha-specific tier-preserving ranks and grammar evidence. Reject
unknown forbidden keys recursively.

- [ ] **Step 4: Run the contract tests**

Run:

```bash
python3 -m unittest experiments.exp400_402.test_contracts -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add experiments/exp400_402
git commit -m "feat: define Exp400 shadow ranking contract"
```

---

### Task 2: Build and Freeze Exp400 Predictions

**Files:**
- Create: `experiments/exp400_402/artifact_io.py`
- Create: `experiments/exp400_402/prediction.py`
- Create: `experiments/exp400_402/test_prediction.py`
- Create: `experiments/exp400_402/test_artifact_io.py`

**Interfaces:**
- Consumes: `load_exp292_rows()` and `load_verified_ranking()` from existing experiment modules.
- Produces: `build_shadow_predictions(exp292_rows, exp397_rows) -> list[dict[str, object]]` and `write_prediction(output_dir, rows, metadata) -> dict[str, object]`.

- [ ] **Step 1: Write failing coverage and invariance tests**

```python
def test_build_preserves_candidate_universe_and_frozen_lanes():
    rows = build_shadow_predictions(exp292_fixture(), exp397_fixture())
    self.assertEqual(sum(len(row["candidates"]) for row in rows), 3)
    self.assertEqual(sum(row["automatic_hard_count"] for row in rows), 2)
    self.assertEqual(sum(row["priority_review_count"] for row in rows), 1)

def test_alpha_changes_rank_not_candidate_universe():
    rows = build_shadow_predictions(exp292_fixture(), exp397_fixture())
    candidate = rows[0]["candidates"][0]
    self.assertEqual(set(candidate["tier_preserving_rank"]), {"0.005", "0.01", "0.02", "0.05"})
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m unittest experiments.exp400_402.test_prediction experiments.exp400_402.test_artifact_io -v
```

Expected: missing implementation failures.

- [ ] **Step 3: Implement verified input loading**

Verify:

- Exp292 prediction SHA256 against its manifest.
- Exp397 prediction SHA256 against its manifest.
- Exp397 manifest declares no labels and no evaluation metrics.
- Exp397 contains all four alpha values and all four variants.
- Every variant/alpha has the same 806 candidate keys.

- [ ] **Step 4: Implement one row per dataset**

Create all 1,117 dataset rows, including zero-candidate datasets. Derive
`run_id` from the dataset key:

```python
run_id = "replay-exp397-" + hashlib.sha256(dataset.encode()).hexdigest()[:20]
```

For every Standard candidate, preserve the Exp292 global rank and tier and add
alpha-keyed tier-preserving ranks. Do not carry evaluated Exp398 fields.

- [ ] **Step 5: Implement atomic JSONL and manifest writing**

The manifest must include:

```python
{
    "experiment_id": "experiment_400_exp397_shadow_prediction",
    "datasets": 1117,
    "standard_review_candidates": 806,
    "automatic_hard_count": 2218,
    "priority_review_count": 31,
    "prediction_sha256": "...",
    "contains_test_labels": False,
    "contains_evaluation_metrics": False,
    "test_metrics_select_policy": False,
    "retrospective": True,
    "prospective": False,
    "deployment_validated": False,
}
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
python3 -m unittest experiments.exp400_402.test_prediction experiments.exp400_402.test_artifact_io -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Exp400 prediction**

```bash
git add experiments/exp400_402
git commit -m "feat: freeze Exp400 shadow rankings"
```

---

### Task 3: Add Exp401 Separate Evaluation

**Files:**
- Create: `experiments/exp400_402/evaluation.py`
- Create: `experiments/exp400_402/test_evaluation.py`

**Interfaces:**
- Consumes: verified Exp400 prediction JSONL and manifest.
- Produces: `evaluate_prediction(output_dir: Path) -> dict[str, object]`, `exp401_candidate_evaluation.csv`, `exp401_precision_at_k.csv`, and `exp401_dataset_top1.csv`.

- [ ] **Step 1: Write a hash-boundary failing test**

```python
def test_evaluation_refuses_modified_prediction(tmp_path):
    write_valid_prediction(tmp_path)
    with (tmp_path / PREDICTION_NAME).open("a") as handle:
        handle.write("{}\n")
    with self.assertRaisesRegex(ValueError, "SHA256"):
        evaluate_prediction(tmp_path)
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
python3 -m unittest experiments.exp400_402.test_evaluation -v
```

Expected: missing evaluation implementation.

- [ ] **Step 3: Implement label loading after hash verification**

Call the existing read-only TEST label loader only after `load_prediction()`
has checked manifest and file hashes. Keep labels only in Exp401 evaluation
CSV; never rewrite Exp400 JSONL.

- [ ] **Step 4: Calculate pre-registered workloads**

Calculate for Exp292 and each tier-preserving alpha:

- global K = 25, 50, 100, 200;
- per-dataset top-1;
- reviewed, TP, FP, precision and review TP capture rate;
- exact candidate universe equality;
- rank movement and Spearman correlation.

Also report controls without selecting them.

- [ ] **Step 5: Run the evaluation tests**

Run:

```bash
python3 -m unittest experiments.exp400_402.test_evaluation -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Exp401 evaluation**

```bash
git add experiments/exp400_402
git commit -m "feat: evaluate Exp401 frozen shadow replay"
```

---

### Task 4: Add Independent Reaggregation, Reports, and Pipeline

**Files:**
- Create: `experiments/exp400_402/independent_reaggregate.py`
- Create: `experiments/exp400_402/reporting.py`
- Create: `experiments/exp400_402/run_pipeline.py`
- Create: `experiments/exp400_402/test_pipeline.py`
- Create: `experiments/exp400_402/test_invariance.py`
- Create: `scratch/run_exp400_402_shadow_validation.py`

**Interfaces:**
- Consumes: Tasks 1-3 functions.
- Produces: CLI stages `predict`, `evaluate`, `audit`, `all`; status JSON; readiness report; independent consistency JSON.

- [ ] **Step 1: Write failing smoke and invariance tests**

Tests must verify:

- prediction is identical after TEST label permutation;
- dataset input order does not change sorted output hash;
- all alpha tracks preserve the same candidate keys;
- automatic Hard and Priority totals do not change;
- prediction rows contain no forbidden evaluation fields;
- independent and official metrics match exactly.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
python3 -m unittest experiments.exp400_402.test_pipeline experiments.exp400_402.test_invariance -v
```

- [ ] **Step 3: Implement status and CLI stages**

Default output:

```text
/Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation
```

Write `status.json` atomically at stage start and completion. Direct execution
must insert the repository root into `sys.path`.

- [ ] **Step 4: Implement Exp402 readiness audit**

Write `exp402_readiness_report.json` and `EXP402_READINESS_REPORT.md` with:

- coverage and candidate integrity;
- label/evaluation field scan;
- deterministic hash result;
- current prospective run count `0`;
- `prospective_upstream_available=false`;
- `prospective_validation_completed=false`;
- explicit blocker: no new incoming run data.

- [ ] **Step 5: Run the full Python suite**

Run:

```bash
python3 -m py_compile experiments/exp400_402/*.py scratch/run_exp400_402_shadow_validation.py
python3 -m unittest discover -s experiments/exp400_402 -p 'test_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the pipeline**

```bash
git add experiments/exp400_402 scratch/run_exp400_402_shadow_validation.py
git commit -m "feat: add Exp400-402 shadow validation pipeline"
```

---

### Task 5: Add the Isolated Supabase Ranking Ledger

**Files:**
- Create: `vercel-dashboard/supabase/migrations/202607270004_exp397_shadow_ranking.sql`
- Create: `vercel-dashboard/tests/shadow-ranking-migration.test.ts`

**Interfaces:**
- Consumes: Exp400 dataset and candidate payloads.
- Produces: immutable `shadow_ranking_runs`, `shadow_ranking_candidates`, and `ingest_shadow_ranking_prediction(jsonb)`.

- [ ] **Step 1: Write the failing migration contract test**

The test must require:

```typescript
expect(sql).toContain("create table public.shadow_ranking_runs");
expect(sql).toContain("create table public.shadow_ranking_candidates");
expect(sql).toContain("exp397-review-ranking-replay-v1");
expect(sql).toContain("prediction hash conflict");
expect(sql).toContain("enable row level security");
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
cd vercel-dashboard
npm test -- --run tests/shadow-ranking-migration.test.ts
```

- [ ] **Step 3: Implement tables, constraints, RLS, and RPC**

Required invariants:

- only `retrospective_replay` is accepted for this policy version;
- `contains_test_labels=false`;
- `contains_evaluation_metrics=false`;
- `automatic_policy_changed=false`;
- candidate uniqueness on `(run_id, candidate_index)`;
- authenticated dashboard members can read;
- only permitted writer sessions can call ingest;
- authenticated users cannot update or delete prediction rows;
- same run/hash is idempotent;
- same run/different hash raises `prediction hash conflict`.

- [ ] **Step 4: Run migration tests**

Run:

```bash
npm test -- --run tests/shadow-ranking-migration.test.ts
```

- [ ] **Step 5: Commit the migration**

```bash
git add vercel-dashboard/supabase/migrations/202607270004_exp397_shadow_ranking.sql vercel-dashboard/tests/shadow-ranking-migration.test.ts
git commit -m "feat: add Exp397 shadow ranking ledger"
```

---

### Task 6: Add Dashboard Contracts and APIs

**Files:**
- Create: `vercel-dashboard/src/lib/shadow-ranking.ts`
- Create: `vercel-dashboard/src/lib/shadow-ranking-data.ts`
- Create: `vercel-dashboard/app/api/shadow/rankings/ingest/route.ts`
- Create: `vercel-dashboard/app/api/shadow/rankings/route.ts`
- Create: `vercel-dashboard/tests/shadow-ranking-contract.test.ts`
- Create: `vercel-dashboard/tests/shadow-ranking-api.test.ts`
- Create: `vercel-dashboard/tests/shadow-ranking-data.test.ts`

**Interfaces:**
- Consumes: Exp400 JSON payload.
- Produces: `assertValidShadowRankingPrediction`, `loadShadowRankingRows`, authenticated ingest and read endpoints.

- [ ] **Step 1: Write failing contract and API tests**

Tests must cover:

- forbidden TEST labels and metrics;
- exact four alpha keys;
- duplicate/out-of-range candidates;
- 1,117-row pagination;
- unauthenticated read and write return `401`;
- hash conflict returns `409`;
- query returns no evaluation metrics.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
cd vercel-dashboard
npm test -- --run tests/shadow-ranking-contract.test.ts tests/shadow-ranking-api.test.ts tests/shadow-ranking-data.test.ts
```

- [ ] **Step 3: Implement strict payload validation**

The TypeScript contract must use the same canonical JSON key sorting as the
Python contract and verify `prediction_sha256` before RPC ingestion.

- [ ] **Step 4: Implement authenticated API routes**

Reuse existing bearer-token and Supabase session patterns. Do not use the
service-role key in browser code or route responses.

- [ ] **Step 5: Run API and data tests**

Run:

```bash
npm test -- --run tests/shadow-ranking-contract.test.ts tests/shadow-ranking-api.test.ts tests/shadow-ranking-data.test.ts
```

- [ ] **Step 6: Commit the contracts and APIs**

```bash
git add vercel-dashboard/src/lib/shadow-ranking*.ts vercel-dashboard/app/api/shadow/rankings vercel-dashboard/tests/shadow-ranking-*.test.ts
git commit -m "feat: add Exp397 shadow ranking APIs"
```

---

### Task 7: Add the Exp397 Shadow Comparison UI

**Files:**
- Create: `vercel-dashboard/app/ui/shadow-ranking-comparison.tsx`
- Modify: `vercel-dashboard/app/ui/shadow-validation.tsx`
- Modify: `vercel-dashboard/app/shadow/shadow.css`
- Create: `vercel-dashboard/tests/shadow-ranking-view.test.ts`

**Interfaces:**
- Consumes: `GET /api/shadow/rankings`.
- Produces: a protected Exp397 comparison subview within `/shadow`.

- [ ] **Step 1: Write the failing view-model test**

The test must require:

- default comparison is Exp292 versus tier-preserving grammar;
- all four alpha tracks are selectable without a “best” label;
- rank movement is shown as up, unchanged, or down;
- research controls are collapsed by default;
- copy states `과거 데이터 재생 · 실제 운영 검증 아님`;
- no precision, recall, or F1 appears without a separate evaluation artifact.

- [ ] **Step 2: Run the view test and confirm failure**

Run:

```bash
cd vercel-dashboard
npm test -- --run tests/shadow-ranking-view.test.ts
```

- [ ] **Step 3: Implement the compact comparison view**

Use an existing segmented-control pattern to switch between:

- Exp265 후보 검토;
- Exp397 순위 비교.

Use a table, not nested cards, for dataset, candidate index, Exp292 tier/rank,
Shadow rank, rank movement, grammar support and p-value. Keep filters for
alpha, dataset, tier, and movement direction.

- [ ] **Step 4: Run type, test, and build checks**

Run:

```bash
cd vercel-dashboard
npm test
npx tsc --noEmit
npm run build
```

Expected: all commands pass.

- [ ] **Step 5: Commit the UI**

```bash
git add vercel-dashboard/app/ui/shadow-ranking-comparison.tsx vercel-dashboard/app/ui/shadow-validation.tsx vercel-dashboard/app/shadow/shadow.css vercel-dashboard/tests/shadow-ranking-view.test.ts
git commit -m "feat: show Exp397 shadow ranking comparison"
```

---

### Task 8: Register the Queue and Operational Runbook

**Files:**
- Create: `scratch/upload_exp400_shadow_replay.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Create: `docs/EXP400_402_RUNBOOK_KO.md`
- Create: `scratch/test_exp400_402_queue.py`

**Interfaces:**
- Consumes: Exp400-402 pipeline and dashboard ingest API.
- Produces: queue ID `experiment_400_402_exp397_shadow_validation`, uploader, local progress registration, Korean runbook.

- [ ] **Step 1: Write the failing queue registration test**

Require:

- exactly one queue command;
- one active process for the batch;
- output root points to `exp400_402_exp397_shadow_validation`;
- dashboard status uses `status.json`;
- no automatic policy promotion metadata.

- [ ] **Step 2: Run the queue test and confirm failure**

Run:

```bash
python3 scratch/test_exp400_402_queue.py
```

- [ ] **Step 3: Register the batch and uploader**

The uploader must:

- verify the Exp400 manifest and prediction SHA256;
- enqueue every row in a local SQLite spool;
- support `--enqueue-only`, `--once`, and retry;
- obtain secrets only from environment variables;
- never log access tokens;
- leave prospective progress at zero.

- [ ] **Step 4: Write the runbook**

Document:

- local smoke and full replay commands;
- Supabase dry-run and migration commands;
- Vercel build and deployment checks;
- authenticated replay upload;
- idempotency and conflict smoke tests;
- how to verify 1,117 runs and 806 candidates;
- why this is retrospective;
- prospective blocker and activation checklist.

- [ ] **Step 5: Run registration tests**

Run:

```bash
python3 scratch/test_exp400_402_queue.py
python3 -m py_compile scratch/upload_exp400_shadow_replay.py
git diff --check
```

- [ ] **Step 6: Commit registration and operations**

```bash
git add scratch/upload_exp400_shadow_replay.py scratch/test_exp400_402_queue.py run_rank_experiments_sequential.py serve_rank_dashboard.py docs/EXP400_402_RUNBOOK_KO.md
git commit -m "chore: register Exp400-402 shadow validation"
```

---

### Task 9: Execute Smoke, Full Replay, and Independent Verification

**Files:**
- Generate only under: `/Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation`

**Interfaces:**
- Consumes: complete Python implementation and frozen Exp397 artifacts.
- Produces: Exp400 prediction, Exp401 evaluation, Exp402 readiness report, status, logs, and manifests.

- [ ] **Step 1: Run a 10-dataset smoke**

```bash
python3 scratch/run_exp400_402_shadow_validation.py \
  --stage all \
  --dataset-limit 10 \
  --output-dir /Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation_smoke
```

Expected:

- 10 dataset rows;
- no forbidden prediction fields;
- prediction/evaluation separation;
- independent mismatch count 0.

- [ ] **Step 2: Run the full replay**

```bash
python3 scratch/run_exp400_402_shadow_validation.py \
  --stage all \
  --output-dir /Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation
```

Expected:

- 1,117 datasets;
- 806 Standard candidates;
- 2,218 automatic Hard unchanged;
- 31 Priority review unchanged;
- errors 0.

- [ ] **Step 3: Independently verify artifacts**

Run the independent reaggregator a second time and verify:

- official versus independent mismatch count 0;
- prediction hash unchanged;
- candidate universe exact match across alpha;
- TP+FP equals evaluated candidates;
- no lane or index errors.

- [ ] **Step 4: Run all regression checks**

```bash
python3 -m unittest discover -s experiments/exp400_402 -p 'test_*.py' -v
python3 -m unittest discover -s experiments/exp265_shadow_validation/tests -p 'test_*.py' -v
cd vercel-dashboard
npm test
npx tsc --noEmit
npm run build
```

- [ ] **Step 5: Record the execution manifest**

Record commands, git commit, input/output SHA256, Python/Node versions,
retrospective scope, autonomous/human-assisted scope, and known prospective
blocker in `MANIFEST.json` and `MANIFEST.md`.

---

### Task 10: Validate the Dashboard Without Claiming Prospective Success

**Files:**
- No source change unless a test exposes a defect.
- Generate deployment and validation logs under the Exp400-402 output directory.

**Interfaces:**
- Consumes: verified Exp400 replay and built Vercel dashboard.
- Produces: local dashboard smoke evidence and, when credentials are available, authenticated remote replay evidence.

- [ ] **Step 1: Start the local dashboard and verify the protected page**

Verify:

- unauthenticated `/api/shadow/rankings` returns `401`;
- signed-in `/shadow` opens Exp397 ranking comparison;
- 1,117 dataset runs and 806 candidates are visible after local/test ingestion;
- alpha switching changes only rank evidence;
- Exp265 view still loads unchanged.

- [ ] **Step 2: Run idempotency and hash-conflict smoke**

Upload one replay row twice and verify one stored run. Change one rank while
reusing the run ID and verify `409`.

- [ ] **Step 3: Perform remote upload only when existing credentials and migration state are valid**

Before upload:

- use Supabase migration dry-run;
- verify only the new migration is pending;
- verify Vercel environment variables exist without printing secret values;
- upload replay rows under `retrospective_replay`;
- never insert rows into `prospective_shadow`.

If any prerequisite is missing, write `BLOCKER_REPORT.md` and keep the verified
local artifacts. Do not weaken authentication or schema checks.

- [ ] **Step 4: Write final claims**

Classify each claim as verified, conditional, or prohibited:

- Exp397 replay is label-free during prediction.
- Automatic Hard and existing lanes are unchanged.
- Historical ranking evaluation was reproduced.
- Dashboard and ledger are ready to receive future label-free runs.
- Exp397 was prospectively validated.
- A particular alpha is operationally optimal.

The last two claims must remain prohibited.

- [ ] **Step 5: Final verification commit**

Commit only source, tests, docs, and small manifests intentionally tracked by
the repository. Do not commit the large database, replay JSONL, tokens, or
runtime logs.

```bash
git status --short
git diff --check
git log --oneline --decorate -12
```

