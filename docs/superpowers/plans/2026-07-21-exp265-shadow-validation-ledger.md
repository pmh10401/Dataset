# Exp265 Shadow Validation Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prospective shadow-validation ledger that preserves Exp253 Hard alerts, assigns A/B/C confidence, routes Exp264 evidence into Shadow/Priority/Standard review, records authenticated user feedback in Supabase, and reports progress over 100 new runs without changing the automatic alert policy.

**Architecture:** A label-free Python boundary validates an upstream run prediction, rebuilds confidence and review lanes with the frozen Exp264 policy, and stores an immutable payload in a local SQLite spool before delivery. A dedicated Vercel API validates the same schema and atomically stores runs and candidates in Supabase; append-only feedback events record candidate and run decisions. A separate `/shadow` dashboard reads the ledger and shows the 100-run progress, confidence, review priority, workload, and only those evaluation metrics justified by available feedback.

**Tech Stack:** Python 3.14 standard library, SQLite, Next.js 16.2.10, React 19.2.7, TypeScript 5.9.3, Supabase JS 2.110.7, Supabase SSR 0.12.3, PostgreSQL/RLS, Vitest 4.1.10.

## Global Constraints

- Existing Exp253 Hard alert indices are immutable; Exp265 must not add or remove autonomous alerts.
- `ALPHA = 0.005`, `MIN_TRAIN_FOR_SHADOW = 100`, required Shadow agreement `3`, and Priority agreement `2` are frozen before prospective collection.
- Fixed evidence sources are `statistical`, `structural`, `change_point`, and `reconstruction`.
- Shadow requires an existing Standard candidate, `n_train >= 100`, at least three agreeing sources, and no quality flag.
- Priority receives an existing Standard candidate with at least two agreeing sources that is not Shadow; all other existing Standard candidates remain Standard.
- Prediction and ingest payloads must not contain TEST labels, actual anomaly positions, TP, FP, FN, precision, recall, or F1.
- Dataset/family names may be lookup keys and display metadata only; they must not control threshold, source selection, confidence, or routing.
- New collection progress counts unique successfully ingested runs, including runs with zero candidates; it does not count candidates.
- Candidate feedback is append-only and limited to `true_anomaly`, `false_alarm`, or `deferred`.
- Run feedback is append-only and limited to `none`, `exists`, or `unknown` for missed anomalies; Recall eligibility additionally requires an exact `missed_anomaly_count` (`0` for `none`, positive for `exists`).
- Recall, FN, and F1 are hidden unless the report explicitly identifies fully annotated runs.
- Reaching 100 runs creates a report only; it never changes a threshold, source, lane, or automatic alert rule.
- Existing experiment status APIs, Vercel dashboard, Exp263/264 results, and Supabase tables must remain backward compatible.
- Supabase service-role credentials must never be sent to or bundled in the browser.
- No synthetic or historical TEST row may be counted toward the prospective 100-run total.

---

## File Map

### Python domain and delivery

- `experiments/exp265_shadow_validation/config.py`: frozen constants, schema version, source names, and local output paths.
- `experiments/exp265_shadow_validation/contracts.py`: strict label-free input and payload validation plus canonical hashing.
- `experiments/exp265_shadow_validation/routing.py`: pure A/B/C confidence and Shadow/Priority/Standard routing.
- `experiments/exp265_shadow_validation/spool.py`: append-only SQLite spool with idempotent run storage and delivery state.
- `experiments/exp265_shadow_validation/exporter.py`: authenticated HTTP delivery without storing secrets in artifacts.
- `experiments/exp265_shadow_validation/reporting.py`: read-only export of the 100-run ledger report.
- `scratch/run_exp265_shadow_exporter.py`: one-shot and polling CLI for validated upstream JSONL payloads.
- `scratch/export_exp265_shadow_report.py`: authenticated report download to JSON and Markdown.

### Supabase and Vercel

- `vercel-dashboard/supabase/migrations/202607210001_shadow_validation.sql`: immutable run/candidate tables, append-only feedback tables, RLS, indexes, views, and ingest RPC.
- `vercel-dashboard/src/lib/shadow.ts`: strict TypeScript contracts and payload validator.
- `vercel-dashboard/src/lib/shadow-metrics.ts`: feedback-derived metric eligibility and display model.
- `vercel-dashboard/app/api/shadow/ingest/route.ts`: machine ingestion boundary.
- `vercel-dashboard/app/api/shadow/runs/route.ts`: authenticated run/candidate query.
- `vercel-dashboard/app/api/shadow/candidates/[candidateId]/feedback/route.ts`: candidate feedback append endpoint.
- `vercel-dashboard/app/api/shadow/runs/[runId]/feedback/route.ts`: run-level missed-anomaly feedback endpoint.
- `vercel-dashboard/app/api/shadow/report/route.ts`: authenticated progress and metrics endpoint.
- `vercel-dashboard/app/shadow/page.tsx`: protected shadow-validation page.
- `vercel-dashboard/app/ui/shadow-validation.tsx`: review workflow and progress UI.
- `vercel-dashboard/app/globals.css`: additive responsive styles scoped under `.shadow-validation`.

### Tests and operations

- `experiments/exp265_shadow_validation/tests/`: Python contracts, routing, spool, exporter, and reporting tests.
- `vercel-dashboard/tests/shadow-contract.test.ts`: TypeScript schema tests.
- `vercel-dashboard/tests/shadow-migration.test.ts`: SQL/RLS/immutability contract tests.
- `vercel-dashboard/tests/shadow-ingest-route.test.ts`: API boundary tests.
- `vercel-dashboard/tests/shadow-feedback-route.test.ts`: authentication and append-only feedback tests.
- `vercel-dashboard/tests/shadow-metrics.test.ts`: metric eligibility tests.
- `docs/EXP265_PREREGISTRATION_KO.md`: frozen prospective rules and claim boundaries.
- `docs/EXP265_RUNBOOK_KO.md`: migration, deployment, smoke, start, stop, recovery, and 100-run reporting commands.

---

### Task 1: Label-Free Shadow Contracts And Routing

**Files:**
- Create: `experiments/exp265_shadow_validation/__init__.py`
- Create: `experiments/exp265_shadow_validation/config.py`
- Create: `experiments/exp265_shadow_validation/contracts.py`
- Create: `experiments/exp265_shadow_validation/routing.py`
- Create: `experiments/exp265_shadow_validation/tests/__init__.py`
- Create: `experiments/exp265_shadow_validation/tests/test_contracts.py`
- Create: `experiments/exp265_shadow_validation/tests/test_routing.py`

**Interfaces:**
- Consumes: label-free upstream mappings containing baseline lanes, four source candidate sets, scores/thresholds, quality flags, `n_train`, and `n_test`.
- Produces: `validate_run_input(raw: Mapping[str, object]) -> ShadowRunInput`, `route_shadow_run(inputs: ShadowRunInput) -> dict[str, object]`, and `prediction_sha256(payload: Mapping[str, object]) -> str`.

- [ ] **Step 1: Write failing strict-contract tests**

```python
def test_prediction_contract_rejects_evaluation_fields():
    raw = valid_run_input()
    raw["y_test"] = [0, 1]
    with self.assertRaisesRegex(ContractError, "forbidden field"):
        validate_run_input(raw)

def test_prediction_hash_is_order_independent():
    left = route_shadow_run(valid_shadow_input())
    right = dict(reversed(list(left.items())))
    assert prediction_sha256(left) == prediction_sha256(right)
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
python3 -m unittest experiments.exp265_shadow_validation.tests.test_contracts -v
```

Expected: import failure because `contracts.py` does not exist.

- [ ] **Step 3: Implement frozen configuration and strict input types**

Use these exact constants:

```python
SCHEMA_VERSION = 1
POLICY_VERSION = "exp265-shadow-v1"
ALPHA = 0.005
MIN_TRAIN_FOR_SHADOW = 100
SHADOW_AGREEMENT = 3
PRIORITY_AGREEMENT = 2
TARGET_NEW_RUNS = 100
SOURCE_NAMES = ("statistical", "structural", "change_point", "reconstruction")
FORBIDDEN_FIELDS = frozenset({
    "y_test", "labels", "actual_anomaly_positions", "tp", "fp", "fn",
    "precision", "recall", "f1", "combined_f1",
})
```

`ShadowRunInput` must contain only immutable scalar metadata and frozensets/mappings. It also carries a label-free `series_preview` with equal-length integer `indices` and finite numeric `values`, capped at 1,024 points and guaranteed to include every candidate index. Reject unknown top-level fields, noncanonical source names, lane overlap, out-of-range indices, mismatched score/threshold keys, empty `run_id`, and non-UTC/invalid `observed_at`.

- [ ] **Step 4: Write failing routing tests**

```python
def test_hard_is_preserved_and_gets_confidence_grade():
    result = route_shadow_run(valid_shadow_input(
        baseline_hard={1, 2, 3},
        sources={
            "statistical": {1, 2}, "structural": {1},
            "change_point": {1}, "reconstruction": set(),
        },
    ))
    assert result["hard_indices"] == [1, 2, 3]
    assert result["confidence"] == {"1": "A", "2": "B", "3": "C"}

def test_shadow_never_enters_hard():
    result = route_shadow_run(valid_shadow_input(
        n_train=100,
        baseline_standard={7},
        sources={name: {7} for name in SOURCE_NAMES},
    ))
    assert result["shadow_indices"] == [7]
    assert result["hard_indices"] == []
```

Also test `n_train=99`, quality blocking, two-source Priority, zero/one-source Standard, pre-existing Priority preservation, candidate-free runs, and all pairwise lane intersections.

- [ ] **Step 5: Implement the pure routing function**

The implementation must use this logic without reading files, environment variables, labels, dataset names, or family names:

```python
support = {index: sum(index in values for values in inputs.sources.values()) for index in all_candidates}
shadow = {
    index for index in inputs.baseline_standard
    if inputs.n_train >= MIN_TRAIN_FOR_SHADOW
    and support[index] >= SHADOW_AGREEMENT
    and index not in inputs.quality_flags
}
priority_added = {
    index for index in inputs.baseline_standard
    if support[index] >= PRIORITY_AGREEMENT and index not in shadow
}
standard = inputs.baseline_standard - shadow - priority_added
hard = set(inputs.baseline_hard)
priority = inputs.baseline_priority | priority_added
```

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest \
  experiments.exp265_shadow_validation.tests.test_contracts \
  experiments.exp265_shadow_validation.tests.test_routing -v
git add experiments/exp265_shadow_validation
git commit -m "feat(exp265): add label-free shadow routing"
```

Expected: all Task 1 tests pass and the commit contains no dashboard or result files.

---

### Task 2: Durable Local Spool And Authenticated Exporter

**Files:**
- Create: `experiments/exp265_shadow_validation/spool.py`
- Create: `experiments/exp265_shadow_validation/exporter.py`
- Create: `experiments/exp265_shadow_validation/tests/test_spool.py`
- Create: `experiments/exp265_shadow_validation/tests/test_exporter.py`
- Create: `scratch/run_exp265_shadow_exporter.py`

**Interfaces:**
- Consumes: canonical payloads returned by `route_shadow_run`.
- Produces: `ShadowSpool.enqueue(payload)`, `ShadowSpool.pending(limit)`, `ShadowSpool.mark_delivered(run_id, receipt)`, `push_prediction(...)`, and a CLI that accepts `--prediction-jsonl`, `--once`, and `--poll-seconds`.

- [ ] **Step 1: Write failing spool tests**

```python
def test_enqueue_is_idempotent_for_same_hash(tmp_path):
    spool = ShadowSpool(tmp_path / "spool.sqlite")
    spool.enqueue(payload("run-1"))
    spool.enqueue(payload("run-1"))
    assert [row.run_id for row in spool.pending(10)] == ["run-1"]

def test_conflicting_prediction_hash_is_rejected(tmp_path):
    spool = ShadowSpool(tmp_path / "spool.sqlite")
    spool.enqueue(payload("run-1"))
    with self.assertRaises(SpoolConflictError):
        spool.enqueue(payload("run-1", shadow_indices=[8]))
```

- [ ] **Step 2: Run the spool tests and verify RED**

Run `python3 -m unittest experiments.exp265_shadow_validation.tests.test_spool -v`.

Expected: import failure for `ShadowSpool`.

- [ ] **Step 3: Implement SQLite spool transactions**

Create a single `shadow_predictions` table with `run_id` primary key, canonical payload JSON, prediction SHA256, state (`pending`, `delivered`, `dead_letter`), attempt count, last error, receipt JSON, and timestamps. Use WAL mode, `BEGIN IMMEDIATE`, parameterized SQL, and atomic state transitions. Never store Supabase passwords, access tokens, ingest tokens, or user feedback in this database.

- [ ] **Step 4: Write failing exporter tests**

Use an in-process HTTP test server to assert:

```python
def test_push_sends_only_bearer_and_user_session_headers():
    receipt = push_prediction(
        url=server.url, ingest_token="ingest", access_token="session", payload=payload("run-1")
    )
    assert receipt["ok"] is True
    assert server.request.headers["Authorization"] == "Bearer ingest"
    assert server.request.headers["X-Supabase-Access-Token"] == "session"
    assert "y_test" not in server.request.body.decode()
```

Test HTTP 409 as a permanent hash conflict, 400 as dead-letter input rejection, and network/429/5xx as retryable without losing the spool row.

- [ ] **Step 5: Implement exporter and CLI**

The CLI must validate and route every line before enqueueing. It must reject the production results of Exp263/264 as prospective input when `retrospective=true`, require `collection_mode="prospective_shadow"`, and print counts without printing credentials.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest \
  experiments.exp265_shadow_validation.tests.test_spool \
  experiments.exp265_shadow_validation.tests.test_exporter -v
python3 -m py_compile scratch/run_exp265_shadow_exporter.py
git add experiments/exp265_shadow_validation scratch/run_exp265_shadow_exporter.py
git commit -m "feat(exp265): add durable shadow exporter"
```

---

### Task 3: Supabase Shadow Ledger Migration

**Files:**
- Create: `vercel-dashboard/supabase/migrations/202607210001_shadow_validation.sql`
- Create: `vercel-dashboard/tests/shadow-migration.test.ts`

**Interfaces:**
- Consumes: authenticated dashboard membership and canonical Exp265 payload JSON.
- Produces: immutable `shadow_validation_runs`, immutable `shadow_validation_candidates`, append-only feedback event tables, `ingest_shadow_prediction(jsonb)`, and read models used by later APIs.

- [ ] **Step 1: Write a failing migration contract test**

```typescript
it("keeps prediction and feedback tables immutable", () => {
  const sql = readFileSync(MIGRATION, "utf8");
  expect(sql).toContain("create table if not exists public.shadow_validation_runs");
  expect(sql).toContain("create table if not exists public.shadow_feedback_events");
  expect(sql).not.toMatch(/grant\s+update.*shadow_feedback_events/i);
  expect(sql).not.toMatch(/grant\s+delete.*shadow_feedback_events/i);
});
```

Also assert RLS is enabled on all four tables, anon has no access, authenticated members can read, writers can ingest/insert feedback, prediction hashes are unique per run, and Realtime publication includes run/candidate/feedback tables.

- [ ] **Step 2: Run the migration test and verify RED**

```bash
cd vercel-dashboard
npm test -- --run tests/shadow-migration.test.ts
```

Expected: missing migration failure.

- [ ] **Step 3: Implement tables and constraints**

Use exact verdict checks:

```sql
verdict text not null check (verdict in ('true_anomaly', 'false_alarm', 'deferred'))
```

and:

```sql
missed_anomaly_state text not null check (missed_anomaly_state in ('none', 'exists', 'unknown')),
missed_anomaly_count integer check (
  (missed_anomaly_state = 'none' and missed_anomaly_count = 0) or
  (missed_anomaly_state = 'exists' and missed_anomaly_count > 0) or
  (missed_anomaly_state = 'unknown' and missed_anomaly_count is null)
)
```

Prediction tables receive no UPDATE or DELETE grants. Feedback tables receive INSERT and SELECT only. Foreign keys cascade only when a run is administratively removed; authenticated users cannot delete runs.

`shadow_validation_runs` stores the validated label-free `series_preview` JSON (`indices` and `values`, at most 1,024 aligned points) so the dashboard can render the waveform without querying TEST labels or raw result files. The ingest RPC must recheck equal lengths, finite values, point limits, and candidate-index inclusion before inserting it.

- [ ] **Step 4: Implement atomic ingest RPC**

`ingest_shadow_prediction(payload jsonb)` must:

1. Require the caller to be an authenticated `dashboard_members.can_write=true` user.
2. Insert a new run and its candidates in one database transaction.
3. Return the existing receipt when `run_id` and prediction SHA256 match.
4. Raise a conflict when the same `run_id` has a different hash.
5. Verify `candidate_count` equals the JSON candidate array length.
6. Reject duplicate candidate IDs and invalid final lanes.

Use `SECURITY INVOKER` where possible. If a `SECURITY DEFINER` helper is necessary, set a fixed `search_path`, revoke direct execution from `anon`, and expose only the exact RPC to `authenticated`.

- [ ] **Step 5: Add read views**

Create one view for latest effective candidate feedback and one for latest effective run feedback using `row_number() over (partition by ... order by created_at desc, id desc)`. Views must not calculate Recall/F1; later application logic decides eligibility.

- [ ] **Step 6: Run tests and commit**

```bash
cd vercel-dashboard
npm test -- --run tests/shadow-migration.test.ts
git add supabase/migrations/202607210001_shadow_validation.sql tests/shadow-migration.test.ts
git commit -m "feat(exp265): add shadow validation ledger schema"
```

---

### Task 4: Strict TypeScript Contract And Ingest API

**Files:**
- Create: `vercel-dashboard/src/lib/shadow.ts`
- Create: `vercel-dashboard/app/api/shadow/ingest/route.ts`
- Create: `vercel-dashboard/tests/shadow-contract.test.ts`
- Create: `vercel-dashboard/tests/shadow-ingest-route.test.ts`

**Interfaces:**
- Consumes: the canonical Python JSON payload and existing `hasValidBearerToken` / `createIngestClient` helpers.
- Produces: `assertValidShadowPrediction(value: unknown): ShadowPrediction` and `POST /api/shadow/ingest`.

- [ ] **Step 1: Write failing TypeScript schema tests**

```typescript
it.each(["y_test", "tp", "fp", "f1", "actual_anomaly_positions"])(
  "rejects forbidden field %s",
  (field) => expect(() => assertValidShadowPrediction({ ...validPayload(), [field]: [] })).toThrow("unknown field"),
);

it("requires pairwise-disjoint lanes", () => {
  const payload = validPayload();
  payload.candidates.push({ ...payload.candidates[0], candidate_id: "run-1:1", final_lane: "Priority" });
  expect(() => assertValidShadowPrediction(payload)).toThrow("duplicate candidate index");
});
```

- [ ] **Step 2: Run schema tests and verify RED**

Run `cd vercel-dashboard && npm test -- --run tests/shadow-contract.test.ts`.

- [ ] **Step 3: Implement strict validator**

Mirror Python field names and limits exactly. Reject unknown keys recursively, nonfinite numbers, scores without thresholds, source names outside the four-name set, candidate indices outside `[0, n_test)`, wrong A/B/C placement, autonomous Hard mutations, retrospective collection mode, payloads above 512 KiB, and policy/config hash mismatches.

- [ ] **Step 4: Write failing ingest boundary tests**

Test this order:

1. Invalid bearer token returns 401 before reading body.
2. Oversized body returns 413 before JSON parsing.
3. Invalid writer session returns 401.
4. Strict schema rejection returns 400.
5. RPC hash conflict returns 409.
6. Supabase availability error returns 503.
7. Valid payload returns the RPC receipt with `Cache-Control: no-store`.

- [ ] **Step 5: Implement ingest route**

Follow the existing status-ingest authentication pattern, but call only:

```typescript
await supabase.rpc("ingest_shadow_prediction", { payload: prediction });
```

Do not use a service-role key and do not perform partial client-side multi-table writes.

- [ ] **Step 6: Run tests and commit**

```bash
cd vercel-dashboard
npm test -- --run tests/shadow-contract.test.ts tests/shadow-ingest-route.test.ts
npx tsc --noEmit
git add src/lib/shadow.ts app/api/shadow/ingest/route.ts tests/shadow-*.test.ts
git commit -m "feat(exp265): add strict shadow ingest API"
```

---

### Task 5: Feedback And Report APIs

**Files:**
- Create: `vercel-dashboard/src/lib/shadow-metrics.ts`
- Create: `vercel-dashboard/app/api/shadow/runs/route.ts`
- Create: `vercel-dashboard/app/api/shadow/candidates/[candidateId]/feedback/route.ts`
- Create: `vercel-dashboard/app/api/shadow/runs/[runId]/feedback/route.ts`
- Create: `vercel-dashboard/app/api/shadow/report/route.ts`
- Create: `vercel-dashboard/tests/shadow-feedback-route.test.ts`
- Create: `vercel-dashboard/tests/shadow-metrics.test.ts`

**Interfaces:**
- Consumes: Supabase-authenticated browser sessions and effective feedback views.
- Produces: paginated run/candidate JSON, append-only feedback endpoints, and `buildShadowReport(rows): ShadowReport`.

- [ ] **Step 1: Write failing feedback API tests**

```typescript
it("uses authenticated user id as reviewer and ignores client reviewer fields", async () => {
  const response = await postCandidateFeedback({ verdict: "true_anomaly", reviewer_id: "forged" });
  expect(response.status).toBe(201);
  expect(mockInsert).toHaveBeenCalledWith(expect.objectContaining({ reviewer_id: AUTH_USER_ID }));
});
```

Also test invalid verdicts, missing candidate/run, RLS denial, notes longer than 2,000 characters, correction references, and absence of UPDATE/DELETE calls.

- [ ] **Step 2: Implement feedback routes**

Both routes must call `supabase.auth.getUser()`, insert a new event, return 201, and never update or delete an old event. Database RLS remains the final authorization boundary.

- [ ] **Step 3: Write failing metric eligibility tests**

```typescript
it("does not expose recall or f1 for candidate-only feedback", () => {
  const report = buildShadowReport([candidateReviewedRun({ missed_state: "unknown" })]);
  expect(report.recall).toBeNull();
  expect(report.f1).toBeNull();
  expect(report.metric_scope).toBe("candidate_feedback_only");
});

it("counts candidate-free runs toward the 100-run target", () => {
  const report = buildShadowReport([run({ candidate_count: 0 })]);
  expect(report.runs_collected).toBe(1);
});
```

- [ ] **Step 4: Implement paginated reads and report model**

The run API must cap page size at 100 and support lane, confidence, missing state, and verdict filters. The report must calculate candidate precision/workload for available candidate feedback and calculate Recall/F1 only over explicitly fully annotated runs where every candidate has an effective non-deferred verdict and run feedback supplies a valid `missed_anomaly_count`. Keep `autonomous`, `shadow`, and `review` scopes separate.

- [ ] **Step 5: Run tests and commit**

```bash
cd vercel-dashboard
npm test -- --run tests/shadow-feedback-route.test.ts tests/shadow-metrics.test.ts
npx tsc --noEmit
git add src/lib/shadow-metrics.ts app/api/shadow tests/shadow-feedback-route.test.ts tests/shadow-metrics.test.ts
git commit -m "feat(exp265): add feedback and report APIs"
```

---

### Task 6: Shadow Review Dashboard

**Files:**
- Create: `vercel-dashboard/app/shadow/page.tsx`
- Create: `vercel-dashboard/app/ui/shadow-validation.tsx`
- Modify: `vercel-dashboard/app/ui/dashboard.tsx`
- Modify: `vercel-dashboard/app/globals.css`
- Create: `vercel-dashboard/tests/shadow-view-model.test.ts`

**Interfaces:**
- Consumes: `/api/shadow/runs`, `/api/shadow/report`, and feedback APIs.
- Produces: an authenticated mobile/desktop shadow review surface and a navigation link from the existing experiment dashboard.

- [ ] **Step 1: Extract a testable view model and write failing tests**

Test ordering exactly:

```typescript
expect(sortCandidates(rows).map((row) => row.final_lane)).toEqual([
  "Shadow", "Priority", "Hard", "Hard", "Standard",
]);
```

Within Hard, sort A before B before C. Unreviewed precedes deferred and reviewed. Add tests for Korean labels, missing badge, progress `runs_collected / 100`, and hidden Recall/F1.

- [ ] **Step 2: Run view-model tests and verify RED**

Run `cd vercel-dashboard && npm test -- --run tests/shadow-view-model.test.ts`.

- [ ] **Step 3: Implement the page and interaction**

The first viewport must show:

- `신규 run 검증 0/100`
- lane counts
- reviewed/deferred counts
- candidate precision when available
- a clear `Shadow는 자동 경보가 아닙니다` notice

Use tabs for `전체`, `Shadow`, `우선 검토`, `자동 경보`, and `일반 검토`; use filter controls for confidence, missing status, and verdict. Candidate rows show the actual waveform only when its authorized endpoint provides it. The three verdict actions use clear text plus Lucide icons and require confirmation before replacing a prior effective verdict with a correction event.

- [ ] **Step 4: Add run-level missed-anomaly control**

Provide a segmented control with `놓친 이상 없음`, `놓친 이상 있음`, and `확인 불가`. Explain in a tooltip that Recall/F1 requires run-level confirmation; do not use a large permanent explanation block.

- [ ] **Step 5: Add scoped responsive styles**

Keep styles under `.shadow-validation`, use existing palette and spacing, avoid nested cards, keep controls within mobile width, and preserve the current experiment dashboard layout.

- [ ] **Step 6: Run tests, build, and commit**

```bash
cd vercel-dashboard
npm test
npm run build
git add app/shadow app/ui/shadow-validation.tsx app/ui/dashboard.tsx app/globals.css tests/shadow-view-model.test.ts
git commit -m "feat(exp265): add shadow review dashboard"
```

Expected: all existing status-dashboard tests and the new shadow tests pass; production build exits 0.

---

### Task 7: Read-Only 100-Run Report Export

**Files:**
- Create: `experiments/exp265_shadow_validation/reporting.py`
- Create: `experiments/exp265_shadow_validation/tests/test_reporting.py`
- Create: `scratch/export_exp265_shadow_report.py`

**Interfaces:**
- Consumes: authenticated `/api/shadow/report` JSON and its server-provided hash.
- Produces: `shadow_report.json`, `SHADOW_REPORT.md`, and `MANIFEST.json` under a user-selected output directory.

- [ ] **Step 1: Write failing report tests**

```python
def test_candidate_only_report_omits_recall_and_f1(tmp_path):
    paths = write_report(tmp_path, candidate_only_response())
    text = paths.markdown.read_text()
    assert "Recall: 계산 불가" in text
    assert "F1: 계산 불가" in text

def test_report_never_selects_policy_at_100_runs(tmp_path):
    result = write_report(tmp_path, report_response(runs_collected=100))
    assert result.payload["automatic_policy_changed"] is False
```

- [ ] **Step 2: Implement strict report validation and output**

Reject responses that mix Hard with Shadow/review, claim deployment eligibility, expose Recall/F1 without full-run coverage, or identify historical TEST rows as prospective. Record API URL, response SHA, code SHA, config hash, collected date range, and limitations in the manifest.

- [ ] **Step 3: Implement authenticated report CLI**

Require output directory, dashboard URL, Supabase session configuration, and `--expected-policy-version exp265-shadow-v1`. Write files atomically and never print passwords/tokens.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m unittest experiments.exp265_shadow_validation.tests.test_reporting -v
python3 -m py_compile scratch/export_exp265_shadow_report.py
git add experiments/exp265_shadow_validation/reporting.py \
  experiments/exp265_shadow_validation/tests/test_reporting.py \
  scratch/export_exp265_shadow_report.py
git commit -m "feat(exp265): add shadow validation reporting"
```

---

### Task 8: Preregistration, Runbook, Security Regression, And Launch Gate

**Files:**
- Create: `docs/EXP265_PREREGISTRATION_KO.md`
- Create: `docs/EXP265_RUNBOOK_KO.md`
- Modify: `vercel-dashboard/.env.example`
- Create: `experiments/exp265_shadow_validation/tests/test_structure.py`
- Create: `vercel-dashboard/tests/shadow-security.test.ts`

**Interfaces:**
- Consumes: all Task 1-7 artifacts.
- Produces: auditable launch instructions and an explicit go/no-go result for real prospective collection.

- [ ] **Step 1: Write structure and forbidden-input tests**

The Python structure test must inspect all Exp265 prediction/export source files and fail if prediction functions accept `y_test`, label paths, actual anomaly positions, or TEST metric selectors. The TypeScript security test must fail if any browser file references `SUPABASE_SERVICE_ROLE_KEY`, `DASHBOARD_INGEST_TOKEN`, or a machine password.

- [ ] **Step 2: Write preregistration**

Copy all frozen constants and claim boundaries from the approved design. Include the activation timestamp, prospective-only rule, exact 100-run denominator, no automatic promotion, no TEST-based tuning, and the distinction between candidate precision and fully annotated Recall/F1.

- [ ] **Step 3: Write runbook**

Document exact commands for:

1. Applying the Supabase migration.
2. Confirming RLS with an unauthorized and authorized session.
3. Building and deploying Vercel.
4. Running a three-run synthetic smoke using IDs prefixed `smoke-`.
5. Deleting smoke rows with an administrator-only SQL command before activation.
6. Starting the exporter against a real prospective upstream JSONL stream.
7. Checking spool backlog and dead-letter rows.
8. Exporting the report at any time and at 100 runs.
9. Stopping collection without deleting predictions or feedback.

- [ ] **Step 4: Add the upstream-feed launch blocker**

The runbook must state that code completion is not collection activation. Real collection may start only when an upstream process can provide new, label-free `ShadowRunInput` payloads with:

- stable `run_id`
- `collection_mode="prospective_shadow"`
- all three baseline lanes
- all four source index/score/threshold maps, or an explicit reconstruction ineligibility caused only by `n_train < 100`
- quality flags
- policy/config hash
- no retrospective/test markers

If this feed is absent, report `BLOCKED: prospective upstream feed unavailable`; do not replay the 1,117 historical TEST datasets as new runs.

- [ ] **Step 5: Run the full verification matrix**

```bash
python3 -m unittest discover -s experiments/exp265_shadow_validation/tests -v
python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v
cd vercel-dashboard
npm test
npx tsc --noEmit
npm run build
git diff --check
```

Expected: all commands pass; no Exp263/264 production artifact hash changes; no existing dashboard snapshot API regression.

- [ ] **Step 6: Run isolated end-to-end smoke**

Use a local/test Supabase project or transactionally isolated schema. Submit three runs: zero candidates, one Hard with A confidence, and one Shadow plus Priority. Verify idempotent resend, user feedback, correction history, run feedback, progress `3/100`, candidate-only metric suppression, and no autonomous Hard mutation.

- [ ] **Step 7: Record launch-gate result and commit**

```bash
git add docs/EXP265_PREREGISTRATION_KO.md docs/EXP265_RUNBOOK_KO.md \
  vercel-dashboard/.env.example \
  experiments/exp265_shadow_validation/tests/test_structure.py \
  vercel-dashboard/tests/shadow-security.test.ts
git commit -m "docs(exp265): preregister shadow validation launch"
```

The completion report must separately state:

- software verification status
- Supabase migration/deployment status
- real upstream feed status
- prospective collection activation status
- current unique run count

Do not call Exp265 operationally validated until 100 genuinely new runs and the required feedback coverage exist.

---

## Final Acceptance Checklist

- [ ] Exp253 Hard indices are byte-for-byte unchanged by Exp265 routing.
- [ ] Shadow candidates never enter autonomous Hard.
- [ ] Confidence A/B/C is assigned only to existing Hard candidates.
- [ ] New run payloads contain no labels or evaluation metrics.
- [ ] Same `run_id` plus same hash is idempotent; a different hash is a conflict.
- [ ] Prediction and feedback records are immutable/append-only under authenticated roles.
- [ ] Nonmembers cannot read; nonwriters cannot ingest or submit feedback.
- [ ] Machine secrets are absent from browser bundles and artifacts.
- [ ] Candidate-free runs count toward the 100-run denominator.
- [ ] Historical 1,117 TEST datasets do not count toward prospective progress.
- [ ] Recall/F1 stays hidden without full-run annotation.
- [ ] Missing and nonmissing results remain separate.
- [ ] Reaching 100 runs creates a report but no automatic policy update.
- [ ] Existing experiment dashboard, Vercel snapshot ingest, and Exp263/264 artifacts remain unchanged.
