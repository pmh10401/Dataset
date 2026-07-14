# Historical Experiment Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display every known experiment through Exp152 with explicit validity status, normalized autonomous Hard metrics, separated review metrics, and no silent missing values.

**Architecture:** Extend the existing read-only `serve_rank_dashboard.py` with an explicit history catalog and small schema adapters. Reconcile catalog metadata, queue state, and artifact presence into normalized history records, then expose a complete history and a comparison-eligible subset to the browser.

**Tech Stack:** Python 3 standard library, CSV/JSON artifacts, vanilla HTML/CSS/JavaScript, function-style Python regression tests.

## Global Constraints

- Do not rerun experiments or modify historical result artifacts.
- Keep autonomous Hard alert metrics separate from Standard/Priority review and `mean_combined_f1`.
- Exp149/150 remain visible but invalid and excluded from default comparison.
- Research, audit, blocked, invalid, and not-run entries remain visible in history.
- TEST metrics must not select alpha, threshold, budget, source, or selector.
- Restart only the dashboard; the active Exp153-157 PID must remain unchanged.

---

### Task 1: Historical Catalog And Status Reconciliation

**Files:**
- Modify: `serve_rank_dashboard.py:193-1144`
- Test: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: `EXPERIMENTS`, sequential state JSON, artifact paths.
- Produces: `HISTORICAL_EXPERIMENT_META: dict[str, dict]` and `experiment_history_status(exp_id, info, state) -> dict`.

- [ ] **Step 1: Write failing catalog tests**

```python
def test_historical_catalog_covers_exp138_through_exp152():
    for number in range(138, 153):
        assert any(exp_id.startswith(f"experiment_{number}") for exp_id in dash.HISTORICAL_EXPERIMENT_META)


def test_invalid_and_blocked_statuses_are_explicit():
    meta = dash.HISTORICAL_EXPERIMENT_META
    assert meta["experiment_149_virtual_run_conformal_policy"]["history_status"] == "invalid"
    assert meta["experiment_150_virtual_run_conformal_policy"]["history_status"] == "invalid"
    assert meta["experiment_145_train_normal_count_budget"]["history_status"] == "blocked"
    assert meta["experiment_148_policy_level_train_only_train_budget"]["history_status"] == "blocked"
```

- [ ] **Step 2: Run tests and verify `HISTORICAL_EXPERIMENT_META` is missing**

```bash
python3 - <<'PY'
import scratch.test_rank_dashboard_operational as t
t.test_historical_catalog_covers_exp138_through_exp152()
t.test_invalid_and_blocked_statuses_are_explicit()
PY
```

Expected: `AttributeError`.

- [ ] **Step 3: Add Exp138-152 metadata**

Each record must contain `label`, `history_status`, `comparison_eligible`, `scope`, `status_reason`, and real artifact paths when they exist. Use `completed_research` for retrospective audits/sensitivities, `invalid` for Exp149/150, `blocked` for unmet contracts, and `completed_valid` only for comparable corrected results.

```python
"experiment_149_virtual_run_conformal_policy": {
    "label": "Exp149 · virtual-run conformal current source",
    "history_status": "invalid",
    "comparison_eligible": False,
    "scope": "retrospective_counterfactual",
    "status_reason": "routing implementation defect corrected by Exp151",
    "detail_csv": ROOT / "outputs/exp137_policy_train_only_validation/virtual_run_conformal/03_exp149_current_source_results.csv",
    "summary_csv": ROOT / "outputs/exp137_policy_train_only_validation/virtual_run_conformal/04_exp149_current_source_summary.csv",
},
```

- [ ] **Step 4: Implement state/artifact reconciliation**

```python
def experiment_history_status(exp_id, info, state):
    if exp_id == (state.get("running") or {}).get("id"):
        return {"history_status": "running", "status_reason": "active process"}
    if exp_id in (state.get("queue") or []):
        return {"history_status": "queued", "status_reason": "sequential queue"}
    configured = info.get("history_status")
    if configured in {"invalid", "blocked", "not_run"}:
        return {"history_status": configured, "status_reason": info.get("status_reason")}
    detail = info.get("detail_csv")
    if detail and Path(detail).exists():
        return {"history_status": configured or "completed_valid", "status_reason": info.get("status_reason")}
    return {"history_status": "not_run", "status_reason": "result artifact missing"}
```

- [ ] **Step 5: Run the targeted tests and commit**

```bash
python3 - <<'PY'
import scratch.test_rank_dashboard_operational as t
t.test_historical_catalog_covers_exp138_through_exp152()
t.test_invalid_and_blocked_statuses_are_explicit()
PY
git add serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "feat: catalog historical dashboard experiments"
```

---

### Task 2: Normalize Historical Metric Schemas

**Files:**
- Modify: `serve_rank_dashboard.py:2470-2670`
- Test: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: selected summary row and detail rows.
- Produces: `normalize_historical_metrics(summary_row, detail_rows) -> dict`.

- [ ] **Step 1: Write failing Hard/review separation tests**

```python
def test_hard_schema_never_uses_combined_f1_as_autonomous_f1():
    metrics = dash.normalize_historical_metrics({
        "mean_hard_f1": "0.60", "mean_combined_f1": "0.82",
        "mean_hard_fp": "0.30", "hard_alert_precision": "0.84",
        "standard_review_total_candidates": "639",
    }, [])
    assert metrics["mean_f1"] == 0.60
    assert metrics["mean_combined_f1"] == 0.82
    assert metrics["mean_fp"] == 0.30
    assert metrics["alert_precision"] == 0.84
    assert metrics["review_candidate_count"] == 639


def test_audit_schema_reports_reason_instead_of_fabricated_f1():
    metrics = dash.normalize_historical_metrics({"binding_dataset_count": "12"}, [])
    assert metrics["mean_f1"] is None
    assert metrics["metric_availability_reason"] == "audit only"
```

- [ ] **Step 2: Run tests and verify the adapter is missing**

Expected: `AttributeError` for `normalize_historical_metrics`.

- [ ] **Step 3: Implement explicit field precedence**

```python
def normalize_historical_metrics(summary_row, detail_rows):
    row = summary_row or {}
    mean_f1 = first_number(row, ["mean_hard_f1", "mean_f1", "f1_evt", "mean_f1_evt", "f1"])
    combined = first_number(row, ["mean_combined_f1"])
    mean_fp = first_number(row, ["mean_hard_fp", "mean_fp", "fp"])
    precision = first_number(row, ["hard_alert_precision", "alert_precision"])
    review_count = first_number(row, ["standard_review_total_candidates", "review_candidates_total", "standard_review_count"])
    audit_keys = {"binding_dataset_count", "removed_candidate_count", "coverage_count"}
    reason = None if mean_f1 is not None else ("audit only" if audit_keys.intersection(row) else "metric not recorded")
    return {
        "mean_f1": mean_f1,
        "mean_combined_f1": combined,
        "mean_fp": mean_fp,
        "alert_precision": precision,
        "review_candidate_count": int(review_count) if review_count is not None else None,
        "metric_availability_reason": reason,
    }
```

Add detail-row fallbacks for Hard totals and micro Hard precision. Never map review/combined fields into autonomous fields.

- [ ] **Step 4: Route `completed_result_snapshot` through the adapter**

```python
normalized = normalize_historical_metrics(best, read_csv_rows(info.get("detail_csv")))
history_row.update(normalized)
history_row["autonomous_metric_scope"] = info.get("autonomous_metric_scope", "hard_alert_or_legacy_primary")
history_row["review_metric_scope"] = info.get("review_metric_scope", "human_assisted_diagnostic_only")
```

- [ ] **Step 5: Run tests and commit**

```bash
python3 scratch/test_rank_dashboard_operational.py
git add serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "fix: normalize historical experiment metrics"
```

---

### Task 3: Complete History API And Eligible Comparison

**Files:**
- Modify: `serve_rank_dashboard.py:2585-2670`
- Modify: `serve_rank_dashboard.py:3277-3325`
- Test: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: catalog metadata and normalized metrics.
- Produces: complete `completed_results` history and filtered `experiment_compare`.

- [ ] **Step 1: Write failing visibility and exclusion tests**

```python
def test_invalid_history_is_visible_but_excluded_from_compare():
    history = [
        {"id": "valid", "history_status": "completed_valid", "comparison_eligible": True, "mean_f1": 0.6},
        {"id": "invalid", "history_status": "invalid", "comparison_eligible": False, "mean_f1": 0.9},
    ]
    rows = dash.experiment_compare_snapshot("", "", [], history, is_running=False)
    assert [row["id"] for row in rows] == ["valid"]


def test_catalog_only_blocked_and_invalid_rows_are_in_history():
    rows = dash.completed_result_snapshot({"completed": [], "queue": [], "history": []})
    by_id = {row["id"]: row for row in rows}
    assert by_id["experiment_145_train_normal_count_budget"]["history_status"] == "blocked"
    assert by_id["experiment_149_virtual_run_conformal_policy"]["history_status"] == "invalid"
```

- [ ] **Step 2: Build history from state/catalog union**

```python
known_ids = set(EXPERIMENTS) | set(HISTORICAL_EXPERIMENT_META)
state_ids = list(state.get("completed") or [])
catalog_only = sorted(known_ids - set(state_ids), key=experiment_sort_key)
experiment_ids = state_ids + catalog_only
```

- [ ] **Step 3: Apply explicit comparison eligibility**

```python
eligible_completed = [
    item for item in completed_results
    if item.get("comparison_eligible")
    and item.get("history_status") == "completed_valid"
    and as_float(item.get("mean_f1")) is not None
]
```

- [ ] **Step 4: Run tests and commit**

```bash
python3 scratch/test_rank_dashboard_operational.py
git add serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "feat: expose complete experiment history"
```

---

### Task 4: Status-Filtered History UI

**Files:**
- Modify: `serve_rank_dashboard.py:1370-2000` (`INDEX_HTML`)
- Test: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Consumes: status, reason, autonomous metrics, and review metrics.
- Produces: compact history table with status filter and explicit missing reasons.

- [ ] **Step 1: Write failing UI contract test**

```python
def test_history_ui_has_status_filter_and_separate_review_metrics():
    html = dash.INDEX_HTML
    assert 'id="historyStatusFilter"' in html
    assert "history_status" in html
    assert "metric_availability_reason" in html
    assert "mean_combined_f1" in html
    assert "review_candidate_count" in html
```

- [ ] **Step 2: Add status filter and explicit columns**

Use a `<select id="historyStatusFilter">` with `all`, `completed_valid`, `completed_research`, `invalid`, `blocked`, and `not_run`. Display Status, Hard F1, Hard FP, Hard Precision, Combined F1, Review Candidates, Scope, and Reason.

```javascript
const metricText = item.mean_f1 == null
  ? (item.metric_availability_reason || 'metric not recorded')
  : fmt(item.mean_f1);
```

Keep `mean_combined_f1` only in the human-assisted column. Preserve horizontal scrolling on mobile and add no new top-level cards.

- [ ] **Step 3: Run UI tests and commit**

```bash
python3 -m py_compile serve_rank_dashboard.py
python3 scratch/test_rank_dashboard_operational.py
git add serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "feat: add experiment history status view"
```

---

### Task 5: Reconciliation And Safe Dashboard Restart

**Files:**
- Test: `scratch/test_rank_dashboard_operational.py`
- Read: `outputs/exp137_policy_train_only_validation/**`
- Read: `outputs/exp151_152_virtual_run_conformal_v2/**`

**Interfaces:**
- Consumes: live `/api/status`, process table, normalized history.
- Produces: verified dashboard with unchanged active experiment PID.

- [ ] **Step 1: Run static verification**

```bash
python3 -m py_compile serve_rank_dashboard.py run_rank_experiments_sequential.py
python3 scratch/test_rank_dashboard_operational.py
git diff --check -- serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
```

- [ ] **Step 2: Reconcile expected statuses**

Assert Exp138-152 coverage, Exp149/150 invalid status, blocked visibility, and absence of invalid/research rows from `experiment_compare`.

- [ ] **Step 3: Restart only the dashboard and preserve training PID**

```bash
before=$(pgrep -f 'experiments/exp153_157/run_experiment_153_157_pipeline.py' | head -1)
python3 dashboardctl.py restart
after=$(pgrep -f 'experiments/exp153_157/run_experiment_153_157_pipeline.py' | head -1)
test "$before" = "$after"
```

- [ ] **Step 4: Verify API and browser**

Confirm complete history count, status filtering, missing reasons, desktop/mobile table behavior, distinct Hard/combined fields, and unchanged active experiment PID.

- [ ] **Step 5: Commit only if verification required a correction**

```bash
git add serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "test: verify historical dashboard reconciliation"
```
