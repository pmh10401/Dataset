# Exp392 Label-Guided Score Discovery Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a retrospective, label-guided audit that discovers promising frozen evidence sources and combinations without changing or selecting an operational alert policy.

**Architecture:** A label-free `prepare` phase verifies Exp389/Exp357/Exp381 artifacts and writes a frozen candidate matrix. A separate `evaluate` phase verifies that matrix SHA256, then reads TEST labels and computes source, combination, complementarity, stability, and association reports. The sequential runner exposes progress, but the experiment never rewrites Hard or review lanes.

**Tech Stack:** Python 3, standard library, NumPy 2.3, pandas 2.3, SciPy 1.17, SQLite read-only access, unittest, CSV/JSONL/JSON.

## Global Constraints

- Reuse Alpha values `0.005, 0.010, 0.020, 0.050, 0.100, 0.200` without alteration.
- Analyze exactly the nine frozen Exp357 evidence sources.
- TEST labels may be read only in `evaluate`, after frozen prediction SHA256 verification.
- Existing Hard, Standard review, Priority review, and exact-normal lanes remain byte-for-byte unchanged.
- Do not select an Alpha, source, combination, threshold, family rule, or size rule from TEST performance.
- Every evaluated output records `label_guided_discovery=true`, `eligible_for_deployment=false`, `test_metrics_select_policy=false`, `retrospective_counterfactual=true`, and `prospective_validated=false`.
- Dataset/family lineage may stratify retrospective reports but may not route candidates.
- Do not add an automatic-alert promotion output.
- Write only to `/Users/minho/Documents/Dataset/exp392_label_guided_score_discovery_audit`.
- Research labels are descriptive only: `unsupported` when recovered FN is zero
  or candidate precision is below 30%; `research_signal` for all remaining
  observations; `preregistration_candidate` only when recovered FN is at least
  10, candidate precision is at least 50%, TP comes from at least five lineage
  groups, top-lineage TP share is below 80%, and the dataset-bootstrap precision
  lower bound is at least 35%.

---

## File Structure

- Create `experiments/exp392_label_guided_discovery/__init__.py`: package marker.
- Create `experiments/exp392_label_guided_discovery/discovery_core.py`: deterministic source/combination enumeration, candidate-set operations, metrics, complementarity, bootstrap, Fisher association, and Benjamini-Hochberg correction.
- Create `experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py`: `prepare`, `evaluate`, and `all` phases; frozen input verification; DB label boundary; outputs, status, and manifest.
- Create `experiments/exp392_label_guided_discovery/test_exp392_label_guided_discovery.py`: unit, invariance, boundary, and queue tests.
- Create `docs/EXP392_PREREGISTRATION_KO.md`: concise experiment contract.
- Create `docs/EXP392_RUNBOOK_KO.md`: smoke/full commands and interpretation boundaries.
- Modify `run_rank_experiments_sequential.py`: add the Exp392 queue entry only.

---

### Task 1: Deterministic Source and Combination Core

**Files:**
- Create: `experiments/exp392_label_guided_discovery/__init__.py`
- Create: `experiments/exp392_label_guided_discovery/discovery_core.py`
- Create: `experiments/exp392_label_guided_discovery/test_exp392_label_guided_discovery.py`

**Interfaces:**
- Produces: `CombinationRule`, `enumerate_combination_rules()`, `candidate_indices_for_rule()`, `evaluate_candidate_set()`, `benjamini_hochberg()`.
- Consumes: fixed source names, Alpha values, and label-free evidence rows.

- [ ] **Step 1: Write failing deterministic-enumeration tests**

```python
def test_enumerates_every_pair_and_triple_without_duplicates():
    rules = enumerate_combination_rules(("a", "b", "c"), alpha=0.005)
    names = [rule.rule_id for rule in rules]
    assert len(names) == len(set(names))
    assert "intersection__a__b__alpha_0p005" in names
    assert "union__a__b__alpha_0p005" in names
    assert "atleast2__a__b__c__alpha_0p005" in names

def test_candidate_rule_never_reintroduces_excluded_indices():
    rule = CombinationRule("intersection", ("a", "b"), 0.005, 2)
    result = candidate_indices_for_rule(
        rule,
        {"a": {1, 2}, "b": {2, 3}},
        excluded={2},
        n_test=5,
    )
    assert result == set()
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery
```

Expected: import failure because `discovery_core.py` does not exist.

- [ ] **Step 3: Implement immutable rule types and exhaustive enumeration**

```python
@dataclass(frozen=True, order=True)
class CombinationRule:
    operator: str
    sources: tuple[str, ...]
    alpha: float
    minimum_votes: int

    @property
    def rule_id(self) -> str:
        joined = "__".join(self.sources)
        return f"{self.operator}__{joined}__alpha_{alpha_slug(self.alpha)}"


def enumerate_combination_rules(
    sources: Sequence[str],
    *,
    alpha: float,
) -> tuple[CombinationRule, ...]:
    rules = []
    ordered = tuple(sorted(sources))
    for size in (2, 3):
        for group in combinations(ordered, size):
            rules.append(CombinationRule("intersection", group, alpha, size))
            rules.append(CombinationRule("union", group, alpha, 1))
            if size == 3:
                rules.append(CombinationRule("atleast2", group, alpha, 2))
    return tuple(sorted(rules, key=lambda item: item.rule_id))
```

- [ ] **Step 4: Implement set evaluation and BH correction**

`evaluate_candidate_set()` must return candidate count, recovered FN, added FP,
precision, FN recovery rate, review overlap TP/FP, novel FN, and contributing
dataset counts. `benjamini_hochberg()` must preserve input order and cap adjusted
values at `1.0`.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery
git add experiments/exp392_label_guided_discovery
git commit -m "feat: add deterministic Exp392 discovery core"
```

Expected: all Task 1 tests pass.

---

### Task 2: Label-Free Prepare Phase

**Files:**
- Create: `experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py`
- Modify: `experiments/exp392_label_guided_discovery/test_exp392_label_guided_discovery.py`

**Interfaces:**
- Consumes: `load_frozen_inputs()` and coverage metadata from the Exp391-compatible artifacts.
- Produces: `prepare(output_root: Path, dataset_limit: int | None) -> dict`, `exp392_frozen_candidate_matrix.jsonl`, and `exp392_prepare_manifest.json`.

- [ ] **Step 1: Write failing boundary and schema tests**

```python
def test_prepare_row_contains_no_labels_or_metrics():
    row = build_prepare_row(hard_row, evidence_by_alpha, coverage_row)
    forbidden = {"y_test", "label", "is_anomaly", "tp", "fp", "fn", "f1"}
    assert forbidden.isdisjoint(row)
    assert row["contains_test_labels"] is False
    assert row["contains_evaluation_metrics"] is False

def test_prepare_manifest_disallows_policy_selection():
    manifest = build_prepare_manifest([], "abc")
    assert manifest["test_metrics_select_policy"] is False
    assert manifest["eligible_for_deployment"] is False
```

- [ ] **Step 2: Run tests and confirm failure**

Run the Exp392 unittest module. Expected: missing runner symbols.

- [ ] **Step 3: Implement prepare rows**

Each dataset row stores:

```python
{
    "dataset": dataset,
    "n_train_normal": n_train_normal,
    "n_test": n_test,
    "baseline_hard_indices": sorted(hard),
    "standard_review_indices": sorted(standard),
    "priority_review_indices": sorted(priority),
    "automatic_normal_exact_indices": sorted(exact_normal),
    "quality_conflict_indices": sorted(quality_conflict),
    "evidence_by_alpha": {
        alpha_slug(alpha): {
            source: {
                "available": bool(payload["available"]),
                "indices": list(payload["indices"]),
                "resolution_blocked": bool(payload["resolution_blocked"]),
            }
            for source, payload in sorted(evidence.items())
        }
        for alpha in ALPHAS
    },
    "contains_test_labels": False,
    "contains_evaluation_metrics": False,
}
```

- [ ] **Step 4: Freeze and hash the prepare artifact**

Write rows sorted by dataset. Store artifact SHA256, input SHA256 values,
row count, dataset-list SHA256, code SHA256, fixed source list, and Alpha list
in the prepare manifest. Reject duplicate/missing datasets and out-of-range
indices.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery
git add experiments/exp392_label_guided_discovery
git commit -m "feat: freeze label-free Exp392 candidate matrix"
```

---

### Task 3: Separate Label-Guided Evaluation

**Files:**
- Modify: `experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py`
- Modify: `experiments/exp392_label_guided_discovery/discovery_core.py`
- Modify: `experiments/exp392_label_guided_discovery/test_exp392_label_guided_discovery.py`

**Interfaces:**
- Consumes: verified prepare manifest and frozen candidate matrix.
- Produces: `evaluate(output_root: Path) -> dict` and all source/combination/complementarity CSVs.

- [ ] **Step 1: Write failing hash-gate and lane-invariance tests**

```python
def test_evaluate_rejects_tampered_prepare_artifact(tmp_path):
    matrix, manifest = write_valid_fixture(tmp_path)
    matrix.write_text(matrix.read_text() + "{}\n")
    with self.assertRaisesRegex(ValueError, "SHA256"):
        evaluate(tmp_path)

def test_evaluation_does_not_change_frozen_lanes():
    evaluated = evaluate_fixture()
    assert evaluated["baseline_hard_indices"] == evaluated["frozen_hard_indices"]
    assert evaluated["standard_review_indices"] == evaluated["frozen_standard_indices"]
    assert evaluated["priority_review_indices"] == evaluated["frozen_priority_indices"]
```

- [ ] **Step 2: Run tests and confirm failure**

Expected: evaluation API is missing.

- [ ] **Step 3: Implement the label boundary**

The evaluation sequence must be:

```python
matrix_rows, manifest = load_and_verify_prepare(output_root)
assert sha256_file(matrix_path) == manifest["prediction_sha256"]
labels_by_dataset = load_test_labels({row["dataset"] for row in matrix_rows})
```

No helper reachable from `prepare()` may call `load_test_labels()`.

- [ ] **Step 4: Implement source and lattice summaries**

Emit:

- `exp392_individual_source_summary.csv`;
- `exp392_pair_triple_summary.csv`;
- `exp392_candidate_level_evaluation.csv.gz`.

Every summary row includes the five safety flags from Global Constraints.
All sources, Alphas, pairs, triples, and fixed operators are emitted even when
candidate count is zero.

- [ ] **Step 5: Implement complementarity**

Emit `exp392_source_complementarity.csv` with exact candidate Jaccard, TP
Jaccard, FP Jaccard, unique recovered FN for A/B, shared recovered FN, and
conditional TP/FP contributions.

- [ ] **Step 6: Implement exploratory association**

For each source/Alpha binary indicator among non-Hard, non-exact-normal
indices, call:

```python
odds_ratio, p_value = scipy.stats.fisher_exact(
    [[candidate_tp, candidate_fp], [noncandidate_tp, noncandidate_fp]],
    alternative="greater",
)
```

Apply Benjamini-Hochberg adjustment across all source/Alpha tests and report
effect sizes and adjusted p-values. Do not create a selection boolean from the
p-values.

- [ ] **Step 7: Run tests and commit**

```bash
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery
git add experiments/exp392_label_guided_discovery
git commit -m "feat: evaluate Exp392 label-guided evidence lattice"
```

---

### Task 4: Stability, Concentration, and Reporting

**Files:**
- Modify: `experiments/exp392_label_guided_discovery/discovery_core.py`
- Modify: `experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py`
- Modify: `experiments/exp392_label_guided_discovery/test_exp392_label_guided_discovery.py`

**Interfaces:**
- Consumes: evaluated dataset-level source and combination rows.
- Produces: bootstrap intervals, lineage concentration diagnostics, Korean report, final manifest.

- [ ] **Step 1: Write failing deterministic-bootstrap and concentration tests**

```python
def test_bootstrap_is_reproducible():
    first = paired_dataset_bootstrap(rows, seed=20260727, iterations=2000)
    second = paired_dataset_bootstrap(rows, seed=20260727, iterations=2000)
    assert first == second

def test_lineage_concentration_flags_single_group_signal():
    result = concentration_diagnostic(
        [{"lineage": "A", "tp": 5}, {"lineage": "B", "tp": 0}]
    )
    assert result["top_lineage_tp_share"] == 1.0
    assert result["concentrated_signal"] is True
```

- [ ] **Step 2: Implement fixed stability strata**

Generate rows for all, TEST `<100`, TEST `>=100`, Zero-F1, no-Hard Zero-F1,
wrong-Hard Zero-F1, and preregistered TRAIN count bands. Dataset lineage is
derived only for retrospective grouping and never passed into candidate
construction.

- [ ] **Step 3: Implement paired dataset bootstrap**

Use `numpy.random.default_rng(20260727)`, 2,000 dataset-resampling iterations,
and percentile 2.5/97.5 intervals for recovered FN, added FP, and candidate
precision. Store seed and iterations in every output and manifest.

- [ ] **Step 4: Implement leave-one-lineage-out concentration diagnostics**

For each observation report the top lineage TP share, number of TP-contributing
lineages, and the min/max recovery after leaving one lineage out. Flag
concentration when one lineage supplies at least 80% of recovered FN; the flag
is diagnostic, not a policy selector.

- [ ] **Step 5: Render the Korean report without a “best” rule**

The report contains:

- frozen baseline;
- source and combination coverage;
- fixed tables sorted by rule ID, not TEST score;
- signals with concentration caveats;
- unsupported observations;
- explicit statement that no operating rule was selected;
- statement that automatic Hard and review lanes were unchanged;
- next-step gate requiring a new preregistration.

Apply the fixed research-label definitions from Global Constraints only after
all raw metrics, bootstrap intervals, and concentration fields have been
computed. No research label changes the ordering or content of raw outputs.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery
git add experiments/exp392_label_guided_discovery
git commit -m "feat: report Exp392 stability and concentration"
```

---

### Task 5: Queue, Documentation, Smoke, and Full Run

**Files:**
- Create: `docs/EXP392_PREREGISTRATION_KO.md`
- Create: `docs/EXP392_RUNBOOK_KO.md`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `experiments/exp392_label_guided_discovery/test_exp392_label_guided_discovery.py`

**Interfaces:**
- Consumes: completed Exp392 runner.
- Produces: queue-visible `experiment_392_label_guided_score_discovery_audit`.

- [ ] **Step 1: Write the failing queue-entry test**

```python
def test_exp392_has_explicit_queue_entry():
    experiment = get_experiment(
        "experiment_392_label_guided_score_discovery_audit"
    )
    assert experiment["script"].name == (
        "run_experiment_392_label_guided_score_discovery_audit.py"
    )
    assert experiment["detail_csv"].name == "exp392_individual_source_summary.csv"
```

- [ ] **Step 2: Add the queue entry**

The entry uses:

```python
{
    "id": "experiment_392_label_guided_score_discovery_audit",
    "script": ROOT / "experiments/exp392_label_guided_discovery"
        / "run_experiment_392_label_guided_score_discovery_audit.py",
    "detail_csv": DATA_DIR / "exp392_label_guided_score_discovery_audit"
        / "exp392_individual_source_summary.csv",
    "summary_csv": DATA_DIR / "exp392_label_guided_score_discovery_audit"
        / "exp392_pair_triple_summary.csv",
    "stdout_log": DATA_DIR / "exp392_label_guided_score_discovery_audit"
        / "exp392_stdout.log",
}
```

The CLI default phase is `all`, so the queue invocation runs `prepare` and then
`evaluate` without additional arguments. If `prepare` fails, `evaluate` must not
start.

- [ ] **Step 3: Write preregistration and runbook**

Document the fixed inputs, phases, exact commands, safety flags, no-selection
rule, retrospective limitation, output schemas, and failure conditions.

- [ ] **Step 4: Run static and unit verification**

```bash
python3 -m py_compile experiments/exp392_label_guided_discovery/*.py
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery \
  experiments.exp391_fn_reachability.test_exp391_fn_reachability
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 5: Run a 10-dataset smoke**

```bash
python3 experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py \
  --phase prepare \
  --dataset-limit 10 \
  --output-dir /Users/minho/Documents/Dataset/exp392_label_guided_score_discovery_audit_smoke

python3 experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py \
  --phase evaluate \
  --output-dir /Users/minho/Documents/Dataset/exp392_label_guided_score_discovery_audit_smoke
```

Verify 10 unique datasets, zero lane mutations, no labels/metrics in the frozen
matrix, all fixed source/combination rows, no out-of-range indices, and matching
artifact hashes.

- [ ] **Step 6: Commit the queue and docs**

```bash
git add run_rank_experiments_sequential.py \
  docs/EXP392_PREREGISTRATION_KO.md \
  docs/EXP392_RUNBOOK_KO.md \
  experiments/exp392_label_guided_discovery/test_exp392_label_guided_discovery.py
git commit -m "docs: register Exp392 label-guided discovery audit"
```

- [ ] **Step 7: Run the full experiment sequentially**

```bash
python3 run_rank_experiments_sequential.py queue \
  experiment_392_label_guided_score_discovery_audit
python3 run_rank_experiments_sequential.py run-next
```

Expected:

- 1,117/1,117 dataset coverage;
- errors `0`;
- queue empty after completion;
- status `completed`;
- no baseline lane changes;
- complete manifest hashes;
- no selected operating policy.

- [ ] **Step 8: Independently reaggregate and final commit**

Use a read-only one-off aggregation to reconcile candidate counts, recovered FN,
added FP, review overlap, and every source/combination row against the
candidate-level matrix. Update the Korean report only for factual consistency,
not to select a rule.

```bash
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery \
  experiments.exp391_fn_reachability.test_exp391_fn_reachability
git diff --check
git add experiments/exp392_label_guided_discovery docs
git commit -m "docs: finalize Exp392 discovery audit results"
```

---

## Completion Criteria

- Prepare phase contains no TEST labels or evaluation metrics.
- Evaluate phase refuses a tampered prepare artifact.
- All 1,117 datasets and every fixed source/combination are represented.
- Existing automatic Hard and review lanes are unchanged.
- No TEST-derived rule selection exists in code or output.
- Source complementarity, stability, concentration, bootstrap, and association
  artifacts reconcile with candidate-level rows.
- Dashboard status reaches completed with zero errors.
- Result report distinguishes research signals from deployable rules.
