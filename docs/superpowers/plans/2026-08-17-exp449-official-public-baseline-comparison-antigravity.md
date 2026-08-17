# Exp449 Official Public Baseline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an auditable paper-ready comparison between the frozen proposed detector and six published public baselines over the same 1,117 datasets and 195,900 TEST rows.

**Architecture:** Preserve the four-method classical evaluation unchanged, derive a byte-preserving one-method Deep SVDD view from its completed mixed-root shards, and evaluate the already-frozen DROCC MPS root. Combine the three independently verified evaluation bundles into one canonical comparison artifact and render one technical HTML report plus audit tables.

**Tech Stack:** Python 3.14, NumPy, scikit-learn metrics, canonical JSON/create-only artifacts, unittest, portable Data Analytics HTML report builder.

## Global Constraints

- Do not retrain or transform any stored score.
- Use exactly 1,117 datasets and 195,900 TEST rows in frozen order.
- Primary comparison is `matched_alert_budget`; secondary protocols are conformal alpha `0.005`, `0.01`, and `0.02`.
- Proposed-method AP/AUROC stays absent because no continuous proposal score is frozen.
- DROCC MPS parity is recorded as user-confirmed; no machine-readable parity artifact is invented.
- Every new file name contains `antigravity`.
- Evaluation is retrospective and cannot automatically promote a model or claim field generalization.

---

### Task 1: Freeze a Byte-Preserving Deep SVDD View

**Files:**
- Create: `experiments/paper_public_baselines/official_view_antigravity.py`
- Create: `experiments/paper_public_baselines/tests/test_official_view_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: mixed score root, destination root, and method id `deep_svdd_1d`.
- Produces: `freeze_single_method_view_antigravity(source_root: Path, destination_root: Path, method: str) -> FrozenScoreManifest` and `derived_view_provenance_antigravity.json`.

- [ ] **Step 1: Write the failing byte-preservation tests**

```python
def test_single_method_view_preserves_every_shard_byte(self):
    manifest = freeze_single_method_view_antigravity(source, destination, "deep_svdd_1d")
    self.assertEqual(manifest.pair_count, 2)
    for source_file in sorted((source / "shards/deep_svdd_1d").glob("*.json")):
        target = destination / "shards/deep_svdd_1d" / source_file.name
        self.assertEqual(target.read_bytes(), source_file.read_bytes())

def test_single_method_view_rejects_missing_or_tampered_source(self):
    source_shard.write_bytes(b"{}\n")
    with self.assertRaisesRegex(ValueError, "source shard"):
        freeze_single_method_view_antigravity(source, destination, "deep_svdd_1d")
```

- [ ] **Step 2: Run the focused test and observe RED**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_official_view_antigravity -v`

Expected: import failure because `official_view_antigravity.py` does not exist.

- [ ] **Step 3: Implement the minimal derived-view freezer**

```python
def freeze_single_method_view_antigravity(source_root, destination_root, method):
    source_config, source_sha = _verified_run_config(Path(source_root))
    if method != "deep_svdd_1d" or method not in source_config["methods"]:
        raise ValueError("derived method identity is invalid")
    target_config = dict(source_config)
    target_config["methods"] = [method]
    target_bytes = canonical_json_bytes(target_config)
    target_sha = hashlib.sha256(target_bytes).hexdigest()
    # Verify each source shard/meta against source_sha, copy JSON bytes exactly,
    # and create target metadata bound to target_sha and the unchanged shard SHA.
    # Publish provenance, then call freeze_score_root(destination_root).
```

The implementation must use create-only writes, reject symlinks/extras, and record the source root, source run-config SHA, destination run-config SHA, method, dataset count, and ordered shard-set SHA in provenance.

- [ ] **Step 4: Run focused and runner tests GREEN**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_official_view_antigravity experiments.paper_public_baselines.tests.test_runner -v`

Expected: all tests pass.

- [ ] **Step 5: Refresh all three Exp449 code-map files and commit**

```bash
git add experiments/paper_public_baselines/official_view_antigravity.py \
  experiments/paper_public_baselines/tests/test_official_view_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.{html,json,lock}
git commit -m "feat(baselines): freeze Deep SVDD official view"
```

### Task 2: Combine Independently Verified Evaluation Bundles

**Files:**
- Create: `experiments/paper_public_baselines/official_comparison_antigravity.py`
- Create: `experiments/paper_public_baselines/tests/test_official_comparison_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes: classical, Deep SVDD, and DROCC `EvaluationBundle` objects plus their verified score/evaluation identities.
- Produces: `combine_official_evaluations_antigravity(classical: EvaluationBundle, deep_svdd: EvaluationBundle, drocc: EvaluationBundle, *, drocc_parity_status: str) -> OfficialComparisonAntigravity` and `write_official_comparison_antigravity(root: Path, comparison: OfficialComparisonAntigravity) -> str`.

- [ ] **Step 1: Write failing identity and combination tests**

```python
def test_combiner_requires_exact_six_unique_public_methods(self):
    comparison = combine_official_evaluations_antigravity(
        classical_bundle, deep_svdd_bundle, drocc_bundle,
        drocc_parity_status="user_confirmed_no_difference",
    )
    self.assertEqual(comparison.public_methods, EXPECTED_METHOD_ORDER_ANTIGRAVITY)
    self.assertEqual(comparison.dataset_count, 1117)

def test_combiner_rejects_proposal_or_database_identity_drift(self):
    drifted = replace(deep_svdd_bundle, metadata={**deep_svdd_bundle.metadata, "database_sha256": "0" * 64})
    with self.assertRaisesRegex(ValueError, "database identity"):
        combine_official_evaluations_antigravity(classical_bundle, drifted, drocc_bundle, drocc_parity_status="user_confirmed_no_difference")
```

- [ ] **Step 2: Run the focused test and observe RED**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_official_comparison_antigravity -v`

Expected: import failure because the combiner is absent.

- [ ] **Step 3: Implement exact-bundle combination and create-only writers**

```python
EXPECTED_METHOD_ORDER_ANTIGRAVITY = (
    "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest",
    "deep_svdd_1d", "drocc_1d",
)

def combine_official_evaluations_antigravity(classical, deep_svdd, drocc, *, drocc_parity_status):
    bundles = (classical, deep_svdd, drocc)
    identities = {
        (
            bundle.metadata["database_sha256"],
            bundle.metadata["proposal_reviewed_commit"],
            bundle.metadata["proposal_artifact_map_sha256"],
            bundle.metadata["dataset_count"],
        )
        for bundle in bundles
    }
    if len(identities) != 1 or next(iter(identities))[3] != 1117:
        raise ValueError("evaluation database/proposal identity differs")
    public_rows = tuple(
        row for bundle in bundles for row in bundle.summaries
        if row["is_external_baseline"] is True
    )
    methods = tuple(dict.fromkeys(str(row["method"]) for row in public_rows))
    if methods != EXPECTED_METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("official public method order differs")
    return OfficialComparisonAntigravity.from_verified_bundles_antigravity(
        bundles=bundles,
        public_rows=public_rows,
        drocc_parity_status=drocc_parity_status,
    )
```

Writer outputs under one new root:

- `official_comparison_summary_antigravity.json`
- `official_comparison_summary_antigravity.csv`
- `official_comparison_dataset_metrics_antigravity.csv`
- `official_comparison_bootstrap_antigravity.json`
- `official_comparison_artifact_sha256_antigravity.json`

- [ ] **Step 4: Run comparison, evaluation, and reporting tests GREEN**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_official_comparison_antigravity experiments.paper_public_baselines.tests.test_evaluation_boundary experiments.paper_public_baselines.tests.test_reporting -v`

Expected: all tests pass.

- [ ] **Step 5: Refresh the Exp449 triplet and commit**

```bash
git add experiments/paper_public_baselines/official_comparison_antigravity.py \
  experiments/paper_public_baselines/tests/test_official_comparison_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.{html,json,lock}
git commit -m "feat(baselines): combine official six-method evaluation"
```

### Task 3: Run the Official Evaluation and Freeze the Numerical Tables

**Files:**
- Create: `scratch/run_official_public_baseline_comparison_antigravity.py`
- Create: `experiments/paper_public_baselines/tests/test_official_comparison_cli_antigravity.py`
- Modify: `docs/codemap/codemap_exp449_antigravity.html`
- Modify: `docs/codemap/codemap_exp449_antigravity.json`
- Modify: `docs/codemap/codemap_exp449_antigravity.lock`

**Interfaces:**
- Consumes the fixed source roots from the approved design.
- Produces commands `prepare-deep-view`, `evaluate`, `combine`, and `verify` with canonical JSON stdout.

- [ ] **Step 1: Write failing CLI boundary tests**

```python
def test_verify_never_imports_label_loader_or_scorers(self):
    completed = subprocess.run([PYTHON, SCRIPT, "verify"], text=True, capture_output=True)
    self.assertEqual(completed.returncode, 0)
    self.assertNotIn("label_access", completed.stderr)

def test_evaluate_uses_exact_fixed_roots(self):
    parser = build_parser_antigravity()
    self.assertEqual(parser.parse_args(["evaluate"]).command, "evaluate")
```

- [ ] **Step 2: Run the focused CLI test and observe RED**

Run: `python3 -m unittest experiments.paper_public_baselines.tests.test_official_comparison_cli_antigravity -v`

Expected: failure because the launcher is absent.

- [ ] **Step 3: Implement the closed dispatcher**

```python
COMMANDS_ANTIGRAVITY = ("prepare-deep-view", "evaluate", "combine", "verify")

def main_antigravity(argv=None):
    arguments = build_parser_antigravity().parse_args(argv)
    if arguments.command == "evaluate":
        verify_score_root(DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY)
        verify_score_root(DROCC_MPS_ROOT_ANTIGRAVITY)
        from experiments.paper_public_baselines.evaluation import (
            evaluate_frozen_scores,
            write_evaluation_bundle,
        )
        for root in (DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY, DROCC_MPS_ROOT_ANTIGRAVITY):
            bundle = evaluate_frozen_scores(root, PROPOSAL_AUTHORITY_ROOT_ANTIGRAVITY)
            write_evaluation_bundle(root, bundle)
        return 0
```

The command binds these exact roots:

- classical: `/Users/mac/Documents/Dataset/paper_public_baselines_classical_v1`
- Deep SVDD source: `/Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1`
- Deep SVDD view: `/Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_1d_derived_full_antigravity_20260817`
- DROCC MPS: `/Users/mac/Documents/Dataset/paper_public_baselines_drocc_mps_unverified_antigravity_20260817`
- proposal authority: `/Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812`
- combined output: `/Users/mac/Documents/Dataset/paper_public_baselines_official_comparison_antigravity_20260817`

- [ ] **Step 4: Run focused and complete public-baseline suites GREEN**

Run: `python3 -m unittest discover -s experiments/paper_public_baselines/tests -v`

Expected: all tests pass.

- [ ] **Step 5: Execute and verify the official result**

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -B \
  scratch/run_official_public_baseline_comparison_antigravity.py prepare-deep-view
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -B \
  scratch/run_official_public_baseline_comparison_antigravity.py evaluate
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -B \
  scratch/run_official_public_baseline_comparison_antigravity.py combine
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -B \
  scratch/run_official_public_baseline_comparison_antigravity.py verify
```

Expected: 1,117 datasets; six unique public methods; no missing evaluation rows; scorer-free verify exit 0.

- [ ] **Step 6: Refresh the Exp449 triplet and commit**

```bash
git add scratch/run_official_public_baseline_comparison_antigravity.py \
  experiments/paper_public_baselines/tests/test_official_comparison_cli_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.{html,json,lock}
git commit -m "feat(baselines): run official comparison lifecycle"
```

### Task 4: Build and Validate the Paper-Ready Technical Report

**Files:**
- Create: `/Users/mac/Documents/Dataset/paper_public_baselines_official_comparison_antigravity_20260817/artifact_antigravity.json`
- Create: `/Users/mac/Documents/Dataset/paper_public_baselines_official_comparison_antigravity_20260817/PAPER_OFFICIAL_PUBLIC_BASELINE_COMPARISON_ANTIGRAVITY.html`

**Interfaces:**
- Consumes the verified official comparison summary, dataset metrics, bootstrap rows, source identities, and report artifact map.
- Produces one portable HTML technical report with exact primary and supporting tables.

- [ ] **Step 1: Route chart selection through `data-analytics:visualize-data`**

Use a grouped horizontal bar chart for matched-budget micro/macro F1 and a confidence-interval dot/whisker chart for baseline-minus-proposal macro F1. AP/AUROC remains a separate exact-value table because the proposed method has no comparable continuous score.

- [ ] **Step 2: Build the canonical report artifact**

The visible report sections are:

1. 기술 요약
2. 동일 경보 예산에서 제안기술이 여섯 공개 기준선을 상회
3. 공개 기준선 연속 점수의 AP와 AUROC
4. 비교 범위와 지표 정의
5. 평가 방법과 재현성
6. 한계와 강건성 확인
7. 논문 반영 권고
8. 후속 질문

- [ ] **Step 3: Package and verify the HTML once**

Run from the Data Analytics plugin root:

```bash
npm run report:deliver -- \
  --input /Users/mac/Documents/Dataset/paper_public_baselines_official_comparison_antigravity_20260817/artifact_antigravity.json \
  --output /Users/mac/Documents/Dataset/paper_public_baselines_official_comparison_antigravity_20260817/PAPER_OFFICIAL_PUBLIC_BASELINE_COMPARISON_ANTIGRAVITY.html
```

Expected: artifact validation passes; HTML payload equals the canonical artifact; browser verification passes or explicitly reports `structural_only` if no compatible browser exists.

- [ ] **Step 4: Independently audit report numbers**

Recompute TP/FP/FN/TN sums from dataset-level rows, recompute derived precision/recall/F1, confirm six method identities, confirm each baseline AP/AUROC equals its evaluation bundle, and confirm proposed AP/AUROC is null.

- [ ] **Step 5: Record final artifact identities**

Write the HTML SHA-256 and comparison artifact-map SHA-256 into the handoff. Do not modify frozen score/evaluation roots after this step.
