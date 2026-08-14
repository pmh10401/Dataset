# Exp449 Public Baseline and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze six public anomaly-detection baseline authorities and evaluate them against the verified Exp449 Candidate C proposal with preregistered dataset-macro AP/AUROC after a strict single-pass label gate.

**Architecture:** Extend the Exp449 namespace with versioned one-method baseline roots, a six-manifest bridge, a proposal score adapter, a lazy label capability, and pure metric/reporting modules. Existing public-baseline scientific scorers are reused without changing their algorithms, but their Exp448-coupled population, seven-worker runner, label authority, evaluation, and prior roots are not reused. All proposal and baseline score manifests verify before the label module is imported.

**Tech Stack:** Python 3.14.6, NumPy, scikit-learn 1.9.0, SciPy 1.18.0, PyTorch 2.13.0, MLX/MLX-metal 0.32.0 assessment, SQLite, unittest, canonical JSON, SHA-256.

## Global Constraints

- This plan depends on completion and independent verification of `2026-08-14-exp449-proposal-reproducibility-antigravity.md`.
- Every newly created filename contains `antigravity`; do not add `__init__.py` files.
- The exact method order is `knn_distance`, `lof_novelty`, `ocsvm_rbf`, `isolation_forest`, `deep_svdd_1d`, `drocc_1d`.
- Create one absent scientific root per method; never reuse mixed-method, partial, deferred-deep, Exp448, or older baseline roots.
- Classical CPU-parallel scoring uses exactly `max(1, os.cpu_count() - 1)` workers; the currently observed 12 logical CPUs imply 11 workers.
- Deep SVDD and DROCC remain deliberately sequential one-process scientific training stages with one Torch thread; record that they are not CPU-parallel stages.
- Run only one MLX/MPS-heavy process at a time; never combine an outer process pool with concurrent MLX/MPS work unless a separately preregistered parity and end-to-end benchmark passes.
- Preserve each published method's current preprocessing, score definition, score direction, calibration, seed, and 50 official epochs for deep methods.
- Preserve the public baselines' all-TRAIN training population: the label-free values loader must include every TRAIN row, including anomalous TRAIN rows, and its SQL must not reference the label column.
- Every method score/config manifest, the verified proposal manifest, and the six-manifest bridge freeze before importing the label capability.
- Exp449 proposal continuous authority is serialized Candidate C (`hard_plus_stable`) `PolicyRoute.scores`; higher is more anomalous; `score >= 0.5` is the frozen alert rule.
- Positive class is exact SQLite TEXT `'1'`; normal class is exact TEXT `'0'`; reject every other value or type.
- Primary continuous estimands are dataset-macro Average Precision and dataset-macro AUROC over two-class datasets only.
- Single-class dataset metrics are `null`, explicitly listed, and excluded from macro denominators.
- Do not report pooled AP/AUROC as confirmatory metrics; retain micro TP/FP/FN/TN and derived threshold metrics only.
- Do not create matched-budget or label-informed alerts after labels open.
- No baseline scoring or label access is authorized by implementation tasks in this plan.

## File Structure

- `experiments/exp449_reproducible_rebaseline_antigravity/baseline_config_antigravity.py`: method order, roots, seeds, epochs, worker policy, and schemas.
- `experiments/exp449_reproducible_rebaseline_antigravity/baseline_contracts_antigravity.py`: immutable method config, score manifest, bridge, proposal authority, and metric bundle.
- `experiments/exp449_reproducible_rebaseline_antigravity/baseline_runner_antigravity.py`: one-method label-free scoring, failure ledger, resume, freeze, and verification.
- `experiments/exp449_reproducible_rebaseline_antigravity/baseline_data_antigravity.py`: descriptor-pinned, label-free all-TRAIN/TEST values loader and raw-row identity binding.
- `experiments/exp449_reproducible_rebaseline_antigravity/baseline_bridge_antigravity.py`: exact six-method plus proposal manifest bridge.
- `experiments/exp449_reproducible_rebaseline_antigravity/proposal_authority_antigravity.py`: scorer-free Candidate C adapter from verified proposal shards.
- `experiments/exp449_reproducible_rebaseline_antigravity/label_claim_antigravity.py`: label-free, fixed-root, create-only single-use claim publisher.
- `experiments/exp449_reproducible_rebaseline_antigravity/label_access_antigravity.py`: lazy Exp449-only single-pass label capability.
- `experiments/exp449_reproducible_rebaseline_antigravity/evaluation_antigravity.py`: pure dataset metrics and gated evaluation orchestration.
- `experiments/exp449_reproducible_rebaseline_antigravity/reporting_antigravity.py`: macro-first canonical report artifacts.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_config_antigravity.py`: frozen method/root/worker contracts.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_runner_antigravity.py`: one-method scoring, recovery, manifest, and worker tests.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_bridge_antigravity.py`: six-manifest and proposal binding tests.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_evaluation_antigravity.py`: lazy labels, score authority, macro metrics, and no-pooled tests.
- `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_reporting_antigravity.py`: report schema and wording tests.
- `scratch/run_exp449_public_baselines_antigravity.py`: thin label-free baseline CLI.
- `scratch/evaluate_exp449_public_baselines_antigravity.py`: post-freeze evaluation CLI with explicit label authority.

---

### Task 1: Freeze Baseline Identities, Roots, and Contracts

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/baseline_config_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/baseline_contracts_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_config_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: `FROZEN_EXP449_CONFIG`, existing `METHOD_IDS`, current scientific scorer methods.
- Produces: `FROZEN_EXP449_BASELINE_CONFIG`, `Exp449MethodRunConfig`, `Exp449MethodManifest`, `Exp449BaselineBridge`, `Exp449ProposalAuthority`, and `Exp449EvaluationBundle`.

- [ ] **Step 1: Write failing identity tests**

```python
def test_method_order_and_roots_are_exact(self):
    self.assertEqual(FROZEN_EXP449_BASELINE_CONFIG.method_ids, (
        "knn_distance", "lof_novelty", "ocsvm_rbf",
        "isolation_forest", "deep_svdd_1d", "drocc_1d",
    ))
    self.assertEqual(FROZEN_EXP449_BASELINE_CONFIG.deep_epochs, 50)
    self.assertEqual(len(set(FROZEN_EXP449_BASELINE_CONFIG.method_roots.values())), 6)
    self.assertTrue(all("exp449" in str(path) and "antigravity" in str(path)
                        for path in FROZEN_EXP449_BASELINE_CONFIG.method_roots.values()))
    self.assertEqual(
        FROZEN_EXP449_BASELINE_CONFIG.evaluation_root,
        Path("/Users/mac/Documents/Dataset/exp449_evaluation_antigravity_20260814"),
    )
    self.assertEqual(
        FROZEN_EXP449_BASELINE_CONFIG.label_claim_path,
        Path("/Users/mac/Documents/Dataset/exp449_label_access_claim_antigravity_20260814.json"),
    )
```

Also test exact schema fields, lowercase SHA-256, immutable mappings, higher-is-anomaly direction, typed sequential/parallel stage kind, dataset count 1,117, and rejection of Exp448 or mixed-method authority fields.

- [ ] **Step 2: Run and observe import failures**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_baseline_config_antigravity -v
```

- [ ] **Step 3: Implement the frozen baseline config**

Use seed `20260812`, alphas `(0.005, 0.01, 0.02)`, deep epochs `50`, and one fixed absent root per method:

```text
/Users/mac/Documents/Dataset/exp449_baseline_knn_distance_antigravity_20260814
/Users/mac/Documents/Dataset/exp449_baseline_lof_novelty_antigravity_20260814
/Users/mac/Documents/Dataset/exp449_baseline_ocsvm_rbf_antigravity_20260814
/Users/mac/Documents/Dataset/exp449_baseline_isolation_forest_antigravity_20260814
/Users/mac/Documents/Dataset/exp449_baseline_deep_svdd_1d_antigravity_20260814
/Users/mac/Documents/Dataset/exp449_baseline_drocc_1d_antigravity_20260814
```

Freeze the single exact evaluation root `/Users/mac/Documents/Dataset/exp449_evaluation_antigravity_20260814` and the fixed claim path `/Users/mac/Documents/Dataset/exp449_label_access_claim_antigravity_20260814.json`; no alternate Exp449 evaluation root or claim path is accepted. All runtime artifact filenames defined by the config end with `_antigravity.json`, `_antigravity.sha256`, or `_antigravity.jsonl`.

- [ ] **Step 4: Implement immutable contracts**

`Exp449MethodRunConfig` binds exactly one method, Exp449 context/catalog/database hashes, ordered dataset identities, raw row-identity digest, seed, epochs, score direction, abstention policy, stage kind, logical CPU count, policy worker count, actual worker count, inner-thread count, environment, acceleration, method-specific source closure, and no-label flags.

`Exp449MethodManifest` binds exactly 1,117 shard entries and the run-config hash. `Exp449BaselineBridge` binds exactly six method IDs in fixed order, each manifest path/hash/config hash, the verified proposal-manifest path/hash, common catalog/context hashes, and no-label flags.

- [ ] **Step 5: Run tests, synchronize map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_baseline_config_antigravity -v
git add experiments/exp449_reproducible_rebaseline_antigravity/baseline_config_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/baseline_contracts_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_config_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): freeze public baseline authorities"
```

### Task 2: Implement One-Method Label-Free Scoring and Recovery

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/baseline_data_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/baseline_runner_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_runner_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: Exp449 catalog dataset identities, the current public-baseline all-TRAIN/TEST values population, public `prepare_dataset`, `score_published_method`, `conformal_alerts`, and strict artifact helpers.
- Produces: `pair_seed_antigravity(seed, method_id, dataset) -> int`, `score_exp449_method_antigravity(config)`, `freeze_exp449_method_antigravity(root)`, and `verify_exp449_method_antigravity(root)`.

- [ ] **Step 1: Write failing scoring and recovery tests**

Cover one-method-only roots, all-TRAIN/TEST conversion to baseline `DatasetValues`, raw blob identity verification, deterministic pair seed, complete and typed-abstain shards, create-only publication, one failure while later datasets continue, verified reuse, retry queue, exact 1,117 freeze gate, extra/missing/duplicate shard rejection, tamper rejection, and old root/schema rejection. A database fixture must contain an anomalous TRAIN row and prove that it remains in the loaded training population; a static SQL assertion must prove the loader query contains no `label` reference.

- [ ] **Step 2: Observe RED**

- [ ] **Step 3: Implement a label-free raw adapter**

Implement a descriptor-pinned Exp449 values-only loader whose TRAIN query selects all `split='TRAIN'` rows and whose TEST query selects all `split='TEST'` rows, with no label predicate or selected label column. Freeze an all-TRAIN/TEST raw-row digest containing dataset identity, split, instance index, and raw values-blob SHA-256. Construct the existing public-baseline `DatasetValues` from detached values and indices. Do not adapt `paper.data_paper.load_dataset_values`, because its normal-TRAIN population is scientifically different.

- [ ] **Step 4: Implement one pair scoring**

Define `pair_seed_antigravity` as the first eight bytes of SHA-256 over the UTF-8 string `f"{seed}\0{method_id}\0{dataset}"`, interpreted as an unsigned big-endian integer. This makes the seed independent of process order and Python hash randomization.

```python
prepared = prepare_dataset(dataset_values)
result = score_published_method(
    config.method_id, prepared.train, prepared.test,
    pair_seed_antigravity(config.seed, config.method_id, dataset),
    epochs=config.epochs,
)
```

For complete results, serialize exact OOF and TEST float64 scores, TEST indices, label-free conformal alerts for all frozen alphas, status, runtime, dataset/raw identity digest, method/config hash, and `score_direction="higher_is_more_anomalous"`. For abstention, require empty score arrays, empty alerts, and a typed nonempty reason.

- [ ] **Step 5: Implement bounded scheduling and failure persistence**

Classical methods use a rolling window of at most `2 * workers`; deep methods execute one dataset at a time. Parent-only publication validates the typed payload and hash before create-only write. Failures append to a method-specific attempt ledger and do not cancel healthy work. Resume revalidates every shard/meta pair and schedules `catalog - verified_success` only.

- [ ] **Step 6: Implement exact method freeze and verification**

Require exact catalog bijection, one shard per dataset, no extras, no labels/metrics, matching raw identities, method/config/context/database/catalog hashes, and finite complete scores. The manifest lists relative path, size, SHA-256, status, dataset, score direction, and raw identity hash for every entry.

- [ ] **Step 7: Run focused and existing method tests, update map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_baseline_runner_antigravity \
  experiments.paper_public_baselines.tests.test_classical_methods \
  experiments.paper_public_baselines.tests.test_deep_methods \
  experiments.paper_public_baselines.tests.test_preprocessing -v
git add experiments/exp449_reproducible_rebaseline_antigravity/baseline_runner_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/baseline_data_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_runner_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): add resumable one-method baseline scoring"
```

### Task 3: Enforce CPU, Deep-Stage, and Acceleration Policies

**Files:**
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/baseline_runner_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_runner_antigravity.py`
- Create: `scratch/run_exp449_public_baselines_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: `os.cpu_count`, public acceleration assessment primitives, fixed method classes.
- Produces: exact runtime worker records and a label-free CLI with `readiness`, `score-method`, `status-method`, `freeze-method`, and `verify-method`.

- [ ] **Step 1: Write failing worker-policy tests**

Patch logical CPU counts 1, 2, and 12. Classical expected workers are 1, 1, and 11. Reject a caller-supplied value different from policy. Deep methods require actual worker 1, `stage_kind="deliberately_sequential_training"`, Torch thread count 1, and a nonempty rationale; they must never enter the classical ProcessPool branch.

- [ ] **Step 2: Write failing acceleration-record tests**

Require PyTorch, MLX, and MPS availability facts, exact method/stage identity, transfer/compile/sync-inclusive benchmark declaration, coordinate parity by alpha, score tolerance `1e-5`, minimum speedup `1.1`, selected backend, device, versions, CPU fallback reason, and an explicit `one_heavy_process=true`/`concurrent_outer_pool=false` record. Without an exact accelerated implementation, require CPU canonical and do not invent timings. A fixture that enables an outer pool while MLX/MPS is selected must fail.

- [ ] **Step 3: Implement policy records and CLI refusal gates**

The CLI detects workers; it does not accept arbitrary official worker overrides. Every execution subcommand requires a reviewed clean commit and a create-only method-specific execution authority. Readiness and status remain read-only and create no root. All old roots and roots belonging to other methods are rejected. Runtime scheduling enforces one MLX/MPS-heavy process and refuses concurrent outer multiprocessing whenever an accelerated backend is selected.

- [ ] **Step 4: Run tests and prove old public-baseline runner remains unchanged**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_baseline_runner_antigravity \
  experiments.paper_public_baselines.tests.test_runner \
  experiments.paper_public_baselines.tests.test_acceleration -v
git diff --check
```

- [ ] **Step 5: Synchronize map and commit**

```bash
git add experiments/exp449_reproducible_rebaseline_antigravity/baseline_runner_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_runner_antigravity.py \
  scratch/run_exp449_public_baselines_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): enforce baseline execution policy"
```

### Task 4: Bind Candidate C and Six Baseline Manifests Before Labels

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/proposal_authority_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/baseline_bridge_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_bridge_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: independently verified Exp449 proposal manifest and six independently verified method manifests.
- Produces: `load_candidate_c_authority_antigravity(root) -> Exp449ProposalAuthority`, `freeze_baseline_bridge_antigravity(...) -> Exp449BaselineBridge`, and `verify_baseline_bridge_antigravity(...) -> Exp449BaselineBridge`.

- [ ] **Step 1: Write failing proposal-authority tests**

Verify exact use of serialized `hard_plus_stable` scores, no recomputation, higher-is-anomaly, finite score length equal to TEST count, exact alerts equal `{index | score >= 0.5}`, score mask identity, tie `0.5` preserved as alert, missing evidence unchanged, ordered row identities, and no expected-confusion/historical-count fields.

- [ ] **Step 2: Write failing bridge tests**

Require exactly six fixed method IDs and one manifest each. Reject missing, duplicate, seventh, reordered, tampered, mixed-context, mixed-catalog, mixed-database, old schema, old root, unverified proposal, and any labels/metrics. Patch the label module import to raise and prove bridge verification never imports it.

- [ ] **Step 3: Implement scorer-free Candidate C loading**

Call the Exp449 proposal verifier first. Open only manifest-declared relative shard files. Select the policy row with `policy_id == "hard_plus_stable"`; copy exact float64 values and alerts into immutable mappings. Bind the proposal manifest SHA-256, context/catalog hashes, dataset order, and TEST row identities.

- [ ] **Step 4: Implement create-only bridge freeze and verification**

The bridge lists proposal manifest path/hash and six `{method_id, manifest_path, manifest_sha256, run_config_sha256}` rows in fixed order. Paths are root-confined and method roots are distinct. Reverification opens and semantically validates every proposal/baseline manifest before returning the bridge.

- [ ] **Step 5: Run tests, update map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_baseline_bridge_antigravity \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_manifest_verifier_antigravity -v
git add experiments/exp449_reproducible_rebaseline_antigravity/proposal_authority_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/baseline_bridge_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_baseline_bridge_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): bind proposal and baseline score authorities"
```

### Task 5: Add the Exp449 Lazy Single-Pass Label Capability

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/label_claim_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/label_access_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_evaluation_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: verified `Exp449BaselineBridge`, verified Exp449 context/database/catalog, paper descriptor-pinned SQLite boundary.
- Produces: `begin_exp449_label_access_antigravity(...) -> LabelAccessClaimCapability` and `open_exp449_labels_antigravity(claim_capability) -> Exp449LabelSession`.

- [ ] **Step 1: Write failing label-boundary tests**

Prove the label module is absent from `sys.modules` during all proposal/baseline/bridge/claim verification. Missing or invalid proposal/method/bridge manifests must fail before claim creation and label-module import. Exact TEXT `'0'/'1'` converts to 0/1; integer 1, booleans, whitespace, empty text, and other text fail. Dataset order is single-pass and exact; close reauthenticates the database descriptor. Add crash/restart tests: after the permanent claim is created, a simulated crash before the first query and a crash after one query both make every later Exp449 attempt refuse label reopening. Recovery requires a new experiment version, not a new Exp449 root.

- [ ] **Step 2: Observe RED**

- [ ] **Step 3: Implement a versioned capability wrapper**

Implement `label_claim_antigravity.py` without importing the label module. After bridge and all seven score authorities verify, `begin_exp449_label_access_antigravity` checks the one frozen evaluation root and fixed claim path are absent, then publishes the permanent claim directly in the already-existing Dataset parent with `O_CREAT|O_EXCL`, file `fsync`, and parent-directory `fsync` before any evaluation-root creation or label query. It returns an in-memory nonserializable single-use capability. The claim binds the bridge hash, proposal and six method-manifest hashes, database/catalog/context hashes, reviewed commit, evaluation authority hash, fixed evaluation-root identity, and a one-way hash of a process-local secret held only by the capability. Any existing Exp449 claim is terminal and refuses publication; neither another path nor another authority may reopen labels under Exp449. A crash after claim publication but before evaluation-root creation intentionally terminates Exp449 label access.

Only after claim publication does the evaluator lazy-import `label_access_antigravity.py`. `open_exp449_labels_antigravity` requires the live capability object, rehashes and verifies the fixed claim and all bound authorities, verifies the process-local secret, atomically consumes the capability, then pins the database descriptor and performs the first label query. A deserialized claim, reused capability, or restarted process cannot open labels. The session authorizes dataset/name/split/index/label reads, reads each dataset once in frozen order, accepts exact TEXT values only, and closes with descriptor and database reauthentication. It does not import any Exp448 module or require Exp404 coordinates.

- [ ] **Step 4: Add label-access audit artifact semantics**

The session returns an in-memory immutable label mapping and a close receipt containing the claim hash, bridge hash, database hash, dataset count, row count, access count exactly 1, and close status. Publish the close receipt create-only after descriptor reauthentication and before the evaluation bundle; the permanent pre-query claim, not the receipt, is the crash-safe authority that prevents a second label session.

- [ ] **Step 5: Run tests, map synchronization, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_evaluation_antigravity \
  paper.equipment_anomaly_detection.tests_paper.test_data_boundary_paper -v
git add experiments/exp449_reproducible_rebaseline_antigravity/label_access_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/label_claim_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_evaluation_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): gate single-pass label access"
```

### Task 6: Implement Macro AP/AUROC Evaluation Without Pooled Ranking

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/evaluation_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/baseline_contracts_antigravity.py`
- Modify: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_evaluation_antigravity.py`
- Create: `scratch/evaluate_exp449_public_baselines_antigravity.py`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: verified bridge, Candidate C authority, six method authorities, lazy label session.
- Produces: `evaluate_dataset_rows_antigravity(...) -> Exp449EvaluationBundle` and `evaluate_exp449_antigravity(bridge_path, output_root) -> Exp449EvaluationBundle`.

- [ ] **Step 1: Write pure metric RED tests**

Use fixed arrays to test AP/AUROC, score ties, Candidate C threshold ID `candidate_c_score_ge_0_5`, baseline threshold IDs `alpha_0_005`, `alpha_0_01`, and `alpha_0_02`, positive label 1, single-class nulls and identities, macro denominator/count, per-threshold micro TP/FP/FN/TN and derived precision/recall/F1/FPR, abstained baseline rows, and paired dataset rows. Assert pooled `average_precision` and pooled `auroc` keys are absent.

```python
result = evaluate_dataset_rows_antigravity(
    alerts_by_threshold={
        "candidate_c_score_ge_0_5": {"mixed": (1,), "normal_only": ()},
    },
    scores={"mixed": np.array([0.25, 0.75]), "normal_only": np.array([0.2, 0.1])},
    labels={"mixed": (0, 1), "normal_only": (0, 0)},
)
self.assertEqual(result["macro_average_precision"], 1.0)
self.assertEqual(result["macro_auroc"], 1.0)
self.assertEqual(result["undefined_auroc_datasets"], ("normal_only",))
self.assertEqual(result["threshold_metrics"]["candidate_c_score_ge_0_5"]["tp"], 1)
self.assertNotIn("auroc", result["threshold_metrics"]["candidate_c_score_ge_0_5"])
```

- [ ] **Step 2: Write evaluation-order RED tests**

Patch bridge/proposal/method verifiers and label import. Verify all seven score authorities complete before import. Reject any scorer invocation, matched-budget function, conformal recalibration, score normalization, route recomputation, or threshold mutation after labels are opened.

- [ ] **Step 3: Implement the pure metric function**

For each dataset, validate aligned exact binary labels, finite scores, fixed threshold-ID-to-alert mappings, unique bounded alerts, and status. Candidate C must expose only `candidate_c_score_ge_0_5`; every baseline must expose exactly the three preregistered alpha IDs. Compute confusion separately for every frozen threshold without choosing or ranking thresholds after label access. When both classes exist, call `average_precision_score(labels, scores)` and `roc_auc_score(labels, scores)` once per continuous score vector; otherwise store `None` and the exact exclusion reason. Macro values are arithmetic means over valid rows only and record denominator counts.

- [ ] **Step 4: Implement gated evaluation orchestration**

Verify the bridge and all seven authorities, call the label-free claim publisher to obtain the live single-use capability, then lazy-import `open_exp449_labels_antigravity(capability)`. Evaluate Candidate C and six methods through the same pure metric function. A typed-abstain method/dataset has no continuous score, receives null AP/AUROC with abstention reason, and remains in every preregistered threshold/method coverage report. Publish one create-only canonical evaluation bundle and artifact map bound to all input manifest hashes, the pre-query claim, and the label close receipt.

- [ ] **Step 5: Implement a non-accidental evaluation CLI**

The entrypoint requires `--bridge-path`, `--reviewed-commit`, and a create-only evaluation-authority artifact whose hash is preregistered after score freeze. It always uses the single frozen Exp449 evaluation root and accepts no output-root override. Any existing claim or evaluation root refuses the command before label import. Importing the module does not import label access.

- [ ] **Step 6: Run tests, update map, and commit**

```bash
python3 -m unittest \
  experiments.exp449_reproducible_rebaseline_antigravity.tests.test_evaluation_antigravity \
  experiments.paper_public_baselines.tests.test_evaluation_boundary -v
git add experiments/exp449_reproducible_rebaseline_antigravity/evaluation_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/baseline_contracts_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_evaluation_antigravity.py \
  scratch/evaluate_exp449_public_baselines_antigravity.py \
  docs/codemap/codemap_exp449_antigravity.*
git commit -m "feat(exp449): evaluate frozen scores with macro metrics"
```

### Task 7: Add Macro-First Reporting and Final Readiness Review

**Files:**
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/reporting_antigravity.py`
- Create: `experiments/exp449_reproducible_rebaseline_antigravity/tests/test_reporting_antigravity.py`
- Modify: `docs/EXP449_RUNBOOK_antigravity_KO.md`
- Modify: `docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md`
- Modify: `RESEARCH_STATE.json`
- Modify together: `docs/codemap/codemap_exp449_antigravity.{html,json,lock}`

**Interfaces:**
- Consumes: verified immutable `Exp449EvaluationBundle`.
- Produces: canonical CSV/JSON/Markdown report artifacts and implementation-ready baseline/evaluation state.

- [ ] **Step 1: Write failing report-schema tests**

Require method, support count, scored count, abstained count, undefined AP/AUROC identities/counts, macro denominators, macro AP, macro AUROC, explicitly keyed threshold metrics for Candidate C `score >= 0.5` and all three baseline alphas, and input manifest hashes. Reject pooled AP/AUROC columns, any unlabeled or post-label-selected threshold summary, and any wording that claims historical parity or promotion.

- [ ] **Step 2: Implement deterministic reporting**

Sort methods in frozen order and datasets in catalog order. Format AP/AUROC without changing stored numeric values. Write create-only `exp449_evaluation_summary_antigravity.json`, `exp449_dataset_metrics_antigravity.csv`, and `exp449_evaluation_report_antigravity.md` plus one artifact SHA map. Verification rereads and re-renders byte-identically.

- [ ] **Step 3: Update the runbook without authorizing execution**

Document the order: proposal verify → six method readiness → six method score/freeze/verify → bridge freeze/verify → separate evaluation authorization → single label session → evaluation/report. Record the six fixed roots and commands. Keep all score/evaluation started flags false.

- [ ] **Step 4: Run the complete non-execution regression suite**

```bash
python3 -m unittest discover \
  -s experiments/exp449_reproducible_rebaseline_antigravity/tests \
  -p 'test_*_antigravity.py' -v
python3 -m unittest discover \
  -s experiments/paper_public_baselines/tests -p 'test_*.py' -v
python3 -m unittest discover \
  -s paper/equipment_anomaly_detection/tests_paper -p 'test_*_paper.py' -v
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp449_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp449_antigravity.lock >/dev/null
git diff --check
```

- [ ] **Step 5: Obtain fresh independent result-integrity review**

The reviewer verifies that all seven score authorities freeze before labels, no Exp448 authority enters Exp449, Candidate C scores are consumed exactly, worker/acceleration records are truthful, single-class handling and macro denominators are exact, pooled AP/AUROC are absent, label access is one-pass, and no run or evaluation occurred.

- [ ] **Step 6: Fix only blocker/high/medium findings and commit**

```bash
git add experiments/exp449_reproducible_rebaseline_antigravity/reporting_antigravity.py \
  experiments/exp449_reproducible_rebaseline_antigravity/tests/test_reporting_antigravity.py \
  docs/EXP449_RUNBOOK_antigravity_KO.md \
  docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md \
  RESEARCH_STATE.json docs/codemap/codemap_exp449_antigravity.*
git commit -m "docs(exp449): verify baseline evaluation readiness"
```

Set state to `implementation_verified_awaiting_proposal_and_baseline_execution`. Keep proposal smoke, proposal official, six method scoring, bridge freeze, evaluation, and label-access flags false.

## Execution Gates Outside This Plan

This implementation plan does not authorize scoring or labels. Later execution requires this order and separate approvals:

1. complete and verify Exp449 proposal smoke A/B and official proposal freeze;
2. authorize and execute each of the six one-method roots, preserving failures and typed abstentions;
3. verify all six exact manifests;
4. freeze and independently verify the six-manifest/proposal bridge;
5. request explicit evaluation authorization;
6. open TEST labels exactly once;
7. compute and freeze dataset-level and macro metrics;
8. render and independently verify the report;
9. never mutate Exp449 after label access; any scientific correction creates Exp450 or later.
