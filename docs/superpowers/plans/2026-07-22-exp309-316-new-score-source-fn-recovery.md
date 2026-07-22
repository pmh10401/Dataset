# Exp309-316 New Score Source FN Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and sequentially evaluate three non-learning and three learning-based TRAIN-normal score sources, then combine them as retrospective Shadow evidence without changing Exp291 automatic Hard alerts.

**Architecture:** Add one shared `experiments/exp309_316_fn_recovery` package with strict prediction/evaluation separation. Every source emits cross-fitted TRAIN-normal calibration scores, TEST run-level scores, diagnostics, and provenance through a common interface; Exp312 and Exp316 consume only frozen upstream predictions. A single sequential runner executes one experiment at a time and publishes atomic status for the dashboard.

**Tech Stack:** Python 3, NumPy, SciPy, PyWavelets, PyTorch with Apple MPS, SQLite input helpers, standard-library JSONL/CSV/status reporting, `unittest`.

## Global Constraints

- Database remains `/Users/minho/Documents/Dataset/univariate_ts.db`.
- Output roots are `/Users/minho/Documents/Dataset/exp309_*` through `/Users/minho/Documents/Dataset/exp316_*`; no previous result is overwritten.
- Preprocessing order is exactly `missing_repair -> train_length_alignment -> normalization`.
- TRAIN normal data alone determines feature fitting, OOF calibration, thresholds, source availability, Gate and model stopping.
- Prediction code never accepts or reads TEST labels, anomaly positions, family performance or TEST-derived policy constants.
- `candidate_index` is a TEST instance/run index; within-run time segments are diagnostic evidence only.
- Exp291 automatic Hard indices are copied and hash-checked without modification.
- All new candidates remain retrospective research Shadow; no automatic alert policy is changed.
- Sensitivity alphas are fixed before evaluation as `(0.005, 0.01, 0.02, 0.05)` and no TEST metric selects one.
- Non-learning experiments use at most 7 CPU workers; learning experiments use MPS when available and record CPU fallback.
- TEST labels are loaded only after the prediction JSONL and manifest SHA256 are frozen.
- Every task follows TDD, passes a 10-dataset smoke test before full execution, and commits only its scoped files.

---

## File Structure

- `experiments/exp309_316_fn_recovery/config.py`: frozen IDs, seeds, alphas, scale and neural hyperparameters.
- `experiments/exp309_316_fn_recovery/contracts.py`: source result dataclass, forbidden-field validation, atomic freeze/verify.
- `experiments/exp309_316_fn_recovery/foundation.py`: DB loading, preprocessing-contract assertion, deterministic folds.
- `experiments/exp309_316_fn_recovery/calibration.py`: cross-fit empirical p-values, resolution and candidate extraction.
- `experiments/exp309_316_fn_recovery/discord.py`: Exp309 multi-scale patch discord score.
- `experiments/exp309_316_fn_recovery/change_point.py`: Exp310 multi-window distribution-change score.
- `experiments/exp309_316_fn_recovery/wavelet_phase.py`: Exp311 wavelet energy/phase/coherence score.
- `experiments/exp309_316_fn_recovery/evidence_fusion.py`: Exp312/316 independent-group evidence aggregation and normal Guard.
- `experiments/exp309_316_fn_recovery/neural_common.py`: deterministic torch/MPS setup, batching and model artifact helpers.
- `experiments/exp309_316_fn_recovery/predictive_residual.py`: Exp313 temporal convolution predictor.
- `experiments/exp309_316_fn_recovery/contrastive_memory.py`: Exp314 temporal contrastive encoder and memory score.
- `experiments/exp309_316_fn_recovery/deep_svdd.py`: Exp315 normal hypersphere score.
- `experiments/exp309_316_fn_recovery/pipeline.py`: phase-separated predict/evaluate orchestration for all IDs.
- `experiments/exp309_316_fn_recovery/evaluation.py`: read-only TEST-label metrics and outside-FN accounting.
- `experiments/exp309_316_fn_recovery/reporting.py`: status, CSV, report and manifest writers.
- `experiments/exp309_316_fn_recovery/tests/`: focused unit, contract, source and pipeline tests.
- `scratch/run_exp309_316_sequential.py`: resumable one-at-a-time queue.
- `run_rank_experiments_sequential.py`: queue command registration.
- `serve_rank_dashboard.py`: Exp309-316 catalog/status/result registration.
- `scratch/test_rank_dashboard_operational.py`: dashboard registration and stale-status tests.

---

### Task 1: Shared TRAIN-only Foundation and Artifact Contract

**Files:**
- Create: `experiments/exp309_316_fn_recovery/__init__.py`
- Create: `experiments/exp309_316_fn_recovery/config.py`
- Create: `experiments/exp309_316_fn_recovery/contracts.py`
- Create: `experiments/exp309_316_fn_recovery/foundation.py`
- Create: `experiments/exp309_316_fn_recovery/calibration.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_foundation_contracts.py`

**Interfaces:**
- Produces: `ScoreArtifact(train_oof_scores, test_scores, diagnostics, available, reason)`.
- Produces: `load_preprocessed_bundle(dataset: str) -> DatasetBundle`.
- Produces: `empirical_pvalues(train_scores, query_scores) -> np.ndarray` and `candidate_indices(pvalues, alpha) -> list[int]`.
- Produces: `freeze_prediction_rows(...)` and `verify_prediction_rows(...)`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_preprocessing_contract_and_candidate_grain():
    assert preprocessing_contract()["order"] == [
        "missing_repair", "train_length_alignment", "normalization"
    ]
    assert CANDIDATE_GRAIN == "test_instance_virtual_run"

def test_empirical_pvalues_are_train_only_and_monotone():
    train = np.array([1.0, 2.0, 3.0])
    actual = empirical_pvalues(train, np.array([0.0, 4.0]))
    np.testing.assert_allclose(actual, [1.0, 0.25])

def test_prediction_contract_rejects_evaluation_fields(tmp_path):
    with self.assertRaisesRegex(ValueError, "forbidden prediction field"):
        freeze_prediction_rows([{"dataset": "D", "tp": 1}], tmp_path / "p", tmp_path / "m")
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_foundation_contracts`

Expected: FAIL because the shared modules do not exist.

- [ ] **Step 3: Implement frozen configuration and interfaces**

```python
SEED = 20260722
ALPHAS = (0.005, 0.01, 0.02, 0.05)
CANDIDATE_GRAIN = "test_instance_virtual_run"
CPU_WORKERS = 7

@dataclass(frozen=True)
class ScoreArtifact:
    train_oof_scores: np.ndarray
    test_scores: np.ndarray
    diagnostics: dict[str, object]
    available: bool = True
    reason: str = ""
```

`load_preprocessed_bundle()` must delegate to the existing aligned bundle loader and assert the exact preprocessing contract before returning finite `float32` arrays. `freeze_prediction_rows()` must reject `y_test`, TP/FP/F1 fields and atomically write canonical JSONL plus SHA256 manifest.

- [ ] **Step 4: Run foundation tests**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_foundation_contracts`

Expected: all tests PASS.

- [ ] **Step 5: Commit foundation**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add TRAIN-only FN recovery foundation"
```

---

### Task 2: Exp309 Multi-scale Patch Discord

**Files:**
- Create: `experiments/exp309_316_fn_recovery/discord.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_discord.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Consumes: `DatasetBundle`, deterministic fold indices.
- Produces: `score_discord(train_normal, test_values, seed) -> ScoreArtifact`.

- [ ] **Step 1: Write failing score tests**

```python
def test_local_patch_change_scores_above_repeated_normal():
    train = np.tile(np.sin(np.linspace(0, 6.28, 64)), (8, 1)).astype("float32")
    test = train[:2].copy(); test[1, 24:32] += 5
    artifact = score_discord(train, test, seed=20260722)
    assert artifact.test_scores[1] > artifact.test_scores[0]
    assert artifact.diagnostics["candidate_grain"] == "test_instance_virtual_run"
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_discord`

Expected: FAIL because `score_discord` is missing.

- [ ] **Step 3: Implement deterministic multi-scale scoring**

Use within-run patch lengths derived only from TRAIN target length with fractions `(1/64, 1/32, 1/16, 1/8)`, clipped to `[8, target_length // 4]`. Deterministically cap each scale's TRAIN memory at 2,048 patches. For every run, calculate the nearest-normal distance for each patch and use the 95th percentile across patches, then the maximum robust TRAIN-normal z-score across scales. Generate OOF TRAIN scores with five deterministic folds.

- [ ] **Step 4: Run unit test and 10-dataset prediction smoke**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_discord
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 309 --phase predict --dataset-limit 10 --workers 2 --output-root /tmp/exp309_smoke
```

Expected: 10 prediction rows, no label/metric fields, no index errors.

- [ ] **Step 5: Commit Exp309**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add multi-scale discord score source"
```

---

### Task 3: Exp310 Multi-window Change-point Score

**Files:**
- Create: `experiments/exp309_316_fn_recovery/change_point.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_change_point.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Produces: `score_change_point(train_normal, test_values, seed) -> ScoreArtifact`.

- [ ] **Step 1: Write a failing level-shift and drift test**

```python
def test_level_shift_and_drift_exceed_flat_normal():
    train = np.zeros((10, 96), dtype="float32")
    test = np.zeros((3, 96), dtype="float32")
    test[1, 48:] = 3
    test[2] = np.linspace(0, 3, 96)
    scores = score_change_point(train, test, seed=20260722).test_scores
    assert scores[1] > scores[0] and scores[2] > scores[0]
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_change_point`

- [ ] **Step 3: Implement TRAIN-scaled multi-window divergence**

For window fractions `(1/32, 1/16, 1/8, 1/4)`, compare adjacent windows using robust location difference, log variance ratio and one-dimensional Wasserstein distance. Aggregate the maximum internally, normalize each component from OOF TRAIN distributions, and store the winning window/component as diagnostic evidence. Exclude receptive-field boundary positions from prefix-invariance assertions.

- [ ] **Step 4: Test and smoke**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_change_point
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 310 --phase predict --dataset-limit 10 --workers 2 --output-root /tmp/exp310_smoke
```

Expected: PASS and 10 frozen prediction rows.

- [ ] **Step 5: Commit Exp310**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add multi-window change-point source"
```

---

### Task 4: Exp311 Wavelet Phase and Coherence Score

**Files:**
- Create: `experiments/exp309_316_fn_recovery/wavelet_phase.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_wavelet_phase.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Produces: `score_wavelet_phase(train_normal, test_values, seed) -> ScoreArtifact`.

- [ ] **Step 1: Write a failing phase-disruption test**

```python
def test_phase_disruption_scores_above_stable_periodic_trace():
    x = np.sin(np.linspace(0, 12 * np.pi, 128)).astype("float32")
    train = np.tile(x, (10, 1))
    test = np.stack([x, np.concatenate([x[:64], -x[64:]])])
    scores = score_wavelet_phase(train, test, seed=20260722).test_scores
    assert scores[1] > scores[0]
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_wavelet_phase`

- [ ] **Step 3: Implement fixed wavelet features**

Use complex Morlet CWT scales `(2, 4, 8, 16, 32)` clipped below the TRAIN target length. Extract log energy, phase-increment circular variance, phase entropy and adjacent-scale coherence. Fit robust center/scale from OOF TRAIN feature vectors and score each run by the maximum component z-score. Record unavailable status rather than changing scales for unsupported lengths.

- [ ] **Step 4: Test and smoke**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_wavelet_phase
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 311 --phase predict --dataset-limit 10 --workers 2 --output-root /tmp/exp311_smoke
```

Expected: PASS and finite frozen scores.

- [ ] **Step 5: Commit Exp311**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add wavelet phase coherence source"
```

---

### Task 5: Exp312 Non-learning Evidence Fusion and Normal Guard

**Files:**
- Create: `experiments/exp309_316_fn_recovery/evidence_fusion.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_evidence_fusion.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Consumes: frozen Exp309-311 predictions and frozen Exp291 routing.
- Produces: `fuse_evidence(source_rows, normal_guard, alpha) -> dict[str, object]`.

- [ ] **Step 1: Write failing independent-group and Guard tests**

```python
def test_two_independent_groups_create_shadow_but_exact_normal_suppresses_it():
    evidence = {"discord": {4}, "change_point": {4}, "wavelet_phase": set()}
    assert fuse_indices(evidence, exact_normal=set(), minimum_groups=2) == {4}
    assert fuse_indices(evidence, exact_normal={4}, minimum_groups=2) == set()
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_evidence_fusion`

- [ ] **Step 3: Implement conservative fusion**

Treat Discord, Change-point and Wavelet as three distinct groups. Preserve per-source conformal p-values for all four frozen alphas. Report both Bonferroni combined p-value and the mean of valid p-to-e calibrators `0.5 / sqrt(max(p, attainable_p))`; do not select one from TEST performance. Create research Shadow only when at least two distinct groups pass the same alpha and remove Exp290 exact-normal indices. Copy Exp291 Hard unchanged and assert exact parity.

- [ ] **Step 4: Test, smoke and verify upstream hashes**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_evidence_fusion
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 312 --phase predict --dataset-limit 10 --output-root /tmp/exp312_smoke
```

Expected: source hash checks pass, no lane overlap, Exp291 Hard exact match.

- [ ] **Step 5: Commit Exp312**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: fuse non-learning FN recovery evidence"
```

---

### Task 6: Shared Deterministic MPS Training Runtime

**Files:**
- Create: `experiments/exp309_316_fn_recovery/neural_common.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_neural_common.py`

**Interfaces:**
- Produces: `choose_device(requested: str) -> torch.device`.
- Produces: `seed_everything(seed: int)`, `train_only_split(n_train, seed)`, atomic model save/load.

- [ ] **Step 1: Write failing device, split and determinism tests**

```python
def test_auto_prefers_mps_and_split_never_uses_test_rows():
    assert choose_device("auto", mps_available=True).type == "mps"
    fit, validation = train_only_split(20, seed=20260722)
    assert set(fit).isdisjoint(validation)
    assert sorted(fit + validation) == list(range(20))
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_neural_common`

- [ ] **Step 3: Implement runtime contract**

Set Python, NumPy and torch seeds to `20260722`, use deterministic TRAIN-only 80/20 split, request MPS by default, record fallback reason, and use fixed `epochs=30`, `patience=5`, `batch_size=128`, `learning_rate=0.001`. Never use TEST loss for early stopping.

- [ ] **Step 4: Run CPU/MPS smoke parity test**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_neural_common`

Expected: PASS; deterministic hash equality on CPU repetitions and finite MPS output within `rtol=1e-4, atol=1e-5` where MPS is available.

- [ ] **Step 5: Commit neural runtime**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add deterministic MPS training runtime"
```

---

### Task 7: Exp313 Predictive Residual Source

**Files:**
- Create: `experiments/exp309_316_fn_recovery/predictive_residual.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_predictive_residual.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Produces: `score_predictive_residual(train_normal, test_values, device, seed) -> ScoreArtifact`.

- [ ] **Step 1: Write failing synthetic sequence test**

```python
def test_predictive_residual_detects_broken_continuation():
    train = periodic_training_rows(count=24, length=96)
    test = train[:2].copy(); test[1, 64:] = test[1, 64:][::-1]
    artifact = score_predictive_residual(train, test, device="cpu", seed=20260722, epochs=2)
    assert artifact.test_scores[1] > artifact.test_scores[0]
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_predictive_residual`

- [ ] **Step 3: Implement a lightweight temporal convolution predictor**

Use context length `min(64, target_length // 2)`, horizon `max(4, context // 8)`, two Conv1d layers with 16 channels and a linear horizon head. Train per dataset only when `n_train >= 20` and target length `>= 32`; otherwise emit explicit abstention. Build OOF residual calibration without scoring a TRAIN row with a model trained on that row.

- [ ] **Step 4: Test and smoke on MPS**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_predictive_residual
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 313 --phase predict --dataset-limit 10 --device mps --output-root /tmp/exp313_smoke
```

Expected: no labels, finite residuals, explicit abstentions, model/config hashes.

- [ ] **Step 5: Commit Exp313**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add predictive residual score source"
```

---

### Task 8: Exp314 Temporal Contrastive Memory Source

**Files:**
- Create: `experiments/exp309_316_fn_recovery/contrastive_memory.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_contrastive_memory.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Produces: `score_contrastive_memory(train_normal, test_values, device, seed) -> ScoreArtifact`.

- [ ] **Step 1: Write failing representation-distance test**

```python
def test_shape_reversal_is_farther_from_normal_memory():
    train = periodic_training_rows(count=24, length=96)
    test = np.stack([train[0], train[0][::-1]])
    artifact = score_contrastive_memory(train, test, device="cpu", seed=20260722, epochs=2)
    assert artifact.test_scores[1] > artifact.test_scores[0]
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_contrastive_memory`

- [ ] **Step 3: Implement encoder and TRAIN memory**

Use a three-block dilated Conv1d encoder with 32-dimensional normalized output. Positive pairs are TRAIN-only jitter, scale and contiguous mask augmentations; negatives are other TRAIN runs in the batch. Use fixed temperature `0.1`, the shared 30-epoch/patience contract and five-fold OOF nearest-memory distances. Abstain for `n_train < 20`.

- [ ] **Step 4: Test and MPS smoke**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_contrastive_memory
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 314 --phase predict --dataset-limit 10 --device mps --output-root /tmp/exp314_smoke
```

Expected: deterministic config, finite 32-D embeddings, no TEST-derived fit.

- [ ] **Step 5: Commit Exp314**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add contrastive normal memory source"
```

---

### Task 9: Exp315 Deep SVDD Normal Boundary

**Files:**
- Create: `experiments/exp309_316_fn_recovery/deep_svdd.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_deep_svdd.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Consumes: Exp314 encoder architecture but not TEST embeddings during fitting.
- Produces: `score_deep_svdd(train_normal, test_values, device, seed) -> ScoreArtifact`.

- [ ] **Step 1: Write failing hypersphere-distance test**

```python
def test_shifted_trace_is_farther_from_train_center():
    train = periodic_training_rows(count=24, length=96)
    test = np.stack([train[0], train[0] + np.linspace(0, 4, 96)])
    artifact = score_deep_svdd(train, test, device="cpu", seed=20260722, epochs=2)
    assert artifact.test_scores[1] > artifact.test_scores[0]
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_deep_svdd`

- [ ] **Step 3: Implement TRAIN-only hypersphere optimization**

Initialize the encoder deterministically, compute the center from fit TRAIN embeddings, optimize squared center distance for 10 fixed epochs after shared early-stopped representation warmup, and obtain OOF TRAIN distances by fold. Store center/radius hashes; abstain for `n_train < 20`.

- [ ] **Step 4: Test and MPS smoke**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_deep_svdd
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 315 --phase predict --dataset-limit 10 --device mps --output-root /tmp/exp315_smoke
```

Expected: finite distances and no TEST data in center/radius fitting.

- [ ] **Step 5: Commit Exp315**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: add Deep SVDD normal boundary source"
```

---

### Task 10: Exp316 Full Independent-group Fusion

**Files:**
- Modify: `experiments/exp309_316_fn_recovery/evidence_fusion.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_full_fusion.py`
- Modify: `experiments/exp309_316_fn_recovery/pipeline.py`

**Interfaces:**
- Consumes: frozen Exp309-315 predictions plus Exp291/Exp290 hashes.
- Produces: Exp316 research Shadow rows for every alpha and evidence threshold.

- [ ] **Step 1: Write failing six-source grouping test**

```python
def test_correlated_sources_count_once_per_method_group():
    groups = {
        "local_shape": {"discord"},
        "transition": {"change_point"},
        "frequency": {"wavelet_phase"},
        "prediction": {"predictive_residual"},
        "representation": {"contrastive_memory", "deep_svdd"},
    }
    support = {"contrastive_memory", "deep_svdd", "wavelet_phase"}
    assert independent_group_count(support, groups) == 2
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_full_fusion`

- [ ] **Step 3: Implement frozen fusion variants**

Use five method groups: local shape, transition, frequency, prediction and representation. Report `two_group_shadow` and `three_group_strong_shadow` for all four alphas. Apply exact-normal and multi-space close-normal Guards, preserve Exp291 Hard, and set `automatic_policy_changed=False`, `shadow_autonomous=False`, `primary_policy_selected=False` on every row and manifest.

- [ ] **Step 4: Test and 10-dataset integration smoke**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_full_fusion
python3 -m experiments.exp309_316_fn_recovery.pipeline --experiment 316 --phase predict --dataset-limit 10 --output-root /tmp/exp316_smoke
```

Expected: exact Hard parity, no lane overlaps, source provenance present.

- [ ] **Step 5: Commit Exp316**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: combine FN recovery sources as research Shadow"
```

---

### Task 11: Evaluation, Independent Reaggregation and Reports

**Files:**
- Create: `experiments/exp309_316_fn_recovery/evaluation.py`
- Create: `experiments/exp309_316_fn_recovery/reporting.py`
- Create: `experiments/exp309_316_fn_recovery/independent_reaggregate.py`
- Test: `experiments/exp309_316_fn_recovery/tests/test_evaluation.py`

**Interfaces:**
- Produces per experiment: dataset CSV, summary CSV, evaluation manifest, `RESULT_REPORT.md`.

- [ ] **Step 1: Write failing evaluation-boundary tests**

```python
def test_evaluation_verifies_hash_before_loading_labels():
    events = []
    with patch.object(evaluation, "verify_prediction_rows", lambda *a: events.append("verify") or []), \
         patch.object(evaluation, "load_labels", lambda *a: events.append("labels") or {}):
        evaluation.evaluate(Path("prediction"), Path("manifest"))
    assert events == ["verify", "labels"]
```

- [ ] **Step 2: Verify failure**

Run: `python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_evaluation`

- [ ] **Step 3: Implement fixed retrospective metrics**

Calculate TP, FP, precision, outside-Exp291-Hard TP, outside-all-existing-candidate TP, recovered fraction of 1,529, zero-Hard-TP dataset recovery, exact/±1/±3/±5 matching, size-bin metrics and workload distribution. Keep autonomous metrics and human-assisted/research Shadow metrics in separate columns and report 80% only as a reference line.

- [ ] **Step 4: Run evaluation tests and read-only reaggregation smoke**

Run:

```bash
python3 -m unittest experiments.exp309_316_fn_recovery.tests.test_evaluation
python3 -m experiments.exp309_316_fn_recovery.independent_reaggregate --output-root /tmp/exp316_smoke
```

Expected: independently recomputed totals match pipeline output exactly.

- [ ] **Step 5: Commit evaluation**

```bash
git add experiments/exp309_316_fn_recovery
git commit -m "feat: evaluate FN recovery Shadow experiments"
```

---

### Task 12: Sequential Queue and Dashboard Visibility

**Files:**
- Create: `scratch/run_exp309_316_sequential.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Produces one-at-a-time queue state and Exp309-316 catalog entries.

- [ ] **Step 1: Write failing queue/dashboard registration tests**

```python
def test_exp309_316_are_registered_as_research_shadow():
    for number in range(309, 317):
        info = dash.experiment_info_by_number(number)
        assert info["comparison_eligible"] is False
        assert "자동 경보 변경 없음" in info["autonomous_metric_scope"]
```

- [ ] **Step 2: Verify failure**

Run: `python3 scratch/test_rank_dashboard_operational.py`

- [ ] **Step 3: Implement resumable sequential runner and catalog**

The runner queue is fixed as `309, 310, 311, 312, 313, 314, 315, 316`; it starts only one child process, uses 7 workers for 309-311, MPS for 313-315, writes status atomically after every dataset, skips only experiments with verified completed status and artifacts, and stops on missing coverage or failed hash verification.

- [ ] **Step 4: Run dashboard and runner tests**

Run:

```bash
python3 scratch/test_rank_dashboard_operational.py
python3 scratch/run_exp309_316_sequential.py --dry-run
```

Expected: eight ordered queue entries, no active overlap, dashboard API exposes status/log/result paths.

- [ ] **Step 5: Commit orchestration**

```bash
git add scratch/run_exp309_316_sequential.py run_rank_experiments_sequential.py serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "feat: queue and monitor Exp309 through Exp316"
```

---

### Task 13: Full Verification and Sequential Launch

**Files:**
- Modify only if a pre-label contract defect is found: scoped Exp309-316 files.
- Generate under `/Users/minho/Documents/Dataset/exp309_*` through `exp316_*`.

**Interfaces:**
- Produces full 1,117-dataset prediction/evaluation artifacts and live dashboard progress.

- [ ] **Step 1: Run complete unit suite**

```bash
python3 -m unittest discover -s experiments/exp309_316_fn_recovery/tests -p 'test_*.py'
python3 scratch/test_rank_dashboard_operational.py
```

Expected: all tests PASS before any full prediction.

- [ ] **Step 2: Run all 10-dataset smoke phases**

```bash
for exp in 309 310 311 312 313 314 315 316; do
  python3 -m experiments.exp309_316_fn_recovery.pipeline \
    --experiment "$exp" --phase all --dataset-limit 10 \
    --workers 2 --device auto --output-root "/tmp/exp${exp}_smoke"
done
```

Expected: 10 datasets per experiment, errors 0, forbidden fields 0, index errors 0, lane overlap 0.

- [ ] **Step 3: Freeze input hashes and enqueue full runs**

```bash
python3 scratch/run_exp309_316_sequential.py --reset-queue --workers 7 --device auto
```

Expected: Exp309 becomes running and Exp310-316 remain queued; dashboard shows current dataset, throughput and ETA.

- [ ] **Step 4: After queue completion, independently reaggregate**

```bash
python3 -m experiments.exp309_316_fn_recovery.independent_reaggregate \
  --data-root /Users/minho/Documents/Dataset --experiments 309 310 311 312 313 314 315 316
```

Expected: all pipeline and independent totals match; no missing dataset.

- [ ] **Step 5: Verify repository changes and commit only code/docs**

```bash
git diff --check
git status --short --branch
git add experiments/exp309_316_fn_recovery scratch/run_exp309_316_sequential.py \
  run_rank_experiments_sequential.py serve_rank_dashboard.py scratch/test_rank_dashboard_operational.py
git commit -m "test: verify Exp309 through Exp316 full pipeline"
```

Do not add DB, model cache, prediction JSONL, logs or unrelated dirty files to Git.
