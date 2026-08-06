# Exp436 Missing-Safe Strict Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute all 1,117 frozen datasets with a label-free detector that skips only nonfinite rows, requires prototype, gradient, and ACF agreement, and remains review-only.

**Architecture:** Exp436 is a new namespace package with its own run identity, contracts, checkpoint, output, and evaluation root. It preserves the verified Exp425 descriptor-bound database snapshot and Exp435 runtime/publication patterns, but adds a values-only raw-blob capability so Exp436 can retain exact row hashes without constructing the finite-only Exp417 `RawTestDataset`. Prediction and evaluation remain separate commands; TEST labels cannot be opened until the prediction tree, manifest, and artifact hashes are frozen and independently verified.

**Tech Stack:** Python 3.14, NumPy float32, SQLite read-only descriptor snapshots, `concurrent.futures.ProcessPoolExecutor`, optional MLX/Metal and PyTorch/MPS behind parity-and-speed gates, canonical JSON/JSONL with SHA256, `unittest`.

## Global Constraints

- Every newly created repository file name contains `antigravity`; do not add `__init__.py` files.
- Exp435 source and failed root are immutable. Exp436 imports no Exp435 checkpoint record and starts all 1,117 datasets from zero.
- Prediction may read normal TRAIN labels only to select the existing normal TRAIN rows. It may read TEST identity and values, but never TEST labels, `labels_blob`, anomaly positions, TEST metrics, or family-specific past TEST performance.
- Nonfinite TRAIN rows are excluded whole; values are never imputed, padded, truncated, copied, or tail-replaced. Their original values-blob SHA256 is retained.
- A length group is calibratable only when at least four finite normal TRAIN rows remain.
- Nonfinite TEST rows produce `nonfinite_test` evidence and no candidate. Finite sibling rows continue.
- Scoring stays float32 with `candidate_alpha=0.01`, `k=3`, TRAIN LOO, `q=0.99`, `method='higher'`, and strict `score > threshold`. Prototype/raw and gradient each use their own TRAIN-group matrix with `center=median(matrix)` and `scale=float32(1.4826 * median(abs(matrix-center)))`; ACF retains its approved formula.
- Candidate rule is `prototype AND gradient AND ACF` outside all four frozen lanes. A missing channel cannot vote true.
- Global candidate budget is 100. The 101st candidate fails the run without trimming or reselection.
- CPU work uses `max(1, (os.cpu_count() or 1) - 1)` dataset workers and one numerical-library thread per worker; this Mac should report 11 workers.
- MLX and PyTorch/MPS use smoke seed exactly `43601`; each backend tests batches `(16, 32, 64, 128, 256)`, runs exactly 2 untimed warm-ups and 5 timed paired CPU/device runs per batch, synchronizes device work inside timing, and takes the median of the 5 runs. A backend selects its passing batch by lowest device median then smaller batch; a shared selector chooses the passing backend by lower device median then lexical backend name (`mlx` before `mps`), otherwise CPU. Adoption remains limited to the exact dense-distance stage when outputs are finite, all preregistered candidate coordinates match CPU, maximum absolute and normalized score error are each at most `1e-5`, and end-to-end stage speedup including transfer and synchronization is at least `1.1x`.
- Only one MLX/MPS-heavy process may run. An adopted accelerator uses no outer process pool.
- Prediction artifacts are review-only and cannot become operational alerts.
- Post-freeze success requires rescued datasets >=10, added false positives <=30, positive paired micro and dataset-balanced macro F1 deltas, and positive lower bounds for both 95% cluster-bootstrap intervals using 20,000 repetitions and seed `43520260806`.
- Before editing a module, use `docs/codemap/codemap.json` to record its callers, downstream dependencies, and tests. Regenerate all three code-map files if stale.
- Each behavior begins with a failing test, is implemented minimally, and is committed only after focused and relevant regression tests pass.

---

## Planned File Structure and Interfaces

### Existing shared boundary, modified additively

- `experiments/exp425_selective_recovery/database.py`
  - Add `Exp425DatabaseSnapshot.load_normal_train_value_blobs(dataset, context) -> tuple[tuple[int, bytes], ...]`.
  - Add `Exp425DatabaseSnapshot.load_test_value_blobs_after_train(dataset, train, context) -> tuple[tuple[int, bytes], ...]`.
  - Both methods reuse the retained descriptor, before/after boundary checks, fixed values-only SQL, sorted identity checks, and no label-bearing return type.
- `experiments/exp425_selective_recovery/tests/test_database.py`
  - Prove exact blob preservation, TEST values-only SQL, TRAIN-before-TEST ordering, manifest identity, and unchanged existing loaders.

### New Exp436 prediction package

- `experiments/exp436_missing_safe_strict_consensus_antigravity/config_antigravity.py`: immutable Exp436 constants and dynamic max-minus-one CPU count.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/contracts_antigravity.py`: raw row/container, filter audit, calibrated groups, strict evidence, candidates, results, backend decision, and run summary.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/inputs_antigravity.py`: frozen population/lane session and exact little-endian float32 decoding from raw blobs.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/features_antigravity.py`: finite-row filtering, audit creation, exact-length grouping, TRAIN-only transforms, and LOO calibration.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/scoring_antigravity.py`: row-local abstention, strict three-channel consensus, lane exclusion, and global budget.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/mlx_antigravity.py`: exact dense-distance MLX implementation and adoption evidence.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/mps_antigravity.py`: exact dense-distance PyTorch/MPS implementation and adoption evidence.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/checkpoint_antigravity.py`: canonical Exp436 context and atomic ordered records.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/progress_antigravity.py`: atomic progress JSON, append-only JSONL/text, heartbeat, PID, dataset metadata, rate, and ETA.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/runtime_antigravity.py`: parent-owned input, serial accelerator path, 11-worker CPU path, ordered commits, and interrupted-only resume.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py`: replay-verified review-only publication and artifact hashes.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/evaluation_antigravity.py`: post-freeze labels, confusion matrices, paired metrics/bootstrap, regressions, and success decision.
- `experiments/exp436_missing_safe_strict_consensus_antigravity/cli_antigravity.py`: fixed `readiness/new/resume/inspect/evaluate` commands and roots.
- `scratch/run_exp436_missing_safe_strict_consensus_antigravity.py`: minimal executable wrapper.

Each new production module has a matching `test_<component>_antigravity.py` under `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/`.

### Governance and generated documentation

- `docs/EXP436_PREREGISTRATION_antigravity_KO.md`
- `docs/EXP436_RUNBOOK_antigravity_KO.md`
- `RESEARCH_STATE.json`
- `EXPERIMENT_CATALOG.json`
- `FAILURE_LEDGER.json` or `CHAMPION_LEDGER.json`, chosen only after frozen evaluation
- `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, and `docs/codemap/codemap.lock`, always regenerated together

---

### Task 0: Freeze the preregistration, runbook, and implementation baseline

**Files:**
- Create: `docs/EXP436_PREREGISTRATION_antigravity_KO.md`
- Create: `docs/EXP436_RUNBOOK_antigravity_KO.md`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`
- Modify together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`

**Interfaces:** The preregistration is the scientific authority; the runbook is the operator authority. Both bind run identity `exp436-missing-safe-strict-consensus-antigravity-20260806`, parent `/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806`, the frozen hashes, missing-row rules, strict consensus, budget, acceleration gates, and success criteria.

- [ ] **Step 1: Write both documents from the approved design before product code.**

Include exact allowed/prohibited inputs, all constants, artifact names, expected commands, failure/resume semantics, and the rule that readiness creates/owns RUN_PARENT while `new` requires that same parent with an exact verified readiness artifact and absent output/checkpoint/evaluation roots.

- [ ] **Step 2: Verify every frozen value is present.**

```bash
rg -n "1117|k.?=.?3|0\.99|higher|100|43520260806|10개|30개|nonfinite_test|prototype.*gradient.*ACF|exp436-missing-safe" \
  docs/EXP436_PREREGISTRATION_antigravity_KO.md \
  docs/EXP436_RUNBOOK_antigravity_KO.md
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
git diff --check
```

Expected: all searches match, both JSON files parse, and whitespace validation exits 0.

- [ ] **Step 3: Regenerate and validate the synchronized code map.**

Use the existing repository code-map generator or the same deterministic procedure recorded in `codemap.lock`. Validate JSON parsing, paths, evidence symbols, node/edge/flow references, HTML payload identity, commit/dirty state, and module fingerprints.

- [ ] **Step 4: Commit the governance baseline.**

```bash
git add RESEARCH_STATE.json EXPERIMENT_CATALOG.json \
  docs/EXP436_PREREGISTRATION_antigravity_KO.md \
  docs/EXP436_RUNBOOK_antigravity_KO.md docs/codemap
git commit -m "docs(exp436): freeze missing-safe strict consensus plan"
```

---

### Task 1: Add an exact values-only raw-blob capability

**Files:**
- Modify: `experiments/exp425_selective_recovery/database.py`
- Modify: `experiments/exp425_selective_recovery/tests/test_database.py`

**Interfaces:** The two new snapshot methods return sorted `(instance_index, original_blob_bytes)` pairs. TRAIN SQL keeps `split='TRAIN' AND label='0'`; TEST SQL selects only `instance_index, values_blob` with `split='TEST'`. Existing `load_train`, `load_test_values_after_train`, and Exp425/Exp435 behavior remain byte-for-byte compatible at their public outputs.

- [ ] **Step 1: Record code-map impact before editing.**

From `docs/codemap/codemap.json`, record that the database boundary is called by experiment input sessions, affects frozen run context and raw TRAIN/TEST loading, and is covered by `experiments/exp425_selective_recovery/tests/test_database.py` plus Exp435 input/runtime regressions.

- [ ] **Step 2: Write failing exact-blob and boundary tests.**

```python
def test_raw_blob_loader_preserves_nonfinite_test_bytes(self):
    raw = np.asarray([1.0, np.nan, np.inf], dtype="<f4").tobytes()
    # Fixture inserts raw into one TEST row and freezes the matching identity.
    rows = snapshot.load_test_value_blobs_after_train("Demo", train, context)
    self.assertEqual(rows, ((0, raw),))

def test_test_blob_loader_requires_bound_train_and_never_selects_labels(self):
    with self.assertRaisesRegex(ValueError, "TRAIN"):
        snapshot.load_test_value_blobs_after_train("Demo", foreign_train, context)
    test_sql = " ".join(statement.lower().split() for statement in traced_statements if "'test'" in statement.lower())
    self.assertNotIn("labels_blob", test_sql)
    self.assertNotRegex(test_sql, r"select .*\blabel\b")
```

- [ ] **Step 3: Run the focused tests and observe failure.**

```bash
python3 -m unittest \
  experiments.exp425_selective_recovery.tests.test_database.Exp425DatabaseSnapshotTests.test_raw_blob_loader_preserves_nonfinite_test_bytes \
  experiments.exp425_selective_recovery.tests.test_database.Exp425DatabaseSnapshotTests.test_test_blob_loader_requires_bound_train_and_never_selects_labels -v
```

Expected: FAIL because the methods do not exist.

- [ ] **Step 4: Implement one fixed private blob query and the two add-only methods.**

```python
def _load_value_blobs(connection: sqlite3.Connection, dataset: str, query: str) -> tuple[tuple[int, bytes], ...]:
    rows = connection.execute(query, (dataset,)).fetchall()
    decoded = tuple((index, bytes(blob)) for index, blob in rows)
    if not decoded or tuple(index for index, _ in decoded) != tuple(sorted(index for index, _ in decoded)):
        raise ValueError("values-only row identities are invalid")
    if len({index for index, _ in decoded}) != len(decoded):
        raise ValueError("values-only row identities contain a duplicate")
    return decoded
```

Both public methods call `verify_boundary()` before and after, require the live frozen context, validate TEST `(index, len(blob)//4)` against `context.test_identity_manifest`, reject non-multiple-of-four blobs, and translate SQLite errors to stable `ValueError` messages. Do not expose the connection or accept caller-provided SQL.

- [ ] **Step 5: Run focused and shared regressions.**

```bash
python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v
python3 -m unittest discover -s experiments/exp435_train_only_consensus_recovery_antigravity/tests -p "test_*_antigravity.py"
```

Expected: PASS; Exp435 still rejects its known nonfinite TEST fixture at the old finite-only container boundary.

- [ ] **Step 6: Commit.**

```bash
git add experiments/exp425_selective_recovery/database.py \
  experiments/exp425_selective_recovery/tests/test_database.py
git commit -m "feat(inputs): expose pinned values-only blobs for exp436"
```

---

### Task 2: Freeze Exp436 configuration and missing-safe raw contracts

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/config_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/contracts_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_config_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_contracts_antigravity.py`

**Interfaces:** `Exp436FrozenConfig`, `FROZEN_CONFIG`, `RawValueSeries`, `MissingSafeRawDataset`, `ExcludedTrainRowAudit`, `TrainPreparation`, `PreparedTrainGroup`, `UnavailableTrainGroup`, `ChannelCalibration`, `CalibratedTrainGroup`, `InstanceScoreEvidence`, `Exp436Candidate`, `Exp436DatasetResult`, `BackendBatchEvidence`, `BackendBenchmarkEvidence`, `BackendDecision`, and `Exp436RunSummary`. Public records provide deterministic JSON-safe payloads; matrix records carry immutable float32 metadata/digests rather than mutable NumPy arrays. The three-level backend evidence retains raw five-pair synchronized timings, the frozen `paired-synchronized-v1` protocol, recomputed medians/speed/parity gates, all five frozen batches and backend provenance, then the dual-backend selection/fallback. Lifecycle proceeds through `initializing`, `running`, `interrupted`/`failed`, `checkpoint_complete`, and `complete_review_only`; it does not change any frozen scientific constant.

- [ ] **Step 1: Write failing config tests.**

```python
def test_frozen_config_and_cpu_policy(self):
    self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp436")
    self.assertEqual(FROZEN_CONFIG.run_identity, "exp436-missing-safe-strict-consensus-antigravity-20260806")
    self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
    self.assertEqual(FROZEN_CONFIG.cpu_workers, max(1, (os.cpu_count() or 1) - 1))
    self.assertEqual(FROZEN_CONFIG.global_candidate_budget, 100)
    self.assertEqual(FROZEN_CONFIG.candidate_alpha, 0.01)
    self.assertEqual(FROZEN_CONFIG.backend_smoke_seed, 43601)
    self.assertEqual(FROZEN_CONFIG.backend_batch_sizes, (16, 32, 64, 128, 256))
    self.assertEqual(FROZEN_CONFIG.backend_warmup_runs, 2)
    self.assertEqual(FROZEN_CONFIG.backend_timed_runs, 5)
    self.assertEqual(FROZEN_CONFIG.robust_scale_multiplier, 1.4826)
    FROZEN_CONFIG.validate()

def test_mutated_config_fails_closed(self):
    with self.assertRaisesRegex(ValueError, "Exp436 frozen config mismatch"):
        replace(FROZEN_CONFIG, neighbor_count=2).validate()
```

- [ ] **Step 2: Write failing raw-container, immutable-byte matrix, exact TEST identity-coverage, strict-consensus, and backend-evidence tests.**

```python
def test_test_container_accepts_nonfinite_values_and_preserves_digest(self):
    row = RawValueSeries(2, (1.0, float("nan")), "a" * 64)
    batch = MissingSafeRawDataset("Demo", "TEST", (row,))
    self.assertTrue(math.isnan(batch.series[0].values[1]))

def test_consensus_requires_all_three_channels(self):
    evidence = InstanceScoreEvidence.fixture(
        prototype_abnormal=True, gradient_abnormal=True, acf_abnormal=False
    )
    self.assertFalse(evidence.consensus_candidate)
    with self.assertRaisesRegex(ValueError, "consensus"):
        replace(evidence, consensus_candidate=True)
```

- [ ] **Step 3: Confirm the tests fail on missing modules.**

```bash
python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_config_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_contracts_antigravity -v
```

- [ ] **Step 4: Implement validated immutable contracts.**

`RawValueSeries` accepts numeric finite or nonfinite values but validates sorted identity at the container level and requires an exact 64-character lowercase blob digest. `MissingSafeRawDataset` accepts only `TRAIN` or `TEST`. `ExcludedTrainRowAudit` stores dataset, instance index, length, nonfinite count, and blob digest. `PreparedTrainGroup` and `CalibratedTrainGroup` copy finite float32 C-order arrays into immutable bytes-backed storage, so `setflags(write=True)` fails. `Exp436DatasetResult` carries exact sorted TEST identities and exactly one evidence record for each, including non-contiguous identities. `InstanceScoreEvidence.__post_init__` recomputes `prototype_abnormal and gradient_abnormal and acf_abnormal`; nonfinite TEST abstention cannot be a candidate. No published scalar score/timing/calibration value may be nonfinite. Backend tests reject malformed timing pairs and caller-supplied derived median/speed values, incomplete batches, and inconsistent batch/backend tie or CPU fallback decisions.

- [ ] **Step 5: Run tests and commit.**

```bash
python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_config_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_contracts_antigravity -v
git add experiments/exp436_missing_safe_strict_consensus_antigravity
git commit -m "feat(exp436): freeze missing-safe scientific contracts"
```

---

### Task 3: Build the label-free Exp436 input session

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/inputs_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_inputs_antigravity.py`

**Interfaces:** `Exp436InputSession.load_train(dataset) -> MissingSafeRawDataset`, `Exp436InputSession.load_test_values(dataset, train) -> MissingSafeRawDataset`, and `open_exp436_input_session(..., verified_code_commit)`. `Exp436InputSession` privately retains the exact raw TRAIN blob tuple returned by this snapshot as the DB authorization proof, binds that tuple to the exact returned `MissingSafeRawDataset` object, and passes the retained tuple to the TEST loader only after verifying the caller supplied that exact public TRAIN object. It does not use an Exp417 `RawTrainDataset` as this proof, and it exposes no TEST label method.

- [ ] **Step 1: Write failing decoder and session tests.**

```python
def test_decoder_hashes_original_blob_before_float_conversion(self):
    raw = bytes.fromhex("0000803f0000c07f0000807f")
    row = decode_raw_value_row_antigravity(7, raw)
    self.assertEqual(row.raw_blob_sha256, sha256(raw).hexdigest())
    self.assertEqual(len(row.values), 3)

def test_test_load_requires_the_exact_train_object(self):
    train = session.load_train("Demo")
    with self.assertRaisesRegex(ValueError, "bound TRAIN"):
        session.load_test_values("Demo", copy.deepcopy(train))
```

Add a spy connection test that fails if a TEST query contains `label`, `labels_blob`, anomaly, or evaluation fields.

- [ ] **Step 2: Run and observe failure.**

```bash
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_inputs_antigravity -v
```

- [ ] **Step 3: Implement exact decoding and the session.**

```python
def decode_raw_value_row_antigravity(index: int, blob: bytes) -> RawValueSeries:
    if len(blob) == 0 or len(blob) % 4:
        raise ValueError("raw values blob length is invalid")
    values = np.frombuffer(blob, dtype="<f4")
    return RawValueSeries(index, tuple(float(value) for value in values), sha256(blob).hexdigest())
```

Use `freeze_lane_inputs` before opening the snapshot, require the exact sorted 1,117-name lane population, bind the exact reviewed commit, and validate TEST identity/lengths twice: in the shared snapshot method and in the Exp436 container. `label_access_count` remains zero throughout prediction.

- [ ] **Step 4: Run focused plus Exp417/425 boundary regressions.**

```bash
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_inputs_antigravity -v
python3 -m unittest \
  experiments.exp417_causal_multihorizon_shadow.tests.test_inputs \
  experiments.exp425_selective_recovery.tests.test_database -v
```

- [ ] **Step 5: Commit.**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/inputs_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_inputs_antigravity.py
git commit -m "feat(exp436): preserve missing-safe label-free inputs"
```

---

### Task 4: Filter only bad TRAIN rows and preserve an audit

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/features_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_features_antigravity.py`

**Interfaces:** `prepare_train_groups(train) -> TrainPreparation`, `fit_calibrated_group(group) -> CalibratedTrainGroup | UnavailableTrainGroup`, `acf_vector`, and `score_channel_cpu`. `TrainPreparation(dataset, train_identity_lengths, groups, excluded_rows)` returns ordered exact-length groups plus ordered excluded-row audits and binds the exact sorted source `(instance_index, length)` manifest: surviving group rows and excluded rows are disjoint and their union must cover every source identity exactly. An all-nonfinite length therefore uses `UnavailableTrainGroup(..., instance_indices=())`.

- [ ] **Step 1: Write failing row-local filtering tests.**

```python
def test_one_nonfinite_train_row_is_excluded_but_four_finite_siblings_continue(self):
    train = raw_train(rows=[finite(0), finite(1), nonfinite(2), finite(3), finite(4)])
    prepared = prepare_train_groups(train)
    group = prepared.groups[0]
    self.assertEqual(group.instance_indices, (0, 1, 3, 4))
    self.assertEqual(prepared.excluded_rows[0].instance_index, 2)
    self.assertEqual(prepared.excluded_rows[0].nonfinite_count, 1)

def test_three_finite_rows_after_filter_abstain_only_that_length(self):
    prepared = prepare_train_groups(raw_train_with_two_lengths())
    self.assertEqual(prepared.groups[0].reason, "insufficient_finite_train_rows")
    self.assertIsInstance(prepared.groups[1], PreparedTrainGroup)
```

Add assertions that retained arrays equal the original finite rows exactly and no output length changes.

- [ ] **Step 2: Run and observe failure.**

```bash
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_features_antigravity -v
```

- [ ] **Step 3: Implement finite-row filtering before matrix construction.**

```python
for series in sorted(train.series, key=lambda item: item.instance_index):
    values = np.asarray(series.values, dtype=np.float32)
    nonfinite_count = int(values.size - np.count_nonzero(np.isfinite(values)))
    if nonfinite_count:
        excluded.append(ExcludedTrainRowAudit.from_series(train.dataset, series, nonfinite_count))
    else:
        finite_by_length.setdefault(len(values), []).append(series)
```

For every original length, emit either a finite `PreparedTrainGroup` or `UnavailableTrainGroup(reason='insufficient_finite_train_rows')`. Freeze `candidate_alpha=0.01` and compute each prototype/raw and gradient TRAIN-group transform as `center=median(matrix)` and `scale=float32(1.4826 * median(abs(matrix-center)))`; do not share a matrix between those channels. ACF keeps its already approved formula. Reuse the remaining Exp435 float32 LOO, digest, and threshold formulas exactly; change no numerical rule.

- [ ] **Step 4: Prove parity on finite fixtures.**

For an all-finite fixed fixture, compare every Exp436 center, scale, LOO score, threshold, and reference digest with Exp435 to `0.0` exact difference where serialization is shared and `<=1e-7` for scalar float conversion.

**Task 4 architecture ruling (2026-08-06):** zero scale is channel-local.  A
`CalibratedTrainGroup` therefore permits `values=None` exactly when the
prototype calibration is structurally unavailable for `zero_scale`; finite
gradient and ACF matrices/calibrations remain publishable when constructible.
The same independent rule applies to gradient and ACF availability.  In
contrast, a nonfinite result created after finite-row filtering (center,
scale, standardization, ACF arithmetic/vector, distance, LOO score, or
threshold) crosses the whole-prediction numerical-failure boundary and raises
a stable `ValueError`; it must never be rewritten as channel unavailability.
Prepared and calibrated TRAIN groups retain raw lengths of one or greater so
the preparation partition covers every source length; the frozen scored TEST
minimum remains two and is enforced later.  Direct calibrated-group validation
allows only `prototype=zero_scale`, `gradient=zero_scale`, and
`acf=short_series` below the ACF minimum or `acf=zero_acf_energy` at/above it.
No other unavailable reason can encode a numerical failure.

```bash
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_features_antigravity -v
```

- [ ] **Step 5: Commit.**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/features_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_features_antigravity.py
git commit -m "feat(exp436): filter nonfinite train rows by exact length"
```

---

### Task 5: Implement row-local TEST abstention and strict three-channel scoring

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/scoring_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_scoring_antigravity.py`

**Interfaces:** `score_test_instance_cpu`, `score_dataset_cpu`, `score_dataset_backend`, `validate_global_candidates`, and `CandidateBudgetExceeded`.

- [ ] **Step 1: Write failing row-local TEST tests.**

```python
def test_nonfinite_test_row_abstains_without_blocking_finite_sibling(self):
    result = score_dataset_cpu(groups, test_rows(nonfinite(0), finite(1)), empty_lanes())
    self.assertEqual(result.evidence[0].exclusion_reason, "nonfinite_test")
    self.assertEqual(result.evidence[0].status, "abstained")
    self.assertEqual(result.evidence[1].status, "scored")

def test_missing_channel_never_forms_candidate(self):
    evidence = score_test_instance_cpu(row, group_with_unavailable_acf(), empty_lanes())
    self.assertFalse(evidence.consensus_candidate)
    self.assertEqual(evidence.exclusion_reason, "three_channel_consensus_unavailable")
```

- [ ] **Step 2: Write failing strict-candidate and budget tests.**

```python
def test_candidate_requires_three_abnormal_channels_and_no_lane(self):
    self.assertTrue(score(all_three_abnormal=True, in_lane=False).consensus_candidate)
    self.assertFalse(score(acf_abnormal=False, in_lane=False).consensus_candidate)
    self.assertEqual(score(all_three_abnormal=True, in_lane=True).exclusion_reason, "frozen_lane")

def test_101_candidates_fail_without_truncation(self):
    with self.assertRaisesRegex(CandidateBudgetExceeded, "observed 101, budget 100"):
        validate_global_candidates(results_with_candidates(101))
```

- [ ] **Step 3: Run and observe failure.**

```bash
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_scoring_antigravity -v
```

- [ ] **Step 4: Implement row precheck and strict consensus.**

The first operation on each TEST row is a float32 dimensionality/finiteness check. A nonfinite row returns `_abstention(..., reason='nonfinite_test')` before group lookup or channel transforms. For finite rows, preserve all available scalar evidence and compute:

```python
consensus = bool(prototype_abnormal and gradient_abnormal and acf_abnormal)
```

If any channel score/threshold is absent, set `consensus_candidate=False` and preserve the channel reason. Lane checks happen only after all three booleans are true and fail closed on lane lookup errors.

**Settled evidence and numerical-failure ruling:** `InstanceScoreEvidence` records
optional exact per-channel unavailability reasons (`zero_scale` for prototype or
gradient; `short_series` or `zero_acf_energy` for ACF) in its canonical payload
and digest. A scalar-free routing abstention is limited to `nonfinite_test`,
`insufficient_series_length`, `missing_length`, or
`insufficient_finite_train_rows`; a TRAIN-unavailable channel instead preserves
all calculable sibling scalar pairs under
`three_channel_consensus_unavailable`. Any nonfinite result after the finite
raw-row check, or an error from the injected dense scorer, terminates prediction
with `ValueError`; it is never converted into an abstention. A finite
zero-energy TEST waveform records `zero_acf_energy` and is never substituted as
raw input to the ACF scorer.

- [ ] **Step 5: Run tests and finite-fixture parity regression.**

```bash
python3 -m unittest experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_scoring_antigravity -v
python3 -m unittest experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_scoring_antigravity -v
```

- [ ] **Step 6: Commit.**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/scoring_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_scoring_antigravity.py
git commit -m "feat(exp436): require strict three-channel candidates"
```

---

### Task 6: Re-run MLX and PyTorch/MPS readiness gates

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/mlx_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/mps_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_mlx_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_mps_antigravity.py`

**Interfaces:** `benchmark_mlx_adoption(fixtures, seed) -> BackendDecision`, `benchmark_mps_adoption(fixtures, seed) -> BackendDecision`, `score_with_backend`, and one deterministic selector that chooses the fastest passing backend or CPU.

- [ ] **Step 1: Write failing parity, candidate-identity, speed, and fallback tests for both backends.**

```python
def test_backend_rejects_speed_without_candidate_identity(self):
    decision = benchmark(fake_engine(speedup=2.0, shifted_candidate=True))
    self.assertFalse(decision.adopted)
    self.assertIn("candidate identity", decision.reason)

def test_backend_requires_end_to_end_speedup(self):
    decision = benchmark(fake_engine(speedup=1.09, exact=True))
    self.assertFalse(decision.adopted)
```

For each backend assert smoke seed `43601`, batches `(16, 32, 64, 128, 256)`, exactly 2 untimed warm-ups and exactly 5 timed paired CPU/device runs per batch, device synchronization inside timing, and median-of-5 timing. Assert a backend chooses the passing lowest device median batch (tie: smaller batch), the shared selector chooses lower device median across passing MLX/MPS (exact tie: `mlx` before `mps`), and neither passing selects CPU.

- [ ] **Step 2: Run and observe failure.**

```bash
python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_mlx_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_mps_antigravity -v
```

- [ ] **Step 3: Implement exact dense RMS-distance kernels only.**

Do not move normalization, ACF, thresholds, consensus, lanes, or candidate limits to a device. MLX uses `mx.eval`; MPS uses `torch.mps.synchronize()` before stopping the timer. Use seed `43601`, exactly 2 untimed warm-ups, exactly 5 timed paired CPU/device runs, and the median of five synchronized device timings for every batch. Record Python, NumPy, MLX, mlx-metal, PyTorch, device, Metal availability, seed, fixture hashes, timings, batch evidence, maximum errors, candidate identity, selected batch, backend-selection tie break, speedup, and fallback reason.

- [ ] **Step 4: Run both backend test suites.**

```bash
python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_mlx_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_mps_antigravity -v
```

Expected: PASS. The real hardware smoke runs in Task 9 after the fixed CLI wrapper exists. Exp435's prior MLX result (~0.10x CPU) is provenance, not a forced Exp436 outcome.

- [ ] **Step 5: Commit.**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/{mlx,mps}_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_{mlx,mps}_antigravity.py
git commit -m "feat(exp436): gate mlx and mps dense scoring"
```

---

### Task 7: Add durable checkpoint, progress, and runtime execution

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/checkpoint_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/progress_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/runtime_antigravity.py`
- Create: matching three test files.

**Interfaces:** `Exp436CheckpointStore`, `ProgressWriter`, `run_exp436_new`, `resume_exp436`, and `inspect_exp436`. Context binds the run identity, reviewed commit, database/lane hashes, backend decision, frozen config, exact 1,117 datasets, TEST identities, and zero Exp435 imports.

- [ ] **Step 1: Write failing checkpoint tests.**

Test canonical create/open, context mismatch, ordered unique records, 101st-candidate rejection before commit, no Exp435 record import field, interrupted resume, failed resume rejection, and checkpoint-ahead-of-event crash reconciliation.

- [ ] **Step 2: Write failing progress tests.**

```python
def test_progress_contains_required_live_fields(self):
    payload = writer.progress
    for key in ("pid", "current_dataset", "dataset_metadata", "stage", "backend",
                "completed_datasets", "total_datasets", "success", "abstention", "error",
                "excluded_train_rows", "nonfinite_test_rows", "datasets_per_second",
                "elapsed_seconds", "eta_seconds", "checkpoint_path", "latest_error"):
        self.assertIn(key, payload)
```

Also test atomic replacement, append-only event/text logs, heartbeat, `current_dataset=None` only during idle/terminal stages, and PID liveness.

- [ ] **Step 3: Write failing runtime tests.**

Test serial/parallel prediction identity, out-of-order worker completion with ordered commits, max-minus-one worker selection, one inner numerical thread, parent-only SQLite/checkpoint writes, row-audit counters, accelerator serial enforcement, and no publication on failure.

- [ ] **Step 4: Run and observe failure.**

```bash
python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_progress_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity -v
```

- [ ] **Step 5: Implement by adapting the verified Exp435 durability pattern with Exp436 types.**

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `NUMBA_NUM_THREADS` to `1` before importing NumPy transitively. Parent loads raw values; workers receive immutable value records and lane snapshots. Keep at most one in-flight dataset per worker and commit results in frozen dataset order. Runtime updates missing-row counts from each committed result and calculates rate/ETA from committed datasets.

- [ ] **Step 6: Run focused and Exp435 durability regressions.**

```bash
python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_progress_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_runtime_antigravity -v
python3 -m unittest \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_progress_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_runtime_antigravity -v
```

- [ ] **Step 7: Commit.**

```bash
git add experiments/exp436_missing_safe_strict_consensus_antigravity/{checkpoint,progress,runtime}_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_{checkpoint,progress,runtime}_antigravity.py
git commit -m "feat(exp436): add ordered resumable execution"
```

---

### Task 8: Separate publication, evaluation, and CLI firewalls

**Files:**
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/reporting_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp436_missing_safe_strict_consensus_antigravity/cli_antigravity.py`
- Create: `scratch/run_exp436_missing_safe_strict_consensus_antigravity.py`
- Create: matching three test files.

**Interfaces:** `publish_exp436`, `inspect_publication`, `evaluate_exp436`, `decide_exp436_success`, `build_parser`, and `main`. Prediction commands must not import the evaluation module; evaluate verifies the frozen publication first and only then opens TEST labels.

- [ ] **Step 1: Write failing publication tests.**

Test independent checkpoint replay, exact artifact inventory, no clobber, candidate order/deduplication/budget, audit-ledger hashes, `review_only=true`, `operational_alert_eligible=false`, and forbidden label/evaluation keys.

- [ ] **Step 2: Write failing evaluation firewall tests.**

```python
def test_label_loader_is_not_called_before_frozen_publication_verifies(self):
    with patch.object(module, "_load_labels", side_effect=AssertionError("label access")):
        with self.assertRaisesRegex(ValueError, "publication"):
            evaluate_exp436(tampered_publication, evaluation_root)

def test_success_requires_every_preregistered_gate(self):
    decision = decide_exp436_success(metrics_fixture(micro_ci_low=0.0))
    self.assertFalse(decision.success)
```

- [ ] **Step 3: Write failing CLI tests.**

Test fixed roots, clean reviewed HEAD, readiness creates/owns RUN_PARENT, `new` requires the same parent and exact verified readiness artifact while output/checkpoint/evaluation roots are absent and never deletes/overwrites readiness, interrupted-only `resume`, read-only `inspect`, evaluate-after-freeze, no hidden tuning flags, and exactly `FROZEN_CONFIG.cpu_workers` for CPU production.

- [ ] **Step 4: Run and observe failure.**

```bash
python3 -m unittest \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_evaluation_antigravity \
  experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity -v
```

- [ ] **Step 5: Implement publication by adapting Exp435 replay with Exp436 schemas.**

Publish prediction, dataset summary, score evidence, `train_exclusion_audit_antigravity.jsonl`, TEST abstention ledger, backend decision, manifest, status, and artifact hash map. The manifest and `artifact_sha256_antigravity.json` must bind `train_exclusion_audit_antigravity.jsonl`. Use create-only writes followed by exclusive directory rename. Re-open and verify every artifact before returning `FrozenPublication`.

- [ ] **Step 6: Implement post-freeze evaluation.**

Verify prediction and manifest hashes, snapshot the frozen source tree, then load TEST labels. Combine Exp436 candidates with the frozen Exp404 Hard baseline, compute per-dataset and global confusion matrices, micro/macro F1, rescued datasets, added false positives, all regressions, strong regressions, and paired cluster bootstraps. Reverify the frozen source tree after label use.

- [ ] **Step 7: Implement the fixed CLI and wrapper.**

```python
from experiments.exp436_missing_safe_strict_consensus_antigravity.cli_antigravity import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run focused tests and commit.**

```bash
python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p "test_*_antigravity.py"
git add experiments/exp436_missing_safe_strict_consensus_antigravity/{reporting,evaluation,cli}_antigravity.py \
  experiments/exp436_missing_safe_strict_consensus_antigravity/tests/test_{reporting,evaluation,cli}_antigravity.py \
  scratch/run_exp436_missing_safe_strict_consensus_antigravity.py
git commit -m "feat(exp436): freeze publication and evaluation firewall"
```

---

### Task 9: Integrated verification, code-map synchronization, and readiness

**Files:**
- Modify together: `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, `docs/codemap/codemap.lock`
- Modify: `RESEARCH_STATE.json`
- Modify: `EXPERIMENT_CATALOG.json`

- [ ] **Step 1: Run all Exp436 tests.**

```bash
python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p "test_*_antigravity.py"
```

Expected: PASS with zero skips for scientific, firewall, checkpoint, and runtime tests. Hardware-unavailable tests must assert a recorded CPU fallback rather than silently skip.

- [ ] **Step 2: Run shared regression suites.**

```bash
python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v
python3 -m unittest discover -s experiments/exp435_train_only_consensus_recovery_antigravity/tests -p "test_*_antigravity.py"
python3 -m unittest discover -s experiments/exp426_verified_baseline/tests -p "test_*_antigravity.py"
python3 -m unittest discover -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
```

- [ ] **Step 3: Run static governance checks.**

```bash
rg -n "TEST.*label|labels_blob|family.*performance|tail replacement" \
  experiments/exp436_missing_safe_strict_consensus_antigravity
rg --files experiments/exp436_missing_safe_strict_consensus_antigravity scratch docs | \
  python3 -c 'import sys; bad=[p.strip() for p in sys.stdin if "exp436" in p and "antigravity" not in p.split("/")[-1]]; assert not bad, bad'
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
git diff --check
```

Inspect every match; only explicit prohibition tests/docs or the post-freeze evaluation module may refer to TEST labels.

- [ ] **Step 4: Generate all three code-map files together and validate them.**

The map must show the new raw-blob boundary, missing-safe input/feature/scoring path, accelerator readiness choice, ordered runtime/checkpoint path, publication freeze, and post-freeze evaluation. Every node/edge needs path and symbol evidence; unknown relations must be marked `unknown`.

- [ ] **Step 5: Obtain a fresh read-only implementation review.**

The reviewer checks the approved spec against code and tests, with special attention to original blob hashes, row-local abstention, strict AND invariant, TEST-label import firewall, 101st-candidate failure, failed-root rejection, CPU worker count, and readiness provenance. Resolve every blocking finding with a new failing test.

- [ ] **Step 6: Commit the verified implementation state and code map.**

```bash
git add RESEARCH_STATE.json EXPERIMENT_CATALOG.json docs/codemap
git commit -m "docs(exp436): record verified implementation readiness"
git status --short --branch
```

Expected: clean worktree after the commit.

- [ ] **Step 7: From a previously absent RUN_PARENT, generate official readiness at the reviewed clean HEAD.**

```bash
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py readiness
```

Readiness must atomically create the previously absent RUN_PARENT and its `readiness_antigravity.json`, then bind the clean commit, DB/lane hashes, focused test evidence, CPU/MLX/MPS benchmark evidence, selected backend, and fallback reason. It fails closed if the parent already exists. `inspect` is not called until a prediction root exists.

---

### Task 10: Run, freeze, and evaluate all 1,117 datasets

**Files outside Git:**
- Create only under `/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806`

- [ ] **Step 1: Prove the readiness-owned root and exact launch preconditions, then launch once.**

```bash
test -d /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806
test -f /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/readiness_antigravity.json
test ! -e /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/output_antigravity
test ! -e /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/checkpoint_antigravity
test ! -e /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/evaluation_antigravity
nohup python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py new \
  >/tmp/exp436_new_antigravity.log 2>&1 &
exp436_launcher_pid=$!
ps -p "$exp436_launcher_pid" -o pid=,command=
```

The background process writes its own authoritative PID, progress JSON, events JSONL, parent text log, checkpoint, current dataset, sizes/lengths, stage/backend, counts, missing-row audits, speed, elapsed time, ETA, and latest error. `/tmp/exp436_new_antigravity.log` captures only launcher-level stdout/stderr. Do not spend model tokens polling it.

- [ ] **Step 2: If interrupted, validate then resume; if failed, stop.**

```bash
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py inspect
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py resume
```

Run `resume` only when inspect proves an interrupted, fully bound root. A failed root is immutable evidence and must lead to a new experiment number, not an Exp436 retry with changed science.

- [ ] **Step 3: Freeze and independently inspect publication.**

The `new`/`resume` command publishes only after 1,117 ordered records complete with zero failed datasets and <=100 candidates. Run read-only `inspect` again and record prediction, manifest, and artifact SHA256 values in `RESEARCH_STATE.json` before evaluation.

- [ ] **Step 4: Evaluate only after the freeze.**

```bash
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py evaluate
```

Confirm the evaluator reports all preregistered confusion matrices, paired deltas/intervals, rescued datasets, false positives, regressions, and the review-only success decision.

- [ ] **Step 5: Record the immutable outcome.**

On success, update `CHAMPION_LEDGER.json`; on failure, update `FAILURE_LEDGER.json`. In either case update `RESEARCH_STATE.json` and `EXPERIMENT_CATALOG.json`, regenerate all three code-map files together, validate JSON and fingerprints, and commit without changing Exp436 thresholds or rerunning it from TEST feedback.

---

## Plan Self-Review Checklist

- [ ] Every section of the approved Exp436 design maps to at least one task and test above.
- [ ] No step changes Exp435 source, imports its checkpoint records, or resumes its failed root.
- [ ] Every new filename includes `antigravity`; no package initializer is needed.
- [ ] Raw blob SHA256 is computed before float conversion.
- [ ] TRAIN and TEST missing-value rules are row-local and contain no imputation path.
- [ ] `InstanceScoreEvidence`, scorer, checkpoint, and publication all enforce the same strict three-channel AND invariant.
- [ ] Prediction imports cannot reach TEST-label loaders; evaluation verifies frozen hashes before label access.
- [ ] CPU, MLX, and MPS signatures use the same float32 score definition and candidate rule.
- [ ] Candidate budget, success thresholds, bootstrap repetitions/seed, roots, and run identity are identical in config, preregistration, runbook, tests, and CLI.
- [ ] Search the completed implementation for unfinished markers:

```bash
python3 - <<'PY'
from pathlib import Path
terms = ("TO" + "DO", "T" + "BD", "FIX" + "ME")
root = Path("experiments/exp436_missing_safe_strict_consensus_antigravity")
hits = [(str(path), term) for path in root.rglob("*.py") for term in terms if term in path.read_text()]
assert not hits, hits
PY
```

Expected: no unfinished implementation marker.
