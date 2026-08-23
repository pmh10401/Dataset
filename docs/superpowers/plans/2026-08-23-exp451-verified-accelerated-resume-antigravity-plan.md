# Exp451 Verified Accelerated Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exp450의 검증된 4,234개 결과를 변경하지 않고 Exp451로 이관하고, 남은 234개를 빠른 native CPU pooling fallback으로 계산한다.

**Architecture:** Exp451은 별도 코드·루트·context·manifest를 소유한다. 이관기는 Git commit `eac0d3b4`의 Exp450 code closure로 원본 checkpoint를 검증한 후 typed payload를 Exp451 create-only checkpoint로 다시 봉인한다. 신규 C/D 계산은 MPS native pooling을 우선하고 관찰된 shape 제한에서만 PyTorch CPU native pooling으로 전환한다.

**Tech Stack:** Python 3.14, NumPy, PyTorch 2.13.0, MPS, SQLite read-only catalog, unittest, SHA-256 canonical JSON

**Spec:** `docs/superpowers/specs/2026-08-23-exp451-verified-accelerated-resume-antigravity-design.md`

## Global Constraints

- Exp450 공식 결과 루트와 checkpoint는 읽기 전용이며 수정·이동·삭제하지 않는다.
- 모든 새 파일 이름에는 `antigravity`를 포함한다.
- Exp451은 독립 root와 `docs/codemap/codemap_exp451_antigravity.{html,json,lock}`를 사용한다.
- 점수, 조건 A–D, 임계값, 라우팅, 데이터베이스와 평가 정의는 Exp450과 동일하다.
- MPS-heavy 프로세스는 한 번에 하나만 실행한다.
- CPU 병렬 단계는 논리 CPU 수에서 1을 뺀 worker와 내부 수치 thread 1개를 사용한다.
- TEST label은 4,468개 성공 결과와 네 manifest 검증 전에는 열지 않는다.
- 실패 좌표는 ledger에 기록하고 다른 좌표 실행을 계속한다.

---

### Task 1: Exp450 원상복구와 Exp451 기본 계약

**Files:**
- Restore: `experiments/exp450_resampling_image_ablation_antigravity/transforms_antigravity.py`
- Restore: `experiments/exp450_resampling_image_ablation_antigravity/tests/test_transforms_backend_antigravity.py`
- Restore: `docs/codemap/codemap_exp450_antigravity.lock`
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/contracts_antigravity.py`
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/config_antigravity.py`
- Test: `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_contracts_config_antigravity.py`

**Interfaces:**
- Consumes: Exp450 frozen condition definitions and paper database identity.
- Produces: `Exp451RunContext`, `Exp451MigrationEntry`, `Exp451MigrationManifest`, `FROZEN_EXP451_CONFIG`.

- [ ] **Step 1: Restore only the three prototype files to commit `c35f4a5e^` bytes**

Use `git show c35f4a5e^:<path>` as the byte authority and `apply_patch` to restore the two Python files; regenerate Exp450 code map with:

```bash
python3 scratch/generate_exp450_codemap_antigravity.py
python3 scratch/generate_exp450_codemap_antigravity.py --check
```

- [ ] **Step 2: Write failing contract tests**

```python
def test_exp451_config_binds_isolated_roots_and_unchanged_population(self):
    assert FROZEN_EXP451_CONFIG.experiment_id == "Exp451"
    assert FROZEN_EXP451_CONFIG.dataset_count == 1117
    assert FROZEN_EXP451_CONFIG.test_row_count == 195900
    assert all("exp451" in str(root).lower() for root in FROZEN_EXP451_CONFIG.roots)

def test_migration_manifest_requires_exact_unique_inventory(self):
    with self.assertRaises(ValueError):
        Exp451MigrationManifest(source_context_sha256=SOURCE_CONTEXT, entries=())
```

- [ ] **Step 3: Verify RED**

Run:

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_contracts_config_antigravity
```

Expected: import failure because the Exp451 contracts do not exist.

- [ ] **Step 4: Implement minimal immutable contracts and frozen paths**

The manifest entry must expose exact strings for `dataset`, `condition_id`, `source_relative_path`, `source_envelope_sha256`, `payload_sha256`, `target_relative_path`, `target_envelope_sha256`, and `execution_origin`. The manifest must enforce 4,234 unique coordinates with counts `A=1117`, `B=1117`, `C=1000`, `D=1000`.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_contracts_config_antigravity
git add experiments/exp450_resampling_image_ablation_antigravity docs/codemap/codemap_exp450_antigravity.lock experiments/exp451_verified_accelerated_resume_antigravity
git commit -m "feat: define isolated Exp451 contracts"
```

### Task 2: Native CPU Pooling Fallback

**Files:**
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/transforms_antigravity.py`
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/backend_antigravity.py`
- Test: `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_transforms_backend_antigravity.py`

**Interfaces:**
- Consumes: finite float32 arrays and `device in {"cpu", "mps"}`.
- Produces: `adaptive_average_1d_antigravity`, `adaptive_average_2d_antigravity`, `assess_exp451_backend_antigravity`.

- [ ] **Step 1: Write failing MPS fallback tests**

```python
def fail_only_on_mps(tensor, output_size):
    if tensor.device.type == "mps":
        raise RuntimeError("Adaptive pool MPS: input sizes must be divisible by output sizes.")
    return native_pool(tensor, output_size)

with patch.object(torch_functional, "adaptive_avg_pool2d", side_effect=fail_only_on_mps), \
     patch.object(transforms, "_adaptive_pool_windows_antigravity", side_effect=AssertionError):
    actual = adaptive_average_2d_antigravity(np.arange(16, dtype=np.float32).reshape(1, 4, 4), 3, device="mps")
np.testing.assert_array_equal(actual, EXPECTED_3X3)
```

Add separate tests for 1D, mixed-axis 2D, unrelated RuntimeError propagation, finite output and exact shape.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_transforms_backend_antigravity
```

Expected: missing Exp451 transform implementation.

- [ ] **Step 3: Implement the exact fallback branch**

```python
try:
    return torch_functional.adaptive_avg_pool2d(tensor, output_size)
except RuntimeError as error:
    if device != "mps" or not _is_mps_nondivisible_pool_error_antigravity(error):
        raise
    return torch_functional.adaptive_avg_pool2d(tensor.cpu(), output_size)
```

Apply the same rule to 1D. Preserve Exp450 resampling definitions unchanged.

- [ ] **Step 4: Verify behavior and bounded timing**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_transforms_backend_antigravity
python3 -c "import numpy as np; from experiments.exp451_verified_accelerated_resume_antigravity.transforms_antigravity import adaptive_average_2d_antigravity as f; print(f(np.ones((1,512,513),dtype=np.float32),(224,224),device='mps').shape)"
```

Expected: tests pass and shape is `(1, 224, 224)` without scalar-window execution.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp451_verified_accelerated_resume_antigravity
git commit -m "perf: add Exp451 native pooling fallback"
```

### Task 3: Exp451 Scientific Scoring Closure

**Files:**
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/imaging_antigravity.py`
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/scoring_antigravity.py`
- Test: `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_scoring_imaging_antigravity.py`

**Interfaces:**
- Consumes: Exp451 conditions, backend decision and raw datasets.
- Produces: Exp450-equivalent A–D typed source and route results using Exp451 pooling.

- [ ] **Step 1: Write failing factor-preservation tests**

Assert A=`linear/linear`, B=`polyphase_pchip/linear`, C=`linear/area`, D=`polyphase_pchip/area`; patch only the public imaging boundary and assert C/D call Exp451 pooling while A/B retain the paper control.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_scoring_imaging_antigravity
```

- [ ] **Step 3: Copy the Exp450 scientific closure and change only imports/identity**

Use the Exp450 implementation as the source. Replace imports with the Exp451 namespace and route area imaging through Task 2 transforms. Do not alter score, threshold, seed, fold, source or routing constants.

- [ ] **Step 4: Verify GREEN and mutation guards**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_scoring_imaging_antigravity
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_imaging_antigravity experiments.exp450_resampling_image_ablation_antigravity.tests.test_scoring_antigravity
```

Dependency-unavailable tests must be reported separately from Exp451 failures.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp451_verified_accelerated_resume_antigravity
git commit -m "feat: preserve Exp450 science in Exp451"
```

### Task 4: Verified Migration Authority

**Files:**
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/artifacts_antigravity.py`
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/migration_antigravity.py`
- Test: `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_migration_artifacts_antigravity.py`

**Interfaces:**
- Consumes: Exp450 root, Git commit `eac0d3b4`, current catalog and Exp451 context.
- Produces: `reconstruct_exp450_source_context_antigravity`, `migrate_verified_exp450_results_antigravity`, `verify_migration_manifest_antigravity`.

- [ ] **Step 1: Write failing source-context tests**

```python
context = reconstruct_exp450_source_context_antigravity(repo, catalog)
self.assertEqual(context.sha256, "5d5b44aec0e738ae61d9dafd85d9eaea1448baba49705c4d042c5ed1d6886f83")
self.assertEqual(context.code_closure[TRANSFORMS_PATH], "28b5fcf731fc7ed31546939bbbaa932e38770f3f6e77864fc353328ddaf059c7")
```

Add tests that a changed commit, digest, catalog, backend, payload and raw identity fail closed.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_migration_artifacts_antigravity
```

- [ ] **Step 3: Implement old closure reconstruction**

List the exact paths used by Exp450 `_code_closure`; read each with `git show eac0d3b4:<path>`, hash bytes, construct the old context and require the fixed context SHA before reading a checkpoint.

- [ ] **Step 4: Implement create-only target envelopes and manifest**

For each verified source typed result, write an Exp451 envelope with source execution origin `imported_exp450_mps`. Use same-directory temporary files, fsync, no-clobber link publication and exact-byte reuse only. Publish `migration_manifest_antigravity.json` only after all 4,234 entries validate.

- [ ] **Step 5: Verify GREEN and source immutability fixture**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_migration_artifacts_antigravity
```

The fixture test must hash the source tree before and after migration and assert identical path/hash maps.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp451_verified_accelerated_resume_antigravity
git commit -m "feat: migrate verified Exp450 checkpoints"
```

### Task 5: Resume Runner, Failure Ledger and Progress Log

**Files:**
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/runner_antigravity.py`
- Create: `scratch/run_exp451_verified_accelerated_resume_antigravity.py`
- Test: `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_runner_cli_antigravity.py`

**Interfaces:**
- Consumes: verified migration manifest, Exp451 checkpoint inventory and scientific scorer.
- Produces: `migrate`, `verify-migration`, `run`, `status`, `verify`, `evaluate`, `report` commands and JSONL progress.

- [ ] **Step 1: Write failing resume-count and progress tests**

```python
self.assertEqual(inventory.completed_count, 4234)
self.assertEqual(inventory.remaining_count, 234)
self.assertEqual(inventory.by_condition, {"A": 1117, "B": 1117, "C": 1000, "D": 1000})
```

Assert every success/failure event contains experiment, dataset, condition, completed, failed, remaining, total, elapsed seconds and ETA.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_runner_cli_antigravity
```

- [ ] **Step 3: Implement deterministic missing-coordinate schedule**

Skip imported and native successes, execute only missing coordinates, persist each result before logging it, and run only one MPS-heavy coordinate at once. Record native results with origin `exp451_native_cpu_pool_fallback`.

- [ ] **Step 4: Implement failure continuation and retry waves**

Write failures as unique `(dataset, condition, wave)` records. Complete wave one before retrying failed coordinates in wave two. Never recompute successful coordinates.

- [ ] **Step 5: Verify GREEN**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_runner_cli_antigravity
```

- [ ] **Step 6: Commit**

```bash
git add experiments/exp451_verified_accelerated_resume_antigravity scratch/run_exp451_verified_accelerated_resume_antigravity.py
git commit -m "feat: resume Exp451 from migrated inventory"
```

### Task 6: Verification, Evaluation and Paper Report

**Files:**
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/verifier_antigravity.py`
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/evaluation_antigravity.py`
- Create: `experiments/exp451_verified_accelerated_resume_antigravity/reporting_antigravity.py`
- Create: `docs/EXP451_METHODS_AND_PROVENANCE_antigravity_KO.md`
- Test: `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_verification_reporting_antigravity.py`

**Interfaces:**
- Consumes: 4,468 verified Exp451 checkpoints and migration manifest.
- Produces: four frozen manifests, AP/AUROC/ROC/PR artifacts, comparison report and Korean provenance document.

- [ ] **Step 1: Write failing label-gate and provenance tests**

Assert evaluation refuses 4,467 successes, refuses a missing condition manifest, and reports imported/native counts separately. Assert ROC x=`false_positive_rate`, y=`true_positive_rate`; PR x=`recall`, y=`precision`.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_verification_reporting_antigravity
```

- [ ] **Step 3: Implement verifier and reuse frozen Exp450 evaluation definitions**

Verify origin, context, payload, count and TEST-row invariants before importing label access. Copy evaluation/report logic without changing method order or metric definitions.

- [ ] **Step 4: Write the methods/provenance document**

Document exact imported counts, native counts, old/new context SHA-256, device fallback rule, accepted floating-point distinction, database identity and reproduction commands.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_verification_reporting_antigravity
git add experiments/exp451_verified_accelerated_resume_antigravity docs/EXP451_METHODS_AND_PROVENANCE_antigravity_KO.md
git commit -m "feat: verify and report Exp451 results"
```

### Task 7: Exp451 Code Map and Full Verification

**Files:**
- Create: `scratch/generate_exp451_codemap_antigravity.py`
- Create: `docs/codemap/codemap_exp451_antigravity.html`
- Create: `docs/codemap/codemap_exp451_antigravity.json`
- Create: `docs/codemap/codemap_exp451_antigravity.lock`
- Test: `experiments/exp451_verified_accelerated_resume_antigravity/tests/test_codemap_antigravity.py`

**Interfaces:**
- Consumes: complete Exp451 source closure.
- Produces: synchronized Exp451 code-map triplet and final verification evidence.

- [ ] **Step 1: Write failing codemap freshness test**

The test runs the generator with `--check`, requires all Exp451 production modules in the lock and requires caller/effect/test answers for every node.

- [ ] **Step 2: Verify RED, implement generator and generate triplet**

```bash
python3 -m unittest experiments.exp451_verified_accelerated_resume_antigravity.tests.test_codemap_antigravity
python3 scratch/generate_exp451_codemap_antigravity.py
```

- [ ] **Step 3: Run complete verification**

```bash
python3 scratch/generate_exp450_codemap_antigravity.py --check
python3 scratch/generate_exp451_codemap_antigravity.py --check
python3 -m unittest discover -s experiments/exp451_verified_accelerated_resume_antigravity/tests -p 'test_*_antigravity.py'
python3 -m unittest experiments.exp450_resampling_image_ablation_antigravity.tests.test_transforms_backend_antigravity experiments.exp450_resampling_image_ablation_antigravity.tests.test_imaging_antigravity
git diff --check
```

- [ ] **Step 4: Fresh read-only final review**

Use `sol_advisor_sol_reviewer` to inspect the complete diff, source immutability evidence, migration authority, failure continuation, label gate and exact commands. Resolve every correctness finding before launch.

- [ ] **Step 5: Commit**

```bash
git add scratch/generate_exp451_codemap_antigravity.py docs/codemap/codemap_exp451_antigravity.html docs/codemap/codemap_exp451_antigravity.json docs/codemap/codemap_exp451_antigravity.lock experiments/exp451_verified_accelerated_resume_antigravity
git commit -m "docs: finalize Exp451 code map"
```

### Task 8: Real Migration and Official Resume

**Files:**
- External create-only artifacts under the approved Exp451 Dataset roots.

**Interfaces:**
- Consumes: reviewed Exp451 implementation and read-only Exp450 root.
- Produces: verified 4,234-entry migration manifest and running 234-coordinate official resume.

- [ ] **Step 1: Snapshot source evidence**

Write the sorted relative-path/SHA-256 map to an `antigravity`-named temporary verification artifact outside the Exp450 root.

- [ ] **Step 2: Run migration and verification**

```bash
python3 -B scratch/run_exp451_verified_accelerated_resume_antigravity.py migrate --mode full
python3 -B scratch/run_exp451_verified_accelerated_resume_antigravity.py verify-migration --mode full
```

Require exactly 4,234 successes and counts A=1,117, B=1,117, C=1,000, D=1,000.

- [ ] **Step 3: Compare source evidence after migration**

Regenerate the Exp450 path/SHA-256 map and require byte-identical equality with Step 1.

- [ ] **Step 4: Launch and observe first durable result**

```bash
python3 -B scratch/run_exp451_verified_accelerated_resume_antigravity.py run --mode full
```

Keep the process attached to persistent logging. Confirm the first new success is durably stored, total becomes 4,235, failure count remains observable, and the process does not enter the scalar-window call stack.

- [ ] **Step 5: Report live status**

Report PID, completed/total, failures, latest dataset-condition, log path and evidence-based ETA. Do not claim experiment completion until all 4,468 results and four manifests verify.
