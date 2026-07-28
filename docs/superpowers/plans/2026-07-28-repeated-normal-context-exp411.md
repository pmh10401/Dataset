# Repeated-Normal Context and Exp411 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 반복 정상을 반도체의 안정 운전 조건으로 공식 문서에 바로잡고, 정상 TRAIN과 TRAIN-only 합성 교란만으로 near-repeat 자동 정상 규칙의 자격을 심사하는 Exp411을 구현한다.

**Architecture:** 공식 맥락 문서와 별도 decision record가 해석 경계를 고정한다. Exp411 코드는 읽기 전용 SQLite loader, identity-grouped fold, NumPy CPU 정본 거리 엔진, 선택적 Numba CPU JIT와 PyTorch CPU/MPS backend, TRAIN-only 합성 안전성 gate, 원자적 게시 계층으로 나뉜다. Exp411은 TEST split과 label을 읽지 않으며, gate를 통과한 단일 quantile과 provenance만 동결하고 Exp412 prediction은 만들지 않는다.

**Tech Stack:** Python 3.14.6, NumPy 2.3.5, SciPy 1.17.1, scikit-learn 1.7.2, Numba 0.66.0 with llvmlite 0.48.0, PyTorch 2.13.0 with optional MPS, SQLite read-only URI, JSON/JSONL/CSV/SHA256, `unittest`.

## Global Constraints

- Dataset universe는 Exp410과 같은 1,117개 dataset 및 128개 family다.
- DB SHA256은 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`이며 main DB만 허용하고 unpinned WAL/SHM을 거부한다.
- prediction/qualification은 정상 TRAIN `instance_index`, `values_blob`만 읽고 TEST split, TEST label, `labels_blob`, 실제 이상 위치를 읽지 않는다.
- 과거 family별 TEST 성능, source 순위, TP/FP/F1을 설정 선택에 사용하지 않는다.
- byte SHA256이 같은 TRAIN instance는 하나의 identity group으로 같은 fold에 둔다.
- identity group `<5`는 abstain, `5-19`는 leave-one-identity-group-out, `20+`는 seed `411`의 deterministic 5-fold group split을 쓴다.
- 같은 길이 waveform만 비교하며 보간과 tail replacement를 사용하지 않는다.
- 거리 view는 `raw_rms`, `diff_rms`, `spectrum_rms`; quantile은 `0.01`, `0.05`, `0.10`; NumPy quantile method는 `lower`다.
- near 후보는 exact byte match가 아니고, `raw_rms`와 사용 가능한 view 중 최소 2개가 해당 quantile 이하일 때만 생성한다.
- TRAIN-only gate는 정상 near 인정률 `>=0.01`, pooled synthetic false-normal rate `<=0.001`, perturbation별 rate `<=0.005`, family bootstrap 95% 상한 `<=0.005`를 모두 요구한다.
- 여러 quantile이 통과하면 가장 넓은 quantile을 선택하고, 없으면 `no_qualified_near_rule`로 fail closed한다.
- 기존 Hard 2,218, Standard 806, Priority 31, exact-normal 35와 모든 동결 artifact를 변경하지 않는다.
- `mean_combined_f1`은 human-assisted 지표로만 취급한다.
- CPU/NumPy가 정본 fallback이다. 가속 backend는 tag와 선택 quantile이 CPU와 완전히 같고 거리 최대 절대차가 `<=1e-6`이며 end-to-end stage가 `>=1.1x` 빠를 때만 정본 후보가 된다.
- Numba benchmark는 최초 JIT compile 시간을 포함한 cold run과 compile을 제외한 warm run을 모두 기록하며, 정본 승격 gate에는 전체 stage cold run을 사용한다.
- PyTorch MPS가 비가용하거나 device open이 실패하면 예외를 외부로 누출하지 않고 availability와 이유를 manifest에 기록한 뒤 CPU를 사용한다.
- 사용자의 2026-07-28 결정에 따라 MLX는 구현, 설치, import, benchmark 또는 정본 후보로 고려하지 않고 `mlx_status: excluded_by_user_decision`만 기록한다.
- 단위 테스트와 representative smoke는 허용하지만 전체 1,117개 Exp411 실행은 사용자의 별도 명시적 승인 전에는 시작하지 않는다.
- 상세 파생 결과와 로컬 artifact를 공개 GitHub에 게시하지 않는다.

---

## File Structure

- `docs/PROJECT_RESEARCH_CONTEXT_KO.md`: 반복 정상의 공식 해석, Exp290/291/410의 위치, repeatability-aware 보고 원칙.
- `docs/REPEATED_NORMAL_INTERPRETATION_KO.md`: 반복 정상 decision record와 허용·금지 주장.
- `docs/EXP411_PREREGISTRATION_KO.md`: Exp411의 입력, 알고리즘, gate, 출력, 실행 승인 계약.
- `docs/EXP411_RUNBOOK_KO.md`: `.venv`, 테스트, backend probe, smoke, 승인된 full-run 명령.
- `experiments/exp411_near_repeat_qualification/config.py`: 모든 동결 수치와 hash.
- `experiments/exp411_near_repeat_qualification/contracts.py`: run, fold, tag, backend, qualification dataclass.
- `experiments/exp411_near_repeat_qualification/folds.py`: byte identity grouping과 deterministic group folds.
- `experiments/exp411_near_repeat_qualification/distances.py`: 세 view, MAD scale, CPU 거리와 quantile tag.
- `experiments/exp411_near_repeat_qualification/backends.py`: Numba CPU JIT와 PyTorch CPU/MPS availability, parity, benchmark, NumPy CPU fallback.
- `experiments/exp411_near_repeat_qualification/perturbations.py`: Exp407 동결 합성 교란 호출과 provenance 검증.
- `experiments/exp411_near_repeat_qualification/inputs.py`: hash-first read-only SQLite TRAIN loader와 snapshot 경계.
- `experiments/exp411_near_repeat_qualification/qualification.py`: dataset 결과 집계, family bootstrap, quantile 선택 gate.
- `experiments/exp411_near_repeat_qualification/publication.py`: 원자적 CSV/JSON/report/manifest 게시.
- `experiments/exp411_near_repeat_qualification/runner.py`: representative fixture와 승인된 full qualification orchestration.
- `experiments/exp411_near_repeat_qualification/reporting.py`: 한국어 결과 보고서 rendering.
- `scratch/run_exp411_near_repeat_qualification.py`: smoke/full CLI 경계.
- `experiments/exp411_near_repeat_qualification/tests/`: 각 계층의 단위·통합·구조 테스트.

### Task 1: 공식 반복 정상 문맥을 정정한다

**Files:**
- Create: `docs/REPEATED_NORMAL_INTERPRETATION_KO.md`
- Modify: `docs/PROJECT_RESEARCH_CONTEXT_KO.md`
- Create: `experiments/exp411_near_repeat_qualification/__init__.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/__init__.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_documentation_contract.py`

**Interfaces:**
- Consumes: Exp290/291 설계 수치, Exp410 local result report, umbrella design.
- Produces: 후속 사전등록과 코드가 인용할 공식 decision record.

- [ ] **Step 1: 문서 계약 실패 테스트를 작성한다**

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class DocumentationContractTests(unittest.TestCase):
    def test_context_defines_repeated_normal_as_intended_operation(self) -> None:
        context = (ROOT / "docs/PROJECT_RESEARCH_CONTEXT_KO.md").read_text("utf-8")
        required = (
            "의도적 repeated-normal 운전 조건",
            "automatic normal exact",
            "35개 모두 FP",
            "TP 손실 0",
            "repeatability-aware reporting",
            "clean-balanced",
            "Exp410",
        )
        for phrase in required:
            self.assertIn(phrase, context)

    def test_decision_record_preserves_claim_boundary(self) -> None:
        record = (ROOT / "docs/REPEATED_NORMAL_INTERPRETATION_KO.md").read_text("utf-8")
        for phrase in (
            "실제 동일 wafer run이라는 뜻이 아니다",
            "prospective 성능을 증명하지 않는다",
            "Exp290",
            "Exp411",
        ):
            self.assertIn(phrase, record)
```

- [ ] **Step 2: 테스트가 현재 문구와 파일 부재 때문에 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_documentation_contract -v
```

Expected: `REPEATED_NORMAL_INTERPRETATION_KO.md` 부재 또는 필수 문구 누락으로 FAIL.

- [ ] **Step 3: 공식 맥락 문서와 decision record를 작성한다**

`docs/PROJECT_RESEARCH_CONTEXT_KO.md`에서 다음 의미를 정확히 반영한다.

```markdown
정상 TRAIN/TEST의 정확 중복은 이 프로젝트에서 안정된
equipment/recipe/sensor/step 상태가 반복되는 의도적 repeated-normal 운전
조건이다. 반복 정상에 대한 낮은 오경보는 반도체 적용 목표와 맞는 긍정적
특성이다. 다만 UCR waveform 중복이 실제 동일 wafer run을 뜻하거나 미관측
설비의 prospective 성능을 증명하지는 않는다.
```

위험 표는 “중복=결함” 대신 “운영 가치와 주장 경계”를 함께 쓰고, 연구
연혁에는 Exp290/291 exact-normal 35개가 모두 FP이고 TP 손실은 0이었다는
사실을 추가한다. Exp410의 high-overlap 성능은 긍정적 관찰이지만 family
구성 차이 때문에 overlap 단독 인과 효과로 부르지 않는다.

- [ ] **Step 4: 문서 계약 테스트와 링크 검사를 통과시킨다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_documentation_contract -v
rg -n "Exp290|Exp291|Exp410|repeated-normal|repeatability-aware|clean-balanced" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md docs/REPEATED_NORMAL_INTERPRETATION_KO.md
```

Expected: 2 tests PASS, 두 문서 모두 근거와 해석 경계를 포함.

- [ ] **Step 5: 문서 정정을 커밋한다**

```bash
git add docs/PROJECT_RESEARCH_CONTEXT_KO.md \
  docs/REPEATED_NORMAL_INTERPRETATION_KO.md \
  experiments/exp411_near_repeat_qualification/__init__.py \
  experiments/exp411_near_repeat_qualification/tests/__init__.py \
  experiments/exp411_near_repeat_qualification/tests/test_documentation_contract.py
git commit -m "docs: clarify repeated-normal operating intent"
```

### Task 2: 동결 config, 계약, identity-group fold를 구현한다

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/config.py`
- Create: `experiments/exp411_near_repeat_qualification/contracts.py`
- Create: `experiments/exp411_near_repeat_qualification/folds.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_folds.py`

**Interfaces:**
- Produces: `FROZEN_CONFIG`, `TrainRun`, `IdentityFold`, `group_crossfit_plan(runs, seed)`.
- Consumes: Task 3-7에서 모든 동결 값과 fold identity를 사용.

- [ ] **Step 1: identity group과 fold 실패 테스트를 작성한다**

```python
from dataclasses import replace
import unittest

from experiments.exp411_near_repeat_qualification.config import FROZEN_CONFIG
from experiments.exp411_near_repeat_qualification.contracts import TrainRun
from experiments.exp411_near_repeat_qualification.folds import group_crossfit_plan


def run(index: int, digest: str, length: int = 4) -> TrainRun:
    return TrainRun(
        dataset="Alpha_normal_0",
        instance_index=index,
        values=(float(index),) * length,
        values_sha256=digest,
    )


class GroupFoldTests(unittest.TestCase):
    def test_duplicate_identity_never_crosses_fit_and_query(self) -> None:
        rows = tuple(
            run(index, digest)
            for index, digest in enumerate(("a", "a", "b", "c", "d", "e"))
        )
        folds = group_crossfit_plan(rows, seed=411)
        for fold in folds:
            fit_hashes = {rows[i].values_sha256 for i in fold.fit_indices}
            query_hashes = {rows[i].values_sha256 for i in fold.query_indices}
            self.assertTrue(fit_hashes.isdisjoint(query_hashes))

    def test_fewer_than_five_identity_groups_abstains(self) -> None:
        rows = tuple(run(i, str(i % 4)) for i in range(8))
        self.assertEqual(group_crossfit_plan(rows, seed=411), ())

    def test_twenty_groups_use_deterministic_five_fold(self) -> None:
        rows = tuple(run(i, str(i)) for i in range(20))
        first = group_crossfit_plan(rows, seed=411)
        second = group_crossfit_plan(tuple(reversed(rows)), seed=411)
        self.assertEqual(
            {tuple(sorted(f.query_identity_sha256)) for f in first},
            {tuple(sorted(f.query_identity_sha256)) for f in second},
        )
        self.assertEqual(len(first), 5)

    def test_frozen_parameters_reject_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, quantiles=(0.02, 0.05, 0.10)).validate()
```

- [ ] **Step 2: 테스트가 모듈 부재로 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_folds -v
```

Expected: import failure.

- [ ] **Step 3: dataclass와 fold 구현을 추가한다**

`config.py`의 public 계약은 다음과 같다.

```python
@dataclass(frozen=True)
class Exp411FrozenConfig:
    experiment_id: str = "Exp411"
    expected_datasets: int = 1117
    expected_families: int = 128
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.10)
    quantile_method: str = "lower"
    seed: int = 411
    bootstrap_iterations: int = 2000
    minimum_identity_groups: int = 5
    five_fold_minimum_groups: int = 20
    minimum_same_length_fit_groups: int = 5
    mad_floor: float = 1e-12
    distance_tolerance: float = 1e-6
    speedup_gate: float = 1.1
    minimum_normal_acceptance: float = 0.01
    pooled_synthetic_false_normal_max: float = 0.001
    per_perturbation_false_normal_max: float = 0.005
    bootstrap_upper_max: float = 0.005

    def validate(self) -> None: ...
```

`folds.py`는 dataset 이름이나 입력 순서 대신 SHA256 정렬과 seed-derived
hash를 사용한다. `5-19` group은 group 하나씩 query로 두고, `20+`는
`sha256(f"{seed}:{digest}")` 순서로 정렬한 뒤 round-robin 5-fold로 나눈다.

- [ ] **Step 4: fold 테스트를 통과시킨다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_folds -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: config와 fold를 커밋한다**

```bash
git add experiments/exp411_near_repeat_qualification/config.py \
  experiments/exp411_near_repeat_qualification/contracts.py \
  experiments/exp411_near_repeat_qualification/folds.py \
  experiments/exp411_near_repeat_qualification/tests/test_folds.py
git commit -m "feat: freeze Exp411 identity-group folds"
```

### Task 3: CPU 거리 정본과 near tag를 구현한다

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/distances.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_distances.py`

**Interfaces:**
- Consumes: `IdentityFold`, `Exp411FrozenConfig`.
- Produces: `view_matrices`, `cpu_pairwise_rms`, `calibrate_thresholds`, `tag_queries`.

- [ ] **Step 1: 세 view, self-exclusion, 합의 규칙 테스트를 작성한다**

```python
import unittest
import numpy as np

from experiments.exp411_near_repeat_qualification.distances import (
    calibrate_thresholds,
    cpu_pairwise_rms,
    tag_queries,
    view_matrices,
)


class DistanceTests(unittest.TestCase):
    def test_views_have_expected_widths_and_finite_values(self) -> None:
        rows = np.asarray([[0., 1., 2., 3.], [0., 1.1, 2., 3.]])
        views = view_matrices(rows)
        self.assertEqual(views["raw_rms"].shape, (2, 4))
        self.assertEqual(views["diff_rms"].shape, (2, 3))
        self.assertEqual(views["spectrum_rms"].shape, (2, 2))
        self.assertTrue(all(np.isfinite(value).all() for value in views.values()))

    def test_pairwise_rms_returns_each_query_minimum(self) -> None:
        fit = np.asarray([[0., 0.], [3., 4.]])
        query = np.asarray([[0., 0.], [4., 4.]])
        np.testing.assert_allclose(
            cpu_pairwise_rms(fit, query),
            np.asarray([0., np.sqrt(0.5)]),
        )

    def test_near_requires_raw_and_two_available_views(self) -> None:
        thresholds = {
            "raw_rms": 0.2,
            "diff_rms": 0.2,
            "spectrum_rms": 0.2,
        }
        distances = {
            "raw_rms": np.asarray([0.1, 0.1]),
            "diff_rms": np.asarray([0.1, 0.3]),
            "spectrum_rms": np.asarray([0.3, 0.1]),
        }
        self.assertEqual(
            tag_queries(distances, thresholds, exact=(False, False)),
            ("near", "near"),
        )
        self.assertEqual(
            tag_queries(distances, thresholds, exact=(True, False)),
            ("exact", "near"),
        )

    def test_lower_quantile_is_frozen(self) -> None:
        values = {"raw_rms": np.asarray([1., 2., 3., 100.])}
        self.assertEqual(calibrate_thresholds(values, 0.10)["raw_rms"], 1.0)
```

- [ ] **Step 2: 거리 테스트가 import failure로 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_distances -v
```

Expected: import failure.

- [ ] **Step 3: CPU 거리와 tag 구현을 추가한다**

핵심 계산은 다음 형태를 유지한다.

```python
def cpu_pairwise_rms(fit: np.ndarray, query: np.ndarray) -> np.ndarray:
    delta = query[:, None, :] - fit[None, :, :]
    return np.sqrt(np.mean(delta * delta, axis=2)).min(axis=1)


def tag_queries(distances, thresholds, *, exact):
    tags = []
    for index, is_exact in enumerate(exact):
        if is_exact:
            tags.append("exact")
            continue
        available = [
            name for name in ("raw_rms", "diff_rms", "spectrum_rms")
            if name in distances and name in thresholds
        ]
        passes = {
            name for name in available
            if distances[name][index] <= thresholds[name]
        }
        tags.append(
            "near"
            if len(available) >= 2 and "raw_rms" in passes and len(passes) >= 2
            else "unfamiliar"
        )
    return tuple(tags)
```

MAD scale은 각 view의 fit matrix 전체에서 계산하고 `<=1e-12`이면 해당
view를 unavailable로 둔다. Spectrum은 `np.abs(np.fft.rfft(row))[1:]`로
DC를 제거한다. 두 개 미만 view만 남으면 query는 `abstain`이다.

- [ ] **Step 4: 거리 테스트와 입력 순서 불변성을 통과시킨다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_distances -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: CPU 거리 정본을 커밋한다**

```bash
git add experiments/exp411_near_repeat_qualification/distances.py \
  experiments/exp411_near_repeat_qualification/tests/test_distances.py
git commit -m "feat: add Exp411 CPU near-repeat distances"
```

### Task 4: Numba·PyTorch CPU/MPS backend probe, parity와 benchmark를 구현한다

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/backends.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_backends.py`
- Create: `experiments/exp411_near_repeat_qualification/requirements-accelerated.txt`

**Interfaces:**
- Consumes: Task 3 `cpu_pairwise_rms`.
- Produces: `probe_backends`, `pairwise_rms`, `assess_backend`, `BackendAssessment`.

- [ ] **Step 1: CPU parity와 device-open failure 테스트를 작성한다**

```python
import unittest
from unittest.mock import patch
import numpy as np

from experiments.exp411_near_repeat_qualification.backends import (
    assess_backend,
    pairwise_rms,
    probe_backends,
)


class BackendTests(unittest.TestCase):
    def test_numba_cpu_matches_numpy(self) -> None:
        fit = np.asarray([[0., 0.], [3., 4.]])
        query = np.asarray([[0., 0.], [4., 4.]])
        np.testing.assert_allclose(
            pairwise_rms(fit, query, backend="numba_cpu"),
            pairwise_rms(fit, query, backend="numpy_cpu"),
            atol=1e-6,
            rtol=0.0,
        )

    def test_torch_cpu_matches_numpy(self) -> None:
        fit = np.asarray([[0., 0.], [3., 4.]])
        query = np.asarray([[0., 0.], [4., 4.]])
        np.testing.assert_allclose(
            pairwise_rms(fit, query, backend="torch_cpu"),
            pairwise_rms(fit, query, backend="numpy_cpu"),
            atol=1e-6,
            rtol=0.0,
        )

    def test_mps_device_failure_is_recorded_not_raised(self) -> None:
        with patch(
            "experiments.exp411_near_repeat_qualification.backends._probe_torch_mps",
            side_effect=RuntimeError("MPS device unavailable"),
        ):
            result = probe_backends()
        self.assertFalse(result["torch_mps"].available)
        self.assertIn("MPS device", result["torch_mps"].reason)
        self.assertEqual(result["mlx_status"], "excluded_by_user_decision")

    def test_backend_cannot_be_canonical_below_speed_gate(self) -> None:
        result = assess_backend(
            name="torch_cpu",
            max_abs_error=0.0,
            tag_identity=True,
            selected_quantile_identity=True,
            cpu_seconds=1.0,
            backend_seconds=0.95,
        )
        self.assertFalse(result.canonical_eligible)
```

- [ ] **Step 2: backend 테스트가 import failure로 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_backends -v
```

Expected: import failure.

- [ ] **Step 3: optional backend와 환경 파일을 구현한다**

`requirements-accelerated.txt`:

```text
numpy==2.3.5
scipy==1.17.1
scikit-learn==1.7.2
numba==0.66.0
torch==2.13.0
```

`probe_backends()`는 import 성공과 실제 3-element device operation을
분리한다. Numba는 실제 `@njit` compile과 결과 확인을 통과해야 available로
표시한다. PyTorch MPS의 `is_built=true`여도 `is_available=false`이면
available로 표시하지 않는다. MLX package를 import하거나 probe하지 않고
`mlx_status: excluded_by_user_decision`만 반환한다.

- [ ] **Step 4: backend 테스트와 실제 환경 probe를 실행한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_backends -v
.venv/bin/python -c \
  "from experiments.exp411_near_repeat_qualification.backends import probe_backends; print(probe_backends())"
```

Expected: tests PASS. Numba CPU JIT smoke는 available이어야 한다. 현재
Codex 세션에서는 PyTorch MPS가 unavailable일 수 있으며 이유가 출력되고
NumPy/PyTorch CPU가 available이어야 한다. MLX 상태는
`excluded_by_user_decision`이다.

- [ ] **Step 5: backend 계층을 커밋한다**

```bash
git add experiments/exp411_near_repeat_qualification/backends.py \
  experiments/exp411_near_repeat_qualification/requirements-accelerated.txt \
  experiments/exp411_near_repeat_qualification/tests/test_backends.py
git commit -m "feat: assess Exp411 CPU and Torch backends"
```

### Task 5: Exp407 합성 교란 재사용과 TRAIN-only gate를 구현한다

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/perturbations.py`
- Create: `experiments/exp411_near_repeat_qualification/qualification.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_qualification.py`

**Interfaces:**
- Consumes: Exp407 `SYNTHETIC_PERTURBATION_TYPES`, `SyntheticAnomalyConfig`, `generate_synthetic_anomalies`; Tasks 2-4 fold/tag.
- Produces: `qualify_dataset`, `aggregate_qualification`, `select_quantile`.

- [ ] **Step 1: provenance, gate, 가장 넓은 quantile 선택 테스트를 작성한다**

```python
import unittest

from experiments.exp411_near_repeat_qualification.qualification import (
    select_quantile,
)


class QualificationTests(unittest.TestCase):
    def test_selects_widest_train_only_qualified_quantile(self) -> None:
        rows = {
            0.01: dict(normal_acceptance=0.02, pooled_synthetic=0.0001,
                       worst_perturbation=0.001, bootstrap_upper=0.002),
            0.05: dict(normal_acceptance=0.03, pooled_synthetic=0.0005,
                       worst_perturbation=0.003, bootstrap_upper=0.004),
            0.10: dict(normal_acceptance=0.04, pooled_synthetic=0.0020,
                       worst_perturbation=0.003, bootstrap_upper=0.004),
        }
        self.assertEqual(select_quantile(rows), 0.05)

    def test_no_safe_quantile_fails_closed(self) -> None:
        rows = {
            0.01: dict(normal_acceptance=0.005, pooled_synthetic=0.0,
                       worst_perturbation=0.0, bootstrap_upper=0.0),
        }
        self.assertIsNone(select_quantile(rows))

    def test_exp407_perturbation_names_are_exact(self) -> None:
        from experiments.exp411_near_repeat_qualification.perturbations import (
            perturbation_types,
        )
        self.assertEqual(
            perturbation_types(),
            ("spike", "drift", "level_shift", "flat_line",
             "amplitude_scale", "frequency_warp", "phase_order"),
        )
```

- [ ] **Step 2: qualification 테스트가 import failure로 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_qualification -v
```

Expected: import failure.

- [ ] **Step 3: Exp407 wrapper와 gate를 구현한다**

`perturbations.py`는 새 교란을 만들지 않고 Exp407 public API를 import한다.
Exp407 source SHA256과 `SyntheticAnomalyConfig`의 canonical JSON hash를
결과에 기록한다. 각 query waveform 길이에 맞춘 `FrozenConfig`를 사용하고,
seed는 `sha256("Exp411:{dataset}:{fold}:{identity}")`의 첫 8 bytes로 만든다.

`select_quantile()`은 아래 predicate만 사용한다.

```python
def passes_gate(row: Mapping[str, float]) -> bool:
    return (
        row["normal_acceptance"] >= 0.01
        and row["pooled_synthetic"] <= 0.001
        and row["worst_perturbation"] <= 0.005
        and row["bootstrap_upper"] <= 0.005
    )
```

family bootstrap은 dataset family를 cluster로 복원추출하고 seed `411`,
2,000회를 사용한다. TEST 정보는 함수 인자와 dataclass에 존재하지 않는다.

- [ ] **Step 4: qualification과 기존 Exp407 테스트를 함께 통과시킨다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_qualification \
  experiments.exp403_408.tests.test_synthetic_anomaly -v
```

Expected: Exp411 tests와 기존 Exp407 9 tests 모두 PASS.

- [ ] **Step 5: TRAIN-only gate를 커밋한다**

```bash
git add experiments/exp411_near_repeat_qualification/perturbations.py \
  experiments/exp411_near_repeat_qualification/qualification.py \
  experiments/exp411_near_repeat_qualification/tests/test_qualification.py
git commit -m "feat: qualify Exp411 near-repeat rules"
```

### Task 6: 읽기 전용 SQLite 입력과 금지 필드 구조 검사를 구현한다

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/inputs.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_inputs.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_structure.py`

**Interfaces:**
- Consumes: DB path, frozen universe, DB hash.
- Produces: `database_snapshot`, `load_train_runs`, `trace_sql_contract`.

- [ ] **Step 1: TRAIN-only loader와 SQL 구조 실패 테스트를 작성한다**

```python
import unittest
from pathlib import Path

from experiments.exp411_near_repeat_qualification.inputs import load_train_runs


class InputTests(unittest.TestCase):
    def test_loader_returns_only_normal_train_runs(self) -> None:
        rows = load_train_runs(self.fixture_db, ("Alpha_normal_0",), config=self.config)
        self.assertEqual(tuple(rows), ("Alpha_normal_0",))
        self.assertEqual([row.instance_index for row in rows["Alpha_normal_0"]], [0, 1])

    def test_loader_rejects_wal_sidecar(self) -> None:
        Path(f"{self.fixture_db}-wal").write_bytes(b"unpinned")
        with self.assertRaisesRegex(ValueError, "snapshot artifact"):
            load_train_runs(self.fixture_db, ("Alpha_normal_0",), config=self.config)
```

`test_structure.py`는 `inputs.py`, `qualification.py`, `runner.py` AST의 SQL
문자열과 identifier를 검사한다.

```python
for forbidden in ("labels_blob", "split = 'TEST'", "test_label", "is_true_anomaly"):
    self.assertNotIn(forbidden, prediction_source.lower())
```

- [ ] **Step 2: 입력 테스트가 import failure로 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_inputs \
  experiments.exp411_near_repeat_qualification.tests.test_structure -v
```

Expected: import failure.

- [ ] **Step 3: hash-first read-only loader를 구현한다**

SQLite는 다음 projection만 사용한다.

```sql
SELECT instance_index, values_blob
FROM instances
WHERE dataset_id = ? AND split = 'TRAIN' AND label = '0'
ORDER BY instance_index
```

별도 count query로 해당 dataset의 전체 TRAIN 수와 `label='0'` 수가 같은지
확인해 비정상 TRAIN label을 조용히 제외하지 않는다. `mode=ro`,
`PRAGMA quick_check`, scan 전후 DB/WAL/SHM snapshot equality를 강제한다.
blob은 schema dtype 근거에 따라 little-endian float array로 decode하고
finite, non-empty, 길이 `>=2`를 검증한다.

- [ ] **Step 4: 입력·구조 테스트를 통과시킨다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_inputs \
  experiments.exp411_near_repeat_qualification.tests.test_structure -v
```

Expected: 모든 테스트 PASS, 금지 field/SQL 0건.

- [ ] **Step 5: 입력 경계를 커밋한다**

```bash
git add experiments/exp411_near_repeat_qualification/inputs.py \
  experiments/exp411_near_repeat_qualification/tests/test_inputs.py \
  experiments/exp411_near_repeat_qualification/tests/test_structure.py
git commit -m "feat: enforce Exp411 TRAIN-only inputs"
```

### Task 7: 원자적 게시, report, runner와 CLI를 구현한다

**Files:**
- Create: `experiments/exp411_near_repeat_qualification/publication.py`
- Create: `experiments/exp411_near_repeat_qualification/reporting.py`
- Create: `experiments/exp411_near_repeat_qualification/runner.py`
- Create: `scratch/run_exp411_near_repeat_qualification.py`
- Create: `experiments/exp411_near_repeat_qualification/tests/test_runner.py`

**Interfaces:**
- Consumes: verified TRAIN rows, qualification rows, backend assessments.
- Produces: `run_representative_smoke`, `run_full_qualification`, atomic output directory.

- [ ] **Step 1: smoke, atomic publication, no-qualified 경로 테스트를 작성한다**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from experiments.exp411_near_repeat_qualification.runner import (
    run_representative_smoke,
)


class RunnerTests(unittest.TestCase):
    def test_representative_smoke_publishes_complete_manifest(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "smoke"
            hashes = run_representative_smoke(output)
            self.assertTrue((output / "MANIFEST.json").is_file())
            summary = json.loads((output / "exp411_summary.json").read_text("utf-8"))
            self.assertFalse(summary["full_qualification_executed"])
            self.assertFalse(summary["operational_policy_changed"])
            self.assertFalse(summary["exp412_prediction_created"])
            self.assertEqual(set(hashes), set(json.loads(
                (output / "MANIFEST.json").read_text("utf-8")
            )["artifact_sha256"]))

    def test_existing_output_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "must not exist"):
                run_representative_smoke(output)
```

- [ ] **Step 2: runner 테스트가 import failure로 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_runner -v
```

Expected: import failure.

- [ ] **Step 3: runner와 게시 계약을 구현한다**

정본 출력은 다음 8개 파일이다.

```text
exp411_frozen_config.json
exp411_dataset_qualification.csv
exp411_quantile_summary.csv
exp411_backend_assessment.json
exp411_selected_rule.json
exp411_summary.json
EXP411_RESULT_REPORT_KO.md
MANIFEST.json
```

모든 파일을 output parent의 임시 sibling에 쓴 뒤 각 SHA256을 계산하고,
DB snapshot과 clean Git commit을 다시 확인한 다음 한 번의 rename으로
게시한다. `selected_rule`은 quantile 또는 `null`, status는
`qualified` 또는 `no_qualified_near_rule`만 허용한다.

CLI mode는 `--representative-smoke`와 `--full-qualification`을 상호
배타적으로 요구한다. full mode는 `--db-path`, `--population-manifest`,
존재하지 않는 `--output-dir`를 모두 요구한다.

- [ ] **Step 4: runner 테스트와 실제 representative smoke를 실행한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_runner -v
smoke_root="$(mktemp -d /tmp/exp411-smoke.XXXXXX)"
.venv/bin/python scratch/run_exp411_near_repeat_qualification.py \
  --representative-smoke \
  --output-dir "$smoke_root/output"
```

Expected: tests PASS, smoke는 32개 합성 dataset, 오류 0, 완전한 8-file
manifest를 게시하고 full execution은 false.

- [ ] **Step 5: runner와 게시 계층을 커밋한다**

```bash
git add experiments/exp411_near_repeat_qualification/publication.py \
  experiments/exp411_near_repeat_qualification/reporting.py \
  experiments/exp411_near_repeat_qualification/runner.py \
  experiments/exp411_near_repeat_qualification/tests/test_runner.py \
  scratch/run_exp411_near_repeat_qualification.py
git commit -m "feat: publish Exp411 qualification atomically"
```

### Task 8: Exp411 사전등록·runbook과 전체 회귀 검증을 완성한다

**Files:**
- Create: `docs/EXP411_PREREGISTRATION_KO.md`
- Create: `docs/EXP411_RUNBOOK_KO.md`
- Modify: `experiments/exp411_near_repeat_qualification/tests/test_documentation_contract.py`

**Interfaces:**
- Consumes: Tasks 1-7의 실제 CLI, config, output contract.
- Produces: 재현 가능한 실행 계약과 full-run 승인 경계.

- [ ] **Step 1: 사전등록·runbook 계약 테스트를 추가한다**

```python
def test_exp411_docs_freeze_execution_boundary(self) -> None:
    prereg = (ROOT / "docs/EXP411_PREREGISTRATION_KO.md").read_text("utf-8")
    runbook = (ROOT / "docs/EXP411_RUNBOOK_KO.md").read_text("utf-8")
    for phrase in (
        "TEST split을 읽지 않는다",
        "no_qualified_near_rule",
        "0.01",
        "0.05",
        "0.10",
        "사용자의 별도 명시적 승인",
    ):
        self.assertIn(phrase, prereg)
    self.assertIn(".venv/bin/python", runbook)
    self.assertIn("--representative-smoke", runbook)
    self.assertIn("--full-qualification", runbook)
```

- [ ] **Step 2: 문서 부재로 테스트가 실패하는지 확인한다**

Run:

```bash
.venv/bin/python -m unittest \
  experiments.exp411_near_repeat_qualification.tests.test_documentation_contract -v
```

Expected: Exp411 문서 부재로 FAIL.

- [ ] **Step 3: 사전등록과 runbook을 실제 구현에 맞춰 작성한다**

사전등록은 umbrella design의 Exp411 계약을 수치 그대로 복사하고,
Exp412/413 및 정책 변경이 범위 밖임을 명시한다. Runbook 환경 확인은 다음
명령을 사용한다.

```bash
.venv/bin/python -c \
  "import numba, numpy, scipy, sklearn, torch; print(numba.__version__, numpy.__version__, scipy.__version__, sklearn.__version__, torch.__version__)"
.venv/bin/python -c \
  "from experiments.exp411_near_repeat_qualification.backends import probe_backends; print(probe_backends())"
```

full 명령은 문서화하되 사용자 승인 전 실행하지 않는다고 바로 위에
경고한다.

- [ ] **Step 4: Exp411과 관련 회귀 테스트 전체를 실행한다**

Run:

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp411_near_repeat_qualification/tests -p "test_*.py"
.venv/bin/python -m unittest \
  experiments.exp403_408.tests.test_synthetic_anomaly
.venv/bin/python -m unittest discover \
  -s experiments/exp410_family_robustness/tests -p "test_*.py"
.venv/bin/python -m compileall -q \
  experiments/exp411_near_repeat_qualification \
  scratch/run_exp411_near_repeat_qualification.py
git diff --check
```

Expected: Exp411 tests 0 failures, Exp407 synthetic 9 tests PASS, Exp410
20 tests PASS, compileall exit 0, whitespace error 0.

- [ ] **Step 5: 금지 필드와 정책 불변성을 독립 검사한다**

Run:

```bash
rg -n "labels_blob|split = ['\\\"]TEST|test_label|is_true_anomaly|tail replacement" \
  experiments/exp411_near_repeat_qualification
rg -n "Hard 2,218|Standard 806|Priority 31|exact-normal 35|별도 명시적 승인" \
  docs/EXP411_PREREGISTRATION_KO.md docs/EXP411_RUNBOOK_KO.md
git status --short
```

Expected: 첫 명령은 구조 테스트의 forbidden literal 이외 production
module에서 0건, 두 번째 명령은 동결 수치와 승인 gate를 모두 찾고, 변경은
계획된 Exp411/docs 파일뿐.

- [ ] **Step 6: 문서와 최종 검증을 커밋한다**

```bash
git add docs/EXP411_PREREGISTRATION_KO.md docs/EXP411_RUNBOOK_KO.md \
  experiments/exp411_near_repeat_qualification/tests/test_documentation_contract.py
git commit -m "docs: preregister Exp411 near-repeat qualification"
```

- [ ] **Step 7: full 실행 없이 최종 상태를 확인한다**

Run:

```bash
git status --short --branch
git log -8 --oneline
```

Expected: clean worktree. `run_full_qualification`은 호출하지 않았고 로컬
1,117-dataset 결과 디렉터리는 생성되지 않음.

## Deferred Follow-up Plans

Exp411 full 결과와 `exp411_selected_rule.json`이 동결된 뒤에만 다음 계획을
작성한다.

1. Exp412 plan: 선택된 단일 rule을 TEST values에 label-blind 적용하고
   `near_repeat_shadow_normal`을 동결한 후 별도 evaluation.
2. Exp413 plan: Exp412가 TP 손실 0, FP 감소 1개 이상 gate를 통과한 경우
   family robustness와 backend parity를 독립 감사.
3. Exp414 plan: Exp413이 `promotion_research_worthy`여도 사용자가 별도로
   승인한 경우에만 자동 정상 lane 변경을 사전등록.
