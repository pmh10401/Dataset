# Exp411 TRAIN-only Near-Repeat 자격 심사 실행서

## 1. 실행 범위

Exp411은 정상 TRAIN만으로 near-repeat 규칙의 자격을 심사한다. 구현 검증
단계에서는 32개 합성 dataset 대표 smoke만 실행한다. 실제 DB의 1,117개
전체 자격 심사는 사용자의 별도 명시적 승인 후에만 실행한다.

현재 Hard 2,218 / Standard 806 / Priority 31과 exact-normal 35는
변경하지 않는다. Exp412 prediction도 만들지 않는다.

## 2. 작업 경로와 환경 확인

worktree 경로를 이름으로 추정하지 말고 branch와 commit을 확인한다.

```bash
git -C "/Users/mac/Documents/Time Series" worktree list
cd "/Users/mac/Documents/Time Series/.worktrees/exp409-prospective-readiness"
test "$(git branch --show-current)" = \
  "codex/exp411-413-near-repeat-normal-validation"
git rev-parse HEAD
git status --short --branch
```

프로젝트 전용 가상환경을 사용한다.

```bash
.venv/bin/python -c \
  "import numba, numpy, scipy, sklearn, torch; print(numba.__version__, numpy.__version__, scipy.__version__, sklearn.__version__, torch.__version__)"

.venv/bin/python -c \
  "from experiments.exp411_near_repeat_qualification.backends import probe_backends; print(probe_backends())"
```

NumPy CPU는 정본과 fallback이다. Numba CPU JIT와 PyTorch CPU는 동일
32-dataset stage에서 수치·query identity/index별 tag·선택 quantile parity와
최소 1.1배 속도 gate를 평가하며 Numba 전체 stage cold/warm 시간을 따로
기록한다. 이 평가는 실제 DB를 열기 전에 끝나야 한다. 전체 population
parity가 동결되기 전에는 가속 backend를 채택하지 않는다. PyTorch MPS가
unavailable이면 이유를 기록하고 CPU를 사용한다. 사용자 결정에 따라 MLX는
제외하며 설치·import·benchmark하지 않는다.

## 3. 단위·회귀 테스트

```bash
LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m unittest discover \
  -s experiments/exp411_near_repeat_qualification/tests \
  -p "test_*.py"

LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m unittest \
  experiments.exp403_408.tests.test_synthetic_anomaly

LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m unittest discover \
  -s experiments/exp410_family_robustness/tests \
  -p "test_*.py"

.venv/bin/python -m compileall -q \
  experiments/exp411_near_repeat_qualification \
  scratch/run_exp411_near_repeat_qualification.py
```

## 4. 대표 합성 smoke

output은 반드시 존재하지 않는 새 경로여야 한다. smoke는 실제 DB나 TEST
split을 읽지 않는다.

```bash
exp411_smoke_parent=$(mktemp -d "/private/tmp/exp411-smoke-parent.XXXXXX")
.venv/bin/python scratch/run_exp411_near_repeat_qualification.py \
  --representative-smoke \
  --output-dir "$exp411_smoke_parent/publication"
```

smoke는 실제 family 크기 `1 / 3 / 7 / 21`로 합계 32개 합성 dataset을
구성한다. dataset별 identity group은 4, 6, 20개를 순환하여 `<5` abstain,
5-19 LOGO, 20+ deterministic 5-fold 경로를 모두 지난다. 실행·fold·거리·
합성 교란·집계·backend assessment·원자적 게시 연결을 검사할 뿐 최종 연구
결과로 해석하지 않는다.

## 5. Artifact와 manifest 검증

성공 시 다음 8개 파일이 모두 있어야 한다.

- `exp411_frozen_config.json`
- `exp411_dataset_qualification.csv`
- `exp411_quantile_summary.csv`
- `exp411_backend_assessment.json`
- `exp411_selected_rule.json`
- `exp411_summary.json`
- `EXP411_RESULT_REPORT_KO.md`
- `MANIFEST.json`

```bash
.venv/bin/python -m json.tool \
  "$exp411_smoke_parent/publication/exp411_summary.json"

.venv/bin/python -m json.tool \
  "$exp411_smoke_parent/publication/MANIFEST.json"

shasum -a 256 "$exp411_smoke_parent/publication"/*

.venv/bin/python - "$exp411_smoke_parent/publication" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
for name, expected in manifest["artifact_sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA256 mismatch: {name}")
if len(manifest["artifact_sha256"]) != 7:
    raise SystemExit("Exp411 manifest must hash seven non-manifest artifacts")
print("Exp411 manifest hashes: OK")
PY
```

`exp411_summary.json`에서 다음 값이 확인되어야 한다.

```json
{
  "mode": "representative_smoke",
  "dataset_count": 32,
  "full_qualification_executed": false,
  "operational_policy_changed": false,
  "exp412_prediction_created": false,
  "test_labels_used": false,
  "test_values_used": false
}
```

## 6. 즉시 중단 조건

- Git worktree가 dirty이거나 실행 중 commit이 바뀜
- population manifest SHA256 또는 DB SHA256 불일치
- 데이터셋 1,117개 / family 128개 불일치
- SQLite `PRAGMA quick_check` 실패, WAL/SHM 출현, snapshot 변화
- TRAIN 이외 split, TEST label, `labels_blob`, 실제 이상 위치 접근
- 잘못된 TRAIN label/blob/index 또는 identity fold 분리
- 고정 quantile `0.01 / 0.05 / 0.10`, seed `411`, gate 또는 교란 변경
- 기존 output 경로, 부분 artifact 또는 staging hash 불일치

실패하면 같은 Exp411에서 threshold, quantile, source, view, 합성 교란 또는
budget을 바꾸어 재선택하지 않는다. `no_qualified_near_rule`은 유효한
종료 결과이며 Exp412를 시작하지 않는다.

## 7. 사용자의 별도 명시적 승인 후에만

아래 명령은 실제 DB의 정상 TRAIN을 읽어 1,117개 전체 자격 심사를 수행한다.
이 문서에 명령이 있다는 사실은 실행 승인이 아니다. 단위·회귀 테스트,
대표 smoke, manifest hash 검증이 모두 통과하고 사용자가 이 전체 실행을
별도로 명시 승인했을 때만 실행한다. output은 미리 존재하면 안 된다.
loader는 dataset 하나씩 처리하고 해제하며, self-excluded 거리는 bounded
block의 직접 elementwise 차분으로 계산한다.

```bash
.venv/bin/python scratch/run_exp411_near_repeat_qualification.py \
  --full-qualification \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --population-manifest "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json" \
  --output-dir "/Users/mac/Documents/Dataset/exp411_near_repeat_qualification"
```

전체 결과가 `qualified`여도 이는 TRAIN-only 자격 통과일 뿐 운영 승격이
아니다. 결과를 먼저 동결한 뒤 별도 사전등록된 Exp412 Shadow prediction
여부를 결정한다. TEST 결과를 보고 Exp411 규칙을 다시 고르지 않는다.
