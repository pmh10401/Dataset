# Exp410 동결 Hard family 견고성 감사 실행서

## 1. 실행 범위

Exp410은 기존 Exp403-408 Hard 2,218개를 변경하지 않는 읽기 전용 회고
감사다. 구현 검증 단계에서는 32개 합성 데이터셋 smoke만 실행한다.
약 1.6GB 실제 SQLite DB를 읽는 전체 감사는 이 문서의 마지막 명령을
사용자의 별도 명시적 승인 후에만 실행한다.

다음 값을 실행 옵션으로 바꿀 수 없다.

- family cap `1 / 3 / 5 / 10 / unlimited`
- bootstrap seed `410`, 반복 `2,000`
- robustness margin `-0.05`, severe margin `-0.10`
- prediction, label, DB, Exp291/292 SHA256
- threshold, alpha, source, route, review budget

## 2. 환경 활성화

저장소의 기존 가상환경과 portable data root를 사용한다.

```bash
cd "/Users/mac/Documents/Time Series/.worktrees/exp409-prospective-readiness"
source "/Users/mac/Documents/Time Series/.venv/bin/activate"
export TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset"
```

현재 정본 환경은 Python 3.14.6, NumPy 2.3.5, PyTorch 2.13.0이다.
PyTorch MPS는 built `true`이지만 available `false`이고 MLX는 설치되어
있지 않다.

Exp410의 지배 작업은 SQLite streaming, 정확한 blob byte hashing,
CSV/JSON parsing, grouped counter, 128-family bootstrap이다. tensor 중심
작업이 아니므로 CPU SQLite/`hashlib`/NumPy가 정본이자 fallback이다.
PyTorch·MLX·MPS 경로는 동일 identity와 수치 parity를 만족하고
transfer/synchronization 포함 최소 1.1배 benchmark가 있을 때만 정본
승격을 검토한다. Exp410을 위해 새 가속 package를 설치하지 않는다.

## 3. 단위·회귀 테스트

```bash
python3 -m unittest discover \
  -s experiments/exp410_family_robustness/tests \
  -p "test_*.py"

python3 -m unittest discover \
  -s experiments/exp403_408/tests \
  -p "test_*.py"

python3 -m unittest discover \
  -s experiments/exp409_prospective_readiness/tests \
  -p "test_*.py"

python3 -m py_compile \
  experiments/exp410_family_robustness/*.py \
  scratch/run_exp410_family_robustness.py
```

macOS semaphore sandbox가 Exp403-408 process-pool 테스트를 막으면 동일
가상환경에서 해당 테스트 명령만 sandbox 밖에서 다시 실행한다.

## 4. 대표 합성 smoke

output은 반드시 존재하지 않는 새 경로여야 한다. smoke는 임시 32-dataset
publication과 SQLite fixture를 만들며 실제 DB를 열지 않는다.

```bash
exp410_smoke_parent=$(mktemp -d "/private/tmp/exp410-smoke-parent.XXXXXX")
python3 scratch/run_exp410_family_robustness.py \
  --representative-smoke \
  --output-dir "$exp410_smoke_parent/publication"
```

합성 family 크기는 `1 / 3 / 7 / 21`이며 다음을 모두 포함한다.

- family 크기 `1`, `2-5`, `6-20`, `21+`
- normal TRAIN `<20`, `20-49`, `50+`
- exact overlap `<50%`, `50-<90%`, `>=90%`
- TEST anomaly `1`, `2-10`, `11+`
- 무경보 데이터셋과 TP·FP를 모두 가진 데이터셋

smoke는 실행·schema·hash·원자적 게시만 검증하며 최종 family 견고성
분류의 연구 결과로 해석하지 않는다.

## 5. Smoke artifact와 hash 검증

성공 시 다음 9개 파일이 모두 있어야 한다.

- `exp410_frozen_config.json`
- `exp410_dataset_metrics.csv`
- `exp410_family_metrics.csv`
- `exp410_weight_sensitivity.csv`
- `exp410_family_influence.csv`
- `exp410_stratified_metrics.csv`
- `exp410_summary.json`
- `EXP410_RESULT_REPORT_KO.md`
- `MANIFEST.json`

```bash
python3 -m json.tool \
  "$exp410_smoke_parent/publication/exp410_summary.json"

python3 -m json.tool \
  "$exp410_smoke_parent/publication/MANIFEST.json"

shasum -a 256 "$exp410_smoke_parent/publication"/*

python3 - "$exp410_smoke_parent/publication" <<'PY'
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
print("Exp410 smoke manifest hashes: OK")
PY
```

summary에서 `smoke`, `full_audit_executed`는 각각 `true`, `false`여야
한다. `operational_policy_changed`, `deployment_eligible`,
`unseen_family_validated`, `prospective_validated`는 모두 `false`여야
한다. difficulty는 다음과 같아야 한다.

```json
{
  "status": "not_evaluated",
  "reason": "source_artifact_unavailable"
}
```

## 6. 즉시 중단 조건

- Git worktree가 dirty이거나 실행 중 commit이 바뀜
- input/manifest/DB/Exp291/292 SHA256 불일치
- Hard identity 또는 Exp291/292 lane parity 불일치
- SQLite `PRAGMA quick_check` 실패 또는 DB/WAL/SHM snapshot 변화
- dataset/family/bucket/Hard/TP/FP/FN/F1 고정값 불일치
- 잘못된 dataset 이름, label, index, overlap 또는 bootstrap 계약
- 기존 output 경로, 부분 artifact, staging hash 불일치

실패 후 같은 Exp410에서 cap, seed, margin, threshold, alpha, source,
budget을 바꾸어 재선택하지 않는다. 개선안은 새 실험 번호와 새
사전등록으로 분리한다.

## 7. 사용자의 별도 명시적 승인 후에만

아래 명령은 약 1.6GB 실제 DB를 streaming scan하는 전체 감사다.
단위·회귀 테스트와 대표 smoke 및 manifest hash 검증이 모두 통과한 뒤,
사용자가 이 전체 실행을 별도로 명시 승인했을 때만 실행한다. output
디렉터리는 미리 존재하면 안 된다.

```bash
python3 scratch/run_exp410_family_robustness.py \
  --full-audit \
  --input-dir "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation" \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --output-dir "/Users/mac/Documents/Dataset/exp410_family_robustness"
```

전체 결과도 unseen family, 새 데이터, 반도체 현장, prospective,
deployment 또는 자동 경보 향상을 입증하지 않는다. Hard, Standard,
Priority lane은 변경되지 않는다.
