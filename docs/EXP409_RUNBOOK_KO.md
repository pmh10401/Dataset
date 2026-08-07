# Exp409 대표 readiness smoke 실행서

## 실행 전 확인

이 runbook은 대표 데이터셋 8개의 readiness smoke만 허용한다. 전체
1,117개 실행은 허용 범위가 아니다.

실행 전에 다음을 확인한다.

1. 현재 브랜치가 `codex/exp409-prospective-readiness`이고 Git worktree가
   깨끗하다.
2. `/Users/mac/Documents/Dataset/univariate_ts.db`가 존재하고 SQLite
   `PRAGMA quick_check`가 `ok`이다.
3. 출력 경로
   `/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke`가
   존재하지 않는다. 이전 출력 경로를 재사용하지 않는다.
4. Exp409 및 Exp403-408 테스트가 모두 통과한다.

## 유일한 실행 명령

```bash
export TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset"
python3 scratch/run_exp409_prospective_readiness.py \
  --representative-smoke \
  --dataset-limit 8 \
  --output-dir \
  "/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke"
```

variant, alpha, seed, independence threshold, review 예산, 계산 장치 또는
checkpoint 동작을 바꾸는 실행 옵션은 제공하지 않는다.

## 출력 파일

원자적 publication이 성공하면 아래 파일이 모두 존재한다.

- `exp409_config.json`: 동결 구성과 비성능 claim
- `exp409_shadow_candidates.jsonl`: 정렬된 review-only 후보
- `exp409_source_statuses.jsonl`: 데이터셋별 13개 source 상태와 output hash
- `exp409_provenance_manifest.json`: canonical provenance와 `closure_sha256`
- `exp409_readiness_summary.json`: readiness 수량과 동결 claim
- `exp409_readiness_report_ko.md`: 한국어 readiness 보고
- `exp409_runtime_envelope.json`: canonical closure와 분리된 실행 환경
- `exp409_artifact_manifest.json`: 파일별 SHA256

`exp409_shadow_candidates.jsonl`은 0행일 수 있지만 후보 수가 100을 넘을
수는 없다.

## 성공 후 검증

```bash
shasum -a 256 \
  "/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/exp409_shadow_candidates.jsonl"

python3 -m json.tool \
  "/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/exp409_provenance_manifest.json"

python3 -m json.tool \
  "/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/exp409_readiness_summary.json"

rg -n -i \
  '"(test_labels|is_true_anomaly|actual_anomaly_positions|tp|fp|fn|f1|precision|recall)"' \
  "/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke"
```

후보 JSONL의 `shasum`은 provenance의 `candidate_sha256`과 같아야 한다.
canonical provenance를 다시 직렬화한 SHA256은 `closure_sha256`과 같아야
한다. provenance의 label access, lane parity, review budget, determinism
gate는 모두 `passed`여야 한다. forbidden-field 검색은 prediction-side
JSON/JSONL에서 match가 없어야 한다.

## 즉시 중단 조건

- TEST label, 실제 이상 위치 또는 TEST 평가값 접근 흔적
- Hard / Standard / Priority lane membership 불일치
- 전체 review 예산 100 초과
- 두 in-memory 조립 사이 후보, source hash, lane 또는 value hash 불일치
- 13개 source status coverage 누락 또는 알 수 없는 abstention 사유
- Git dirty 상태, DB/lane/source/code SHA256 불일치
- 기존 출력 경로 발견 또는 staging/file hash 오류

예산을 넘으면 일부 상위 후보를 고르지 않는다. publication 디렉터리가 없어야
하며, 별도의 candidate-free failure JSON만 허용한다. gate 실패 후 alpha,
variant, threshold 또는 예산을 바꾸어 같은 Exp409를 다시 실행하지 않는다.

## 해석 제한

성공 결과의 `performance_evaluation`은 `not_evaluated`이고,
`prospective_validated`, `deployment_eligible`,
`operational_policy_changed`는 모두 false다. 이는 새 데이터가 없는 현재
상태에서 실행 준비를 확인한 것이며 성능 개선이나 배포 가능성을 입증하지
않는다.

