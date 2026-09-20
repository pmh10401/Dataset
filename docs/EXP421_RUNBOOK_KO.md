# Exp421 실행 런북

이 런북은 동결된 Exp417 계산을 이용한 **작은 TRAIN-only 진단**을 한 번
실행하는 방법이다. 공식 Exp419 재실행, 1,117개 전체 회귀, TEST 평가를
시작하는 방법은 포함하지 않는다.

## 1. 실행 전 확인

먼저 공식 작업이 살아 있는지 확인하고, Exp419의 프로세스·출력·checkpoint를
건드리지 않는 별도 터미널과 새 output root를 사용한다. 데이터베이스는
읽기만 한다.

```bash
DB="/Users/mac/Documents/Dataset/univariate_ts.db"
shasum -a 256 "$DB"
sqlite3 "$DB" "PRAGMA quick_check;"
git status --short --branch
```

SHA-256이 동결 값과 다르거나 `quick_check`가 `ok`가 아니면 중지한다. 현재
공식 Exp419 root를 `--output-root`로 지정하지 않는다. 결과를 담을 디렉터리는
존재하지 않거나 비어 있어야 하며, 기존 파일이 있으면 CLI가 시작 전에
거부한다.

## 2. 승인된 형태의 작은 실행

아래는 형식 예시다. 실제 데이터에 실행하려면 이 진단 실행에 대한 별도
승인이 필요하다.

```bash
PYTHON="/Users/mac/Documents/Time Series/.venv/bin/python"
DB="/Users/mac/Documents/Dataset/univariate_ts.db"
OUT="/tmp/exp421-attempt-$(date +%Y%m%d-%H%M%S)"

"$PYTHON" scratch/run_exp421_mps_batched_acceleration.py \
  --database "$DB" \
  --output-root "$OUT" \
  --dataset CricketX_normal_1 \
  --repeats 1
```

`--dataset`은 정상 TRAIN dataset 이름을 지정할 때만 반복한다. 생략하면
동결된 입력 경계가 허용하는 정상 TRAIN 모집단에서 제한된 표본을 고른다.
`--repeats`는 1~7의 측정 반복 수이며 결과 선택 규칙이나 threshold를 바꾸지
않는다. CLI는 CPU thread 수를 1로 고정하고 한 프로세스에서 CPU와 MPS를
순차 측정한다. MLX는 metadata만 확인한다.

## 3. 결과 확인

성공적으로 매니페스트가 만들어지면 마지막 줄에 다음 세 가지가 JSON으로
출력된다.

- `manifest_path`: 새 output root 안의 `manifest.json`
- `state`: 모든 고정 게이트가 통과한 `passed` 또는 보수적으로 종료한 `failed`
- `policy_reasons`: batch별 `mps_unavailable`, `speedup_below_1.1`,
  `qualification_mismatch` 같은 이유

매니페스트에는 database/config/source/Git digest, 시작·종료 시각, PID/host,
fit 모집단 count/digest, CPU/MPS timing, finite·좌표·수치·rank·qualification
검증, MLX 버전과 fallback 사유가 들어간다. `failed`는 문제를 숨기지 않은
정상 결과이며, 운영 prediction이나 lane 판정으로 복사하지 않는다.

## 4. 흔한 중단 원인

- `database` SHA 또는 SQLite `quick_check` 불일치: 동결 데이터가 아니므로 중단
- output root가 공식 Exp419이거나 비어 있지 않음: 다른 경로를 새로 지정
- TEST/label/ground-truth 이름을 지정함: 정상 TRAIN 이름만 사용
- MPS가 없거나 출력이 비유한·좌표 불일치·오차 초과·rank/qualification 실패:
  해당 batch는 CPU fallback, 원인은 매니페스트에 기록
- 메모리 부족이나 runtime 오류: 시도를 보존한 뒤 원인을 확인하고, 임의로
  batch·alpha·threshold를 바꾸지 않음

## 5. 안전한 취소와 금지 사항

진행 중 취소가 필요하면 해당 터미널에서 `Ctrl-C`로 진단 프로세스만
종료한다. 부분 output root는 바로 지우지 말고 매니페스트·로그가 남았는지
확인한 뒤 새 root에서 다시 시도한다. 공식 Exp419 프로세스나 디렉터리는
종료·삭제·덮어쓰지 않는다.

이 프로토타입은 Exp419 checkpoint/prediction/TEST label을 재사용하지 않으며,
1,117개 전체 실행을 자동으로 시작하지 않는다. 전체 실행이나 운영 반영은
새 실험 번호, 사전등록, 별도 사용자 승인을 거친 뒤에만 계획한다.
