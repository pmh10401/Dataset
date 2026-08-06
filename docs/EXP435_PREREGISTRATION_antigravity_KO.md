# Exp435 정상 패턴 합의 기반 복구 사전등록

## 1. 목적과 상태

Exp435는 정상 TRAIN 데이터로 만든 세 가지 거리 점수의 합의를 사용해 현재 Hard
기준이 놓친 이상을 찾는 label-free 실험이다. 동결 lane population의 정확히
1,117개 데이터셋과 총 195,900개 TEST row를 대상으로 한다. 기존 Hard 좌표는
바꾸지 않으며, 새 좌표는 `review_only`로만 공개한다.

- 실험 번호: Exp435
- 데이터셋: 동결 lane population의 정확한 1,117개
- prediction 단계: TEST 라벨 사용 금지
- 자동 경보 승격: 금지
- tail replacement: 금지
- 같은 실험의 TEST 기반 재선택: 금지

DB에는 동결 population 밖의 데이터셋 2개가 존재할 수 있다. Exp435는 DB 전체
이름 수를 실험 모집단으로 사용하지 않고 Exp291/292가 공동으로 고정한 1,117개
population만 사용한다. 추가 이름이 공식 결과에 들어오면 실패한다.

## 2. 고정 입력

- DB: `/Users/mac/Documents/Dataset/univariate_ts.db`
- DB SHA256:
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Exp291 prediction:
  `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl`
- Exp291 prediction SHA256:
  `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848`
- Exp291 manifest:
  `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json`
- Exp291 manifest SHA256:
  `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e`
- Exp292 prediction:
  `/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl`
- Exp292 prediction SHA256:
  `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b`
- Exp292 manifest:
  `/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json`
- Exp292 manifest SHA256:
  `0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0`
- Hard retrospective comparator prediction SHA256:
  `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`
- Hard retrospective comparator path (Exp404):
  `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl`
- Exp416 evaluation:
  `/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit/exp416_confusion_matrix.json`

사후 evaluator는 TEST label을 열기 전에 Exp404 comparator와 Exp291 동결 Hard
좌표를 다시 대조한다. Exp404 파일은 888개 데이터셋에 비어 있지 않은 2,218개
좌표를 가져야 하며, 이 좌표 집합은 Exp291 prediction의
`automatic_hard_indices`와 정확히 같아야 한다. 나머지 229개 데이터셋은 빈
Exp404 목록이어야 하며, 합계는 동결된 1,117개 전체와 일치해야 한다.

실행 Git commit은 readiness와 실제 `new` 실행에서 같은 clean commit이어야 한다.
Python, NumPy, Numba, Torch, MLX와 mlx-metal 버전은 설치 여부를 포함해 manifest에
기록한다.

모든 readiness, runtime context, checkpoint, progress, publication은 단 하나의 동결
run identity `exp435-train-only-consensus-recovery-antigravity-20260806`을 기록하고
검증한다. source commit에서 identity를 파생하지 않는다.

## 3. prediction 정보 경계

prediction이 사용할 수 있는 정보는 다음뿐이다.

- 정상 TRAIN(`split='TRAIN' AND label='0'`) 원시 값
- 정답과 이상 위치가 제거된 TEST 원시 값
- dataset 이름, instance index, TRAIN/TEST 수와 시계열 길이
- Exp291/292 동결 lane과 정확한 1,117개 population
- 이 문서에 고정한 점수식, alpha, 후보 규칙과 budget

다음은 prediction SHA256과 manifest SHA256이 동결되기 전까지 사용할 수 없다.

- TEST label과 `labels_blob`
- 실제 이상 위치
- 과거 family별 TEST 성능
- 혼동행렬과 precision/recall/F1
- human-assisted 지표
- TEST 결과를 보고 바꾼 alpha, threshold, score source, backend 또는 budget

prediction code는 label column 또는 `labels_blob`을 SELECT하지 않는다. 사후 평가
loader는 별도 모듈과 별도 CLI 명령에만 존재한다. 모든 해시·population·comparator
검증을 통과한 뒤 evaluator가 `mode=ro`인 SQLite 연결 하나만 열어
`instance_index`와 `label`을 읽고, 전체 데이터셋을 처리한 뒤 한 번 닫는다.

Exp404 comparator와 Exp291 artifact는 각각 하나의 `O_NOFOLLOW` descriptor에서
바이트를 읽어 SHA256을 계산하고 그 동일한 캡처 바이트를 parse한다. SQLite도
Exp417의 descriptor-bound 검증 read-only connection 하나만 사용하며, 정확한
해시 inode를 연 뒤에만 label query를 수행한다.

## 4. instance와 길이 경계

- 한 후보는 `(dataset, TEST instance_index)` 한 개다.
- 여러 TRAIN/TEST instance를 연결하지 않는다.
- 정상 TRAIN은 정확한 시계열 길이별로 묶는다.
- TEST instance는 같은 길이의 TRAIN 그룹하고만 비교한다.
- TRAIN 그룹 행이 4개 미만이면 그 길이는 `insufficient_train_rows`로 판단 보류한다.
- 원시 길이가 2 미만이면 `insufficient_series_length`로 판단 보류한다.
- TRAIN에 비유한 값이 있거나 공유 MAD 크기를 만들 수 없으면 해당 채널을 사용할
  수 없는 것으로 기록한다. 대표 패턴 채널이 없으면 그룹 전체를 판단 보류한다.

완전히 같은 정상 TRAIN 파형은 제거하지 않는다. leave-one-row-out 계산에서도
현재 행 하나만 제외하며, 같은 파형의 다른 행은 그대로 남긴다.

## 5. 고정 전처리와 점수

### 5.1 공통 숫자

- canonical 입력 dtype: little-endian float32
- CPU 계산 dtype: float32
- 최근접 정상 파형 수 `k = 3`
- 공식 alpha `0.01`
- TRAIN threshold quantile `q = 0.99`
- quantile 방식: `method='higher'`
- 점수 비교: `test_score > threshold`인 경우만 채널 이상
- 거리: `sqrt(mean((x - reference) ** 2))`
- k개 거리 결합: 가장 작은 3개 거리의 median
- 공유 크기 하한: `1e-12`
- ACF lag: `1..min(32, floor(length / 4))`
- ACF 최소 원시 길이: 16
- ACF 분산 하한: `1e-12`

### 5.2 대표 정상 패턴 점수

한 길이 그룹의 모든 TRAIN 값을 합쳐 중앙값과
`1.4826 * median(abs(x - median(x)))`을 계산한다. 크기가 `1e-12` 이하이면 대표
패턴 채널은 사용할 수 없고 그 그룹을 판단 보류한다.

TRAIN 중심과 크기로 TRAIN/TEST를 변환한 뒤 각 TEST 파형과 모든 정상 TRAIN
파형의 RMS 거리를 계산한다. 가장 작은 3개 거리의 median이 대표 패턴 점수다.

### 5.3 기울기 점수

각 파형에 `diff(values, prepend=values[0])`을 적용한다. 정상 TRAIN 차이값 전체의
중앙값과 MAD 크기로 변환한다. 크기가 `1e-12` 이하이면 기울기 채널만 사용할 수
없는 것으로 기록한다. 거리와 k-neighbor 결합은 대표 패턴 점수와 같다.

### 5.4 ACF 점수

길이가 16 이상인 파형에서 평균을 뺀 뒤 lag 1부터 `min(32, length//4)`까지
자기상관을 계산한다. 분모가 `1e-12` 이하이면 해당 파형의 ACF를 사용할 수 없다.
정상 TRAIN ACF 벡터와의 RMS 거리 중 가장 작은 3개의 median을 점수로 사용한다.

### 5.5 TRAIN threshold

각 TRAIN 행에 대해 그 행 하나만 reference에서 제외하고 같은 점수를 계산한다.
유한한 leave-one-row-out 점수가 4개 모두 존재할 때 그 점수들의 0.99 quantile을
`higher` 방식으로 계산한다. TEST 결과를 threshold 계산에 사용하지 않는다.

## 6. 후보 합의와 제외

한 TEST instance는 다음이 모두 참일 때만 후보가 된다.

```text
prototype_abnormal
AND (gradient_abnormal OR acf_abnormal)
AND outside_all_frozen_lanes
```

`outside_all_frozen_lanes`는 Hard, Standard, Priority와 automatic-normal 네 lane
모두의 좌표 밖이라는 뜻이다. 기존 검토 좌표를 새 발견으로 다시 세지 않는다.

기울기와 ACF가 모두 계산 불가능하면 판단 보류한다. 점수가 threshold와 같으면
정상으로 둔다. 모든 후보와 근거는 dataset, instance index 순서로 정렬한다.

## 7. 전체 후보 budget

- 공식 새 후보 상한: 100개
- 0~100개: canonical prediction 동결 가능
- 101개 이상: `candidate_budget_exceeded`로 Exp435 prediction 실패
- 초과 시 top-100 절단 금지
- 초과 결과로 alpha, threshold 또는 source 변경 금지

후보 0개는 완성 가능한 prediction이지만 개선 성공은 아니다.

## 8. MLX/CPU 결정

MLX 후보 단계는 amplitude, gradient와 ACF의 batched dense distance 계산이다.
TRAIN-only 고정 smoke에서 seed `43501`을 사용해 같은 float32 입력을 만들고,
그 입력과 seed를 CPU와 MLX에 그대로 전달해 비교한다. 이 smoke seed는 사후
bootstrap seed(`43520260806`)와 다르며, TEST 결과로 바꾸지 않는다.
production score call과 같은 granularity로, fixture는 channel 하나의 TEST query
row 정확히 하나를 사용한다. reference batch sizes만 `16, 32, 64, 128, 256`으로
고정한다.

여기서 batch size는 query 수가 아니라 reference 수를 뜻한다. MLX는 reference를
선택된 크기로 나눠 RMS 거리를 계산하고, 각 묶음의 결과를 합쳐 전체 reference에
대한 정확한 최근접 k개를 유지한다. 따라서 전체 reference를 한 번에 올리지 않아도
CPU와 같은 전역 k-neighbor median을 계산하며, readiness에서 고른 batch size가
production에도 그대로 전달된다.

- benchmark batch sizes: `16, 32, 64, 128, 256`
- warm-up: 2회
- 측정: 각 batch 5회, median wall time
- 동기화와 device transfer 시간 포함
- 점수 최대 절대오차: `1e-5`
- 각 유한 score 쌍의 정규화오차:
  `abs(cpu_score - mlx_score) / max(abs(cpu_score), abs(mlx_score), 1e-12)`
- 정규화오차 gate: 비교한 모든 유한 score 쌍에서 위 값의 최댓값이 `1e-5` 이하
- 공식 alpha 후보 좌표: 완전 동일
- 최소 속도 향상: 1.1배
- finite 출력: 필수

모든 조건을 통과한 batch 중 median 시간이 가장 짧은 batch를 readiness에서
채택하고 그대로 실행한다. TEST 결과로 batch를 바꾸지 않는다. 조건 하나라도
실패하거나 MLX/Metal이 없으면 CPU를 사용한다.

readiness에서 채택한 MLX는 실행 중 device/version/parity/output 오류가 나면 CPU
점수로 조용히 바꾸지 않고 전체 run을 실패시킨다. 따라서 manifest가 MLX인데 CPU
score가 섞이는 경우는 없다.

CPU는 최대 7 dataset worker를 사용한다. 각 worker 내부 OMP, OpenBLAS, MKL,
VecLib, NumExpr와 Numba thread는 1개로 제한한다. MLX-heavy process는 하나만
실행하며 MLX와 외부 multiprocess scoring을 동시에 사용하지 않는다.
CPU parallel path는 submitted/in-flight future도 최대 7개다. parent가 worker event를
받지 않으므로 `current_dataset`은 loading, dispatching, parent가 아직 result를
수집하지 않은 `awaiting_worker_result`, 또는 committing 중인 dataset만 가리킨다;
queued 또는 이미 completed/committed dataset을 `scoring`으로 표시하지 않는다.

## 9. 실행과 artifact 경로

- run parent:
  `/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806`
- output root:
  `/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/output_antigravity`
- checkpoint root:
  `/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/checkpoint_antigravity`
- readiness artifact:
  `/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/readiness_antigravity.json`
- evaluation root:
  `/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/evaluation_antigravity`
- run identity: `exp435-train-only-consensus-recovery-antigravity-20260806`

이 경로들은 사전등록 시점에 존재하지 않는다. readiness 직전에 비어 있는 run
parent만 한 번 만들고, readiness는 그 아래 output, checkpoint, readiness 파일,
evaluation의 네 고정 root가 모두 없을 때만 `ready`를 기록한다. production
`evaluate`는 `RUN_PARENT` 바로 아래의 정확한 `FROZEN_OUTPUT`과
`FROZEN_EVALUATION` 형제 경로만 허용하며, 서로 같거나 중첩된 helper root와
기존 root를 허용하지 않는다. 기존 root를 덮어쓰지 않는다.

## 10. 진행 로그와 재개

프로그램이 직접 다음을 기록한다.

- `progress_antigravity.json`: 원자적 현재 상태
- `events_antigravity.jsonl`: fsync append-only event
- `parent_antigravity.log`: 사람이 읽는 로그
- `run_antigravity.pid`: PID와 active 상태
- dataset별 canonical checkpoint record

필수 필드는 PID, heartbeat, current dataset, TRAIN/TEST 수와 길이, stage, backend,
completed/total, success/abstention/error, 후보 수, 처리속도, 경과시간, ETA,
checkpoint와 latest error다. 데이터셋 결과 저장 순서는 이름 순서로 고정한다.
progress와 event에 저장되는 모든 stage는 `loading`, `dispatching`,
`awaiting_worker_result`, `committing`, `idle`의 다섯 값만 사용한다. 시작·복구·완료처럼
현재 dataset이 없는 lifecycle은 `idle`로 기록한다. worker가
실제로 scoring 중이라고 parent가 증명할 event는 없으므로 parallel progress에서
`scoring`을 사용하지 않는다.

허용 모드는 `readiness`, `new`, `resume`, `inspect`, `evaluate`다. `evaluate`는
완료된 prediction/manifest/artifact SHA256을 독립 검증한 뒤에만 TEST label을 연다.

## 11. prediction 완료와 실패

prediction 완료 조건은 다음과 같다.

- 정확히 1,117개 데이터셋 처리
- 누락·중복·population 밖 데이터셋 0
- failed dataset 0
- 후보 100개 이하, 정렬·고유·범위 안·동결 lane 밖
- TEST label/labels_blob 접근 0
- checkpoint, event, summary와 prediction 수 일치
- 독립 재파싱 SHA256 일치
- inactive PID, current dataset `null`, ETA 0

입력 지문 변경, label 접근, 101개 이상 후보, 비유한 점수, worker 결과 불일치,
checkpoint 손상 또는 저장 실패는 전체 prediction 실패다. 실패 root를 지우거나
재사용하지 않는다. durable state가 `failed`인 root의 `resume`은 input session open,
PID rewrite, progress write, event append 전에 read-only로 거절되고 tree bytes는
바뀌지 않는다. `interrupted` root만 같은 frozen identity/binding을 검증한 뒤 재개할
수 있다.

## 12. 사후 평가 고정

prediction과 manifest SHA256을 먼저 동결한다. evaluator는 그 해시와 population을
검증한 뒤 TEST label을 한 번 연다. baseline은 Exp404 Hard/Exp416 결과다.

사후 평가의 순서는 고정된다. evaluator는 먼저 publication과 checkpoint를
read-only로 재생하고, Exp404/Exp291 좌표 equality gate와 모든 입력·코드·config
hash gate를 통과시킨다. TEST label을 열기 전에 output 전체 트리의 경로·종류·바이트
snapshot을 저장한다. label을 읽고 평가를 계산한 뒤에는 그 전체 output 바이트를
다시 검사하고, evaluation 임시 디렉터리의 8개 파일을 만든 직후에도 같은 검사를
수행한 다음 no-clobber atomic rename으로 publish한다. prediction publication은
절대 수정하지 않는다.

평가 root의 파일은 정확히 다음 8개다.

- `evaluation_antigravity.json`
- `dataset_metrics_antigravity.jsonl`
- `rescued_datasets_antigravity.jsonl`
- `regressed_datasets_antigravity.jsonl`
- `confusion_antigravity.json`
- `bootstrap_antigravity.json`
- `decision_antigravity.json`
- `artifact_sha256_antigravity.json`

- bootstrap 단위: dataset cluster
- resampling: 1,117개 dataset을 복원 추출
- paired 계산: 같은 bootstrap 표본에서 Hard와 Hard+Exp435를 함께 계산
- 반복 수: 20,000
- seed: `43520260806`
- 신뢰구간: percentile 2.5%와 97.5%
- macro F1에서 분모 0인 dataset의 F1: 0.0
- rescued dataset: baseline TP=0, 결합 TP>=1
- regressed dataset: 결합 dataset F1 < baseline dataset F1
- strong baseline dataset: baseline dataset F1 >= 0.8

## 13. 성공 조건

다음을 모두 만족해야 Exp435를 개선으로 인정한다.

1. 1,117개 전체 prediction과 평가 보존
2. 새 후보 100개 이하
3. rescued dataset 10개 이상
4. 추가 FP 30개 이하
5. micro F1 point delta > 0이고 paired 95% CI lower > 0
6. dataset-balanced macro F1 point delta > 0이고 paired 95% CI lower > 0
7. prediction 동결 전 TEST label 접근 0
8. 같은 Exp435 안의 TEST 기반 재선택 0
9. Exp404/Exp291 888 non-empty/2,218-coordinate/229-empty-complement equality gate 통과
10. output 전체 바이트 재검사와 evaluation no-clobber publish 통과
11. 결과는 `review_only`이고 자동 경보 승격을 하지 않음

일부 dataset 악화는 위 조건이 모두 참일 때만 허용한다. strong baseline 악화 목록을
포함한 모든 regression을 공개한다.

실패하면 FAILURE_LEDGER에 보존하고 새 실험 번호로 다음 가설을 만든다. 성공하면
CHAMPION_LEDGER에 review-only candidate로 기록하되 자동 경보 정책은 바꾸지 않는다.
