# Exp444 TRAIN-only 안정성 정밀화 설계

## 1. 한 문장 설명

Exp444는 Exp442가 선택한 30개 후보를 새로 늘리지 않고, 정상 TRAIN 구성을 조금씩
바꿔도 계속 이상하게 보이는지 반복 확인하여 `stable review`와
`unstable review`로 나누는 설명용 실험이다.

이 실험에는 합격·실패 기준, scientific winner, champion 승격, 운영 경보 승격이
없다. 결과가 좋아도 나빠도 수치를 그대로 기록하며, 같은 Exp444 안에서 결과를
보고 규칙을 다시 고르지 않는다.

## 2. 배경

Exp442는 기존 Hard 기준에 30개 후보를 더해 이상 15개를 새로 찾았고 오경보도
15개 늘렸다. 기존 강한 데이터셋의 악화는 없었고 micro·macro F1은 증가했다.
그러나 이 결과는 적응형 review-only 근거이며 독립 검증이나 운영 승격 근거가
아니다.

Exp443은 결측 정상 TRAIN 처리 세 방법을 비교했다. leave-one-out peer median이
이상 1개를 오경보 없이 추가로 찾았지만 효과는 한 데이터셋에만 나타났다.
따라서 Exp444는 Exp443의 보완법을 섞지 않는다. Exp440/442의 원래 결측 처리와
점수 정의를 유지하여, 후보 안정성이라는 한 가지 질문만 분리해 본다.

## 3. 고정 실험 신원

```text
experiment_id = Exp444
run_identity = exp444-train-only-stability-precision-gate-antigravity-20260810
run_root = /Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810
dataset_count = 1117
source_candidate_count = 30
candidate_per_dataset = 1
maximum_stability_views = 16
stability_fraction = 0.875
mask_seed = 44420260810
bootstrap_seed = 44420260811
bootstrap_samples = 20000
acceleration_tolerance = 1e-5
acceleration_speedup_min = 1.1
cpu_workers = max(1, logical_cpu_count - 1)
expected_cpu_workers_on_current_mac = 11
inner_numeric_threads = 1
review_only = true
operational_alert_eligible = false
scientific_winner_selected = false
```

새 root는 readiness 전까지 존재하지 않아야 한다. Exp440, Exp442, Exp443의 기존
root와 artifact는 읽기 전용이며 수정·복사·resume하지 않는다.

## 4. 동결 입력

Exp444 prediction이 읽을 수 있는 입력은 다음뿐이다.

1. Exp442의 동결된 30개 prediction 좌표
2. Exp442의 49개 dataset winner 순위와 source/selection audit
3. Exp440의 동결된 checkpoint와 저주파 점수 정의
4. Exp404의 동결 Hard 좌표
5. SQLite의 정상 TRAIN `values_blob`
6. 30개 후보 좌표에 해당하는 TEST `values_blob`과 TEST identity

고정해야 할 주요 SHA256은 다음과 같다.

```text
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
exp404_hard_prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
exp442_prediction_sha256 = b79bad6012ad6bc07050ade9968e815796a2cc871b5c85d63f0bd7dbdf6f81f4
exp442_candidate_ranking_sha256 = d8d858cd74553aa64f6f0a5424064872b59f682fed1c8b2ace63fffba6ad20cc
exp442_selection_sha256 = 06004f18dbd9b0df3fa857bb8b1a9fd7eafd4d927f3736fb8cfb25db56b72408
exp442_selected_coordinate_identity_sha256 = db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048
```

readiness는 파일 경로만 믿지 않고 전체 bytes, artifact map, population, 좌표 순서,
후보 수, dataset별 최대 1개 규칙을 다시 검증한다.

## 5. 절대 금지 입력

prediction, readiness, checkpoint, publication은 다음을 import·query·stat·hash하거나
규칙에 사용할 수 없다.

- TEST `label` 또는 `labels_blob`
- 실제 이상 위치
- Exp442 또는 이전 실험의 TP·FP·F1·confusion·평가 파일
- family별 과거 TEST 성능
- TEST 결측 여부를 이용한 후보 선택
- TEST index가 앞·중간·끝인지에 따른 규칙
- tail replacement 또는 tail 우선 규칙
- 평가 후 threshold, view 수, 안정성 비율, source, budget 재선택

정상 TRAIN을 선택하기 위한 기존 `TRAIN label = 0` 조건만 허용한다. TEST 라벨은
prediction publication과 SHA256이 완전히 동결된 뒤 별도 evaluator만 읽는다.

## 6. 원본 Exp442 재현 제어

안정성 view를 만들기 전에 30개 후보를 원래 전체 정상 TRAIN으로 다시 계산한다.

- Exp440의 exact-length grouping을 그대로 사용한다.
- whole-row nonfinite TRAIN 제외를 그대로 사용한다.
- TEST 전처리, FFT 저주파 depth, robust center/scale, nearest `k=3`, OOF 0.99
  higher quantile, strict `score > threshold`, 최소 2-depth 합의를 그대로 사용한다.
- 30개 좌표, agreeing depth, candidate 여부가 Exp442와 정확히 같아야 한다.
- 같은 고정 환경의 CPU scalar score·threshold·excess ratio·empirical rank는 canonical
  float 값까지 같아야 한다. MLX/Numba 비교에만 절대오차 `1e-5`를 허용한다.

이 제어가 실패하면 안정성 결과를 만들지 않고 root를 오류 증거로 보존한다.

## 7. TRAIN 안정성 view

각 후보는 자기 dataset과 자기 `raw_length`에 해당하는 정상 TRAIN group만 사용한다.
공식 30개 후보의 exact-length group 크기는 5 이상이므로 한 행 이상 제외한 뒤에도
Exp440의 최소 4개 TRAIN 조건을 지킬 수 있다.

group 크기를 `n`이라고 할 때 한 view에서 제외할 행 수는 다음과 같다.

```text
omit_count = max(1, floor(0.10 * n))
```

view mask는 dataset, raw length, TRAIN instance identity, `mask_seed`, view counter를
canonical JSON으로 묶어 SHA256 순서를 만든 뒤 작은 순서의 행을 제외한다.
동일 mask는 중복 사용하지 않는다. 가능한 서로 다른 mask가 16개보다 적으면
가능한 mask만 사용한다.

```text
view_count = min(16, combination(n, omit_count))
required_survival_count = ceil(0.875 * view_count)
```

각 view는 남은 TRAIN만으로 Exp440 calibration을 처음부터 다시 만들고, 원래 TEST
후보 한 행만 다시 점수화한다. 다음 두 조건을 모두 만족하면 그 view에서
`survived = true`다.

1. Exp440의 최소 2-depth strict consensus를 다시 만족한다.
2. 원래 Exp442 agreeing depth 중 최소 2개가 해당 view에서도 threshold를 넘는다.

후보의 `survival_count`가 `required_survival_count` 이상이면 `stable_review`,
그보다 작으면 `unstable_review`다. 두 lane 모두 review-only이며 어느 lane도 자동
경보가 아니다. Exp442 후보를 새로 추가하거나 다른 좌표로 교체하지 않는다.

## 8. publication 결과

prediction publication은 최소 다음을 create-only로 저장한다.

- 1,117개 dataset summary
- 원본 Exp442 30개 좌표
- candidate별 view mask identity와 SHA256
- view별 TRAIN 사용·제외 identity
- view별 agreeing depth, score, threshold, excess ratio, empirical rank
- candidate별 survival count/rate
- `stable_review` 좌표
- `unstable_review` 좌표
- source audit, control replay, backend decision
- progress/checkpoint/PID/log identity
- manifest, status, self-excluding artifact SHA256 map

모든 JSON/JSONL은 canonical하고 정렬 순서가 고정되어야 한다. publication 전후에
TEST-label access count는 0이어야 한다. `inspect`는 SQLite나 worker를 열지 않고
동결 bytes만 다시 계산한다.

## 9. 사후 평가와 보고 방식

별도 evaluator는 publication과 모든 SHA256을 두 번 확인한 뒤에만 TEST
`instance_index, label`을 읽는다. 평가 결과는 다음을 그대로 보고한다.

- 기존 Hard 기준 confusion과 micro/macro F1
- Exp442 30개 union confusion과 micro/macro F1
- Exp444 `stable_review` union confusion과 micro/macro F1
- stable/unstable 후보 수
- 각 lane의 TP와 FP
- Exp442에서 유지된 TP·FP와 제외된 TP·FP
- rescued/regressed dataset 목록
- 기존 강한 데이터셋의 변화
- Exp442 대비 dataset-paired micro/macro delta와 20,000회 bootstrap 구간
- 후보별 post-freeze label을 연결한 감사표

사용자 지시에 따라 수치 목표, 합격선, 실패선, winner 선택 규칙은 두지 않는다.
evaluation status는 결과와 무관하게 `descriptive_complete_no_promotion`만 사용할 수
있다. 결과가 좋아도 Exp442를 대체하거나 champion으로 승격하지 않으며, 다음
실험 여부는 결과를 함께 본 뒤 별도로 논의한다.

## 10. backend와 자원 사용

전체 실행 전에 정확한 안정성 stage를 CPU와 MLX로 같은 입력·seed에서 비교한다.
MLX는 다음을 모두 만족할 때만 사용한다.

1. mask, calibration, candidate lane이 CPU와 완전히 같다.
2. 모든 수치가 finite이고 사전등록 허용 오차 안이다.
3. 전송과 synchronize를 포함해 1.1배 이상 빠르다.
4. MLX/Metal 버전, device, batch, 시간과 fallback 이유가 manifest에 남는다.

조건 하나라도 실패하면 CPU를 사용한다. CPU는 현재 Mac의 논리 코어 수에서 하나를
뺀 11 worker를 사용하고, BLAS/OpenMP/Numba 계열 내부 thread는 모두 1로 고정한다.
Numba도 동일한 candidate/view 결과와 1.1배 속도 조건을 통과할 때만 선택한다.
MLX/MPS-heavy process는 동시에 하나만 실행한다.

## 11. checkpoint와 자체 진행 로그

장시간 실행은 Codex 감시가 아니라 프로세스가 다음을 원자적 JSON과 text log에
직접 기록한다.

- PID와 process identity
- 현재 dataset과 candidate index
- TRAIN/TEST 크기와 raw length
- 현재 view, 전체 view, 현재 단계와 backend
- 완료/전체, import/recompute/success/error 수
- 처리속도, 경과시간, ETA
- checkpoint 경로와 마지막 성공 record
- 최신 오류와 traceback 요약

부모 프로세스만 checkpoint, progress, PID, events, text log를 쓴다. worker는 계산
결과만 반환한다. 결과는 30개 후보의 고정 순서로 commit하며, 중단된 inactive
attempt만 exact-prefix 검증 뒤 resume할 수 있다.

## 12. 실패 처리

- source hash 또는 Exp442 좌표 불일치: root 생성 전 readiness 실패
- 원본 30개 control replay 불일치: attempt 오류 보존, stability 미실행
- TEST label 접근: 즉시 실패, prediction publication 금지
- worker 오류: dataset/candidate/view를 기록하고 ordered prefix에서 중단
- checkpoint/log 불일치: 자동 수리하지 않고 실패 증거 보존
- publication 실패: complete checkpoint를 수정하지 않는 finalization-only resume만 허용
- evaluation 중단: prediction은 그대로 보존하고 새 evaluation staging에서 재시도

Exp443에서 관찰한 `pinned database close failed`를 숨기거나 Exp443 root를 고치지
않는다. Exp444는 자신이 실제 소유한 descriptor와 pin만 닫고, close 후 pin 부재와
원본 DB link count를 별도 검증한다.

## 13. 검증 범위

구현은 최소 다음을 자동 검증한다.

- frozen source와 30개 좌표 tamper 차단
- TEST label/labels_blob/evaluation import 방화벽
- exact-length 정상 TRAIN과 nonfinite 제외 parity
- 원본 Exp442 30개 control replay
- mask 생성의 seed·순서·중복 제거 결정성
- 작은 group `n=5`와 큰 group의 view 수·생존 수 계산
- 원래 agreeing depth 두 개 유지 규칙
- stable/unstable 보존 법칙: 두 수의 합은 항상 30
- 1-worker와 11-worker의 view bytes·lane·SHA256 동일성
- CPU/MLX/Numba parity와 1.1배 fallback 계약
- checkpoint 중단·resume·orphan·stale PID 처리
- 원자적 progress와 terminal ETA `null`
- create-only publication, read-only inspect, post-freeze evaluator
- 평가 재실행의 byte identity와 prediction 불변성
- 전체 선행 Exp440/441/442 회귀

## 14. 이 설계가 의도적으로 하지 않는 일

- Exp443 peer median을 Exp442와 결합하지 않는다.
- 새로운 score source나 새로운 TEST 후보를 만들지 않는다.
- 후보 index 위치를 규칙으로 사용하지 않는다.
- budget 30을 TEST 결과에 맞춰 줄이지 않는다.
- stable 후보를 운영 경보로 승격하지 않는다.
- 결과를 보고 같은 Exp444의 16 views 또는 0.875 비율을 바꾸지 않는다.
- 외부 일반화나 독립 검증을 주장하지 않는다.

Exp444의 목적은 승자를 선언하는 것이 아니라, Exp442의 30개 후보 중 어떤 후보가
정상 TRAIN 구성 변화에도 흔들리지 않는지를 재현 가능한 자료로 만드는 것이다.
