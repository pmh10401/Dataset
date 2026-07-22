# Exp290 TRAIN Raw Exact-Match Normal Guard Design

## 목적

현재 Exp253 자동 경보 후보가 동일 데이터셋의 TRAIN 정상 인스턴스와 원본 DB 값 수준에서 완전히 같으면 정상으로 판정하여 자동 이상 경보에서 제외한다. Exp290은 false alarm 감소 효과와 실제 이상 손실 여부를 후향 평가한다.

## 기준 모집단

- 입력 prediction: Exp253 `frozen_predictions.jsonl`
- 데이터셋: 1,117개
- 기존 자동 경보: `final_hard_indices`, 총 2,253건
- 기존 성능: 1,915 TP, 338 FP, micro precision 84.9978%, mean dataset Hard F1 0.608659

Exp253 자체가 retrospective이며 prior TEST review에서 선택된 upstream 구성을 포함한다. 따라서 Exp290 성공은 전체 파이프라인의 prospective 또는 end-to-end strict TRAIN-only 검증을 의미하지 않는다.

## 고정 정상 판정 규칙

후보 TEST 인스턴스의 SQLite `values_blob`이 같은 데이터셋의 TRAIN 정상 `values_blob` 중 하나와 byte-for-byte 완전히 같으면 `train_raw_exact_normal`로 판정한다.

다음은 허용하지 않는다.

- 정규화 후 동일 판정
- 길이 정렬 또는 보간 후 동일 판정
- 허용 오차를 둔 근접 일치
- correlation 또는 거리 threshold
- family 또는 dataset 이름별 분기
- TEST 라벨, TEST 성능 또는 실제 이상 위치 사용

## Prediction/evaluation 경계

Prediction 단계는 다음 정보만 읽는다.

- Exp253 동결 prediction과 hash
- dataset key
- TRAIN/TEST `instance_index`, `values_blob`

Prediction 단계 SQL은 `label`과 `labels_blob`을 SELECT하지 않는다. 각 데이터셋에서 기존 Hard, 정상 exact-match로 제거된 Hard, 최종 Hard를 저장하고 JSONL SHA256을 동결한다.

Evaluation 단계는 동결된 prediction SHA256을 검증한 뒤에만 TEST label을 읽는다. 기존 Hard와 guarded Hard의 TP, FP, FN, precision, recall, dataset F1을 계산한다.

## Lane 의미

- `guarded_hard_indices`: 자동 이상 경보
- `train_raw_exact_normal_indices`: TRAIN 정상과 완전히 같아 자동 정상 처리된 인덱스

제외된 인덱스를 Standard/Priority review로 이동하지 않는다. 사용자가 승인한 목적은 완전 동일 정상 파형을 자동 경보 과정에서 정상으로 판정하는 것이기 때문이다. 별도 데이터 품질 경보도 이번 실험에는 추가하지 않는다.

## 불변 조건

- `baseline_hard = guarded_hard ∪ train_raw_exact_normal`
- 두 집합은 상호 배타적
- baseline Hard 총수는 2,253
- 데이터셋 coverage는 1,117
- prediction JSONL에 TP, FP, FN, F1, label이 없음
- Exp253 및 기존 결과 파일을 덮어쓰지 않음

## 성공 해석

Primary 판단은 TEST 성능 하나로 운영 규칙을 재조정하는 것이 아니다. 고정 규칙의 결과를 그대로 보고한다.

- FP가 감소하고 TP 제거가 0이면 강한 성공 신호
- FP와 TP가 함께 감소하면 안전성 trade-off로 보고하고 자동 적용을 보류
- 제거가 없으면 Exp289 현상은 Recovery Priority에만 존재하며 Exp253 Hard 개선에는 영향 없음
- TP만 제거되거나 precision/mean F1이 악화되면 기각

## 출력

- 동결 prediction JSONL 및 manifest
- 데이터셋별 평가 CSV
- 전체 summary CSV
- 제거 후보 상세 CSV
- lane transition CSV
- 독립 재집계 JSON
- status/log/RESULT_REPORT

## 제한

동일한 기존 TEST 데이터에 대한 retrospective counterfactual ablation이다. 결과를 보고 exact 기준을 완화하거나 다른 유사도 기준을 추가하지 않는다. 추가 규칙은 새 실험 번호와 별도 사전등록이 필요하다.
