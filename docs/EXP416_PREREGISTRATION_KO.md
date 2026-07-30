# Exp416 Exp404 Hard 실제 TEST 혼동행렬 감사 사전등록

## 목적과 고정 범위

Exp416은 이미 동결된 Exp404 Hard prediction 좌표를 frozen TEST label과
한 번 대조하는 retrospective audit이다. 새 prediction을 만들지 않으며
Exp404의 alpha, threshold, source, candidate budget 또는 lane을 다시
선택하지 않는다. TEST-informed reselection은 금지한다. 이 결과는
human-assisted evidence이며 운영 경보 승격이나 운영 정책 변경의 자동
근거가 아니다.

Exp414는 TRAIN-only near-repeat qualification에서
`no_qualified_near_rule`로 종료했다. **Exp415 prediction not created**.
Exp416은 Exp414 성능을 주장하지 않는다.

## 읽기 전 고정한 입력 계약

| 입력 | SHA256 | 고정 계약 |
| --- | --- | --- |
| Exp404 Hard prediction | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` | 2,218개 고유 좌표, 888개 prediction dataset |
| Exp404 prediction manifest | `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8` | 1,117개 모집단 dataset 및 동결 lane |
| frozen TEST labels | `b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4` | 195,900개 row, 1,117개 dataset |
| source DB reference | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` | provenance reference only |

입력 경로는 design에 등록된 읽기 전용 절대 경로를 사용한다. frozen TEST
labels의 정본은 `dataset,candidate_index,is_true_anomaly` CSV이며, DB에서
재생성·보정·대체하지 않는다. Tasks 1–4에서는 실제
`exp403_408_frozen_test_labels.csv`를 열거나 hash, stat, inspect하지 않으며
synthetic fixture만 사용할 수 있다.

## prediction-first 동결 순서

다음 여섯 단계가 모두 성공하기 전에는 TEST label loader, TEST label
argument 또는 metric 계산이 존재해서는 안 된다.

1. Exp404 prediction과 manifest의 SHA256을 검증한다.
2. Manifest의 `contains_test_labels=false`,
   `contains_evaluation_metrics=false`, `datasets=888`을 검증한다.
3. 모든 prediction row의 `contains_test_labels=false`와
   `contains_evaluation_metrics=false`를 검증한다.
4. prediction 좌표가 고유하며 정확히 2,218개인지 검증한다.
5. manifest `frozen_hard_lanes`의 비어 있지 않은 좌표 집합이 prediction
   좌표 집합과 정확히 같은지 검증한다.
6. 검증된 좌표를 canonical 정렬하고 SHA256을 계산하여 Exp416 prediction
   identity로 동결한다.

그 뒤에만 독립 TEST audit이 label hash와 header를 검증하고, duplicate 및
invalid label을 거부하며, label dataset scope와 1,117개 manifest population의
동일성을 검증한다. prediction 좌표는 label universe에 정확히 한 번 존재해야
한다. TEST 결과를 본 뒤 prediction을 다시 만들거나 filter하지 않는다.

## 혼동행렬과 metric

Positive class는 `is_true_anomaly=1`, predicted positive는 Exp404 Hard
prediction 좌표 집합이다.

- TP: prediction 좌표이며 label이 1
- FP: prediction 좌표이며 label이 0
- FN: prediction 좌표가 아니며 label이 1
- TN: prediction 좌표가 아니며 label이 0

전체 195,900개 label universe를 분모로 다음 micro metric을 exact integer
count에서 직접 계산한다.

- precision = `TP / (TP + FP)`
- recall = `TP / (TP + FN)`
- specificity = `TN / (TN + FP)`
- false positive rate = `FP / (FP + TN)`
- false negative rate = `FN / (FN + TP)`
- F1 = `2TP / (2TP + FP + FN)`

분모가 0인 metric은 `null`이며 0으로 대체하지 않는다. 모든 count는 bool이
아닌 음이 아닌 exact integer이고 `TP + FP + FN + TN = 195,900`을 보존해야
한다. 각 dataset에도 같은 정의를 적용한다. Macro metric은 해당 분모가
정의된 dataset의 산술평균만 사용하고, `defined_dataset_count`를 함께
기록한다. Undefined dataset metric은 macro 평균에 포함하지 않으며 0으로
치환하지 않는다.

## 금지된 재선택과 publication 계약

TEST label을 열거나 결과를 본 뒤 alpha, threshold, source, candidate budget,
lane, family 또는 과거 TEST 성능에 따라 variant를 재선택하지 않는다.
Exp404 prediction은 수정하지 않으며 `prediction_modified=false`,
`test_informed_reselection=false`, `operational_policy_changed=false`,
`exp415_prediction_created=false`를 결과 provenance에 기록한다.

새 output directory에 다음 정확히 네 파일만 게시한다.

1. `exp416_confusion_matrix.json`
2. `exp416_dataset_metrics.csv`
3. `EXP416_RESULT_REPORT_KO.md`
4. `MANIFEST.json`

기존 output directory가 있으면 fail closed한다. 새 staging directory에서
위 세 non-manifest payload의 canonical UTF-8/CSV content와 SHA256을 모두
재검증한 뒤 `MANIFEST.json`을 만들고, 파일 집합과 hash를 다시 확인한 후
새 directory로 atomic rename한다. overwrite publication은 금지하며, callback
이후에도 staging hash와 정확한 파일 집합을 재검증한다.
