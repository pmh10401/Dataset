# Exp416 Exp404 Hard 실제 TEST 혼동행렬 감사 설계

## 1. 목적과 주장 경계

Exp416은 이미 동결된 Exp404 Hard 경보 좌표를 실제 TEST 라벨과 한 번
대조하여 포인트 단위 혼동행렬을 만드는 독립 retrospective audit이다.
새 prediction을 만들거나 Exp404의 alpha, threshold, source, candidate
budget을 변경하지 않는다.

Exp414는 정상 TRAIN-only near-repeat 자격 검증에서
`no_qualified_near_rule`로 종료했으므로 Exp415 prediction은 생성하지
않는다. Exp416은 Exp414 규칙의 성능을 주장하지 않으며, 기존 Exp404
Hard 경보만 평가한다.

이 감사의 결과는 설명과 연구 판단을 위한 human-assisted evidence다.
운영 경보 승격, lane 변경, 정책 재선택 또는 새로운 성능 주장의
자동 근거로 사용하지 않는다.

## 2. 고정 입력

모든 입력은 읽기 전용이며 다음 절대 경로와 SHA256으로 고정한다.

| 입력 | 경로 | SHA256 |
| --- | --- | --- |
| Exp404 prediction | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` |
| Exp404 manifest | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json` | `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8` |
| frozen TEST labels | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv` | `b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4` |
| source DB reference | `/Users/mac/Documents/Dataset/univariate_ts.db` | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |

Exp404 prediction은 정확히 2,218개 고유
`(dataset, candidate_index)` 좌표와 888개 prediction 보유 데이터셋을
가져야 한다. Manifest의 `frozen_hard_lanes`는 1,117개 모집단
데이터셋을 포함하고, 그 안의 비어 있지 않은 좌표 집합은 prediction
JSONL의 좌표 집합과 정확히 같아야 한다.

Frozen TEST labels는 스키마
`dataset,candidate_index,is_true_anomaly`를 사용한다. Dataset과
candidate index의 조합은 전역에서 고유해야 하며
정확히 195,900개 row와 1,117개 dataset을 가져야 한다.
`is_true_anomaly`는 정확한 정수 `0` 또는 `1`이어야 한다. DB에서 TEST
라벨을 다시 생성하거나 보정하지 않고 이 동결 CSV만 평가 정본으로
사용한다.

## 3. 누출 방지 실행 순서

실행은 두 단계로 분리하고 순서를 바꾸지 않는다.

### 3.1 Prediction 동결 검증

1. Exp404 prediction과 manifest의 SHA256을 검증한다.
2. Manifest의 `contains_test_labels=false`,
   `contains_evaluation_metrics=false`, `datasets=888`을 검증한다.
3. Prediction 각 row에서 `contains_test_labels=false`,
   `contains_evaluation_metrics=false`를 검증한다.
4. Prediction 좌표가 고유하고 정확히 2,218개인지 검증한다.
5. Manifest `frozen_hard_lanes`의 좌표와 prediction 좌표가 정확히
   같은지 검증한다.
6. 검증된 좌표를 canonical 정렬하고 SHA256을 계산하여 Exp416
   prediction identity로 동결한다.

이 단계에는 TEST label loader, TEST label argument와 평가 metric
계산이 존재하지 않는다.

### 3.2 독립 TEST 감사

Prediction identity가 동결된 뒤에만 frozen TEST labels를 연다.

1. Label 파일의 SHA256과 헤더를 검증한다.
2. 중복 좌표, 음수 index, `0/1` 이외 라벨과 빈 dataset을 거부한다.
3. Manifest의 1,117개 dataset 모집단과 label dataset 집합이 정확히
   같은지 검증한다.
4. 모든 Exp404 prediction 좌표가 label universe에 정확히 한 번
   존재하는지 검증한다.
5. 전체 label universe를 분모로 micro 혼동행렬을 계산한다.

TEST 결과가 보인 뒤 Exp404 prediction을 다시 만들거나 필터링하지
않는다. Confidence-by-alpha, family, 과거 TEST 성능을 이용한 사후
variant 선택도 수행하지 않는다.

## 4. 혼동행렬 정의

Positive class는 실제 이상(`is_true_anomaly=1`)이다. Predicted
positive는 Exp404 Hard 좌표 집합에 포함된 TEST 포인트다.

| | 실제 이상 | 실제 정상 |
| --- | ---: | ---: |
| Exp404 Hard 경보 | TP | FP |
| 경보 없음 | FN | TN |

- `TP`: prediction 좌표이며 label이 1
- `FP`: prediction 좌표이며 label이 0
- `FN`: prediction 좌표가 아니며 label이 1
- `TN`: prediction 좌표가 아니며 label이 0

다음 micro metric을 정수 count에서 직접 계산한다.

- precision = `TP / (TP + FP)`
- recall = `TP / (TP + FN)`
- specificity = `TN / (TN + FP)`
- false positive rate = `FP / (FP + TN)`
- false negative rate = `FN / (FN + TP)`
- F1 = `2TP / (2TP + FP + FN)`

분모가 0인 metric은 `null`로 기록하며 0으로 대체하지 않는다. 모든
count는 음이 아닌 exact integer여야 하고
`TP + FP + FN + TN`은 frozen label row 수와 같아야 한다.

## 5. Dataset 균형 보조 진단

같은 정의로 1,117개 dataset별 TP, FP, FN, TN과 metric을 계산한다.
이 값은 대형 dataset이 micro 결과를 지배하는 정도를 설명하는
진단용이다.

Macro metric은 해당 metric의 분모가 정의된 dataset만 대상으로
산술평균한다. 각 macro 결과에는 반드시 `defined_dataset_count`를
함께 기록하며 undefined를 0으로 대체하지 않는다. Dataset별 또는
family별 결과를 보고 Exp404 정책을 재선택하지 않는다.

## 6. 산출물 계약

새 출력 디렉터리는
`/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit`이며
기존 경로를 덮어쓰지 않는다. 정확히 다음 네 파일만 게시한다.

1. `exp416_confusion_matrix.json`
   - experiment ID와 audit scope
   - 입력 파일 SHA256
   - Git commit
   - canonical prediction identity SHA256
   - dataset, prediction coordinate와 frozen label row 수
   - micro TP, FP, FN, TN
   - micro metric
   - macro metric과 defined dataset count
   - `prediction_modified=false`
   - `test_labels_loaded_after_prediction_freeze=true`
   - `test_informed_reselection=false`
   - `operational_policy_changed=false`
   - `exp415_prediction_created=false`
2. `exp416_dataset_metrics.csv`
   - dataset, TP, FP, FN, TN
   - precision, recall, specificity, FPR, FNR, F1
   - undefined metric은 빈 CSV field
3. `EXP416_RESULT_REPORT_KO.md`
   - 실제 정수 혼동행렬
   - micro metric과 dataset-balanced macro 보조값
   - 결과 해석과 TEST 사용 경계
   - 운영 정책 및 자동 승격이 없다는 명시
4. `MANIFEST.json`
   - 앞의 세 non-manifest payload SHA256
   - 입력 provenance와 Git commit

JSON은 canonical UTF-8, CSV는 고정 header와 dataset 오름차순, Markdown은
결정론적으로 렌더링한다. Staging에서 세 payload를 모두 재검증한 뒤
manifest를 만들고 새 디렉터리로 원자적 rename한다. Publication
callback 이후에도 staging hash와 파일 집합을 다시 확인한다.

## 7. 실패 처리

다음 조건에서는 fail closed하며 최종 출력 디렉터리를 만들지 않는다.

- 입력 SHA256 불일치
- Git worktree가 dirty
- manifest와 prediction 좌표 불일치
- prediction 또는 label 중복 좌표
- prediction 좌표의 label 누락
- label dataset 모집단이 manifest의 1,117개와 불일치
- 음수 candidate index 또는 정확한 `0/1`이 아닌 label
- count 보존식 불일치
- metric이 count에서 재계산한 값과 불일치
- staging 파일 누락, 추가, hash 변경 또는 root substitution
- 출력 디렉터리 선존재
- 실행 중 입력 파일 snapshot 변경

원본 prediction, manifest, frozen TEST labels, DB, Exp414 checkpoint와
Exp414 산출물은 수정하거나 삭제하지 않는다.

## 8. 코드 구조

새 패키지
`experiments.exp416_exp404_hard_test_confusion_audit`를 만든다.

- `config.py`: 고정 경로, SHA, population과 count 계약
- `contracts.py`: frozen prediction, label, confusion count와 metric 타입
- `inputs.py`: prediction 단계 검증과 별도 label audit loader
- `audit.py`: micro 및 dataset별 혼동행렬 계산
- `reporting.py`: JSON, CSV와 한국어 report 렌더링
- `publication.py`: exact-file atomic publication
- `tests/`: 입력, 계산, reporting와 publication 회귀 테스트
- `scratch/run_exp416_exp404_hard_test_confusion_audit.py`: one-shot CLI
- `docs/EXP416_PREREGISTRATION_KO.md`: 실행 전 고정 계약과 금지사항
- `docs/EXP416_RUNBOOK_KO.md`: 검증, 실행과 결과 확인

Exp403-408과 Exp414 패키지는 수정하지 않고 읽기 전용 consumer로만
다룬다.

## 9. 검증 전략

### Unit tests

- 손으로 계산 가능한 2×2 예제의 TP, FP, FN, TN과 metric
- zero-denominator metric의 `null`
- macro defined-dataset count
- bool label, 중복 좌표, 음수 index와 label 누락 거부
- manifest/prediction 좌표 불일치 거부
- prediction에 TEST/evaluation field가 있으면 거부
- canonical JSON/CSV 결정론

### Failure-path tests

- prediction, manifest와 label 각각의 hash mutation
- label loader가 prediction freeze 전에 호출되면 거부
- count 또는 metric tampering
- staging callback mutation, extra file과 symlink substitution
- 모든 실패 경로에서 최종 output 부재

### Real-input verification

- prediction SHA `437dc637...`
- manifest SHA `41a91341...`
- label SHA `b483ba00...`
- 2,218개 고유 Hard 좌표
- 888개 prediction 보유 dataset
- 195,900개 frozen label row
- 1,117개 manifest/label dataset
- source snapshot 전후 동일
- 게시된 세 payload hash와 manifest 일치

## 10. 완료 조건

다음 조건을 모두 만족해야 Exp416 audit 완료로 본다.

1. 사전등록 문서가 실제 label load 전에 커밋되어 있다.
2. Prediction identity가 label loader 호출 전에 동결된다.
3. Micro count 보존과 모든 metric 재계산이 일치한다.
4. Dataset별 1,117개 row와 macro defined count가 일치한다.
5. 네 산출물이 원자적으로 게시되고 manifest hash가 모두 맞는다.
6. Exp404 prediction과 모든 기존 산출물은 byte-identical하다.
7. TEST-informed reselection, Exp415 prediction과 운영 정책 변경이 없다.
