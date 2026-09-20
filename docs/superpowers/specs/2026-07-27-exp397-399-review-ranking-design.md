# Exp397-399 Review Ranking Design

## 1. 목적

Exp292가 만든 기존 일반 검토 후보 806건을 유지하면서, 사용자가 실제 이상을 더 빨리 확인할 수 있도록 검토 순서를 재정렬한다. Exp393의 Normal Process Grammar Score(NPGS)는 새로운 후보 생성이나 자동 경보 확대에 사용하지 않고, 기존 검토 후보의 우선순위 근거로만 사용한다.

이 실험은 retrospective ranking audit이다. TEST 라벨은 순위 artifact를 동결하고 SHA256을 검증한 뒤 평가 단계에서만 읽는다.

## 2. 고정 입력

- Exp292 frozen review tiers:
  `/Users/minho/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl`
- Exp393 frozen grammar prediction:
  `/Users/minho/Documents/Dataset/exp393_396_normal_process_grammar/exp393_predictions.jsonl`
- Exp393 prediction manifest:
  `/Users/minho/Documents/Dataset/exp393_396_normal_process_grammar/exp393_prediction_manifest.json`
- 평가 모집단: Exp389/Exp393 계약과 동일한 1,117개 데이터셋
- 검토 후보 모집단: Exp292 `standard_review_indices` 806건
- 자동 경보: 2,218건을 그대로 유지
- 기존 우선 검토 lane: 변경하지 않음

입력 hash가 바뀌거나 1,117개 coverage, 806개 일반 검토 후보, lane 불변성이 재현되지 않으면 실행을 중단한다.

## 3. 점수와 근거

Exp292는 기존 독립 점수원 지지 수에 따라 일반 검토 후보를 다음 세 단계로 나눈다.

- `review_priority_1`: 기존 점수원 2개 이상 지지
- `review_priority_2`: 기존 점수원 1개 지지
- `review_priority_3`: 기존 추가 지지 없음

Exp393은 각 후보에 대해 다음 TRAIN-normal conformal p-value를 제공한다.

- 상태 전이 surprise
- 체류 시간 anomaly
- 정상 복귀 failure
- 위 세 점수의 composite

문법 근거는 내부 구성요소가 서로 독립이라고 간주하지 않는다. 상태 전이, 체류 시간, 정상 복귀가 여러 개 동의해도 운영상 하나의 신규 `process_grammar` 근거군으로만 계산한다.

## 4. 사전 등록 정렬 방식

### 4.1 `exp292_baseline`

Exp292 등급을 `priority_1`, `priority_2`, `priority_3` 순서로 유지하고, 같은 등급에서는 dataset 이름과 candidate index로 결정적 정렬한다. 새 방식의 비교 기준이다.

### 4.2 `tier_preserving_grammar` - 권장 구조

Exp292 등급 순서를 절대 변경하지 않는다. 같은 등급 안에서만 다음 순서로 정렬한다.

1. 해당 alpha에서 문법 근거 존재 여부
2. primitive 문법 근거 수
3. composite conformal p-value 오름차순
4. primitive 최소 p-value 오름차순
5. dataset 이름, candidate index

기존 검증된 등급을 보존하므로 가장 보수적인 구조다.

### 4.3 `additive_grammar_family` - 연구 대조군

문법 근거가 있는 후보에 독립 근거군 하나를 추가한다. 이 근거로 후보가 최대 한 단계 위로 이동할 수 있다.

- priority 3 + grammar -> effective priority 2
- priority 2 + grammar -> effective priority 1
- priority 1은 그대로 유지

문법 내부 구성요소 수로 두 단계 이상 승격하지 않는다. 자동 경보 또는 기존 우선 검토 lane으로 승격하지 않는다.

### 4.4 `grammar_only_control`

Exp292 등급을 사용하지 않고 문법 근거와 p-value만으로 정렬한다. 문법 점수 자체의 정보량을 확인하기 위한 연구 대조군이며 운영 후보가 아니다.

## 5. Alpha와 workload

다음 alpha를 모두 독립적으로 보고한다.

`0.005, 0.01, 0.02, 0.05`

TEST 결과로 alpha를 선택하지 않는다. `exp292_baseline`은 alpha와 무관하지만 비교표의 형태를 맞추기 위해 각 alpha에서 같은 순위를 반복 기록할 수 있다.

고정 global review workload:

`K = 25, 50, 100, 200`

추가로 각 데이터셋에서 첫 일반 검토 후보 하나만 보는 `per_dataset_top1`을 별도 workload 진단으로 제공한다. 이 값도 운영 정책으로 선택하지 않는다.

## 6. 실험 단계

### Exp397: Frozen review ranking prediction

- Exp292와 Exp393 artifact hash를 검증한다.
- TEST 라벨을 읽지 않는다.
- 네 정렬 방식과 네 alpha의 전체 806건 순서를 생성한다.
- 순위 JSONL과 manifest를 원자적으로 저장한다.
- 모든 variant에서 후보 집합과 lane이 Exp292와 동일한지 검증한다.

### Exp398: Locked ranking evaluation

- Exp397 prediction SHA256을 먼저 검증한다.
- 이후에만 TEST 라벨을 읽는다.
- 각 variant, alpha, workload에서 다음을 계산한다.
  - Precision@K
  - TP@K, FP@K
  - 전체 review TP 중 K 안에서 확인한 비율
  - Exp292 baseline 대비 precision lift
  - 기존 priority 1/2/3 구성
- 데이터셋 단위 결과와 후보 단위 결과를 모두 저장한다.

### Exp399: Operational workload audit

- 자동 경보, 기존 우선 검토, 일반 검토 후보 집합이 모두 불변인지 검증한다.
- global K와 per-dataset top-1의 workload 차이를 보고한다.
- 결과는 `검증 완료`, `조건부 사용`, `사용 금지` 문구로 구분한다.
- 어떤 variant나 alpha도 TEST 결과로 primary, optimal, recommended로 선택하지 않는다.

## 7. 평가 기준

이번에 재정렬하는 일반 검토 806건의 기준 성능은 TP 365, FP 441, precision 약 45.3%다. 기존 우선 검토 31건까지 합친 전체 human-assisted review는 TP 391, FP 446, precision 약 46.7%이지만, 우선 검토 lane은 이번 순위 모집단에 포함하지 않는다. 두 값은 분리해서 보고하며 정렬 결과로 어느 후보 집합도 바뀌지 않는다.

정렬의 성공 여부는 다음으로 판단한다.

- 앞쪽 K에서 baseline보다 TP가 더 많이 나타나는가
- Precision@K가 전체 검토 precision보다 높은가
- 성능 개선이 특정 데이터셋 몇 개에만 집중되는가
- 기존 priority 1/2의 성능을 훼손하지 않는가
- per-dataset top-1 workload가 과도하지 않은가

결과가 나쁘더라도 alpha, tie-break, K 또는 승격 규칙을 수정하지 않는다. 개선은 새 실험 번호와 새 사전등록으로만 제안한다.

## 8. 불변성과 오류 처리

다음 조건은 테스트와 런타임 assert로 고정한다.

- prediction 함수는 `y_test`를 받지 않는다.
- prediction artifact는 TP, FP, precision, F1을 포함하지 않는다.
- dataset/family 이름은 결정적 tie-break와 조회 key로만 사용한다.
- dataset/family 이름으로 정책을 분기하지 않는다.
- TEST 길이 기반 top-N budget을 사용하지 않는다.
- tail replacement를 사용하지 않는다.
- 806개 후보의 중복과 누락이 없다.
- 후보 index가 각 데이터셋의 `n_test` 범위 안에 있다.
- 네 variant 모두 후보 집합이 동일하다.
- 자동 경보와 기존 우선 검토 lane이 바뀌지 않는다.
- 입력 순서를 바꿔도 정렬 결과 hash가 동일하다.
- 같은 입력과 config에서 재실행 hash가 동일하다.

## 9. 산출물

출력 경로:

`/Users/minho/Documents/Dataset/exp397_399_review_ranking`

필수 파일:

- `exp397_frozen_review_ranking.jsonl`
- `exp397_prediction_manifest.json`
- `exp398_precision_at_k.csv`
- `exp398_candidate_evaluation.csv`
- `exp398_dataset_diagnostics.csv`
- `exp398_evaluation_manifest.json`
- `exp398_report.md`
- `exp399_workload_summary.csv`
- `exp399_claims.md`
- `independent_consistency_report.json`
- `status.json`
- `exp397_399_stdout.log`

## 10. 해석 제한

- 전체 결과는 retrospective다.
- 실제 설비 prospective validation이 아니다.
- 자동 경보 성능 개선 실험이 아니다.
- human-assisted 검토 순서 진단이다.
- TEST 결과로 운영 variant, alpha 또는 workload를 선택할 수 없다.
- 배포 전에는 신규 잠금 데이터 또는 shadow mode에서 순위 품질을 검증해야 한다.
