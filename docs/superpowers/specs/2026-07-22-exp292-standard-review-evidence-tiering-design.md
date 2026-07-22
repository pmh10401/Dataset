# Exp292 Standard Review Evidence Tiering Design

## 목적

Exp291의 자동 경보와 검토 후보 집합을 변경하지 않고, 일반 검토 806건을 독립적인 동결 점수원 근거 수로 정렬한다. 사용자가 더 강한 근거가 있는 후보를 먼저 확인하도록 돕는 review-only 실험이다.

## 고정 입력

- Exp291 최종 routing의 `standard_review_indices`
- Exp252 통계 특징 후보
- Exp254 구조 특징 후보
- Exp255 변화점 특징 후보
- Exp257 마스킹 복원 후보
- Exp250 공식 MultiROCKET 후보
- alpha는 모든 점수원에서 `0.005`로 고정
- 각 입력 파일의 SHA256을 prediction 전에 검증

MiniROCKET은 Exp253에서 일반 검토 후보 승격에 이미 사용됐으므로 중복 근거로 다시 넣지 않는다. Exp272 support map은 기존 lane 밖 후보만 포함하므로 일반 검토 등급화에 사용하지 않는다.

## 계열 내부 안정성

같은 물리적 점수원 계열의 여러 설정을 독립 근거로 중복 계산하지 않는다. 데이터셋별로 alpha 0.005에서 사용 가능한 설정 중 절반 이상이 같은 index를 지목할 때 그 계열이 해당 index를 지지한다고 본다.

```text
required_votes = ceil(available_variants / 2)
```

Exp257은 사전 고정된 large-data subset 251개에만 존재한다. source가 없는 데이터셋은 빈 후보가 아니라 unavailable로 기록하며, 다른 네 계열만으로 등급을 계산한다.

## 일반 검토 등급

- `review_priority_1`: 서로 다른 점수원 계열 2개 이상이 재확인
- `review_priority_2`: 정확히 1개 계열이 재확인
- `review_priority_3`: 추가 재확인 없음

세 등급은 서로 겹치지 않고 합집합이 Exp291 일반 검토 806건과 정확히 같아야 한다. label-free 사전 감사에서 예상한 수는 각각 17, 57, 732건이다.

## 금지 사항

- TEST label, 실제 이상 위치, family별 TEST 성능 사용 금지
- 결과를 보고 alpha, 계열, 과반 규칙 또는 등급 경계를 변경하지 않음
- 자동 경보, 자동 정상, Priority/Shadow/Recovery Shadow lane 변경 금지
- 등급을 자동 경보로 승격하지 않음
- tail 위치나 TEST 길이로 우선순위를 변경하지 않음

## Prediction과 Evaluation 분리

Prediction은 frozen candidate index와 hash만 읽고 TEST label이나 성능 지표를 포함하지 않는다. prediction JSONL과 SHA256을 동결한 뒤 별도 evaluation 단계에서만 TEST label을 읽는다.

## 평가

각 등급에 대해 후보 수, TP, FP, precision, 포함 데이터셋 수를 보고한다. 누적 P1, P1+P2, 전체 검토 workload와 이상 발견량도 함께 보고한다. 모든 지표는 retrospective human-assisted 진단이며 autonomous Hard 성능과 합치지 않는다.

## 해석 제한

Exp292는 반복 관찰된 기존 TEST에서 수행하는 retrospective review-ordering 평가다. 신규 설비 prospective validation이나 end-to-end strict TRAIN-only 증명이 아니다.
