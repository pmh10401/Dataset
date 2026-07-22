# Exp291 Final Operational Routing Freeze Design

## 목적

Exp290의 TRAIN raw exact 자동 정상 guard를 최종 자동 경보 출력 단계에 고정하고, 현재 승인된 review-only lane을 함께 보존한 하나의 운영 routing artifact를 만든다.

## 고정 입력

- 자동 경보 및 자동 정상: Exp290 frozen prediction
- 기존 검토 lane: Exp272 frozen prediction
- Exp290 prediction SHA256과 Exp272 prediction SHA256을 실행 전에 검증
- 데이터셋 coverage: 1,117

## 최종 lane

- `automatic_hard_indices`: Exp290 `guarded_hard_indices`
- `automatic_normal_exact_indices`: Exp290 `train_raw_exact_normal_indices`
- `standard_review_indices`: Exp272 `standard_indices`
- `priority_review_indices`: Exp272 `priority_indices`
- `shadow_review_indices`: Exp272 `shadow_indices`
- `recovery_shadow_review_indices`: Exp272 `recovery_shadow_indices`
- `research_only_recovery_priority_indices`: Exp272 `recovery_priority_indices`

Recovery Priority는 이전 평가에서 기본 review selector로 기각됐으므로 운영 UI 기본 검토 건수에 포함하지 않고 연구 전용으로만 보존한다.

## 최종 자동 정상 guard

모든 자동 경보 후보 조합이 끝난 뒤 UI 전달 직전에 다음 규칙을 적용한다.

```text
automatic_hard = proposed_hard - train_raw_exact_normal
```

`train_raw_exact_normal`은 Exp290에서 고정한 SQLite `values_blob` byte equality만 사용한다. 유사도, 보간, 정규화, family, TEST label은 사용하지 않는다.

## 불변 조건

- 기존 Hard = automatic Hard와 automatic normal exact의 서로소 합집합
- automatic Hard는 모든 review lane과 겹치지 않음
- automatic normal exact는 모든 review lane과 겹치지 않음
- 기본 review lane은 standard, priority, shadow, recovery shadow만 포함
- research-only Recovery Priority는 자동 경보 또는 기본 review로 승격하지 않음
- prediction artifact는 TEST label과 평가 metric을 포함하지 않음

## 평가

Prediction SHA256을 동결한 후에만 TEST label을 읽는다. Autonomous Hard 성능은 Exp290과 정확히 같아야 한다. Review 지표는 human-assisted로 분리하여 기록하며 autonomous metric과 합치지 않는다.

## 성공 조건

- coverage 1,117/1,117
- automatic Hard 2,218
- automatic normal exact 35
- automatic normal exact에서 TP 0, FP 35
- automatic Hard TP 1,915, FP 303
- lane overlap 0
- Recovery Priority 137건은 research-only 유지
- 오류 0

## 제한

Exp291은 기존 retrospective 결과를 운영 형태로 재동결하는 integration artifact다. 신규 데이터 prospective 검증이나 upstream end-to-end strict TRAIN-only 증명이 아니다.
