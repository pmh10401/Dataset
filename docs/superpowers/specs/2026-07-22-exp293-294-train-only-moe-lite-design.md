# Exp293-294 TRAIN-only MoE-lite Design

## 목적

Exp291 자동 경보를 변경하지 않고, Exp292 일반 검토 후보를 TRAIN 정상
신뢰도에 따라 다시 정렬한다. 전체 1,117개 결과와 `n_test >= 100` 후향
진단을 함께 보고하지만 TEST 크기는 Gate 입력으로 사용하지 않는다.

## Expert

- 통계 특징 편차 (Exp252)
- 구조 패턴 편차 (Exp254)
- 변화점 편차 (Exp255)
- 공식 MultiROCKET 편차 (Exp250)
- 마스킹 복원 오차 (Exp257, provenance 미완료로 연구 민감도 전용)

## Exp293 TRAIN-only 감사

각 데이터셋과 expert에 대해 cache의 TRAIN OOF score만 읽는다. TEST score,
TEST label, family 이름과 과거 성능은 읽지 않는다.

- OOF 설정 수와 유한값 비율
- 설정 간 median absolute Spearman
- leave-one-out conformal family-majority 정상 후보 수
- one-sided 95% Wilson upper bound
- attainable p-value
- TRAIN 전반/후반 robust median drift
- producer manifest 및 config provenance

`n_train < 5`, cache 누락, 비유한 score는 abstention이다.

## Exp294 Gate variants

네 개의 provenance 검증 expert를 데이터셋별로 순위화해 상위 2개를 선택한다.
동률은 expert 이름으로 결정한다.

1. `risk_first_top2`: Wilson upper가 낮고 stability가 높은 순서
2. `stability_first_top2`: stability가 높고 Wilson upper가 낮은 순서
3. `balanced_rank_top2`: risk, stability, resolution의 동등 평균 순위

세 variant는 sensitivity로 모두 평가하며 TEST 성능으로 하나를 선택하지
않는다. 마스킹 복원을 포함한 결과는 research-only sensitivity로 분리한다.

## 후보 정렬

Exp292 Standard review 806건의 집합은 바꾸지 않는다. 각 후보는 다음 순서로
정렬한다.

1. Gate가 선택한 expert의 지지 수
2. 전체 provenance 검증 expert의 지지 수
3. 지지 expert의 Gate 순위
4. dataset과 candidate index

비교 workload는 Exp292의 label-free 등급 크기와 같은 `17`, `74`, `806`을
사전에 고정한다. 이 값은 운영 K 선택이 아니라 ranking parity 진단이다.

## 금지 사항

- 자동 경보 승격 금지
- TEST label, 실제 이상 위치, TEST 값, TEST 길이 Gate 입력 금지
- family/dataset 이름 정책 분기 금지
- 결과를 본 뒤 Gate, top-2, workload 또는 expert 변경 금지
- tail replacement 금지

Prediction을 먼저 SHA256으로 동결하고 evaluation에서만 TEST label을 읽는다.
결과는 retrospective human-assisted diagnostic이며 prospective 검증이 아니다.

