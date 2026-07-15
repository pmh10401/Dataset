# Exp167/168 사전등록: 점수-보정 조건 일치 검증

## 목적

Exp166에서 확인된 교차적합 TRAIN 정상 점수와 full-TRAIN TEST 점수의 생성 조건
차이를 제거한다. 동일 fold 모델이 제외된 TRAIN 정상 trace와 모든 TEST trace를
함께 점수화하고, fold가 서로 대응하는 empirical p-value를 계산한다.

이 실험은 `5 <= n_train < 20`인 316개 데이터셋의 소표본 문제를 검증하는
후향 연구 실험이다. 자동 경보 정책을 선택하거나 실제 설비 성능을 주장하지 않는다.

## 동결 계약

- 데이터베이스: `/Users/minho/Documents/Dataset/univariate_ts.db`
- 대상: Exp165/166과 동일한 316개 데이터셋
- fold: leave-one-out, seed `20260715`
- alpha: `0.005`, `0.01`, `0.02`, `0.05`
- 점수원: 주파수 변화, 정상 패턴 거리, 구간 통계
- 조합: 세 점수원 쌍, `2-of-3`, `3-of-3`
- p-value: `(1 + paired_oof_ge_test) / (n_train + 1)`
- 후보 lane: 연구용 검토 후보만 사용
- 데이터셋당 출력: 4 alpha x 5 조합 = 20행

## 금지 사항

- TEST 라벨 또는 실제 이상 위치를 prediction에 사용하지 않는다.
- TEST 길이 기반 top-N budget을 사용하지 않는다.
- family 또는 dataset 이름으로 정책을 분기하지 않는다.
- 과거 family별 TEST 성능을 사용하지 않는다.
- tail replacement를 사용하지 않는다.
- 결과를 보고 alpha나 점수원 조합을 선택하지 않는다.
- 연구 후보를 자동 경보로 승격하지 않는다.

## 평가 순서

1. Exp167에서 라벨 없이 fold score와 p-value 해상도를 감사한다.
2. Exp168 prediction 6,320행을 생성하고 SHA256을 동결한다.
3. prediction hash를 확인한 뒤에만 TEST 라벨을 읽는다.
4. 모든 alpha와 조합을 민감도 결과로 함께 보고한다.

## 해석 경계

Exp168의 TP, FP, precision, recall, F1은 사용자가 연구 후보를 검토했다고 가정한
후향 지표다. autonomous Hard alert 지표가 아니며, 특정 alpha 또는 조합의 운영
채택 근거로 사용하지 않는다.
