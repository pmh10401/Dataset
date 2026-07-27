# Exp393-396 사전등록

## 목적

기존 점수 조합이 놓친 이상을 복구할 수 있도록 정상 운전의 상태 순서,
체류시간, 정상 복귀 실패를 측정하는 정상 운전 문법 이탈 점수(NPGS)를
검증한다.

이번 실험은 자동 경보를 확대하지 않는다. 신규 점수는 일반 검토 후보와
검토 순위에만 사용한다.

## 동결 설정

- alpha: `0.005, 0.01, 0.02, 0.05`
- state count: `8`
- seed: `20260727`
- MiniBatchKMeans: `n_init=10`, `max_iter=200`
- cross-fit: instance index modulo 5 기반 최대 5-fold
- high-support state: TRAIN window 점유율 5% 이상
- abnormal transition: TRAIN transition surprise 95th percentile 초과
- 합성 점수: 전환, 체류, 복귀 percentile의 동일 가중 평균
- `n_train < 5`: calibration abstention

## 단계

1. Exp393: TEST 라벨 없는 점수와 후보 생성
2. Exp394: TRAIN pseudo-normal calibration 감사
3. Exp395: prediction SHA256 동결 후 상보성 사후 평가
4. Exp396: 기존 Hard를 유지한 review-only 통합

## 금지 사항

- TEST 라벨과 이상 위치를 prediction에 사용하지 않는다.
- TEST 성능으로 alpha, state count, descriptor, 가중치를 선택하지 않는다.
- family 또는 dataset 이름으로 정책을 바꾸지 않는다.
- TEST 길이 기반 top-N budget을 사용하지 않는다.
- Priority review를 자동 경보로 승격하지 않는다.
- tail replacement를 사용하지 않는다.

## 해석 제한

모든 결과는 retrospective counterfactual이다. 실제 설비 prospective
validation이나 end-to-end strict TRAIN-only 개발 과정 전체를 증명하지
않는다. Exp396의 human-assisted 지표는 자동 경보 성능과 분리한다.
