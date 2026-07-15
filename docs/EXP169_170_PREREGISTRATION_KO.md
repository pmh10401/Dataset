# Exp169/170 OOB pooled calibration 사전등록

## 목적

TRAIN 정상 샘플이 5개 이상 20개 미만인 316개 데이터셋에서 기존 local conformal reference의 해상도 부족을 완화할 수 있는지 검증한다. Exp169는 TEST 라벨을 사용하지 않는 정상 위험 audit이고, Exp170은 동결된 prediction을 사후 평가하는 연구용 review sensitivity다.

이 문서는 실행 전에 커밋한 설계 문서 `docs/superpowers/specs/2026-07-15-exp169-170-oob-pooled-calibration-design.md`와 구현 계획을 요약한다. 설계 커밋은 `bc4e3bf2`, 구현 계획 커밋은 `07b4dafa`다.

## 고정 대상과 설정

- 대상: `5 <= n_train < 20`인 316개 데이터셋
- score source: 주파수 특징, 형태 프로토타입 특징, 구간 통계 특징
- OOB 모델: 32개
- OOB holdout fraction: 0.35
- seed: 20260715
- alpha: 0.005, 0.01, 0.02, 0.05
- donor minimum reference: 200
- donor별 최대 기여: 20
- pool: similarity, coarse strata, global balanced
- source configuration: 세 pair, 2-of-3, 3-of-3
- 데이터셋당 고정 결과 수: 60

## 정보 사용 계약

- feature와 score는 TRAIN 정상 데이터로 구성한다.
- donor descriptor와 donor 선택에는 TRAIN count, TRAIN target length, TRAIN OOB score summary만 사용한다.
- target 데이터셋은 자신의 reference pool에서 제외한다.
- TEST 값은 frozen prediction 생성에만 사용한다.
- TEST 라벨과 실제 이상 위치는 prediction 동결 이후 평가에서만 읽는다.
- family 이름, 과거 family TEST 성능, TEST 길이 budget, top-N, tail 위치 규칙을 사용하지 않는다.

## 해석 계약

- 모든 60개 variant를 sensitivity로 보고한다.
- TEST 성능을 보고 alpha, pool, source configuration을 선택하지 않는다.
- 후보는 `research_review_candidate`이며 autonomous Hard alert가 아니다.
- Exp170 지표는 retrospective human-review 지표다.
- 실제 설비 prospective validation 또는 배포 적합성을 주장하지 않는다.
