# Exp392 정답 기반 점수원 역 탐색 사전등록

## 목적

Exp392는 기존 자동 경보가 놓친 이상과 오탐 결과를 이용해 유망한
점수원 및 점수원 조합의 연구 가설을 찾는 후향 감사다. 자동 경보
정책을 최적화하거나 변경하는 실험이 아니다.

## 동결 입력

- Exp389 자동 경보·검토 lane
- Exp386~388의 Exp357 동결 근거
- Alpha: `0.005, 0.010, 0.020, 0.050, 0.100, 0.200`
- 근거: delta, 1차·2차 delta, EWMA, frequency, normal distance,
  structural, temporal, topology
- Exp381 TRAIN coverage
- TEST 라벨은 prepare 산출물 SHA256 검증 후 evaluate에서만 사용

## 분석 범위

- 9개 점수원의 단독 결과
- 모든 2개·3개 점수원 조합
- 교집합, 합집합, 3개 중 2개 이상 동의
- 후보·TP·FP 상보성과 Jaccard
- 데이터 크기, TRAIN 수, Zero-F1 유형별 안정성
- 데이터셋 계보 집중도
- 데이터셋 부트스트랩 2,000회
- 단일 점수원 Fisher exact test 및 Benjamini-Hochberg 보정

## 연구 분류

- `unsupported`: FN 회수 0 또는 후보 정밀도 30% 미만
- `research_signal`: 나머지 후향 신호
- `preregistration_candidate`: FN 10건 이상, 정밀도 50% 이상,
  TP 기여 계보 5개 이상, 최상위 계보 TP 비중 80% 미만,
  부트스트랩 정밀도 하한 35% 이상

이 분류는 별도 사전등록 필요성을 표시할 뿐 운영 정책을 선택하지 않는다.

## 금지 사항

- TEST 결과로 Alpha, 점수원, 조합, threshold 또는 routing을 선택하지 않는다.
- family나 dataset 이름으로 후보를 라우팅하지 않는다.
- 기존 Hard, Standard review, Priority review, exact-normal lane을 변경하지 않는다.
- Priority 또는 연구 신호를 자동 경보로 승격하지 않는다.
- Exp392 결과를 prospective 또는 실제 설비 검증으로 표현하지 않는다.

## 결과 해석

기존 1,117개 TEST 데이터는 이미 반복 관찰됐다. Exp392는
`label_guided_discovery=true`, `eligible_for_deployment=false`인 가설
발견 실험이다. 후속 적용은 새 실험 번호, 동결 규칙, Shadow/review
검증 및 신규 잠금 데이터가 필요하다.
