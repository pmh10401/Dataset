# Exp389-390 이중 신뢰 등급 사전등록

## 목적

기존 자동 경보를 변경하지 않고, 동결된 기존 근거를 이용해 자동 경보의
신뢰도를 두 단계로 표시한다.

## 동결 규칙

- 최고 신뢰: alpha `0.005` strong Hard
- 확장 신뢰: alpha `0.010` strong Hard에서 최고 신뢰를 제외한 집합
- 일반 신뢰: 기존 Hard에서 위 두 집합을 제외한 집합
- 기존 Hard, Standard review, Priority review, exact-normal lane은 변경하지 않는다.
- Exp389 prediction은 TEST 라벨과 평가 지표를 읽거나 저장하지 않는다.
- Exp390은 Exp389 prediction SHA256 검증 후에만 TEST 라벨을 읽는다.
- TEST 결과로 alpha나 등급 규칙을 변경하지 않는다.
- 모든 결과는 retrospective counterfactual이며 prospective 검증이 아니다.

## 성공 조건

- 1,117개 데이터셋 누락 및 중복 0
- alpha `0.005` strong 집합이 alpha `0.010` strong 집합의 부분집합
- 세 confidence tier가 서로 겹치지 않고 합집합이 기존 Hard와 동일
- 기존 Hard 지표가 Exp358 baseline과 정확히 동일
- prediction label/metric field 0
