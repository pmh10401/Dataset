# Exp381-385 Coverage-First Feature Lattice 사전등록

## 목적

기존 자동 경보를 바로 교체하지 않고 점수원 적용 범위를 먼저 확인한 뒤,
새 점수원 단독 및 가능한 모든 조합과 파이프라인 배치를 비교한다.

## 고정 입력

- DB: `/Users/minho/Documents/Dataset/univariate_ts.db`
- 평가 모집단: Exp357에 동결된 1,117개 데이터셋
- 기존 근거: 추세 변화, 동역학 변화, 주파수 변화, 정상 거리
- 신규 근거: 주파수·차분, 정상 프로토타입 형상, 구간·분위수
- alpha: `0.005`, `0.01`, `0.02`, `0.05`
- TRAIN 정상 표본이 5개 미만이면 calibration 기권

## 실험

1. Exp381: 모집단, provenance, Exp162 캐시 무결성 감사
2. Exp382: 신규 세 점수원 단독 및 12개 전체 조합
3. Exp383: TRAIN 점수 Spearman 상관과 독립 근거군 감사
4. Exp384: 기존·신규 7점수원의 448개 전체 조합과 TRAIN risk
5. Exp385: 검토 추가, 위험 제한 확장, 신뢰도 배지, 연구용 교체 구조

## 금지 사항

- TEST 결과를 이용한 alpha, 규칙, risk target 또는 구조 선택
- TEST 길이 기반 top-N budget
- family 또는 dataset 이름에 따른 정책 분기
- exact-normal 후보의 재승격
- review 후보의 자동 경보 승격
- tail replacement

## 평가 경계

- Prediction을 먼저 라벨 없이 동결하고 SHA256을 기록한다.
- TEST 라벨은 동결 후 사후 평가에서만 읽는다.
- 자동 경보와 사람 검토 지표를 분리한다.
- 모든 결과는 retrospective counterfactual이며 prospective 검증이 아니다.

## 알려진 제한

- Exp357은 alpha 0.005 최종 후보 인덱스만 남아 있고 원점수 생성 계보가 없다.
- 따라서 기존 네 점수원을 다른 alpha로 재생성하지 않는다.
- 기존 네 점수 포함 규칙의 TRAIN pseudo-normal risk는 계산 불가로 기록한다.
- 검증된 TRAIN descriptor manifest가 없는 similarity pooling은 실행하지 않는다.
