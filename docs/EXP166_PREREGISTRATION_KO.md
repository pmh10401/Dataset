# Exp166 사전등록: 소표본 유사도 보정의 이상 보존 검증

## 목적

Exp165에서 TRAIN 정상 pseudo-run 오경보가 감소한 8개 고정 조합(4 alpha x 2 agreement)이 실제 TEST 이상 후보를 얼마나 보존하는지 후향적으로 확인한다.

## 범위

- 대상: `5 <= n_train < 20`인 316개 데이터셋
- 점수원: 주파수 변화 점수, 정상 패턴 거리 점수, 구간 통계 점수
- 보정: Exp164 로컬 교차적합 기준과 Exp165 유사도 풀 기준을 모두 고정 비교
- alpha: `0.005, 0.01, 0.02, 0.05`
- agreement: `2of3`, `3of3`
- 총 조합: 보정 방식 2 x alpha 4 x agreement 2 = 16

## 변경 금지

- TEST 성능으로 alpha, agreement, donor, threshold를 선택하지 않는다.
- TEST 길이 기반 top-N budget을 사용하지 않는다.
- family 또는 과거 TEST 성능으로 라우팅하지 않는다.
- tail replacement를 사용하지 않는다.
- Priority review나 Exp166 후보를 자동 경보로 승격하지 않는다.

## 예측과 평가 분리

1. TRAIN 정상 OOF 점수와 TRAIN 정상 donor만으로 기준 분포와 threshold를 만든다.
2. TEST 값으로 세 점수만 계산한다.
3. 라벨 없는 후보 JSONL을 생성하고 SHA-256으로 동결한다.
4. 동결 hash를 검증한 뒤 별도 단계에서 TEST instance label을 읽어 후향 평가한다.

Exp166의 후보는 `research_review_candidate`이며 autonomous Hard alert가 아니다.

## 점수 척도 연결

Exp164 OOF 점수의 median/MAD로 TRAIN OOF와 full-TRAIN fit TEST 점수를 동일하게 표준화한다. 이는 교차적합 기준과 외부 TEST 점수의 척도를 연결하기 위한 사전 고정 방식이며, 결과를 보고 수정하지 않는다.

## 해석 제한

- 모든 결과는 retrospective sensitivity 결과다.
- 가장 좋은 alpha 또는 agreement를 운영값으로 선택하지 않는다.
- Exp166은 end-to-end strict TRAIN-only 또는 prospective validation을 증명하지 않는다.
- `n_train < 5`인 162개 데이터셋은 이번 범위 밖이며 기존 abstention을 유지한다.
