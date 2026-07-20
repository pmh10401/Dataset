# Exp255 Change-point Score Source Design

## 목적

Exp255는 시계열 전체의 거리만 보는 기존 점수원을 보완하기 위해 한 파형 내부의
상태 전환을 요약하는 변화점 특징을 추가한다. 변화점 결과를 곧바로 자동 경보로
사용하지 않고, 독립 점수원으로 적격성을 평가한다.

## 고정 원칙

- TRAIN은 정상 데이터만 사용한다.
- TEST label과 실제 이상 위치는 prediction 완료 후 평가에서만 읽는다.
- TEST 길이, dataset/family 이름, 과거 TEST 성능으로 설정이나 라우팅을 바꾸지 않는다.
- 변화점 위치는 설명 자료로만 저장하며 정책 분기에 사용하지 않는다.
- Exp254 실행과 결과를 변경하지 않는다.
- 모든 결과는 retrospective score-source qualification으로 표시한다.

## 접근 방식

세 접근을 검토했다.

1. CUSUM/MOSUM만 사용: 가장 빠르고 해석이 쉽지만 복잡한 분포 변화가 빠진다.
2. PELT만 사용: 평균과 분포 변화를 다루지만 국소 변화의 다중 scale 정보가 약하다.
3. CUSUM/MOSUM과 PELT를 독립 block으로 만들고 결합: 계산량은 늘지만 각 방법의
   기여도를 분리할 수 있다.

Exp255는 3번을 사용한다.

## 데이터 흐름

1. DB에서 Exp236의 동결된 1,117개 데이터셋 목록을 읽는다.
2. 기존 `load_bundle()`로 정상 TRAIN과 TEST 파형을 읽고 길이를 정렬한다.
3. TRAIN 정상 파형만으로 point-wise median과 robust scale을 적합한다.
4. 동일한 TRAIN 기준으로 TRAIN과 TEST를 표준화한다.
5. 각 파형에서 다음 변화점 특징을 독립 계산한다.
6. TRAIN feature sanitizer를 적합하고 TEST에 적용한다.
7. 기존 five-fold TRAIN OOF KNN local-gap score를 만든다.
8. prediction JSONL을 동결한 뒤에만 TEST label로 사후 평가한다.

## 특징 block

### 1. Robust CUSUM / Multi-scale MOSUM

- 평균 변화 CUSUM
- 중앙값 기반 강건 변화
- 4개 고정 window scale의 좌우 평균 차이
- 좌우 중앙값 차이
- 좌우 기울기 차이
- 좌우 분산 log-ratio

Window scale은 정렬 길이의 고정 비율로 계산하며 TEST 결과로 변경하지 않는다.

### 2. PELT-L1 / PELT-L2

- 변화점 개수
- 가장 큰 인접 segment 평균 차이
- 가장 큰 인접 segment 중앙값 차이
- 가장 큰 인접 segment 분산 log-ratio
- 가장 짧은 segment 비율

L1과 L2는 같은 `min_size`, `jump`, 사전 고정 penalty를 사용한다.

### 3. PELT-RBF

비선형 분포 변화용 block이다. 동일한 요약을 만들되 별도 점수원으로 유지하여
비용 대비 추가 이득을 확인한다.

### 4. Combined

위 세 block을 연결한다. 독립 block과 combined를 모두 평가하여 특정 방법의 효과와
결합 효과를 분리한다.

## 설정 동결

- working length: 32 이상 256 이하
- MOSUM window fractions: `(1/32, 1/16, 1/8, 1/4)`; 실제 window는 최소 3
- PELT `min_size`: `max(4, working_length // 32)`
- PELT `jump`: `max(1, working_length // 64)`
- PELT penalty: L1/L2/RBF 모두 `3.0 * log(working_length)`
- seed: Exp255 전용 고정 seed
- cross-fit: 5 folds
- KNN: 3-neighbor local-gap
- alpha: `(0.005, 0.01, 0.02, 0.05)` 모두 보고
- detector `min_size`, `jump`, penalty multiplier는 구현 전에 config에 고정
- 결과를 본 뒤 설정을 변경하지 않는다.

## 출력 및 판단

각 block/alpha에 대해 전체 1,117개 coverage, score 유한성, 후보 수, Hard TP/FP,
precision, mean F1, zero-F1 수를 보고한다. Exp253 frozen policy와의 교집합·추가 후보는
진단하되 Exp255 결과만으로 자동 경보를 확대하지 않는다.

성공 신호는 다음과 같다.

- 기존 점수원이 놓친 데이터셋에서 새로운 TP 근거가 생긴다.
- FP가 특정 데이터셋에 과도하게 집중되지 않는다.
- TRAIN OOF score와 TEST score가 모두 유한하다.
- 변화점 block 간 정보 중복이 낮아 결합 가치가 있다.

## 오류 처리와 재현성

- 짧은 파형과 `n_train < 5`는 명시적으로 abstain한다.
- detector 실패를 빈 후보로 숨기지 않고 dataset error로 기록한다.
- 일부 누락 상태에서는 실험 성공으로 처리하지 않는다.
- config, package version, DB hash, input/output SHA256을 manifest에 기록한다.
- prediction에는 label과 평가 지표를 넣지 않는다.

## 실행 방식

Exp254와 동시에 무거운 계산을 시작하지 않는다. Exp255 구현과 smoke 검증만 진행한
뒤 순차 큐에 등록한다. 본 실행은 CPU 7 workers를 사용한다. PELT와 NumPy 기반
통계 연산은 MLX exact-parity 1.1x benchmark를 통과하지 않았으므로 MLX를 사용하지
않는다.
