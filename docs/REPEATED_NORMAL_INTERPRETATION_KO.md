# 반복 정상(repeated-normal) 해석 결정 기록

결정일: 2026-07-28

## 결정

현재 벤치마크의 정상 TRAIN/TEST 정확 중복은 단순한 데이터 결함이 아니라,
같은 equipment/recipe/sensor/step 조합의 안정된 정상 상태가 반복되는
운전 조건을 모사한 의도적 repeated-normal 구조로 해석한다.

반도체 설비에서는 같은 recipe를 반복 실행했을 때 정상 trace가 같거나 매우
비슷하게 재현되는 것이 바람직하다. 탐지기가 이런 정상 반복에 계속 경보를
내면 사용자는 경보를 신뢰하지 않게 된다. 따라서 반복 정상을 안정적으로
자동 정상 처리하여 false alarm을 억제하는 능력은 이 프로젝트의 핵심
목표와 직접 연결된다.

## 기존 근거

- 과거 작업 기록은 TEST 정상에 TRAIN 정상과 같거나 매우 유사한 데이터가
  들어 있는 것을 실제 설비·recipe의 반복 정상 패턴을 흉내 낸 구조로
  명시한다.
- Exp164는 dataset 하나를 가상 equipment/recipe/sensor/step 환경으로,
  정상 TRAIN instance 하나를 정상 virtual wafer run trace로 해석했다.
- Exp290은 정상 TRAIN과 byte-for-byte 같은 기존 Hard 후보를
  `train_raw_exact_normal`로 분리했다.
- Exp290의 사후 평가에서 분리된 35개는 모두 FP였고 TP 손실은 0이었다.
- Exp291은 이를 `automatic normal exact` lane으로 현재 routing에 고정했다.
- Exp410은 동결 Hard가 family별로 고르지 않다는 사실과 별개로, 정상
  반복이 높은 구간에서 상대적으로 강하다는 관찰을 제공했다.

## 허용되는 해석

- original repeated-normal은 현재 프로젝트의 주 운영 시나리오다.
- 같은 정상 상태가 반복될 때 낮은 false alarm은 긍정적인 성능이다.
- exact-normal 자동 정상 guard는 label-blind prediction 규칙이다.
- 반복성이 다른 구간의 결과는 repeatability-aware reporting으로 분리한다.
- `clean-balanced`는 반복성이 약한 조건을 보는 보조 스트레스·하한
  검증이다.

## 허용되지 않는 해석

- UCR의 같은 waveform이 실제 동일 wafer run이라는 뜻이 아니다.
- 현재 결과가 새 설비나 새 recipe의 prospective 성능을 증명하지 않는다.
- 높은 반복 구간과 낮은 반복 구간의 차이를 overlap 하나의 인과 효과라고
  단정하지 않는다.
- TEST 결과를 보고 exact 기준을 near 기준으로 완화하지 않는다.
- 과거 family별 TEST 성능으로 새로운 guard, source 또는 threshold를
  고르지 않는다.

## Exp410과의 관계

Exp410의 `severe_family_imbalance`는 family마다 성능이 고르지 않고 큰
family가 dataset 평균에 더 많은 영향을 준다는 뜻이다. 이는 반복 정상의
운영상 가치를 부정하지 않는다.

따라서 두 결론을 함께 유지한다.

1. 반복 정상 조건에서 강한 성능은 반도체의 안정 운전 목표와 맞는다.
2. 모든 family와 덜 반복되는 조건에 같은 성능을 일반화해서는 안 된다.

## 후속 실험

Exp411은 exact-normal guard를 바로 확장하지 않는다. 정상 TRAIN과
TRAIN-only 합성 교란만으로 near-repeat 규칙을 먼저 자격 심사한다.
Exp411에서 안전한 규칙이 없으면 `no_qualified_near_rule`로 종료한다.

Exp412는 Exp411을 통과한 단 하나의 규칙만 label-blind Shadow로 적용한다.
Exp413은 그 결과가 특정 family에 치우친 우연인지 감사한다. 세 실험이 모두
긍정적이어도 자동 정상 lane 변경은 별도 Exp414 사전등록과 사용자 승인
없이는 수행하지 않는다.
