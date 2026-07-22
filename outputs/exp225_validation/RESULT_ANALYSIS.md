# Exp225 결과 분석

## 목적

Exp219 고정 shape dictionary의 사전등록 후보를 자동 경보로 사용하지 않고,
TRAIN-only로 동결된 두 독립 점수원이 같은 index를 확인하는 경우에만 review
후보로 남겼다.

- 원 후보: Exp219 `fixed_dictionary_control_conservative_max__0.005`
- 확인 점수원: 시간 랜덤 커널, 정규 형태 통계
- 확인 alpha: Exp224가 TRAIN-normal risk로 선택한 `0.01`
- 후보 비교: exact index
- 자동 경보 승격: 없음
- Exp137 자동 경보: 변경 없음

## 재현성 계약

- Exp219 frozen prediction SHA256을 검증한 뒤 후보를 읽었다.
- Exp224 raw replay는 동일 DB SHA256에서 639/639, mismatch 0, gate passed였다.
- Exp153 temporal score cache와 Exp180 canonical score cache는 manifest SHA256 및
  dataset별 cache SHA256을 검증했다.
- prediction 파일은 TEST label과 평가 지표를 포함하지 않는다.
- prediction SHA256을 동결한 뒤 별도 evaluation 단계에서만 TEST label을 읽었다.

## 결과

| 지표 | Exp219 원 후보 | Exp225 독립 확인 후 |
|---|---:|---:|
| 후보 수 | 257 | 38 |
| TP | 39 | 22 |
| FP | 218 | 16 |
| Precision | 15.18% | 57.89% |
| Zero-F1 데이터셋 | 232 | 239 |

세부 lane:

- 두 점수원 모두 확인: 4건, TP 3, FP 1, precision 75.0%
- 한 점수원만 확인: 34건, TP 19, FP 15, precision 55.88%
- 전체 확인 review: 38건, TP 22, FP 16, precision 57.89%

독립 확인은 Exp219 대비 TP의 56.41%를 보존하면서 FP의 92.66%를 제거했다.
전체 표시 후보는 85.21% 감소했고 precision은 42.72%p 증가했다.

## 해석

고정 dictionary 점수는 단독 후보 생성기로 쓰기에는 정상과 이상의 겹침이 너무
크다. 그러나 시간 랜덤 커널 또는 정규 형태 통계가 같은 위치를 확인하면 후보의
품질이 크게 좋아진다. 따라서 이 점수원은 자동 경보원이 아니라 기존 점수원의
근거를 보강하는 review source로 사용하는 것이 타당하다.

두 점수원이 모두 확인한 후보는 precision 75%였지만 4건뿐이다. 표본이 너무 작아
운영 승격 근거로 사용할 수 없다. 전체 확인 review도 기존 251개 TEST를 다시 본
retrospective 결과이므로 prospective 성능을 증명하지 않는다.

탐지 범위는 줄었다. Zero-F1 데이터셋이 232개에서 239개로 증가했고 mean review
F1도 낮다. 이는 Exp225의 목표가 전체 탐지기 교체가 아니라 Exp219 오탐 억제였기
때문이다.

## 다음 실험

다음 단계는 Exp226 TRAIN-normal conditional risk audit이다. TEST 결과를 사용하지
않고 정상 TRAIN pseudo-run에서 다음 전체 결합 규칙의 false-alert 발생량을 계산한다.

1. 고정 dictionary 후보 생성
2. 시간 랜덤 커널 또는 정규 형태 통계의 동일 index 확인
3. 두 점수원 모두 확인과 한 점수원 확인을 별도로 집계

Exp226에서 정상 false-alert 위험과 workload가 사전 정의한 허용 범위에 들어올 때만
이 규칙을 신규 데이터 shadow review로 넘긴다. Exp225 결과만으로 Hard alert 또는
Priority review에 승격하지 않는다.
