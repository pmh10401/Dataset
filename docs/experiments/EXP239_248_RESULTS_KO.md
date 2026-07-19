# Exp239-248 추가 점수원 검증 결과

## 1. 실험 목적

Exp239-248은 Exp236의 자동 경보, 일반 검토, 우선 검토 후보를 변경하지
않고 열 개의 새로운 TRAIN-normal 점수원이 다음 용도로 가치가 있는지
검증한 후향 연구 실험이다.

1. 기존 자동 경보의 신뢰도 표시
2. 기존 일반 검토 후보의 우선순위 정렬
3. 기존 탐지 범위 밖의 이상 후보 발견

TEST 라벨은 prediction 파일을 동결한 뒤 평가에만 사용했다. 이번 결과로
source, variant, alpha 또는 운영 정책을 선택하지 않는다.

## 2. 기준 성능

Exp236의 전체 1,117개 데이터셋에는 실제 이상 3,918개가 있다.

| 구분 | 후보 | TP | FP | Precision |
|---|---:|---:|---:|---:|
| Exp236 자동 경보 | 2,116 | 1,789 | 327 | 84.55% |
| Exp236 일반 검토 | 967 | 511 | 456 | 52.84% |

추가 점수원의 자동 경보 확인 gate는 100건 이상을 확인하면서 Precision
89.55% 이상을 달성하는 것이다. 일반 검토 gate는 50건 이상, Precision
70% 이상이다. 탐지 범위 확대 gate는 기존 lane 밖에서 TP 25건 이상,
Precision 80% 이상이다.

## 3. 핵심 결과

### 자동 경보 신뢰도

| 점수원 | Variant / alpha | 확인 경보 | TP | FP | Precision | Exp236 대비 |
|---|---|---:|---:|---:|---:|---:|
| Masked Patch Context Error | seed intersection / 0.005 | 482 | 454 | 28 | 94.19% | +9.64%p |
| Ordinal Pattern Distribution Shift | default / 0.005 | 334 | 312 | 22 | 93.41% | +8.87%p |
| Masked Patch Context Error | default / 0.005 | 611 | 557 | 54 | 91.16% | +6.62%p |
| Complexity Signature Deviation | default / 0.005 | 508 | 463 | 45 | 91.14% | +6.60%p |
| Smooth Fourier Normal Residual | default / 0.005 | 424 | 383 | 41 | 90.33% | +5.78%p |

Masked Patch seed intersection은 가장 높은 precision을 보였고 Masked Patch
default는 더 넓은 기존 Hard 범위를 확인했다. Ordinal Pattern은 더 적은
후보를 확인하지만 FP 비율이 낮았다. 이 값들은 기존 경보를 삭제하거나
추가한 결과가 아니라, 기존 자동 경보 중 새 점수원이 동의한 부분집합의
품질이다.

### 일반 검토 우선순위

| 점수원 | Variant / alpha | 우선 검토 | TP | FP | Precision | Exp236 일반 검토 대비 |
|---|---|---:|---:|---:|---:|---:|
| Masked Patch Context Error | seed intersection / 0.005 | 44 | 36 | 8 | 81.82% | +28.97%p |
| Ordinal Pattern Distribution Shift | default / 0.005 | 57 | 44 | 13 | 77.19% | +24.35%p |
| Masked Patch Context Error | default / 0.005 | 84 | 61 | 23 | 72.62% | +19.78%p |

Masked Patch seed intersection의 precision은 가장 높지만 사전등록된 최소
표본 50건에 미달한다. Ordinal Pattern과 Masked Patch default는 최소 표본과
precision gate를 모두 통과했다.

두 유력 점수원의 교집합을 사후 진단하면 다음과 같다.

| 후향 교집합 | 구간 | 후보 | TP | FP | Precision |
|---|---|---:|---:|---:|---:|
| Ordinal default + Masked Patch default | 기존 Hard | 220 | 213 | 7 | 96.82% |
| Ordinal default + Masked Patch seed intersection | 기존 Hard | 187 | 182 | 5 | 97.33% |
| Ordinal default + Masked Patch default | 기존 Standard | 18 | 16 | 2 | 88.89% |

이 교집합은 다음 사전등록 실험의 가설을 만드는 후향 진단일 뿐이다. 같은
TEST에서 확인한 수치로 즉시 운영 규칙을 선택하지 않는다.

### 탐지 범위 확대

탐지 범위 확대 gate를 통과한 점수원은 없다. 가장 나은 결과도 Masked
Patch default / alpha 0.005의 기존 lane 밖 195건 중 TP 64, FP 131,
Precision 32.82%였다. 이 후보를 자동 경보로 추가하면 사용자 피로도가
크게 증가한다.

## 4. 점수원별 판정

| Exp | 점수원 | 판정 |
|---:|---|---|
| 239 | Multiscale Wavelet Energy Deviation | 전체 후보와 lane 밖 FP가 많아 현재 형태로는 부적합 |
| 240 | Ordinal Pattern Distribution Shift | Hard 신뢰도와 일반 검토 우선순위에 모두 유효 |
| 241 | Symbolic Motif Rarity | precision과 coverage가 모두 부족 |
| 242 | Complexity Signature Deviation | Hard 신뢰도 보조에는 유효, review와 확대에는 부족 |
| 243 | Recurrence Dynamics Deviation | Hard precision 상승이 최소 gate에 못 미침 |
| 244 | Robust State-space Innovation | TP 대비 FP가 많아 기각 후보 |
| 245 | Smooth Fourier Normal Residual | Hard 신뢰도 보조에는 제한적으로 유효 |
| 246 | Transformation Consistency | 일부 구간 precision은 높지만 표본 coverage가 부족 |
| 247 | Masked Patch Context Error | 이번 묶음에서 가장 강한 Hard 및 review 보조 점수원 |
| 248 | Nonlinear Normal Density | 사실상 후보를 만들지 못했고 생성 후보도 모두 FP |

## 5. 무결성과 제한

- 10개 실험 모두 1,117개 데이터셋을 기록했다.
- frozen prediction 58,084행에서 Exp236 Hard/Standard/Priority lane 불일치는 0건이다.
- 독립 재집계의 lane overlap, index 범위, summary 불일치는 모두 0건이다.
- Exp239-243은 TRAIN 5개 미만인 162개 데이터셋에서 기권했다.
- Exp246-247은 TRAIN 20개 미만인 478개 데이터셋에서 기권했다.
- Exp248은 TRAIN 50개 미만인 616개 데이터셋에서 기권했다.
- 작은 TRAIN 집합에서는 empirical p-value 해상도 때문에 낮은 alpha를 평가할 수 없는 경우가 많다.
- 모든 수치는 반복 사용된 기존 TEST에 대한 retrospective 결과이며 prospective 운영 성능이 아니다.

## 6. 결론과 다음 방향

이번 실험은 자동 경보 확대보다 신뢰도 등급과 검토 우선순위 표시가 더
타당하다는 결론을 강화한다. 다음 실험은 Masked Patch와 Ordinal Pattern의
동의 관계를 TRAIN-only 규칙으로 동결하고, 다음 세 계층을 별도로 검증하는
것이 적절하다.

1. 두 점수원이 함께 동의한 기존 Hard: 매우 높은 신뢰도 후보
2. 한 점수원만 동의한 기존 Hard: 일반 신뢰도 후보
3. Masked Patch 또는 Ordinal이 동의한 기존 Standard: 우선 검토 후보

기존 lane 밖 후보는 자동 경보로 올리지 않는다. 새 조합은 기존 TEST에서
좋은 alpha를 선택하는 방식이 아니라 새 실험 번호와 사전등록된 규칙으로
만들고, 최종 운영 주장은 신규 holdout 또는 shadow validation 이후에만 한다.
