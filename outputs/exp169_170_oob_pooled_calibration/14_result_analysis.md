# Exp169/170 결과 분석

## 결론

Exp169/170은 작은 TRAIN 집합에서 기존 Exp168이 겪었던 conformal 해상도 막힘을 해소했다. 316개 데이터셋의 모든 pool/alpha/source configuration에서 `resolution_blocked_datasets=0`이었다. 다만 coverage와 false positive 사이의 trade-off가 남아 있으며, 이번 TEST 결과로 정책을 선택하지 않는다.

## 정상 위험 audit

TRAIN 정상 OOB instance는 총 3,863개다. 전체 60개 sensitivity에서 정상 review 후보 합계는 0~97개였다.

`alpha=0.05, 2-of-3` 예시는 다음과 같다.

| Pool | 정상 review 후보 | 평균 정상 review rate | 평균 reference 수 |
|---|---:|---:|---:|
| Similarity | 97 | 2.492% | 206.2 |
| Coarse strata | 89 | 2.239% | 707.6 |
| Global balanced | 96 | 2.951% | 3,850.8 |

이 표는 정상 데이터에서 발생할 수 있는 검토 workload를 보여주는 진단이며, TEST 성능과 결합해 운영값을 선택하지 않는다.

## 사후 review sensitivity

전체 60개 결과 범위는 다음과 같다.

- review 후보: 3~209
- TP: 3~125
- FP: 0~84
- micro precision: 55.56%~100%
- mean F1: 0.0095~0.3657
- zero-F1 데이터셋: 191~313

`alpha=0.05, 2-of-3`의 pool별 관찰값은 다음과 같다.

| Pool | 후보 | TP | FP | Precision | Mean F1 | Zero-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Similarity | 209 | 125 | 84 | 59.81% | 0.3657 | 191 |
| Coarse strata | 175 | 116 | 59 | 66.29% | 0.3417 | 200 |
| Global balanced | 172 | 118 | 54 | 68.60% | 0.3499 | 198 |

반대로 `similarity, alpha=0.005, 2-of-3`은 후보 51개, TP 49개, FP 2개, precision 96.08%, mean F1 0.1535였다. 이는 낮은 alpha가 더 보수적이라는 sensitivity를 보여줄 뿐 운영 alpha 선택 근거가 아니다.

## 이전 실험과의 관계

- Exp168은 작은 local reference 때문에 여러 alpha에서 후보가 0이고 316개 모두 resolution-blocked였다.
- Exp169/170은 target-excluded pooled reference로 이 수학적 해상도 제한을 제거했다.
- Exp166 local 2-of-3은 후보 265, TP 131, FP 134, precision 49.43%, mean F1 0.3797이었다.
- Exp170의 `alpha=0.05, 2-of-3` sensitivity들은 Exp166 local보다 FP가 적고 precision이 높았지만 TP와 mean F1도 일부 감소했다.

따라서 이번 실험은 “pooling이 무조건 우월하다”가 아니라, 작은 표본에서 후보를 전혀 만들지 못하던 문제를 해결하면서 workload와 coverage를 조절할 수 있다는 증거다.

## 다음 검증

1. 60개 중 하나를 TEST 결과로 선택하지 않는다.
2. TRAIN 정상 위험 목표와 사용자 review 용량을 먼저 운영 계약으로 정한다.
3. 그 계약만으로 alpha/pool/source configuration을 고정하는 새 실험을 사전등록한다.
4. 고정 정책을 신규 시간 구간 또는 shadow mode에서 prospective 검증한다.

모든 지표는 retrospective research review 성능이며 autonomous Hard alert 또는 배포 성능이 아니다.
