# Exp163 대안 점수원 꼬리 합의 검토 결과

## 한 줄 결론

Exp163은 Exp160의 자동 경보를 전혀 바꾸지 않은 채 후보 0 데이터셋 일부를 일반 검토로 복구했다. `2-of-3`은 coverage가 넓지만 FP가 많고, `3-of-3`은 낮은 alpha에서 더 정밀하지만 복구량이 작다. TEST 결과로 운영 alpha나 variant를 선택하지 않았다.

## 실험 구조

세 점수원을 각각 TRAIN 경험적 상위 분위수로 변환했다.

1. 미분·스펙트럼 통계 점수
2. 부분 형태 일치 점수
3. 구간 분포 점수

원시 점수를 더하지 않고 같은 TEST index를 몇 점수원이 지목했는지만 사용했다.

- `tail_rank_2of3`: 2개 이상 점수원이 동의
- `tail_rank_3of3`: 3개 점수원 모두 동의
- 신규 후보: 일반 검토만 추가
- 자동 경보/우선 검토: Exp160 그대로
- alpha: `0.005, 0.01, 0.02, 0.05` 모두 보고

## 자동 경보 불변성

Exp163 variant에 관계없이 Hard 지표는 Exp160과 완전히 같다.

| alpha | Hard alerts | TP | FP | precision | mean Hard F1 |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 906 | 849 | 57 | 93.709% | 0.055785 |
| 0.010 | 1,260 | 1,140 | 120 | 90.476% | 0.092992 |
| 0.020 | 1,756 | 1,506 | 250 | 85.763% | 0.208912 |
| 0.050 | 2,153 | 1,723 | 430 | 80.028% | 0.292501 |

Hard와 Priority index의 Exp160 대비 mismatch는 0이다. Exp163은 autonomous alert 개선 실험이 아니라 human review coverage 보충 실험이다.

## 신규 일반 검토 효과

### 2-of-3 합의

| alpha | 신규 검토 | TP | FP | precision | 후보 0 전 -> 후 | mean combined F1 전 -> 후 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 187 | 99 | 88 | 52.941% | 699 -> 556 | 0.280585 -> 0.359517 |
| 0.010 | 154 | 73 | 81 | 47.403% | 587 -> 481 | 0.359067 -> 0.415599 |
| 0.020 | 119 | 48 | 71 | 40.336% | 467 -> 383 | 0.442470 -> 0.480932 |
| 0.050 | 78 | 21 | 57 | 26.923% | 307 -> 260 | 0.524818 -> 0.542276 |

### 3-of-3 합의

| alpha | 신규 검토 | TP | FP | precision | 후보 0 전 -> 후 | mean combined F1 전 -> 후 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 36 | 27 | 9 | 75.000% | 699 -> 665 | 0.280585 -> 0.304309 |
| 0.010 | 21 | 17 | 4 | 80.952% | 587 -> 566 | 0.359067 -> 0.374287 |
| 0.020 | 14 | 9 | 5 | 64.286% | 467 -> 453 | 0.442470 -> 0.450528 |
| 0.050 | 9 | 1 | 8 | 11.111% | 307 -> 302 | 0.524818 -> 0.525713 |

`mean combined F1`은 사용자가 검토 결과를 확인했다고 가정한 human-assisted 진단이다. Hard F1과 섞어 해석하지 않는다.

## 무엇이 좋아졌는가

- Exp160에서 남은 후보 0 상태를 추가로 줄였다.
- 자동 경보 FP는 한 건도 늘리지 않았다.
- 낮은 alpha에서 3-of-3 합의는 36건 중 27 TP, 21건 중 17 TP로 비교적 깨끗한 review 신호를 보였다.
- 모든 신규 후보가 일반 검토에만 들어가므로 사용자 확인 전 자동 조치로 이어지지 않는다.

## 무엇이 나빠졌는가

- 2-of-3은 coverage를 늘리는 대신 신규 review precision이 26.9%~52.9%로 낮다.
- 3-of-3도 alpha 0.05에서는 9건 중 TP가 1건뿐이었다. 점수원 합의 자체가 항상 신뢰성을 보장하지 않는다.
- 전체 Standard review precision은 신규 후보가 추가되면서 대체로 낮아졌다.
- 후보 0 문제를 완전히 해결하지 못했다. `n_train < 5` 완전 abstention 162개는 그대로 남는다.

## 점수원 진단

alpha 0.005의 후보 단독 precision은 다음과 같다.

- 미분·스펙트럼 통계 점수: 45.960%
- 부분 형태 일치 점수: 50.725%
- 구간 분포 점수: 31.173%
- 2-of-3 합의: 52.941%
- 3-of-3 합의: 75.000%

낮은 alpha에서는 합의가 단독 점수보다 precision을 높이는 신호가 있다. 반면 alpha가 커질수록 세 source 모두 noisy해지고, alpha 0.05의 3-of-3도 이를 상쇄하지 못했다. 이 관찰은 후향 진단이며 특정 alpha 선택 근거로 사용하지 않는다.

## Threshold 해상도

- dataset-source 조합: 3,351
- 네 alpha가 모두 다른 threshold를 만든 조합: 732
- threshold가 중복된 조합: 2,619
- 하나 이상의 source에서 해상도 제한이 있는 데이터셋: 881/1,117
- 네 alpha가 모두 같은 threshold인 조합: 1,476

TRAIN 표본이 작을 때 `method="higher"` 경험적 분위수는 여러 alpha에서 같은 최대값을 선택한다. 따라서 alpha 네 개가 항상 네 가지 다른 정책을 의미하지 않는다. 이것은 구현 오류가 아니라 표본 해상도 한계다.

## Workload

2-of-3의 dataset별 전체 표시 후보 p95는 alpha 순서대로 6, 8, 11, 16건이다. 최대값은 75, 80, 102, 170건이다. 최대 workload는 Exp163 신규 후보만의 문제가 아니라 기존 Exp160 lane을 포함한 값이다. 신규 추가의 dataset별 최대는 6, 8, 9, 8건이다.

3-of-3의 신규 추가 최대는 3, 1, 1, 5건으로 더 보수적이다.

## 검증

- unit/structure tests: 18/18 통과
- prediction coverage: 8,936/8,936
- prediction/evaluation 독립 재집계: 통과
- lane overlap/union/count/range 오류: 모두 0
- prediction SHA256: `54fb3f1a94a268428b00835c7433d2ef0b033c8880a254c96c59d49e1477c655`
- 평가 전후 prediction SHA: 동일
- deterministic full replay SHA: 동일

## 해석 제한과 다음 판단

이번 결과만으로 `tail_rank_3of3, alpha=0.01`을 운영값으로 선택하면 안 된다. 동일 TEST benchmark를 반복 관찰한 후향 sensitivity이기 때문이다. 다음 단계는 신규 잠금 데이터 또는 shadow mode에서 alpha/variant를 사전에 동결해 검증하는 것이다.

또한 candidate-zero gate가 frozen TEST score 결과에 의존하므로 Exp163은 end-to-end strict TRAIN-only가 아니다. D2 feature/configuration provenance와 실제 equipment/recipe/sensor/step prospective 검증도 여전히 미해결이다.
