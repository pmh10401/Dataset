# Exp173/174 결과 분석

## 결론

권장 균형점인 TRAIN 정상 위험 동시 상한 1%를 유지하면서 Exp171보다 점수원
다양성과 fold 안정성을 우선한 정책을 동결했다.

- pool: `similarity_pool`
- alpha: `0.01`
- 확인 구조: 주파수 패턴, 형상 패턴, 구간 통계 중 임의의 두 점수원 합의
- TRAIN 정상 OOB 후보: 10 / 3,863 (`0.2589%`)
- 60개 동시 비교 보정 단측 95% 상한: `0.6722%`
- 위험 목표: `1.0%`
- 5-fold 정상 검토율: `0.3731%`, `0.1443%`, `0.4043%`, `0.2632%`, `0.1157%`

Exp173은 정상 후보 수를 최대화하지 않았다. 가장 큰 eligible alpha를 먼저 고른
뒤 fold 간 정상 검토율 변동이 가장 작은 pool을 선택했다. 선택 과정에서 TEST
라벨, 이상 위치 또는 Exp170 성능 CSV를 읽지 않았다.

## Exp174 사후 review 결과

| 지표 | Exp172 | Exp174 | 변화 |
|---|---:|---:|---:|
| 데이터셋 | 316 | 316 | 0 |
| Review 후보 | 40 | 71 | +31 |
| TP | 30 | 67 | +37 |
| FP | 10 | 4 | -6 |
| Micro review precision | 75.000% | 94.366% | +19.366%p |
| Mean recall | 0.094937 | 0.212025 | +0.117089 |
| Mean F1 | 0.089662 | 0.208861 | +0.119198 |
| 후보가 0개인 데이터셋 | 284 | 249 | -35 |
| F1이 0인 데이터셋 | 286 | 249 | -37 |

Exp174는 review 후보를 31개 늘렸지만 TP는 37개 늘고 FP는 6개 줄었다. 후보가
있는 데이터셋의 범위도 32개에서 67개로 넓어졌다. 데이터셋별 review 후보 수는
median 0, p90 1, p95 1, 최대 3개다. 따라서 전체 coverage는 아직 낮지만, 후보가
발생한 경우의 사용자 부담은 작다.

## 장점

1. 정상 TRAIN 위험 동시 상한 1%를 유지했다.
2. 고정된 두 점수원만 보는 대신 세 점수원 중 두 개의 합의를 허용해 이상 유형
   coverage를 넓혔다.
3. Exp172보다 review TP가 늘면서 FP가 줄어, 탐지 범위와 사용자 피로도가 동시에
   개선되는 retrospective 신호가 나왔다.
4. 데이터셋당 review 후보가 p95 기준 1개라 검토 workload가 작다.

## 남은 문제

1. 316개 중 249개는 여전히 후보가 없다. 이번 selector 개선만으로 작은 TRAIN
   데이터의 calibration/representation 문제를 모두 해결하지 못했다.
2. Exp170의 60개 TEST 결과는 과거에 이미 관찰됐다. 따라서 Exp174는 새롭고
   독립적인 holdout 검증이 아니라 retrospective counterfactual이다.
3. 94.366%는 자동 Hard alert precision이 아니라 연구용 review 후보를 사용자가
   확인한다고 가정한 사후 precision이다.
4. 1%는 3,863개 TRAIN OOB run을 합친 aggregate 상한이다. 데이터셋별 보장이나
   실제 설비 SLA가 아니다.
5. 과거 feature/source configuration 선택 근거까지 포함한 end-to-end TRAIN-only
   provenance는 검증되지 않았다.

## 다음 검증

이 정책은 신규 시간 구간의 shadow mode에서 그대로 동결해야 한다. 정상 오경보
상한, alpha, `2of3`, fold 수, pool을 신규 라벨이나 사용자 피드백을 보기 전에
고정하고, 이후 review precision과 candidate-zero 비율을 확인해야 한다. 그 전에는
Exp174를 운영 자동 경보로 승격하지 않는다.
