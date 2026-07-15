# Exp166 결과 분석

## 결론

Exp165의 유사도 풀링은 소표본 threshold 해상도와 TRAIN 정상 pseudo-run
오경보를 개선했지만, Exp166 TEST 이상 검증에서는 그 개선이 유지되지 않았다.
유사도 풀링 8개 조합 중 기존 local 기준보다 TP/FP 균형과 Mean F1을 함께
개선한 조합은 없었다.

따라서 Exp165 유사도 풀링을 현재 형태로 기존 calibration의 대체 정책이나
자동 경보로 사용하면 안 된다. 다음 단계는 alpha를 다시 고르는 것이 아니라,
교차적합 calibration score와 full-TRAIN TEST score 사이의 생성 조건을 맞추는
score-calibration parity 검증이다.

## 평가 범위와 지표

- 대상: `5 <= n_train < 20`인 316개 데이터셋
- TEST instance: 15,800개
- 정상 TEST instance: 15,484개
- 이상 TEST instance: 316개
- 각 데이터셋: 정상 49개와 이상 1개
- 평가 조합: local/pool 2종 x alpha 4종 x agreement 2종 = 16개
- 후보 lane: retrospective research review candidate
- autonomous Hard alert: 평가하지 않음
- 특정 alpha 또는 조합 선택: 없음

각 데이터셋에 이상 instance가 하나이므로 TP는 이상을 찾은 데이터셋 수와 같다.
Mean F1은 데이터셋별 F1의 macro 평균이고, precision은 전체 후보의 micro
precision이다.

## 전체 결과

| 보정 | alpha | 합의 | 후보 | TP | FP | Precision | Recall | Mean F1 | Zero-F1 | 후보 0 |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Local | 모든 alpha | 2-of-3 | 265 | 131 | 134 | 49.43% | 41.46% | 0.3797 | 185 | 166 |
| Local | 모든 alpha | 3-of-3 | 50 | 41 | 9 | 82.00% | 12.97% | 0.1271 | 275 | 273 |
| Pool | 0.005 | 2-of-3 | 180 | 70 | 110 | 38.89% | 22.15% | 0.2040 | 246 | 235 |
| Pool | 0.010 | 2-of-3 | 224 | 85 | 139 | 37.95% | 26.90% | 0.2487 | 231 | 211 |
| Pool | 0.020 | 2-of-3 | 288 | 100 | 188 | 34.72% | 31.65% | 0.2864 | 216 | 191 |
| Pool | 0.050 | 2-of-3 | 483 | 128 | 355 | 26.50% | 40.51% | 0.3411 | 188 | 148 |
| Pool | 0.005 | 3-of-3 | 18 | 10 | 8 | 55.56% | 3.16% | 0.0285 | 306 | 305 |
| Pool | 0.010 | 3-of-3 | 22 | 14 | 8 | 63.64% | 4.43% | 0.0411 | 302 | 301 |
| Pool | 0.020 | 3-of-3 | 33 | 20 | 13 | 60.61% | 6.33% | 0.0601 | 296 | 293 |
| Pool | 0.050 | 3-of-3 | 62 | 33 | 29 | 53.23% | 10.44% | 0.1013 | 283 | 277 |

Local의 네 alpha 결과가 같은 것은 소표본 quantile threshold가 같은 최댓값으로
붕괴했기 때문이다. Pool은 alpha별 민감도 곡선을 회복했지만 실제 이상과 정상의
분리 성능은 개선하지 못했다.

## Local 대비 Pool 변화

| alpha | 합의 | TP 변화 | FP 변화 | Mean F1 변화 | 해석 |
|---:|:---:|---:|---:|---:|---|
| 0.005 | 2-of-3 | -61 | -24 | -0.1757 | FP 1개 감소당 TP 약 2.5개 손실 |
| 0.010 | 2-of-3 | -46 | +5 | -0.1310 | TP와 FP 모두 악화 |
| 0.020 | 2-of-3 | -31 | +54 | -0.0933 | TP와 FP 모두 악화 |
| 0.050 | 2-of-3 | -3 | +221 | -0.0386 | recall 유지 대가로 FP 급증 |
| 0.005 | 3-of-3 | -31 | -1 | -0.0986 | FP 1개 감소에 TP 31개 손실 |
| 0.010 | 3-of-3 | -27 | -1 | -0.0860 | FP 1개 감소에 TP 27개 손실 |
| 0.020 | 3-of-3 | -21 | +4 | -0.0670 | TP와 FP 모두 악화 |
| 0.050 | 3-of-3 | -8 | +20 | -0.0258 | TP와 FP 모두 악화 |

paired dataset bootstrap 기준 Pool과 Local의 Mean F1 차이는 낮은 alpha에서
명확히 음수였다. `alpha=0.005, 2-of-3`의 95% interval은
`[-0.2246, -0.1279]`였다. `alpha=0.05, 2-of-3`은
`[-0.0783, 0.0003]`으로 F1 차이가 거의 같을 가능성은 남지만, FP가
데이터셋당 평균 0.699개 증가했고 이 증가의 95% interval은
`[0.424, 1.000]`이었다.

## 왜 정상 audit과 TEST 결과가 달랐는가

### 1. 후보를 줄이는 대신 실제 이상을 더 많이 제거했다

`alpha=0.005, 2-of-3` Pool은 Local 후보에서 TP 66개와 FP 105개를 제거했지만,
새 위치에서 TP 5개와 FP 81개를 추가했다. 최종적으로 FP는 24개만 줄고 TP는
61개 줄었다. Pool은 기존 후보의 안전한 부분집합이 아니라 후보 위치를 상당히
교체했다.

### 2. 교차적합 calibration과 full-TRAIN TEST scoring의 조건이 다르다

Exp165 정상 pseudo-run은 일부 TRAIN run을 제외한 fold 모델로 점수를 만들었다.
Exp166 TEST는 모든 TRAIN run으로 fit한 모델에서 점수를 만들었다. 특히 정상 패턴
거리 점수는 fit 데이터에 따라 prototype이 다시 선택된다. 따라서 OOF 점수 분포와
full-fit TEST 점수가 같은 척도라는 가정이 충분히 성립하지 않았을 가능성이 높다.

### 3. pooled robust score tail이 지나치게 무겁다

주파수 변화 점수와 구간 통계 점수의 일부 donor는 MAD가 매우 작아 robust
normalization 후 극단값이 커진다. 예를 들어 Pool 구간 통계 threshold 중앙값은
`alpha=0.005`에서 3,332.8, `alpha=0.05`에서도 1,315.4였다. 반면 Local의
중앙값은 25.1이었다. 낮은 alpha에서는 유효한 이상까지 억제하고, alpha를 높이면
다른 두 score의 동시 초과가 늘면서 FP가 급증하는 불안정한 전환이 발생했다.

### 4. 세 점수의 합의가 이상과 정상 위치를 충분히 구분하지 못했다

Pool `alpha=0.05, 2-of-3`은 Local과 비슷한 TP를 유지했지만 후보가 265개에서
483개로 늘었다. 후보 0 데이터셋은 166개에서 148개로 줄었지만, 후보가 있으면서
실제 이상은 못 찾은 all-FP 데이터셋은 19개에서 40개로 증가했다. coverage가
늘어난 것이 아니라 잘못된 위치의 후보가 늘어난 경우가 많았다.

## 운영 관점

### 장점

1. TEST 결과를 보기 전에 8개 조합을 고정했고 특정 조합을 선택하지 않았다.
2. alpha별 민감도 변화는 정상적으로 복구됐다.
3. 낮은 alpha에서 TEST 정상 FP 수 자체는 일부 감소했다.
4. 예측과 라벨 평가가 분리되어 결과 계보가 명확하다.

### 문제점

1. 낮은 alpha의 FP 감소보다 TP 손실이 훨씬 크다.
2. 높은 alpha는 TP를 복구하지만 FP가 과도하게 증가한다.
3. 3-of-3 Pool은 지나치게 보수적이며 Local 3-of-3보다 이점이 없다.
4. FP가 일부 데이터셋에 집중된다. Pool `alpha=0.05, 2-of-3`의 FP 절반은
   16개 데이터셋에서 발생했다.
5. score similarity donor가 실제 equipment/recipe/sensor/step 유사성을 뜻한다는
   보장이 없다.
6. 162개 `n_train < 5` 데이터셋은 여전히 기권 상태다.

## 판단

- Exp165 Pool을 운영 기본 calibration으로 채택: **기각**
- Exp165 Pool 후보를 autonomous Hard alert로 사용: **금지**
- Local 결과를 곧바로 운영값으로 채택: **보류**. F1은 더 높지만 alpha 붕괴로
  민감도를 조절할 수 없고 동일 benchmark를 반복 사용한 후향 결과다.
- Exp165의 donor similarity 아이디어 자체: **연구 후보 유지**. calibration과
  TEST score 생성 조건을 맞춘 뒤 재검증할 가치가 있다.

## 다음 실험 권고

다음 실험은 feature를 추가하기 전에 score-calibration parity를 바로잡아야 한다.

1. 각 cross-fit fold 모델로 TEST도 점수화하여 calibration pseudo-run과 TEST가
   같은 fit-size와 prototype 생성 조건을 사용하게 한다.
2. raw robust z-score를 직접 합치지 않고 source별 empirical rank 또는 conformal
   p-value로 변환한 뒤 합의한다.
3. 주파수 변화+정상 패턴 거리, 주파수 변화+구간 통계, 정상 패턴 거리+구간 통계,
   3-source 2-of-3를 모두 사전 고정해 source 불안정을 분리한다.
4. 먼저 TRAIN 정상 pseudo-run audit을 완료하고 prediction hash를 동결한 뒤에만
   TEST label을 평가한다.
5. TEST 결과로 alpha, source pair, donor 수를 선택하지 않는다.

## 근거 파일

- `02_frozen_predictions.jsonl`: label-free frozen prediction 5,056행
- `03_prediction_manifest.json`: prediction SHA-256과 실행 계약
- `04_dataset_metrics.csv`: 데이터셋별 후향 평가 5,056행
- `05_global_summary.csv`: 16개 고정 조합 요약
- `06_candidate_transition.csv`: Local과 Pool 후보 이동
- Exp165 `04_global_summary.csv`: TRAIN 정상 pseudo-run false-review audit

이 분석은 기존 TEST benchmark에 대한 retrospective 결과이며 prospective 또는
실제 설비 검증이 아니다.
