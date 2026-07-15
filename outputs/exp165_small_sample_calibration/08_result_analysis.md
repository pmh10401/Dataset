# Exp165 결과 분석

## 결론

Exp165의 TRAIN-normal score-similarity pooling은 `5 <= n_train < 20`인
316개 소표본 데이터셋에서 Exp164의 threshold resolution collapse를
해결했다. 네 alpha가 모두 같은 threshold였던 기존 방식과 달리 pooled
방식은 거의 모든 pseudo-run과 score source에서 네 개 threshold를 구분했다.

정상 pseudo-run false-review도 크게 감소했다. 다만 이 실험은 TEST 값이나
라벨을 사용하지 않은 정상 calibration audit이므로 anomaly TP, FP, recall,
F1 개선을 주장할 수 없다. 어떤 alpha나 agreement variant도 운영값으로
선택하지 않는다.

## 평가 범위

- 전체 동결 population: 1,117개
- 평가 대상: `5 <= n_train < 20`인 316개
- 정상 pseudo-run: 3,863개
- donor: `n_train >= 20`인 639개
- `n_train < 5`: 162개 완전 기권 유지
- 오류 및 누락: 0

## 정상 false-review 비교

| alpha | 합의 | Exp164 local | similarity pool | 감소율 |
|---:|:---:|---:|---:|---:|
| 0.005 | 2-of-3 | 210 / 3,863 (5.436%) | 34 / 3,863 (0.880%) | 83.8% |
| 0.005 | 3-of-3 | 16 / 3,863 (0.414%) | 2 / 3,863 (0.052%) | 87.5% |
| 0.010 | 2-of-3 | 210 / 3,863 (5.436%) | 48 / 3,863 (1.243%) | 77.1% |
| 0.010 | 3-of-3 | 16 / 3,863 (0.414%) | 3 / 3,863 (0.078%) | 81.3% |
| 0.020 | 2-of-3 | 210 / 3,863 (5.436%) | 69 / 3,863 (1.786%) | 67.1% |
| 0.020 | 3-of-3 | 16 / 3,863 (0.414%) | 5 / 3,863 (0.129%) | 68.8% |
| 0.050 | 2-of-3 | 210 / 3,863 (5.436%) | 151 / 3,863 (3.909%) | 28.1% |
| 0.050 | 3-of-3 | 16 / 3,863 (0.414%) | 11 / 3,863 (0.285%) | 31.3% |

Exp164 local 결과가 모든 alpha에서 동일한 것은 소표본 quantile이 같은
최댓값으로 뭉쳤기 때문이다. Similarity pool은 alpha 증가에 따라 false-review가
단조 증가해 alpha가 다시 실질적인 민감도 조절값으로 작동한다.

## Threshold 해상도

- Exp164 local: 세 source 모두 3,863/3,863 pseudo-run에서 네 alpha threshold가 동일했다.
- Similarity pool shapelet/interval: 평균 4.000개의 고유 threshold였다.
- Similarity pool spectral: 평균 3.997개의 고유 threshold였다.
- Similarity pool에서 네 alpha가 모두 같았던 pseudo-run은 0개였다.

따라서 Exp165의 직접 목표인 threshold resolution repair는 달성됐다.

## 데이터셋별 안정성

가장 보수적인 `alpha=0.005`, `2-of-3`에서:

- 개선: 147개 데이터셋
- 동일: 163개 데이터셋
- 악화: 6개 데이터셋
- 하나 이상의 false-review가 있던 데이터셋: 171개에서 32개로 감소

악화 6개에는 `WordSynonyms_normal_9`, `GestureMidAirD2_normal_17`,
`Lightning7_normal_6`, `PickupGestureWiimoteZ_normal_5`,
`ShakeGestureWiimoteZ_normal_5`, `ShapesAll_normal_25`가 포함된다.
즉 pooled 후보가 local 후보의 단순 부분집합은 아니며 일부 정상 run의 판정
위치가 새로 바뀐다.

## Pool 건전성

- pseudo-run당 donor 수: 중앙값 3, 평균 3.15, p95 6, 최대 8
- source별 reference 수: 최소 200, 중앙값 256, p95 1,044
- 실제 사용 donor: 639개 중 495개
- 상위 10개 donor의 선택 비중: 14.3%

최소 reference 계약은 모두 충족했고 한두 donor가 전체 결과를 독점하지는
않았다. 다만 cross-dataset score similarity가 실제 설비의 동일 공정 구조를
뜻한다는 보장은 없으므로 연구용 pooling이라는 제한은 유지한다.

## Wilson 신뢰구간 해석

모든 데이터셋의 `meets_0.005/0.010/0.020`가 false인 것은 pooled 방식이
전부 실패해서가 아니다. 데이터셋별 정상 표본이 5~19개뿐이라 false-review가
0개여도 one-sided 95% Wilson upper bound를 2% 이하로 증명할 수 없다.

false-review 0건일 때 필요한 최소 표본 수는 대략 다음과 같다.

- 2% 목표: 133개
- 1% 목표: 268개
- 0.5% 목표: 539개

따라서 dataset별 risk guarantee는 현재 데이터로 검증 불가능하다. 전체
3,863 pseudo-run을 합친 진단에서는 pooled `3-of-3`의 Wilson upper bound가
모든 alpha에서 0.5% 미만이지만, 이는 개별 데이터셋 보장이 아니다.

## 장점

1. TEST 정보 없이 소표본 threshold 해상도를 복구했다.
2. 정상 false-review를 특히 낮은 alpha에서 크게 줄였다.
3. alpha가 민감도 조절값으로 다시 작동한다.
4. target pseudo-run을 normalization과 descriptor에서 제외해 자기참조를 막았다.
5. 316개 전체 coverage와 최소 200 reference를 충족했다.

## 한계와 위험

1. anomaly detection 성능은 평가하지 않았다.
2. `3-of-3`의 낮은 false-review는 anomaly miss 증가와 맞바꾼 결과일 수 있다.
3. 일부 데이터셋은 pooled 방식에서 정상 false-review가 증가했다.
4. 실제 equipment/recipe/sensor/step metadata가 아닌 score similarity로 donor를 정했다.
5. `n_train < 5`인 162개 데이터셋 문제는 해결하지 않았다.
6. 기존 TEST benchmark를 반복 사용하는 후속 평가는 retrospective로만 해석해야 한다.

## 권장 후속 검증

다음 단계는 Exp165의 여덟 sensitivity 조합을 그대로 동결해 TEST prediction을
먼저 생성하고 hash를 고정한 뒤, 마지막에만 TEST label로 평가하는 것이다.
모든 alpha와 `2-of-3`/`3-of-3`를 함께 보고하며 TEST F1을 보고 하나를 선택하지
않는다. 확인할 핵심은 false alert 감소가 anomaly TP 보존과 함께 유지되는지,
candidate-zero와 abstention이 어떻게 변하는지다.

별도 안정성 실험으로 donor perturbation, leave-one-donor-out, reference-size
sensitivity를 수행해 특정 donor 선택에 대한 결과 민감도를 확인할 가치가 있다.
