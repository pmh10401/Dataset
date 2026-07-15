# Exp171/172 결과 분석

## 결론

Exp171은 TEST 결과를 사용하지 않고 정상 run 100개당 검토 요청 최대 1개라는
계약 아래 정책 하나를 동결했다. 선택된 정책은 다음과 같다.

- pool: `global_balanced_pool`
- alpha: `0.02`
- 확인 점수원: 주파수 특징 + 구간 통계 특징
- TRAIN 정상 OOB 후보: 17 / 3,863 (`0.4401%`)
- 60개 동시 비교 보정 후 단측 95% 상한: `0.9241%`
- 위험 목표: `1.0%`

따라서 TRAIN audit 기준으로는 사전등록한 정상 검토 요청 상한을 충족했다.
이 선택은 이상 탐지 TEST F1을 보지 않고 완료됐다.

## Exp172 사후 평가

동결된 정책을 316개 작은 TRAIN 데이터셋에 적용한 결과다.

| 지표 | 결과 |
|---|---:|
| 일반 검토 후보 | 40 |
| TP | 30 |
| FP | 10 |
| Micro precision | 75.0% |
| Mean recall | 0.094937 |
| Mean F1 | 0.089662 |
| 후보가 0개인 데이터셋 | 284 |
| F1이 0인 데이터셋 | 286 |

오탐은 적고 검토 precision은 비교적 높지만 coverage가 매우 낮다. 즉 Exp172는
사용자 피로도를 낮추는 방향에는 맞지만, 316개 중 대부분에서 아무 후보도
제시하지 못한다. 자동 경보 기본 모델로 승격할 근거는 없다.

## Exp170과의 일치

Exp172는 Exp170의 60개 sensitivity 중 TRAIN audit으로 미리 고정한 한 행만
다시 동결하고 평가했다. 동일 key의 Exp170 값과 Exp172 값은 정확히 일치한다.
이는 TEST 결과를 보고 새 threshold나 alpha를 조정하지 않았음을 보여준다.

## 다음 단계

1. Exp172를 작은 TRAIN 데이터셋의 review-only 안전 기준선으로 유지한다.
2. zero-F1 286개를 줄이기 위해 기존 threshold를 느슨하게 재선택하지 않는다.
3. Exp173에서는 TRAIN-only로 만들 수 있는 독립적인 추가 특징 점수원을
   사전등록하고, Exp172의 1% 위험 계약을 guard로 유지한다.
4. 최종 운영 주장은 신규 시간 구간의 shadow/prospective 검증 뒤에만 한다.

모든 Exp172 성능은 retrospective human-assisted review 지표다. Autonomous
Hard alert 성능, 실제 설비 성능, end-to-end strict TRAIN-only 개발 계보를
의미하지 않는다.

