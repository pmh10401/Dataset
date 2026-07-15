# Exp158/159 Codex 검증 보고서

## 범위

- 연구 구분: retrospective counterfactual sensitivity
- prospective validation: 수행하지 않음
- TEST metric 기반 alpha/descriptor/pool/lane 선택: 수행하지 않음
- Hard alert: autonomous
- Standard/Priority review 및 combined F1: human-assisted diagnostic only

## Gemini 1차 구현 검토

Gemini 3.5 Flash (High)의 1차 구현에서 다음 세 문제를 확인했다.

1. 10-dataset smoke가 donor universe도 10개로 제한하여 최소 reference 200을 만족하지 못했고 실제 pooling 경로를 통과하지 않았다.
2. pool availability가 source별 finite OOF reference 수가 아니라 단순 n_train 합으로 판정됐다.
3. Exp152 baseline 1,117 x 4 coverage와 Hard/Priority exact preservation이 실행 중 강제되지 않았다.

테스트를 먼저 실패시킨 뒤 다음과 같이 수정했다.

- smoke target만 제한하고 donor universe는 전체 1,117개로 유지
- 모든 source가 실제 finite reference 200개 이상일 때만 pool 허용
- target dataset을 모든 source reference에서 제외
- baseline coverage, duplicate, lane overlap, candidate union, index range를 실행 전에 검증
- similarity-only 후보는 Exp159 Standard review에만 추가
- Exp159 Hard/Priority는 frozen Exp152 그대로 보존
- n_train < 5의 `abstained` 집계를 prediction row에 명시

## 입력 무결성

- score-cache manifest SHA256: `bf73796c046a6102fce53d3b6fc606afa1ed0a5161ebb6e92d94698ab56a68a0`
- Exp152 prediction SHA256: `9684f6bae6e561535f190969a4c71b1f258c97d0b7cf9899b8d9147a74c13499`
- Exp152 manifest SHA256: `bce61baee9f08a68e1bebef5b74ec93b230ed18634a66798b2baa293cb372b48`
- cache coverage: 1,117/1,117
- descriptor complete: 955
- n_train < 5 abstention: 162
- 5 <= n_train < 20 similarity-pool target: 316
- target pool available: 316/316

## Full prediction 검증

- Exp158 rows: 4,468
- Exp159 rows: 4,468
- dataset 누락: 0
- lane overlap: 0
- candidate/lane union mismatch: 0
- candidate index range error: 0
- prediction 평가 필드: 0
- target self-pooling: 0
- source별 pooled reference 200 미만: 0
- Exp159 Hard baseline mismatch: 0
- Exp159 Priority baseline mismatch: 0
- 대상 범위 밖 Standard baseline mismatch: 0

## Small-train review 추가 효과

다음 수치는 `5 <= n_train < 20`인 316개 데이터셋에만 해당한다. TEST label은 prediction 동결 후 사후 평가에만 사용했다.

| alpha | 추가 review | TP | FP | 추가 precision | candidate-zero 전 | 후 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 106 | 100 | 6 | 94.340% | 316 | 216 |
| 0.010 | 154 | 138 | 16 | 89.610% | 316 | 173 |
| 0.020 | 201 | 170 | 31 | 84.577% | 316 | 137 |
| 0.050 | 284 | 221 | 63 | 77.817% | 311 | 75 |

이 표는 alpha 선택 근거가 아니다. 네 alpha는 사전 고정 sensitivity 결과로 함께 유지한다.

## 해석 제한

- Exp159은 기존 autonomous Hard alert를 개선한 실험이 아니다. Hard 결과를 보존하면서 작은 데이터셋의 review coverage를 추가한 실험이다.
- 실제 equipment/recipe/sensor/step 메타데이터가 없어 score-shape similarity가 물리적 동질성을 뜻하는지는 확인되지 않았다.
- 동일 benchmark TEST를 반복 사용한 retrospective 결과다.
- end-to-end strict TRAIN-only 또는 실제 설비 배포 성능을 주장할 수 없다.
