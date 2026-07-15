# Exp160 Codex 검증 보고서

## 실험 범위

- 실험: `experiment_160_candidate_zero_similarity_recovery`
- 기준: frozen Exp152 B2 prediction
- 대상: dataset/alpha별 Exp152 후보가 0개이고 `n_train >= 5`인 경우
- 추가 후보: 일반 검토 전용
- 자동 경보 및 우선 검토: Exp152 결과 그대로 보존
- 구분: retrospective counterfactual sensitivity
- prospective validation: 수행하지 않음
- 특정 alpha 선택: 수행하지 않음

## 점수원 명칭

| 내부 key | 사용자 명칭 | 알고리즘 |
|---|---|---|
| `rocket_exp40` | 시간 패턴 점수 | ROCKET + KNN local-gap |
| `exp55_best` | 주파수 변화 점수 | Spectrogram + PCA + KNN |
| `exp56_best` | 반복 구조 점수 | Recurrence Plot + GLCM + PCA + KNN |
| `exp84` | 다중 커널 점수 | MultiROCKET + HYDRA + stable-tail pruning + KNN local-gap |

## Antigravity 1차 구현 검토 및 수정

Antigravity의 1차 구현은 기본 candidate-zero gate와 Standard-only 통합을 구현했으며 10-target smoke를 통과했다. Codex 검토에서 다음 문제를 확인하고 테스트를 먼저 실패시킨 뒤 수정했다.

1. descriptor, donor 선택, pooled conformal p-value를 alpha마다 반복해 데이터셋당 네 번 계산했다. TRAIN-only pooling 증거는 alpha와 무관하므로 데이터셋당 한 번만 계산하고 alpha별 2-of-4 판정만 반복하도록 변경했다.
2. donor eligibility에 `n_train >= 5`가 코드 계약으로 강제되지 않았다. 순수 helper와 단위 테스트를 추가해 `n_train < 5` donor를 차단했다.
3. prediction hash tamper와 candidate-empty 정책 필드 테스트가 `pass`뿐인 무효 테스트였다. 실제 freeze/tamper 실패와 policy metadata 값을 검증하도록 교체했다.
4. 사용자용 점수 명칭이 상수에만 있고 prediction에 기록되지 않았다. 모든 prediction 행의 `source_display_names`에 네 명칭을 기록했다.
5. 정책이 candidate-empty gate를 사용한다는 사실과 특정 행에서 gate가 적용됐다는 사실이 한 필드에 섞여 있었다. `uses_test_candidate_empty_gate=1`과 `candidate_empty_gate_applied`를 분리했다.

## 검증 결과

- 단위 및 구조 테스트: 23/23 통과
- prediction rows: 4,468
- datasets: 1,117
- dataset/alpha 중복: 0
- lane overlap: 0
- candidate/lane union mismatch: 0
- candidate count mismatch: 0
- candidate index range error: 0
- target self-pooling: 0
- pool available인데 source reference 200 미만: 0
- prediction TEST label/evaluation field: 0
- 사용자용 점수 명칭 누락: 0
- non-target baseline lane mismatch: 0
- Hard/Priority baseline mismatch: 0
- prediction SHA256 재실행 일치: `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`

## Exp152 대비 추가 일반 검토 효과

| alpha | candidate-zero target | 추가 일반 검토 | 추가 TP | 추가 FP | 추가 정밀도 | 후보 0 전 -> 후 | mean combined F1 전 -> 후 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 827 | 400 | 343 | 57 | 85.750% | 989 -> 699 | 0.057579 -> 0.280585 |
| 0.010 | 770 | 450 | 360 | 90 | 80.000% | 932 -> 587 | 0.097754 -> 0.359067 |
| 0.020 | 603 | 354 | 270 | 84 | 76.271% | 765 -> 467 | 0.223700 -> 0.442470 |
| 0.050 | 429 | 352 | 246 | 106 | 69.886% | 591 -> 307 | 0.321635 -> 0.524818 |

후보 0 집계에는 `n_train < 5` 완전 기권 162개가 포함된다. Exp160은 이 162개를 복구하지 않는다. `mean combined F1`은 사용자가 일반/우선 검토를 확인했다고 가정한 human-assisted diagnostic이며 autonomous Hard F1으로 해석하지 않는다.

## 자동 경보 불변성

모든 alpha에서 Exp152 대비 다음 값의 변화는 0이다.

- Hard alert 수
- Hard TP
- Hard FP
- Mean Hard F1
- Priority review 후보와 index

따라서 Exp160은 자동 경보 모델을 개선한 실험이 아니라, 자동 경보를 유지한 채 후보 0 상태의 일반 검토 coverage를 보충한 실험이다.

## 해석 제한

- candidate-zero gate는 TEST label이나 이상 위치를 사용하지 않지만 frozen TEST score 결과에서 간접적으로 유래한다. Exp160을 end-to-end strict TRAIN-only라고 부르지 않는다.
- 네 alpha는 사전 고정 sensitivity이며 TEST 결과로 운영 alpha를 선택하지 않았다.
- 실제 equipment/recipe/sensor/step 메타데이터가 없어 score-shape similarity가 물리적 동질성을 의미하는지는 확인되지 않았다.
- 동일 benchmark를 반복 사용한 후향 실험이며 실제 설비 prospective 성능을 주장할 수 없다.
- 일반 검토 TP/FP와 combined F1은 human-assisted 진단 지표다.

