# Exp160 사전등록 (PREREGISTRATION)

## 연구 구분 및 연구 계약

- **연구 구분**: 소표본 데이터셋 후보 복구를 위한 retrospective counterfactual sensitivity (사후적 반사실적 민감도 분석)
- **제한 정책**: 실제 운영 검증(real-world operational validation) 또는 prospective validation(예측적 검사)으로 표현하거나 보고하지 않는다.

## 점수원 명칭 매핑

내부 key는 기존 결과와의 호환성을 위해 그대로 유지하되, 문서와 사용자용 출력에는 다음 명칭을 함께 쓴다.

| 내부 key | 사용자 명칭 | 알고리즘 설명 |
|---|---|---|
| `rocket_exp40` | 시간 패턴 점수 | ROCKET + KNN local-gap |
| `exp55_best` | 주파수 변화 점수 | Spectrogram + PCA + KNN |
| `exp56_best` | 반복 구조 점수 | Recurrence Plot + GLCM + PCA + KNN |
| `exp84` | 다중 커널 점수 | MultiROCKET + HYDRA + stable-tail pruning + KNN local-gap |

*주의: 새 코드에서 `exp55`, `exp56`, `exp84`만 사용자 표시 문자열로 노출하지 않으며, 한글 명칭과 내부 키를 명확히 함께 지칭한다.*

## 고정 연구 계약 (Fixed Contract)

1. **Alpha 그리드 유지**: `ALPHAS = (0.005, 0.01, 0.02, 0.05)`를 그대로 사용하며, TEST 결과에 따라 특정 alpha를 임의 선택하지 않는다.
2. **Frozen Baseline**: Exp152 B2 prediction 및 manifest를 기준으로 동결하여 재사용하며, 어떠한 기존 수정 사항도 되돌리거나 정리하지 않는다.
3. **Target Gate**: per dataset/alpha별로, frozen Exp152 `candidate_count == 0`이고 `n_train >= 5`인 경우에만 target으로 지정한다.
4. **Calibration 기권**: `n_train < 5`인 데이터셋은 complete calibration abstention을 적용하고 후보를 생성하지 않는다.
5. **Candidate-Zero Gate의 성격**: TEST label이 아닌 frozen prediction 결과를 사용하지만 TEST score에서 간접적으로 유래한다. prediction row에 `uses_test_candidate_empty_gate=1`을 기록하고 end-to-end strict TRAIN-only라고 표현하지 않는다.
6. **TRAIN-only 정보 사용**: similarity descriptor, donor selection, calibration reference에는 TRAIN OOF score만 사용한다.
7. **TEST 및 메타데이터 정보 사용 제한**: TEST label, 실제 이상 위치, TEST 길이, family 이름, dataset 이름 기반 정책 분기, 과거 TEST 성능을 이웃·threshold·candidate·lane 결정에 사용하지 않는다.
8. **데이터셋 명칭의 역할**: dataset 이름은 조회 key와 출력 식별자로만 사용한다.
9. **자가 참조 배제 (Target Exclusion)**: target dataset은 donor reference에서 제외한다.
10. **최소 레퍼런스 크기**: 각 점수원별 finite pooled reference가 최소 200개여야 한다.
11. **합의 규칙**: 네 점수원 중 같은 index에서 2개 이상이 `p <= alpha`인 경우에만 similarity 후보가 된다.
12. **Standard Review 전용 추가**: 새 후보는 Standard review에만 추가한다. Hard와 Priority는 frozen Exp152와 exact match여야 한다.
13. **Non-Target exact match**: non-target dataset/alpha의 모든 lane은 frozen Exp152와 exact match여야 한다.
14. **Prediction과 TEST 정보 차단**: prediction 함수는 `y_test`를 받거나 TEST label을 읽지 않는다.
15. **Prediction 출력 제한**: prediction JSONL에는 TP/FP/FN/precision/recall/F1/combined metric을 저장하지 않는다.
16. **Evaluation 분리**: evaluation은 frozen prediction SHA256을 검증한 뒤 별도 단계에서만 TEST label을 읽는다.
17. **규칙 엄격성**: Priority review를 Hard alert로 승격하지 않으며, tail replacement와 TEST-length top-N budget을 사용하지 않는다.
18. **사후 튜닝 금지**: 결과가 나쁘더라도 상수나 규칙을 임의로 수정하지 않는다.
