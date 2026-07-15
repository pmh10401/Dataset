# Exp163: 대안 점수원 꼬리 합의 검토 (Alternative-Source Tail Agreement Review) 사전등록 문서

## 1. 목적 및 배경

본 실험은 Exp160에서 이상치 후보가 단 하나도 검출되지 않은(candidate_count == 0) 데이터셋을 대상으로, Exp162에서 산출하여 동결 보관 중인 세 대안 점수원(spectral, shapelet, interval)의 경험적 분위수 임계값에 기반한 꼬리 합의(Tail Agreement)를 적용해 일반 검토(Standard review) 후보를 보충하고자 하는 후향적 반사실 실험이다. 

본 실험의 궁극적인 목적은 자동 경보(Hard) 및 우선 검토(Priority) 레인을 전혀 변경하지 않고(baseline 보존), 신뢰할 수 있는 대안 점수원들의 합의를 통해 일반 검토 대상의 커버리지를 보완하고 작업 부담을 분석하는 것이다.

## 2. 동결 입력 및 무결성 조건

본 실험에 사용되는 입력은 다음과 같으며, 파일의 해시가 일치하지 않거나 1,117개 데이터셋의 커버리지가 불완전할 경우 예측 단계를 실행하지 않고 중단한다.

- **Exp160 예측 결과 파일**: `outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`
  - SHA256: `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`
- **Exp162 대안 점수 캐시 매니페스트**: `outputs/exp162_full_coverage_alternative_score_cache/02_cache_manifest.csv`
  - SHA256: `ec8e1a1c617aa2216a81da1959bd1aec529ad7aadb908ababe3f7e05b7f288c1`
- **Exp162 대안 점수 소스 계약서**: `outputs/exp162_full_coverage_alternative_score_cache/01_source_contract.json`
  - SHA256: `0a80d8a26387d81cd7b2e18ae36f3116c829d20742dd1fa79bb8a3c5bc3d6461`

## 3. 대안 점수원 및 한글 명칭

세 가지 대안 점수원은 상호 규모(scale)가 크게 다르므로 단순 원시 점수 합산이나 평균 연산을 금지하며, 각각 독립적으로 임계값을 연산하여 인덱스를 추출한다.
- `spectral`: **미분·스펙트럼 통계 점수**
- `shapelet`: **부분 형태 일치 점수**
- `interval`: **구간 분포 점수**

## 4. 분위수 임계값 및 후보 생성 규칙

각 데이터셋(dataset), 소스(source), 허용오차(alpha)에 대해 다음 단계를 적용한다:
1. 해당 소스의 finite TRAIN 점수값만 취합한다.
2. 분위수 $q = 1 - \alpha$를 정의한다.
3. 임계값(threshold)은 `numpy.quantile(finite_train_scores, q, method="higher")`로 계산하여 고정한다.
4. 테스트 점수가 임계값을 초과하는(`test_score > threshold`) 테스트 인덱스(TEST index)만 해당 소스의 단독 후보 집합으로 삼는다. 동점(tie)은 strict greater-than 조건에 의해 제외된다.

### 알파 민감도 그리드
- $\alpha \in (0.005, 0.01, 0.02, 0.05)$

## 5. 사전등록 대안 검토 variants (합의 규칙)

- **`tail_rank_2of3`**: 세 소스 중 **2개 이상**의 소스가 동일한 TEST index를 후보로 지정한 경우 최종 합의 후보로 인정한다.
- **`tail_rank_3of3`**: 세 소스 **전부(3개)**가 동일한 TEST index를 후보로 지정한 경우 최종 합의 후보로 인정한다.

두 variant는 모두 독립적인 민감도 분석 결과로써 평가하며, 임의로 사후 최적의 variant를 하나만 선택하는 행위를 금지한다.

## 6. 대상 필터링(Gate) 및 레인(Lane) 통합 규칙

- **적용 대상 (Target Gate)**:
  - $n_{\text{train}} \ge 5$ 이며,
  - Exp160 결과에서 `candidate_count == 0`인 행(row)에 대해서만 대안 후보를 계산한다.
  - $n_{\text{train}} < 5$인 데이터셋은 기각 보정(Abstention) 상태를 유지하여 후보를 전혀 추가하지 않는다.
- **레인 통합**:
  - 자동 경보(Hard) 및 우선 검토(Priority) 레인의 인덱스는 기존 Exp160 결과의 것을 **100% 보존**한다.
  - 신규 합의 후보는 **일반 검토(Standard review)에만 추가**된다.
  - 통합된 일반 검토 레인 = `(기존 일반 검토 | 신규 합의 후보) - 자동 경보 - 우선 검토`
  - 상호 배타성 검증: `Hard`, `Priority`, `Standard` 레인은 항상 서로소(disjoint)여야 하며, 세 레인의 합집합은 전체 `candidate_indices`와 일치해야 한다.
  - `uses_test_candidate_empty_gate = 1`을 기록한다. (테스트 셋의 후보 부존재 여부를 gate 기준으로 사용하므로 본 평가는 retrospective sensitivity 분석이다.)

## 7. 기타 제약 조건

- Top-N 후보 제한이나 총 TEST 길이 버젯(TEST-length budget) 등은 사용하지 않는다.
- 데이터셋명이나 계열(family)명에 기반한 수동 정책 분기 처리를 전면 금지한다.
- 예측(Prediction) 단계에서는 어떠한 형태로도 TEST label, 이상치 실제 위치, 평가 지표(F1, TP, FP 등)에 접근할 수 없다.

## 8. 평가와 해석 경계

Prediction JSONL을 원자적으로 생성하고 SHA256 manifest를 동결한 뒤에만 TEST label을 읽는다. 자동 경보 지표와 일반/우선 검토 및 combined F1을 분리한다. `mean_combined_f1`은 사용자가 검토 후보를 확인했다고 가정한 human-assisted 진단값이다.

모든 alpha와 두 variant를 sensitivity로 함께 보고한다. TEST TP, FP, precision, F1을 근거로 특정 alpha 또는 variant를 primary, optimal, recommended로 선택하지 않는다.

Exp163은 frozen TEST score에서 파생된 candidate-empty gate를 사용한 retrospective counterfactual이다. 성공하더라도 end-to-end strict TRAIN-only 또는 실제 설비 prospective validation으로 표현하지 않는다.
