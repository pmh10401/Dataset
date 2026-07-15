# Exp162 Alternative Score Cache 사전등록 문서 (Preregistration)

본 문서는 Exp162 실험에 대한 공식 사전등록 문서입니다.

## 1. 배경 및 목적
Exp161 실험을 통해 과거 수집된 `미분·스펙트럼 통계 점수`, `부분 형태 일치 점수`, `구간 분포 점수` 산출물이 서로 다른 데이터셋 서브셋 및 설정을 바탕으로 계산되어 상호 비교가 어려움이 확인되었습니다. 본 Exp162 실험은 세 대체 점수원(Alternative Sources)을 **동일한 고정 설정(frozen configuration)**으로 **동일한 1,117개 동결 데이터셋 population** 전체에 대해 사전 빌드(Score Cache)하여, 향후 Exp163 등의 복구 정책 평가 시 계통(family) 편향이나 TEST 성능 기반 유출(leakage) 없이 공정하게 후보군 복구를 검증하기 위한 기초 Retrospective 인프라를 구축하는 것을 목적으로 합니다.

## 2. 실험 설계 및 population 계약
- **동결 데이터셋 population (Frozen Population):** Exp160 prediction 파일 (`outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`, SHA256: `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`)에 명시된 고유한 1,117개 데이터셋입니다.
- **데이터베이스:** `/Users/minho/Documents/Dataset/univariate_ts.db`를 그대로 사용합니다.
- **데이터 로드 무결성 (Leakage Guard):** cache/prediction 단계에서 TEST label과 anomaly position 정보를 조회하거나 함수 인자로 주입받지 않습니다. (values-only SQL 쿼리 적용)
- **메타데이터 계약:** family 이름, dataset 이름, TEST 길이, Exp160 candidate-zero 여부를 feature 추출이나 소스 선택 분기에 일절 배제합니다. Dataset 이름은 오직 조회 Key 및 deterministic seed 생성(`20260714 + stable_hash(dataset_name)`)에만 사용됩니다.
- **길이 정렬 (Alignment):** target length는 TRAIN-normal 시계열 길이들의 median으로 결정하며 `[8, 2048]` 범위로 클리핑합니다. TEST 길이는 모델 및 정렬 길이를 설정하는 데 관여하지 않습니다.

## 3. 동결 소스 설정 (Frozen Configurations)
세 소스는 모든 데이터셋에 동일한 설정으로 계산됩니다.

1. **미분·스펙트럼 통계 점수 (Spectral Derivative):**
   - 시계열 피처: raw, 1차 미분, 2차 미분의 mean, std, median, p90-p10, max-absolute + 8 normalized FFT bands + spectral centroid, entropy, peak.
   - Robust deviation 점수화: TRAIN-normal feature로만 scaling 및 center/spread를 피팅(RobustScaler).
2. **부분 형태 일치 점수 (Shapelet Normal Prototype):**
   - Z-normalized TRAIN 시계열에서 길이 `[8, 16, 32, 64]`인 32개 shapelet prototypes를 결정론적 샘플링(seed: `20260714 + stable_hash`).
   - 32개 prototype들과의 최소 거리(minimum-distance) 피처에 대해 robust deviation score 적용.
3. **구간 분포 점수 (Interval Quantile):**
   - Z-normalized raw, first difference, log-periodogram 3개 뷰 생성.
   - 각 뷰마다 16개 deterministic interval 및 1개 full-series interval (총 17 spans).
   - 각 interval마다 mean, std, min, max, q05, q25, q50, q75, q95, slope(총 10개) 추출.
   - Robust deviation score 적용.

## 4. 금지 사항 및 분석 경계
- threshold, candidate index, lane, TP, FP, FN, F1, anomaly count 등의 통계는 이 단계에서 생성하거나 저장하지 않습니다.
- 실패 데이터셋을 빈 score로 우회하지 않고, 에러 기록을 남겨 full coverage 성공 선언을 방지합니다.
- `n_train < 5`인 데이터셋에 대해서도 score cache는 연산합니다. (calibration 정책 검증은 Exp163의 범위입니다.)

## 5. 평가지표 및 검증 프로세스
- **Smoke Test:** 10개 데이터셋 실행을 통해 deterministic feature output, 6개 finite score arrays, values-only SQL, metadata key validation, resume hash stability 등을 단위 테스트 및 smoke 실행을 통해 선 검증합니다.
- **Verification Report:** full run(1,117 datasets)은 이 구현 범위가 아니며, Smoke Test 완료 후 별도 감사자가 검토를 마친 후 수행 여부를 결정합니다.
