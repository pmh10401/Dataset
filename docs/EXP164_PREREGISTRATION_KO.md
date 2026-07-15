# Exp164 사전 등록서 (Preregistration)

본 문서는 Exp164 실험의 설계와 절대 금지 조항을 작업 시작 전 동결하기 위해 작성된 사전 등록서이다.

## 1. 목적 및 가설

Exp164는 Exp163에서 제안된 세 가지 대안 점수원(미분·스펙트럼 통계, 부분 형태 일치, 구간 분포)의 `2-of-3` 및 `3-of-3` 합의 규칙이 정상 TRAIN run 환경에서 얼마나 자주 오경보(일반 검토 후보 지정)를 발생시키는지 측정하여 교정 감사(Calibration Audit)를 수행한다.

- **가설 1**: TRAIN normal pseudo-run을 이용한 교차적합 하에서, alpha의 보수성이 높을수록(예: 0.005) 정상 false-review rate가 유의하게 감소할 것이다.
- **가설 2**: `3-of-3` 합의 규칙은 `2-of-3` 합의 규칙에 비해 false-review rate가 대폭 감소하지만, 기권(Abstention) 및 민감도(Sensitivity)에 미치는 영향 또한 클 것이다.

## 2. 동결 입력 및 데이터 계약

- **DB 경로**: `/Users/minho/Documents/Dataset/univariate_ts.db`
- **DB SHA256**: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- **Exp162 Contract SHA256**: `0a80d8a26387d81cd7b2e18ae36f3116c829d20742dd1fa79bb8a3c5bc3d6461`
- **Exp162 Manifest SHA256**: `ec8e1a1c617aa2216a81da1959bd1aec529ad7aadb908ababe3f7e05b7f288c1`
- **대상 인스턴스 (Population)**: Exp162의 1,117개 데이터셋
- **동결 alphas**: `(0.005, 0.01, 0.02, 0.05)`
- **동결 variants**: `train_audit_2of3`, `train_audit_3of3`
- **위험 sensitivity targets**: `(0.005, 0.01, 0.02)` per normal run
- **One-sided Wilson 신뢰수준**: `0.95`
- **폴드 난수 시드 (Fold Seed)**: `20260715`
- **Abstention 규칙**: `n_train < 5`인 경우 완전 기권으로 처리

## 3. 절대 금지 조항 (Strict Constraints)

- **TEST split 값 및 TEST label 조회 불가**: SQL에서 어떠한 방식으로도 TEST split의 values_blob 이나 label, anomaly_position을 읽어오거나 추론에 사용하지 않는다. (AST 파싱 및 Constant 구조 테스트로 검증함)
- **Exp160/163 candidate-zero 또는 lane 결과 사용 금지**: Exp164는 오직 TRAIN normal의 자체 교차적합 정보만을 사용하여 위험도를 분석한다.
- **Family 정보 및 과거 TEST 성능 정보 사용 금지**: 도메인/가상 설비 분류 기준(Family)이나 데이터셋별 과거 성능 점수를 사용하지 않는다.
- **TEST 결과를 기반으로 한 파라미터 선택 금지**: 본 실험은 alpha, variant, threshold, risk target 등을 최적화하여 "선택"하는 실험이 아니며, 오직 모든 타겟에 대해 sensitivity 분석 데이터만을 보고한다.
- **Hard/Priority lane 생성 금지**: 운영 검토 위험도만을 평가하므로 신규 레인(Lane) 분기를 생성하지 않는다.
- **실패한 데이터셋 예외 처리 금지**: 누락되거나 에러가 난 데이터셋을 빈 결과로 채워 성공한 것으로 처리하는 것을 절대 금지한다.

## 4. 교차적합 (Cross-Fitting) 구조

- **기권**: `n_train < 5`
- **Leave-One-Out (LOO)**: `5 <= n_train < 20` 환경에서 deterministic하게 1개씩 query로 사용하여 총 n_train개의 fold를 수행한다.
- **Shuffled 5-Fold**: `n_train >= 20` 환경에서 `seed 20260715`를 이용하여 shuffled deterministic 5-fold를 구성한다.
- **Query Pseudo-run의 완벽한 격리**:
  - 각 fold에서 query pseudo-run은 feature fitting, target length 결정, scaling(RobustScaler), score center/spread 계산, 그리고 threshold reference 결정에서 완벽하게 제외된다.
  - target length는 오직 각 fold의 fit raw series 길이들의 median을 기준으로 계산되고 `[8, 2048]` 범위로 클리핑된다.
  - 각 alpha에 대해 fold fit score의 `np.quantile(..., 1-alpha, method="higher")`를 계산하여 threshold로 삼는다.
  - `query_score > threshold` (strict 초과) 인 경우에만 해당 소스가 후보를 지목한 것으로 flag 처리한다.
