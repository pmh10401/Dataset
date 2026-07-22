# Exp206-210 사전등록 과학적 계약 (Scientific Contract)

## 1. 개요 및 목적
본 실험 배치(Exp206-210)는 Exp137의 자율 alert 중 Hard-alert Zero-F1을 기록하는 `train_normal_count >= 100` 조건의 데이터셋들에 대하여, TRAIN-normal 상태로 캘리브레이션된 연속형 앙상블/퓨전(Continuous Fusion) 방식이 추가적인 잔여 이상 탐지력(residual anomaly coverage)을 제공할 수 있는지 평가합니다.

본 실험은 Exp137의 실시간 자율 alert 규칙을 직접 변경하지 않으며, 사후적 counterfactual 진단 신호로만 동작합니다.

## 2. 데이터 그레인(Data-Grain) 계약
- **1개 데이터셋 (Dataset)**: 1개의 장비/센서/스텝 감지 환경을 나타냅니다.
- **1개 인스턴스 (Instance)**: 1개의 웨이퍼 trace 데이터(TRAIN 또는 TEST)를 의미합니다.
- **후보 인덱스 (Candidate Index)**: 개별 TEST trace 인스턴스를 식별하는 번호이며, 시계열 내부의 타임스탬프가 아닙니다.
- **인접 인덱스 제한**: 인덱스 간 순서나 인접성에 따른 시간적 허용 오차 매칭, 이벤트 클러스터링, 꼬리 대체(tail replacement) 등은 완전히 금지됩니다.

## 3. 엄격한 레이블 격리 및 하이퍼파라미터 계약
- **TEST 레이블 독립성**: 모든 예측 과정(Stage 206~209)에서 TEST 레이블(`y_test`) 또는 이상 위치 정보에 접근하거나 읽을 수 없습니다.
- **캘리브레이션 데이터**: 임계치 설정, 가중치 산정, conformal reference는 오직 Exp206의 TRAIN OOF(Out-Of-Fold) 스코어만을 활용하여 결정됩니다.
- **의사결정 변수 차단**: 데이터셋 이름, 계열(family) 이름, TEST 데이터 길이 등은 정책이나 임계치 결정에 영향을 미칠 수 없습니다.
- **고정된 알파 그리드**: 알파 임계치 그리드는 `(0.005, 0.01, 0.02, 0.05)`로 완전히 고정되며, TEST 메트릭을 바탕으로 알파를 선택하지 않습니다.

## 4. 리스크 계약 (Risk Contracts)
TRAIN-OOF 상의 pseudo-normal false-alert 발생률의 Wilson score upper bound를 기준으로 다음과 같이 신호 등급을 분류합니다.
- **shadow_hard_diagnostic (그림자 진단)**: Wilson upper bound가 `<= 0.001`을 만족하는 경우.
- **standard_review_diagnostic (검토 진단)**: Wilson upper bound가 `<= 0.005`를 만족하는 경우.
- **research_sensitivity_only (연구용)**: 위의 리스크 요건을 충족하지 못하는 모든 경우.
