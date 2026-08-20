# Exp450 사전등록: 재표본화와 이미지 구성요소 ablation

## 고정된 설계

- 모집단은 동결된 SQLite의 1,117개 데이터셋과 TEST 195,900행이다.
- 조건 순서는 `A`, `B`, `C`, `D`로 고정한다.
- A: 선형 길이 정렬 + 선형 이미지 제어군, B: polyphase/PCHIP 길이 정렬 + 선형 이미지, C: 선형 길이 정렬 + area 이미지, D: polyphase/PCHIP 길이 정렬 + area 이미지다.
- 모든 조건은 같은 `hard_plus_stable` 정책, 소스, seed, 임계값과 TEST 좌표를 사용한다.

## 가설

- H-A: A는 동결된 Exp449 연속-score 제어 의미론을 유지한다.
- H-B: B−A는 길이 재표본화 요인만의 차이다.
- H-C: C−A는 이미지 area-pooling 요인만의 차이다.
- H-D: D는 두 요인의 결합이며, B/C와의 paired 차이도 함께 기술한다.

## 평가와 그림 축

- 데이터셋별, macro, micro Average Precision(AP)와 AUROC를 보고한다. AP와 AUROC의 제외 분모도 함께 보고한다.
- paired 표는 동일 데이터셋의 AP/AUROC 차이와 사전 고정 bootstrap을 사용한다.
- ROC의 x축은 False-positive rate, y축은 True-positive rate다.
- PR의 x축은 Recall, y축은 Precision이다. 좌표는 sklearn이 반환한 짝을 순서 변경 없이 저장한다.
- A–D와 여섯 공개 baseline(`knn_distance`, `lof_novelty`, `ocsvm_rbf`, `isolation_forest`, `deep_svdd_1d`, `drocc_1d`)을 모두 표시한다.

## 선택 금지

이 ablation에는 TEST 결과를 이용한 조건 재선택, champion 선택, 임계값 변경, 자동 승격 또는 운영 배포가 없다. 결과는 설명적 ablation이며 후속 독립 검증 전 자동 promotion 근거가 아니다.
