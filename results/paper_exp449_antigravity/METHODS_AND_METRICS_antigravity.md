# 방법 및 지표 정의

## 1. 제안기술의 연속 점수

제안기술의 연속 점수는 이진 경보를 새로 학습한 별도 모델의 출력이 아니다.
동결된 경보 집합은 그대로 두고, 경보 내부와 비경보 내부에서 증거 강도로 순위를
만드는 결정론적 점수다.

1. 다섯 hard 원천의 upper-tail p-value를 `p_j`라 한다.
2. `u_j = 1 - p_j`로 방향을 “클수록 이상”으로 통일한다.
3. 2-of-5 규칙의 연속 대응으로 두 번째로 큰 `u_j`를 hard 강도로 둔다.
4. 저주파 증거는 유효 depth 중 두 번째로 강한 upper-tail 증거를 사용한다.
5. 두 강도의 최댓값을 최종 증거 강도 `u`로 사용한다.
6. 경보이면 `[0.5, 1)`, 비경보이면 `[0, 0.5)`에 배치한다.

```text
score = 0.5 + 0.5u  (alert)
score = 0.5u        (non-alert)
```

결과적으로 `score >= 0.5`와 공식 이진 경보가 일치한다. 이 점수의 의미는
**composite anomaly ranking**이며, 고장 확률·잔여수명·위험률이 아니다.

## 2. 공개 기준선

| 방법 | 고정 점수 정의 | 주요 설정 |
|---|---|---|
| kNN distance | 정상 최근접 이웃 거리 평균 | `k=min(5,n_train)` |
| LOF novelty | `-score_samples` | `k=min(20,n_train-1)` |
| RBF One-Class SVM | `-score_samples` | RBF, `gamma=scale`, `nu=0.01` |
| Isolation Forest | `-score_samples` | 256 trees, 고정 seed |
| Deep SVDD 1D | 정상 중심까지의 임베딩 거리 | bias-free 1D encoder, 16차원, 50 epochs |
| DROCC 1D | 강건 one-class 경계 점수 | warm-up 10 + robust 40 epochs |

모든 기준선은 정상 TRAIN만 학습한다. calibration reference는 고정 seed의
5-fold TRAIN OOF 점수이며, TEST 점수는 전체 TRAIN으로 별도 fit한 모델에서
얻는다. 연속 비교용 공개 기준선 점수는 `1 - upper-tail p-value`다.

## 3. 경보 프로토콜

### Matched alert budget

각 데이터셋에서 제안기술이 발생시킨 경보 수와 같은 `k`개의 기준선 점수를
선택한다. 동점은 TEST 인덱스 오름차순으로 해소한다. 경보 수 차이와 순위 품질을
분리하는 주 비교다.

### Conformal

```text
p(s) = (1 + #{r in TRAIN-OOF : r >= s}) / (N_reference + 1)
alert iff p(s) <= alpha
alpha ∈ {0.005, 0.01, 0.02}
```

세 alpha를 모두 보고하며 하나만 사후 선택하지 않는다.

## 4. 평가 지표

- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1: precision과 recall의 조화평균
- FPR: `FP / (FP + TN)`
- AP: Precision–Recall 곡선의 ranking 요약
- AUROC: 모든 임계값에서 TPR 대 FPR의 ranking 요약
- Micro: 모든 TEST 행의 confusion count를 합쳐 계산
- Macro: 데이터셋별 지표를 동일 가중 평균

AP는 이상 비율이 낮은 문제에서 특히 중요하다. AUROC는 전체 순위 분리를
보여주지만 class imbalance가 클 때 단독으로 사용하지 않는다. 따라서 본 패키지는
AP와 AUROC, ROC와 Precision–Recall 곡선을 함께 제공한다.

## 5. 지원 모집단

- Full population: 1,117개 데이터셋, 195,900개 TEST 행
- Seven-method common support: 922개 데이터셋, 186,150개 TEST 행
- 공통지원 이상 비율: 2.0%

Deep SVDD와 DROCC는 922개 데이터셋만 완주했다. 이들의 full-population 표기
AP/AUROC도 실제 연속 점수가 존재하는 922개 데이터셋에서 계산되므로, 방법 간
연속 점수의 가장 공정한 직접 비교는 공통지원 표와 ROC/PR 곡선이다.

## 6. Bootstrap

데이터셋 1,117개를 동일한 draw로 2,000회 재표집한다. 추정량은 데이터셋 단위
Macro F1의 `기준선−제안기술` 차이다. 음수이면 제안기술이 높다는 뜻이다. seed는
`20260812`이며 draw identity는 표 3의 SHA-256으로 고정된다.
