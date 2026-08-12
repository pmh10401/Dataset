# 공개 이상 탐지 기법 비교 사전등록서

## 1. 목적과 논문 내 위치

이 비교의 목적은 사내 개발 버전끼리 우열을 가리는 것이 아니라, 제안기술
`AI-Driven Equipment Intelligence: Equipment Anomaly Detection`을 공개된
one-class 이상 탐지 기법과 동일 조건에서 비교하는 것이다.

- 외부 비교군: kNN distance, LOF, One-Class SVM, Isolation Forest,
  Deep SVDD, DROCC
- 제안기술: Exp448에서 동결된 안정성 선별 적용 최종 구성
- 내부 Exp404/416, Exp442, Exp448: 제안기술 계보 또는 ablation에만 사용
- 현재 상태: retrospective research candidate
- 수상 가능성 최종 판단: **[평가 완료 후 기입]**

어떤 TEST 결과도 아래 방법, 하이퍼파라미터, alpha 또는 지원 데이터셋을
바꾸는 데 사용하지 않는다.

## 2. 평가 모집단과 데이터 경계

- DB: `/Users/mac/Documents/Dataset/univariate_ts.db`
- DB SHA-256:
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- 모집단: 라벨 접근 횟수가 0으로 검증된 Exp446 공개 산출물의 순서에 있는
  정확히 1,117개 데이터셋
- DB에 존재하나 Exp446 모집단에 없는 2개 항목은 평가에서 제외
- 학습: 정상 TRAIN만 사용
- 점수 산출: TEST의 값과 연속 `instance_index`만 사용
- 평가 라벨: 모든 score shard와 manifest가 동결·검증된 뒤 별도 모듈에서
  처음 접근

전처리 길이는 TRAIN 길이의 중앙값으로만 결정하고 `[8, 2048]`로 제한한다.
비유한 값은 시계열 내부 선형 보간으로 복구하며, 전부 비유한 행은 0으로
복구한다. 모든 행은 독립적으로 z-normalization한다. TEST 값은 TRAIN
전처리 상태나 calibration score를 바꿀 수 없다.

## 3. 공개 비교 기법과 고정 구현

| 논문 표기 | 공개 근거 | 고정 구현 |
|---|---|---|
| kNN distance | Ramaswamy et al., SIGMOD 2000 | `k=min(5,n_train)`, 최근접 정상 이웃 거리의 평균을 anomaly score로 사용 |
| LOF | Breunig et al., SIGMOD 2000 | novelty mode, `k=min(20,n_train-1)`, `-score_samples` |
| RBF One-Class SVM | Schölkopf et al., Neural Computation 2001 | RBF, `gamma=scale`, `nu=0.01`, `-score_samples` |
| Isolation Forest | Liu et al., ICDM 2008 | 256 trees, auto samples/contamination, 고정 seed, `-score_samples` |
| Deep SVDD | Ruff et al., ICML 2018 | bias-free 1D encoder, 16차원 embedding, center distance, Adam 50 epochs |
| DROCC | Goyal et al., ICML 2020 | 동일 1D encoder, warm-up 10 + robust 40 epochs, annulus `[1,2]`, 20 projected steps |

시계열 비교를 위해 Deep SVDD와 DROCC에는 동일한 사전등록 1D encoder를
사용한다. 이는 원 논문의 목적함수를 보존한 domain adapter이며 원 논문의
이미지 network를 그대로 재현했다는 의미가 아니다. 작은 TRAIN 때문에
방법을 실행할 수 없을 때 다른 detector로 대체하지 않고 명시적으로
abstain한다. 심층 기법은 `n_train < 8`, LOF는 `n_train < 3`일 때 abstain한다.

## 4. 점수 보정과 경보 규칙

각 방법의 정상 TRAIN reference score는 고정 seed의 shuffled 5-fold OOF로
생성한다. TRAIN 행을 자신에게 fit한 모델로 점수화하지 않는다. 전체 TRAIN에
별도로 fit한 모델이 TEST score를 만든다. 모든 점수는 클수록 이상으로
방향을 맞춘다.

사전등록 alpha는 `0.005`, `0.01`, `0.02`이며 하나만 사후 선택하지 않는다.
upper-tail conformal p-value는 다음과 같다.

```text
p(s) = (1 + #{r in TRAIN-OOF : r >= s}) / (N_reference + 1)
alert iff p(s) <= alpha
```

제안기술과 경보 수가 다른 영향을 분리하기 위해, 각 데이터셋에서 제안기술의
경보 수와 같은 `k`개를 baseline score 상위부터 선택하는 matched-alert-budget
결과도 함께 보고한다. 동점은 TEST `instance_index` 오름차순으로 해소한다.

## 5. 평가 지표와 통계

주 표에는 다음을 모두 포함한다.

- 전체 1,117개 population과 모든 방법이 완주한 common support
- coverage, abstention, failure와 경보 수
- TP, FP, FN, TN, precision, recall, micro F1, FPR
- 데이터셋 균형 macro precision, recall, F1, FPR
- 연속 score의 Average Precision와 AUROC
- 제안기술 대비 데이터셋 단위 paired 차이와 동일 bootstrap draw의 95% CI

비교군 실행 실패나 abstention을 제외한 표만 단독으로 제시하지 않는다.
AP/AUROC가 정의되지 않는 단일-class 데이터셋은 해당 데이터셋 macro 계산에서
명시적으로 제외하고 정의 가능 데이터셋 수를 병기한다.

## 6. Artifact 및 누수 방지

수명주기는 `prepare → assess-mlx → score → freeze → evaluate → report`이다.
score package는 label reader를 import하지 않는다. score shard에 label,
ground truth, TP/FP/FN/TN, precision, recall, F1, AP, AUROC 필드가 들어가면
즉시 실패한다. 모든 JSON은 canonical UTF-8로 기록하고, DB·코드·설정·shard
SHA-256을 묶는다. 기존 파일은 동일 byte일 때만 재사용하며 변경된 파일을
덮어쓰지 않는다.

AGENTS.md의 MLX 정책에 따라 공식 scikit-learn estimator는 MLX로 재작성하지
않는다. Deep SVDD와 DROCC도 동결된 PyTorch 초기화, optimizer, autograd,
학습 경로와 정확히 같은 MLX 구현이 없으므로 조건 1을 통과하지 못한 CPU
fallback으로 기록한다. 따라서 거짓 parity나 속도 향상 수치는 보고하지 않는다.

## 7. 공개 원문

- S. Ramaswamy, R. Rastogi, K. Shim, “Efficient Algorithms for Mining
  Outliers from Large Data Sets,” SIGMOD 2000,
  [DOI 10.1145/335191.335437](https://doi.org/10.1145/335191.335437).
- M. M. Breunig et al., “LOF: Identifying Density-Based Local Outliers,”
  SIGMOD 2000,
  [DOI 10.1145/335191.335388](https://doi.org/10.1145/335191.335388).
- B. Schölkopf et al., “Estimating the Support of a High-Dimensional
  Distribution,” Neural Computation 13(7), 2001,
  [Google Research publication](https://research.google/pubs/estimating-the-support-of-a-high-dimensional-distribution/).
- F. T. Liu, K. M. Ting, Z.-H. Zhou, “Isolation Forest,” ICDM 2008,
  [DOI 10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).
- L. Ruff et al., “Deep One-Class Classification,” ICML 2018,
  [PMLR 80:4393–4402](https://proceedings.mlr.press/v80/ruff18a.html).
- S. Goyal et al., “DROCC: Deep Robust One-Class Classification,” ICML
  2020, [PMLR 119:3711–3721](https://proceedings.mlr.press/v119/goyal20c.html).
