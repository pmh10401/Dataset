# 공개 이상탐지 기법 논문 비교 설계

## 1. 목적

사내 논문의 제안기술을 공개된 one-class 이상탐지 기법과 동일한 조건에서
비교한다. 저장소의 Exp404/416, Exp442, Exp448은 모두 저자가 개발한 동일
제안기술의 내부 계보이므로 외부 baseline으로 사용하지 않는다.

이 설계의 최종 산출물은 다음 세 종류다.

1. 공개 기법별 동결 예측·점수와 재현 manifest
2. 제안기술 최종 구성과 공개 기법의 공정한 성능 비교표
3. 제안기술 내부 구성요소의 ablation 표

실제 반도체 설비 효과, 조기 경보 시간, 비용 절감 및 양산 적용 효과는 이
실험의 범위가 아니며 별도 현장 검증 칸으로 남긴다.

## 2. 기준 코드와 연구 경계

- 기준 커밋: `2f5229ccd471bba907562b3131d293b46b0b3afc`
- 기준 결과: Exp448의 `hard_plus_stable_26`
- Exp449 이후 미평가 코드는 비교 실험에 포함하지 않는다.
- 공개 baseline 결과는 기존 논문의 수치를 옮기지 않고 동일한 1,117개
  데이터셋에서 새로 계산한다.
- Exp448 결과는 공개 baseline의 설정 선택이나 threshold 선택에 사용하지
  않는다. 단, 별도로 선언한 matched-alert-budget 진단에서 경보 개수만
  사용하며 TEST label은 사용하지 않는다.

## 3. 문제 정의

평가 단위는 개별 시계열 전체다. 각 데이터셋에서 정상 시계열만 포함된
`TRAIN` 집합으로 정상 상태를 학습하고, `TEST`의 각 시계열을 정상 또는
이상 상태로 판정한다. 따라서 주된 비교 대상은 한 개의 긴 시계열 내부의
이상 시점을 찾는 point-wise detector가 아니라, 정상 예시만 학습하는
one-class 또는 novelty detector다.

입력은 다음 계약을 따른다.

- `X_train`: 정상 시계열만 포함하는 2차원 배열
- `X_test`: 정상과 이상 시계열이 섞인 2차원 배열
- `y_test`: 예측 동결 이후 평가 단계에서만 접근하는 이진 label
- population: 정확히 1,117개 데이터셋
- 데이터셋 순서와 각 TEST 행의 index는 동결한다.

## 4. 비교 접근안

### 접근안 A: 고전 기법만 비교

k-NN, LOF, One-Class SVM, Isolation Forest만 실행한다. 가장 빠르고 구현
위험이 낮지만, 심층 one-class 기법과 비교하지 않아 논문 방어력이 약하다.

### 접근안 B: 고전 기법과 대표 심층 기법 비교 — 채택

네 개의 고전 기법에 Deep SVDD와 DROCC를 추가한다. 거리, 밀도, 경계,
분리, 심층 경계, 강건 심층 학습을 모두 포함하면서 범위를 여섯 기법으로
제한한다. 사내 논문의 기술적 설득력과 실행 비용 사이 균형이 가장 좋다.

### 접근안 C: 대규모 benchmark suite

10개 이상의 재구성·예측·Transformer 계열까지 포함한다. 현재 문제와 다른
point-wise detector를 인스턴스 단위로 바꾸는 추가 규칙이 필요하고, 논문
핵심보다 비교 구현이 커지므로 이번 범위에서 제외한다.

## 5. 동결 공개 비교군

| ID | 기법 | 계열 | 공개 근거 |
|---|---|---|---|
| `knn_distance` | k-nearest-neighbor distance | 거리 | Ramaswamy et al., SIGMOD 2000 |
| `lof_novelty` | Local Outlier Factor | 국소 밀도 | Breunig et al., SIGMOD 2000 |
| `ocsvm_rbf` | One-Class SVM | 경계 | Schölkopf et al., Neural Computation 2001 |
| `isolation_forest` | Isolation Forest | 분리 | Liu et al., ICDM 2008 |
| `deep_svdd_1d` | Deep SVDD | 심층 경계 | Ruff et al., ICML 2018 |
| `drocc_1d` | DROCC | 강건 심층 one-class | Goyal et al., ICML 2020 |

공개 기법의 목적함수와 score 정의를 유지한다. 시계열 입력 차원을 맞추기
위한 1D encoder 또는 배열 shape 변경 이외에 TEST 결과를 보고 구조를
변경하지 않는다. 공식 구현을 사용할 수 있으면 공식 구현을 우선한다.

## 6. 전처리와 hyperparameter 정책

### 공통 전처리

1. 비유한 값은 TRAIN에서 계산한 유한 대체값으로 처리한다.
2. 각 시계열은 기존 benchmark 계약과 동일하게 z-normalization한다.
3. feature scaler가 필요한 기법은 TRAIN에만 fit하고 TEST에는 transform만
   적용한다.
4. 행 길이 보정, index 정렬 및 dataset 순서는 예측 전에 hash로 동결한다.

### Hyperparameter 원칙

- 논문 또는 공식 구현의 기본값을 우선한다.
- 데이터 크기 때문에 유효 범위를 벗어나는 이웃 수만
  `min(configured_k, n_train - 1)`로 축소한다.
- seed는 모든 확률 기법에서 고정하고 manifest에 기록한다.
- TEST metric으로 기법별 hyperparameter를 고르지 않는다.
- 필요한 sensitivity는 사전등록한 값 전체를 보고하며 최고 TEST 값만
  골라 본문에 쓰지 않는다.

### 학습 표본 부족

- 수학적으로 실행 가능한 기법은 축소된 유효 이웃 수로 실행한다.
- 실행 불가능한 기법은 임의의 대체 detector로 바꾸지 않고 `abstain`으로
  기록한다.
- 전체 population 결과에서는 abstention을 무경보로 계산하고, 별도로 모든
  비교 기법이 실행된 common-support 결과와 coverage를 보고한다.

## 7. Score와 threshold 비교 규칙

서로 다른 기법의 native threshold를 그대로 쓰면 경보량 차이가 성능
차이로 섞인다. 따라서 score 품질과 운영 경보 품질을 분리한다.

### 7.1 Score 품질

- TEST label은 평가 단계에서만 열어 AUROC와 Average Precision(AP)을
  계산한다. 기존 자산과의 연결을 위해 사다리꼴 AUC-PR을 추가하면 AP와
  별도 열·별도 이름으로 보고한다.
- 데이터셋별 macro와 전체 pooled 결과를 모두 보고한다.
- 제안기술이 단일 연속 score를 제공하지 않는 경우, score 지표는 공개
  기법끼리의 보조 비교로 명시하고 제안기술과의 주 비교에는 사용하지
  않는다.

### 7.2 TRAIN-only 자동 경보

- 정상 TRAIN의 out-of-fold score로 경험적 conformal p-value를 보정한다.
- 사전등록 alpha `0.005`, `0.01`, `0.02`를 모두 보고한다.
- alpha는 TEST metric으로 선택하지 않는다.
- 이 결과는 실제 자동 경보 가능성을 보여주는 주 비교다.

### 7.3 Matched alert budget 진단

- 각 데이터셋에서 Exp448 최종 구성의 경보 개수와 같은 수의 상위 baseline
  score를 선택한다.
- 경보 개수만 사용하고 위치나 label은 사용하지 않는다.
- 이는 동일 검토 비용에서 순위 품질을 비교하는 진단이며 autonomous
  deployment 결과로 표현하지 않는다.

## 8. 평가 지표와 통계

각 기법과 threshold 규칙에 대해 다음을 저장한다.

- TP, FP, FN, TN 및 총 경보 수
- precision, recall, micro F1, false-positive rate
- 데이터셋 균형 macro precision, recall, F1, false-positive rate
- AUROC와 Average Precision(AP), 선택적으로 별도 표기한 사다리꼴 AUC-PR
- 데이터셋 coverage, abstention 및 failure count
- Exp448 최종 구성 대비 데이터셋 단위 F1 차이
- 동일 bootstrap draw를 사용한 paired 95% confidence interval
- 필요 시 Friedman 검정 후 사전 정의된 쌍별 Wilcoxon 검정과 다중비교 보정

주 성능표는 전체 population과 common support를 함께 제공한다. 어느 한쪽만
제시해 실행 실패를 숨기지 않는다.

## 9. 데이터 누수 차단과 artifact lifecycle

파이프라인은 다음 단계로 분리한다.

1. `prepare`: DB identity, population, dependency, method config 동결
2. `score`: TRAIN과 TEST feature/score 계산, label 접근 금지
3. `freeze`: dataset/index/score/config hash와 manifest 생성
4. `evaluate`: 동결 hash 검증 후 처음으로 TEST label 접근
5. `report`: immutable 결과에서 표와 bootstrap 재집계

`score` 단계의 모듈은 label loader를 import하지 않는다. 예측 artifact는
dataset, test index, method, score, status만 포함한다. label 또는 TP/FP 관련
필드는 금지한다. 완료 marker, 원자적 dataset shard, resume와 no-clobber를
지원한다.

## 10. MLX 및 실행 정책

AGENTS.md의 acceleration policy를 적용한다.

- scikit-learn 공식 CPU detector를 MLX로 다시 작성하지 않는다. 목적함수나
  수치 경계가 달라져 공개 기법 재현성이 훼손되므로 MLX eligibility의 첫
  조건을 충족하지 못한 것으로 기록한다.
- Deep SVDD와 DROCC도 공식 PyTorch 구현과 정확히 같은 연산·초기화·학습
  경로를 보존할 수 있는지 먼저 평가한다. 불가능하면 CPU를 유지하고 그
  사유를 manifest에 기록한다.
- MLX 후보가 생기면 동일 input/seed CPU-MLX smoke, 모든 alpha의 동일 후보
  index, 유한 출력, 수치 tolerance, device transfer와 synchronize를 포함한
  1.1배 속도 조건을 모두 통과해야 한다.
- MLX 또는 MPS heavy process는 한 번에 하나만 실행한다.
- CPU detector는 유효한 범위에서 최대 7개 dataset worker를 사용한다.

## 11. 구현 경계

새 package는 기존 Exp448 evaluation package를 수정하지 않고 독립적으로
구성한다.

```text
experiments/paper_public_baselines/
  config.py
  data.py
  preprocessing.py
  methods/
    classical.py
    deep_svdd.py
    drocc.py
  scoring.py
  artifacts.py
  evaluation.py
  statistics.py
  reporting.py
  cli.py
  tests/
```

외부 실행 root와 DB path는 CLI 인자로 받고 저장소에 대용량 score/result를
커밋하지 않는다. 저장소에는 코드, 테스트, preregistration, runbook, 작은
요약 보고서와 manifest hash만 남긴다.

## 12. 검증 기준

전체 실행 전에 synthetic smoke에서 다음을 검증한다.

- 동일 seed 재실행의 score와 candidate index가 결정적이다.
- score 단계에서 label access가 불가능하다.
- index 순서 변경, config 변경, score 변조를 freeze/evaluate가 거부한다.
- tiny TRAIN에서 축소 또는 abstention 규칙이 정확히 적용된다.
- alpha별 conformal 경보와 matched-budget 경보가 preregistration과 일치한다.
- 중단 후 완료 shard는 재사용하고 불완전 shard만 다시 계산한다.
- 평가 재집계가 저장된 confusion과 정확히 일치한다.
- MLX manifest에 버전, device, timing, parity와 fallback reason이 기록된다.

## 13. 논문 표기

외부 비교표에서는 공개 기법명과 인용을 그대로 사용한다. 저장소 실험
번호는 제안기술 내부 ablation 표에만 다음 의미로 매핑한다.

| 논문 표기 | 내부 provenance |
|---|---|
| 제안기술 기본 구성 | Exp404/416 계보 |
| 후보 확장 적용 구성 | Exp442 계보 |
| 안정성 선별 적용 제안기술 | Exp448 `hard_plus_stable_26` |

Exp404/416을 `기존 기법`, `public baseline` 또는 타 연구자의 방법으로
표현하지 않는다.

## 14. 참고 문헌

- Ramaswamy, S., Rastogi, R., and Shim, K. “Efficient Algorithms for Mining
  Outliers from Large Data Sets.” SIGMOD, 2000.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. “LOF: Identifying
  Density-Based Local Outliers.” SIGMOD, 2000.
- Schölkopf, B. et al. “Estimating the Support of a High-Dimensional
  Distribution.” Neural Computation, 2001.
- Liu, F. T., Ting, K. M., and Zhou, Z.-H. “Isolation Forest.” ICDM, 2008.
- Ruff, L. et al. “Deep One-Class Classification.” ICML, 2018.
- Goyal, S. et al. “DROCC: Deep Robust One-Class Classification.” ICML, 2020.
