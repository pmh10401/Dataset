# 논문용 제안기술 재실험 및 연속 점수 설계

## 1. 결정 사항

논문에 필요한 성능 지표는 새 모델을 만들거나 기존 결과 파일을 다시 집계해서
얻지 않는다. 현재 최종 제안기술의 알고리즘, 전처리, threshold, 후보 예산,
경로와 안정성 판정을 고정한 뒤 원본 공개 benchmark DB에서 다시 실행한다.

재실험은 두 출력을 동시에 만든다.

1. 기존 최종 판정을 정확히 재현하는 이진 경보
2. 경보 판정은 바꾸지 않고 원래 점수 근거를 모든 TEST 행에 남기는 연속 근거 점수

이진 경보의 precision, recall, F1과 FPR은 제안기술의 주 결과다. AP와 AUROC는
원래 시스템에 단일 native score가 없다는 사실을 밝힌 상태에서 다음 두 값으로
보고한다.

- 원 판정의 `binary-policy AP/AUROC`: 이진 경보 자체를 score로 사용한 무결성 값
- `route-preserving continuous-evidence AP/AUROC`: 이 설계에서 사전 고정한 측정
  adapter를 모든 비교 기법에 동일하게 적용한 정책 순위 값

연속 근거 점수는 새 학습 모델, 새 이상 판정기 또는 기존 기술의 native score로
표현하지 않는다. 이 구분은 논문 본문, 표 각주와 재현 부록에 모두 유지한다.

## 2. 논문의 검증 주장

이번 데이터는 반도체 설비 실측 데이터가 아니라 UCR one-vs-all 구성과
CornellWhaleChallenge로 만든 공개 시계열 benchmark다. 따라서 이번 재실험이
뒷받침하는 주장은 다음으로 제한한다.

> 공개 one-class 시계열 benchmark에서 정상 TRAIN만으로 기준을 형성하는 제안
> 이상탐지 기술의 핵심 판별 성능을 공개 기법과 비교 검증했다.

반도체 설비 적용 구조, 현장 신호 연결과 운영 활용 방안은 제안할 수 있다. 그러나
설비 고장 예지 성능, 조기 경보 시간, 생산성·비용 효과, 현장 일반화 및 양산 성능은
이번 수치로 주장하지 않는다. 사내 수상 가능성에 대한 최종 판단도 이번 실험
설계의 범위에서 비워 둔다.

과거 TEST를 반복 검토하며 최종 경로가 형성된 이력은 재실행으로 사라지지 않는다.
재실험은 label 접근을 엄격히 차단한 결정론적 재현이지만, 신규 holdout 또는
prospective 검증으로 표현하지 않는다.

## 3. 논문 표기와 내부 provenance

논문 본문에서는 다음 표기를 사용한다.

- 시스템: `AI-Driven Equipment Intelligence`
- 기술: `Equipment Anomaly Detection`
- 비교표: `Proposed method`

`Exp404`, `Exp448`, `Stable-26`, `retrospective research candidate`와 같은 저장소
관리 표현은 논문 제목, 초록과 주 비교표에 쓰지 않는다. 재현 부록에서만 내부
계보를 다음과 같이 연결한다.

| 논문 표기 | 내부 재현 대상 |
|---|---|
| 제안기술 기본 구성 | 최종 Hard 경로 |
| 후보 확장 구성 | Hard와 동결 source 후보 30개 union |
| 안정성 선별 제안기술 | Hard와 안정 후보 26개 union |
| 불안정 후보 진단 | Hard와 불안정 후보 4개 union |

재실험 ID는 논문 버전명이 아니라 artifact provenance로만
`paper-proposal-clean-rerun-20260812`를 사용한다. 제품형 `v1.0`, `Stable-26`
또는 임의 release 명칭은 논문 성능 근거에 붙이지 않는다.

## 4. 동결 평가 모집단

입력 DB는 다음 identity를 검증한 read-only 파일이다.

```text
database = /Users/mac/Documents/Dataset/univariate_ts.db
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
```

모집단은 기존 논문 경계와 같은 정렬된 1,117개 데이터셋이다.
`CornellWhaleChallenge`와 `Wafer_normal_1`은 제외한다. TEST는 정확히 195,900행이며
좌표 identity는 `(dataset, instance_index)`다. post-freeze 평가 모집단에는 이상
3,918행과 정상 191,982행이 있다.

- TRAIN: 각 데이터셋의 정상 행만 학습·보정에 사용
- TEST values: label-free score와 경보 생성에만 사용
- TEST label: prediction과 score가 동결된 뒤 evaluator만 사용
- dataset 이름, 행 순서, TRAIN/TEST 개수와 원본 blob SHA256: prepare manifest에 동결

SQLite TEST label은 exact `str`인 `'0'` 또는 `'1'`만 허용한다. evaluator 경계에서
각각 정수 0과 1로 변환하고, 다른 type·문자열·공백·null은 실패시킨다.

## 5. 알고리즘 복원 경계

### 5.1 계보 lock

구현 전에 최종 경보를 만드는 모든 의존성을 하나의 machine-readable lineage
lock으로 만든다. 각 항목은 다음 세 종류 중 하나여야 한다.

1. `raw-computable`: DB의 정상 TRAIN과 label-free TEST로 원 점수부터 다시 계산
2. `frozen-rule`: 과거에 선택된 alpha, consensus 수, 예산, tie-break, 안정성 기준
3. `parity-oracle`: 재실행 결과가 같음을 확인할 기존 label-free artifact

과거 prediction 좌표를 새 prediction 입력으로 사용하지 않는다. parity oracle은
새 score·route가 완전히 동결된 뒤 비교에만 사용한다. 어떤 최종 route 의존성도
raw-computable 또는 frozen-rule로 복원할 수 없으면 좌표를 복사해 메우지 않고
재실험을 중단한다.

### 5.2 고정 경보 경로

제안기술의 주 구성은 다음을 그대로 재현한다.

```text
Hard 경로
  = 기존 정상-TRAIN 점수원 합의와 lane routing
  + Standard 후보에 대한 MiniROCKET 1024 또는 4096 재확인
  - 같은 데이터셋 정상 TRAIN values_blob과 byte-exact인 TEST 행

Stable 추가 경로
  = 저주파 다중 depth 중 최소 2-depth strict consensus
  + Hard 경보가 0개인 데이터셋만 허용
  + 데이터셋별 가장 강한 후보 1개
  + 전역 상위 30개 예산
  + 최대 16개 정상 TRAIN omission view에서 생존율 0.875 이상

최종 경보 = Hard 경로 OR Stable 추가 경로
```

원래 source별 정규화, 특징, OOF split, seed, score, 경험적 p-value, threshold,
strict 비교, 결측/길이 eligibility 및 tie-break를 변경하지 않는다. source 간에
임의의 공통 전처리를 새로 덮어쓰지 않는다.

### 5.3 필수 parity

1,117개 전체 재실행은 label을 열기 전에 다음 좌표 수와 exact coordinate identity를
동시에 만족해야 한다.

| 구성 | 예상 경보 수 |
|---|---:|
| Hard only | 2,218 |
| Hard + source candidates | 2,248 |
| Hard + stable candidates | 2,244 |
| Hard + unstable candidates | 2,222 |

stable과 unstable은 각각 26개와 4개이며 서로 겹치지 않고 합집합이 source 후보
30개여야 한다. 좌표가 하나라도 다르면 새 결과를 기존 제안기술 성능으로 평가하지
않는다. 원인과 label-free diff만 남기고 새 run identity로 수리한다.

## 6. 연속 근거 점수

### 6.1 원칙

최종 기술은 하나의 scalar score를 threshold하는 구조가 아니라 여러 source의
threshold 판정을 합의·예산·stability route로 결합한다. 따라서 원래 없던 순수
score fusion을 만들어 기존 알고리즘이라고 주장하지 않는다.

대신 각 source가 원래 계산하던 TEST score와 정상 TRAIN OOF reference를 모든 TEST
행에 저장한다. source의 empirical upper-tail p-value는 다음과 같다.

```text
p = (1 + count(TRAIN_OOF_score >= TEST_score)) / (n_OOF + 1)
leaf_margin = log10(alpha / p)
leaf_strength = 1 - p
```

empirical p-value source는 `p <= alpha`, 즉 `leaf_margin >= 0`을 통과로 사용한다.
Exp440처럼 raw score에 strict threshold가 있는 source는 원래의
`TEST_score > threshold`와 그 signed margin을 그대로 사용한다. score가 낮을수록
이상인 source는 원래 구현의 방향을 먼저 통일한다. 경계의 `>=`와 `>`를 하나로
바꾸지 않으며 각 leaf의 native pass Boolean도 함께 저장한다.

원래 Boolean 판정의 연속 margin은 다음 규칙으로만 계산한다.

- `A AND B`: `min(margin_A, margin_B)`
- `A OR B`: `max(margin_A, margin_B)`
- `k-of-n`: 큰 margin부터 정렬한 k번째 값
- 저주파 2-depth consensus: 두 번째로 큰 depth margin

동일한 AND/OR/k-of-n 규칙을 `leaf_strength`에도 적용한다. Hard의 score-source
합의 strength와 저주파 2-depth strength 중 큰 값을 그 행의
`continuous_evidence_strength`로 사용한다. 이 값은 `[0, 1)` 범위이고 공개
baseline과 같은 empirical p-value 척도다. source가 하나도 실행되지 못한 행은
임의 숫자나 index jitter를 넣지 않고 `evidence_missing=true`로 묶인 동일 순위를
사용한다. global budget, Hard-empty gate, byte-exact veto와 stability lane은
수치 근거를 새로 발명하지 않고 최종 이진 route에만 반영한다.

### 6.2 판정 경로 보존 adapter

`continuous_evidence_strength`를 `u`로 사용한다. 근거가 없는 행은 `u = 0.5`로
두어 모두 tie로 처리한다. 최종 이진 경보를 `alert`라고 할 때 논문용 정책
score는 다음으로 고정한다.

```text
route_preserving_score = 0.5 * u                 if alert == 0
route_preserving_score = 0.5 + 0.5 * u           if alert == 1
```

따라서 모든 비경보 score는 0.5 미만이고 모든 경보 score는 0.5 이상이다.
`score >= 0.5`에서 원 경보 좌표가 정확히 재현되며, 각 band 안에서만 원래 연속
근거가 순위를 제공한다. 좌표나 TEST label로 tie를 깨지 않는다.

같은 adapter는 공개 baseline에도 적용한다. baseline의 `u`는 해당 기법의 정상
TRAIN OOF empirical p-value에서 계산한 `1-p`이고, `alert`는 비교표에 사용하는
동일 TRAIN-only 또는 matched-budget 판정이다. 이 동일 변환을 적용한 값끼리만
`route-preserving AP/AUROC` 표에서 비교한다.

### 6.3 함께 보고할 score 결과

1. 모든 기법의 이진 경보를 0/1 score로 계산한 `binary-policy AP/AUROC`
2. 모든 기법에 같은 adapter를 적용한 `route-preserving AP/AUROC`
3. 공개 baseline의 native continuous AP/AUROC
4. 제안기술 각 원 source의 native/empirical-p AP/AUROC와 coverage

3번과 4번은 score 성질을 설명하는 보조표다. 제안기술 source 중 가장 좋은 TEST
결과를 골라 제안기술 대표값으로 대체하지 않는다. AP는
`average_precision_score` 정의를 사용하고 사다리꼴 PR-AUC와 섞지 않는다.

## 7. 주 비교와 ablation

### 7.1 공개 비교군

공개 논문에 근거한 다음 기법만 외부 baseline으로 사용한다.

- k-nearest-neighbor distance
- Local Outlier Factor novelty detection
- RBF One-Class SVM
- Isolation Forest
- Deep SVDD 1D
- DROCC 1D

저장소 안의 다른 버전과 실험은 모두 저자의 미공개 기술이므로 외부 baseline 표에
넣지 않는다. 공개 baseline은 기존 결과 root를 재사용하지 않고 새 root에서 같은
1,117개 모집단과 새로 동결한 환경으로 다시 score한다.

네 고전 기법은 필수 전체 실행이다. Deep SVDD와 DROCC는 합계 48시간의 사전 고정
실행 예산 안에서 수행한다. 심층 기법이 전체 모집단을 끝내지 못하면 부분 수치를
주 비교표에 넣지 않고 failure, runtime과 coverage만 부록에 남긴다. 부분 coverage의
좋은 결과만 선택해 보고하지 않는다.

### 7.2 비교 판정

주 정책 비교는 동일 검토 부담을 위해 데이터셋별 제안기술 경보 수와 같은 수의
baseline 상위 score를 선택하는 matched-alert-budget을 사용한다. 이 규칙은 제안기술
경보 수만 사용하며 위치와 label은 사용하지 않는다.

각 공개 기법의 정상 TRAIN OOF p-value를 alpha `0.005`, `0.01`, `0.02`에서 판정한
TRAIN-only 자동 경보도 모두 부표에 보고한다. TEST 성능으로 alpha를 선택하지 않는다.

### 7.3 제안기술 ablation

주 구성과 함께 다음 네 가지를 같은 frozen score에서 평가한다.

1. Hard only
2. Hard + source candidates 30
3. Hard + stable candidates 26 — 주 제안기술
4. Hard + unstable candidates 4 — 안정성 분리 진단

`unstable 4`는 대안 모델이나 비교 우승 후보가 아니라 stability gate가 분리한
반대편 진단이다. TEST 결과를 본 뒤 stable 수, survival threshold 또는 source 후보
수를 변경하지 않는다.

## 8. 평가 지표와 통계

모든 이진 정책에 대해 다음을 계산한다.

- TP, FP, FN, TN, 경보 수와 coverage
- precision, recall, specificity, FPR, micro F1
- 데이터셋 균형 macro precision, macro recall, macro F1, macro FPR
- dataset별 경보 수, TP, FP, FN, TN

모든 score 결과에 대해 다음을 계산한다.

- pooled AUROC와 Average Precision(AP)
- dataset macro AUROC와 macro AP
- score coverage와 tied/missing row 수

AUROC는 TEST에 두 class가 모두 있는 데이터셋에서만 dataset 값을 정의한다. AP는
양성 TEST가 있는 데이터셋에서만 정의한다. 제외 dataset 수와 이유를 반드시 함께
보고하며, undefined 값을 0이나 1로 대체하지 않는다. pooled 결과는 동결된 전체
행에서 계산한다.

제안기술과 각 baseline 차이는 같은 dataset bootstrap draw를 사용해 20,000회
paired 95% percentile interval로 계산한다. bootstrap 단위는 dataset이고 seed는
`20260812`, bit generator는 `PCG64`, index dtype은 `int64`, endpoint는 false로
고정한다. NumPy 버전과 draw matrix SHA256도 manifest에 기록한다. interval은
반복적으로 본 benchmark에 대한 descriptive uncertainty이며 독립적인 confirmatory
신뢰구간으로 표현하지 않는다.

## 9. 실행 lifecycle과 artifact

사용자 승인에 따라 논문 코드는 최상위 독립 `paper/equipment_anomaly_detection/`
namespace에 둔다. 필요한 제안기술의 과학 로직은 원본 module·symbol·SHA256를 기록한
뒤 paper 파일로 복사하며, 논문 실행 경로는 내부 Exp package를 runtime dependency로
사용하지 않는다. 기존 실험 package와 공개 baseline package는 수정하지 않는다.
이 논문 폴더의 신규 파일은 기존 `antigravity` 명명 대신 `paper`를 파일명에 포함한다.

```text
paper/equipment_anomaly_detection/
  package_paper.py
  config_paper.py
  lineage_paper.py
  lineage_lock_paper.json
  codemap_paper.{html,json,lock}
  data_paper.py
  sources_paper/
  calibration_paper.py
  hard_route_paper.py
  routing_paper.py
  score_adapter_paper.py
  artifacts_paper.py
  evaluation_paper.py
  statistics_paper.py
  reporting_paper.py
  pipeline_paper.py
  tests_paper/
```

사용자가 실행하는 공식 진입점은 `pipeline_paper.py run` 하나다. 내부 과학 함수는
테스트 가능하게 분리하지만 stage별 CLI는 제공하지 않는다. 하나의 pipeline state가
다음 순서를 강제한다.

1. `preflight`: root·DB·code map·환경 identity 검증
2. `prepare`: DB·population·코드·환경·lineage·metric 정의 동결
3. `acceleration`: PyTorch·MLX·MPS 평가와 CPU worker 동결
4. `proposal_score`: raw DB score와 네 route 생성, label 접근 금지
5. `proposal_freeze`: dataset/index/score/route/config hash manifest 생성
6. `proposal_parity`: 기존 label-free 좌표와 수·identity 비교
7. `public_baselines`: 공개 기법 score와 frozen sub-root 생성
8. `baseline_bridge`: 공통 정책 score 동결
9. `evaluate`: 모든 선행 freeze/parity 성공 후 처음으로 TEST label 접근
10. `report`: immutable 평가 artifact에서 표, CI와 그림 재집계
11. `independent_verify`: 원 score module·SQLite 없이 결과 재집계

`pipeline_state_paper.json`은 이 exact-prefix 상태와 각 artifact SHA를 기록하고 최종
`pipeline_manifest_paper.json`이 전 단계를 묶는다. 동일 config/hash의 완료 prefix만
resume하며 failed root, 중간 누락 또는 순서 역전은 재사용하지 않는다. read-only
`status`와 `verify` 외 public command는 없다.

`score`, `route`, `freeze`, `parity` 모듈은 label loader를 import할 수 없다. score
artifact에는 dataset, instance index, source score, OOF p-value, margin, strength,
native pass, eligibility, route와 provenance만 허용한다. label, TP, FP, FN, metric
필드는 금지한다.

대용량 결과는 `/Users/mac/Documents/Dataset/paper_proposal_clean_rerun_20260812`
아래 새 root에 저장하고 Git에는 넣지 않는다. 저장소에는 코드, 테스트, spec,
preregistration, runbook, 작은 manifest/summary와 hash만 남긴다. dataset별 atomic shard,
no-clobber, exact-prefix resume, progress·heartbeat·ETA와 failure evidence를 지원한다.

## 10. 실패 처리

다음 중 하나라도 발생하면 label을 열지 않고 중단한다.

- DB, population, 코드, 환경 또는 frozen-rule identity 불일치
- lineage dependency 누락 또는 old prediction 좌표를 입력으로 요구함
- TRAIN 외 데이터로 fit·scale·threshold를 계산함
- source score가 비유한 값인데 원 eligibility 규칙으로 설명되지 않음
- dataset/index 순서, score 길이 또는 candidate 범위 불일치
- 동일 seed 재실행의 score tolerance 또는 candidate identity 불일치
- 네 ablation의 예상 좌표 수·partition·기존 parity 불일치
- freeze 전 TEST label 또는 이전 evaluation artifact 접근

실패한 root는 덮어쓰거나 같은 identity로 재시도하지 않는다. 원인 수정은 새 run
identity와 새 manifest를 사용한다. 특정 dataset 실패를 조용히 제외하지 않고 전체
population과 common support를 모두 보고한다.

## 11. PyTorch·MLX·MPS와 자원 정책

모든 새 score stage는 전체 실행 전에 PyTorch, MLX, MPS 각각의 tensor eligibility,
availability, transfer/compile overhead와 canonical CPU fallback을 평가한다.

- 원 알고리즘, feature, score, threshold와 route가 바뀌면 MLX를 사용하지 않는다.
- 동일 input/seed CPU-MLX smoke에서 모든 alpha의 candidate index가 같아야 한다.
- score는 모두 finite이고 절대오차 `1e-5` 안이어야 한다.
- transfer와 synchronize를 포함한 실제 stage가 CPU보다 1.1배 이상 빨라야 한다.
- MLX/mlx-metal 버전, device, timing, parity와 fallback 이유를 manifest에 기록한다.

공식 scikit-learn·aeon·PyTorch 구현을 MLX로 다시 작성해 수치 경계를 바꾸지 않는다.
가속 조건을 만족하지 못하면 CPU를 사용한다. 새 공식 CPU stage는 감지 logical CPU
수에서 1을 뺀 값(최소 1)을 dataset worker로 사용하고 감지값과 선택값을 provenance에
기록하며, worker당 수치 thread는 1개다. MLX/MPS heavy process는 한 번에 하나만 실행한다.

## 12. 테스트와 완료 조건

구현은 TDD로 진행하고 최소 다음을 자동 검증한다.

- DB/population/좌표 identity와 exact TEXT label converter
- source별 기존 score·threshold·candidate parity synthetic fixture
- AND/OR/k-of-n margin과 native boundary의 원 Boolean 판정 일치
- adapter의 0.5 경계에서 원 경보 exact reproduction
- missing evidence가 tie이며 index jitter가 들어가지 않음
- score 단계에서 label import/query 차단
- 네 ablation partition, 예상 수와 exact coordinate parity
- AP 정의, AUROC undefined 처리, pooled/macro 집계
- dataset-paired bootstrap 결정성 및 draw hash
- PyTorch/MLX/MPS parity·속도 gate, CPU fallback과 CPU worker provenance
- atomic shard, resume, no-clobber, hash 변조 거절
- 전체 실행 뒤 저장 confusion과 재집계 confusion의 exact 일치

논문용 제안기술 재실험은 다음이 모두 충족될 때만 완료다.

1. 1,117개 raw-data rerun과 네 ablation 좌표 parity 성공
2. 제안기술 source score와 route-preserving score 완전 동결
3. 필수 네 공개 고전 기법의 전체 재실행 완료
4. freeze 이후 한 번의 label evaluation과 독립 재집계 일치
5. AP/AUROC score type, coverage와 retrospective 한계가 표에 명시됨
6. 재현 manifest, runbook, 논문 표와 결과 hash가 서로 연결됨

이 조건을 통과한 뒤에만 논문 결과 표를 확정한다.
