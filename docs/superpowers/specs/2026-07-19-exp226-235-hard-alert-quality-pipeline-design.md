# Exp226-235 Hard Alert Quality Pipeline Design

## 목적

Exp137 Hard alert의 품질을 개선하기 위한 10개 실험을 하나의 재현 가능한 프로그램으로 연결한다. 프로그램은 SQLite 데이터베이스 조회부터 공통 전처리와 점수 캐시, 독립 실험 prediction, prediction SHA-256 동결, TEST label 사후 평가, 독립 재집계와 대시보드 상태 기록까지 담당한다.

운영 목표는 단일 mean F1 최대화가 아니다. 자동 경보 FP를 줄이고 사용자 신뢰를 높이되, 강한 근거가 있는 TP를 가능한 한 보존하거나 제한적으로 복구한다. 모든 후보 생성, feature fitting, score calibration, threshold, source reliability, alert gate는 TRAIN 정상 데이터와 사전 고정 상수만 사용한다.

이 설계는 `2026-07-18-exp226-train-normal-conditional-confirmation-risk-design.md`와 해당 구현 계획을 포함하고 확장한다. 기존 문서는 삭제하지 않으며 Exp226의 사전등록 이력으로 보존한다.

## 패키지 구조

새 코드는 `experiments/exp226_235_hard_alert_quality/` 아래에만 추가한다.

```text
experiments/exp226_235_hard_alert_quality/
├── __init__.py
├── config.py
├── database.py
├── preprocessing.py
├── contracts.py
├── score_cache.py
├── baseline_exp137.py
├── exp226_train_risk.py
├── exp227_independent_confirmation.py
├── exp228_margin_confidence.py
├── exp229_correlation_aware_agreement.py
├── exp230_seed_stability.py
├── exp231_normal_mode_calibration.py
├── exp232_calibration_reliability_gate.py
├── exp233_conformal_fdr.py
├── exp234_precision_first_rescue.py
├── exp235_shadow_export.py
├── reporting.py
├── independent_reaggregate.py
├── run_pipeline.py
└── tests/
```

각 파일은 하나의 책임만 가진다. 실험 모듈은 DB나 TEST label을 직접 읽지 않고, 동결된 공통 입력 계약을 받아 label-free prediction row를 반환한다.

## 데이터 흐름

1. `database.py`가 `/Users/minho/Documents/Dataset/univariate_ts.db`를 read-only로 연다.
2. `audit` 단계가 schema, 1,117개 dataset coverage, instance ordering, TRAIN 정상 여부, DB SHA-256을 검증한다.
3. `preprocessing.py`가 TRAIN에서 결정한 target length로 variable length를 정렬하고 NaN/Inf를 보정한다.
4. `score_cache.py`가 공통 score를 한 번 계산하거나 기존 동결 캐시를 hash 검증 후 재사용한다.
5. `baseline_exp137.py`가 Exp137의 2,005 Hard indices를 byte-for-byte 재현한다.
6. Exp226~235 모듈이 동일한 baseline과 공통 cache에서 독립 prediction을 생성한다.
7. 모든 prediction JSONL을 정렬하고 SHA-256을 manifest에 동결한다.
8. 동결 뒤에만 evaluation 경로가 TEST label을 별도 조회한다.
9. `independent_reaggregate.py`가 runner와 독립적으로 TP/FP/FN/precision/recall/F1을 재계산한다.
10. `reporting.py`가 실험별 결과와 전체 비교표를 만들고 dashboard status JSON을 갱신한다.

## 정보 경계

Prediction 함수의 공통 형태는 다음과 같다.

```python
prediction = predict_dataset(
    train_normal=train_normal,
    test_values=test_values,
    frozen_scores=frozen_scores,
    frozen_config=frozen_config,
)
```

Prediction 단계는 다음 정보를 받지 않는다.

- `y_test`
- 실제 이상 index 또는 tail 위치
- family별 과거 TEST 성능
- dataset 이름에 따른 정책 분기
- TEST 성능으로 선택된 alpha, threshold, K 또는 source

Dataset 이름은 DB와 cache 조회 key로만 사용한다. Family와 dataset 이름은 evaluation 후 사후 진단 표에 표시할 수 있지만 정책 입력에는 사용할 수 없다.

## 공통 점수원

기존 명칭 대신 다음 운영 명칭을 사용한다.

- 시간 랜덤 커널 점수: ROCKET 기반 시계열 패턴 거리
- 주파수 이미지 점수: Spectrogram 기반 표현 거리
- 텍스처 형상 점수: 이미지 texture/shape 표현 거리
- 표준 정상 상태 점수: 정상 상태 통계와 KNN local-gap 거리
- 고정 사전 형상 점수: data-independent shape dictionary 거리

공통 cache에는 TRAIN OOF score와 TEST score를 분리해 저장한다. Threshold와 reliability는 TRAIN OOF score만으로 계산한다.

## 실험 정의

### Exp226: 전체 규칙 TRAIN-normal 위험 감사

Exp137 Hard 규칙 및 후속 확인 규칙을 TRAIN 정상 pseudo-run에 재생한다. 정상 오경보 수, 100 run당 workload, one-sided 95% Wilson upper bound를 측정한다. 정책은 변경하지 않는다.

### Exp227: 독립 점수원 재확인

Exp137 Hard 후보를 시간 랜덤 커널 점수와 표준 정상 상태 점수가 같은 index에서 확인하는지 분해한다. Dual, single, unconfirmed를 모두 보고하며 TEST 결과로 승격 규칙을 선택하지 않는다.

### Exp228: Margin confidence

Binary threshold 통과 여부뿐 아니라 TRAIN threshold 대비 score margin과 conformal p-value 여유도를 계산한다. 사전 고정 margin grid 전체를 sensitivity로 보고하고 primary margin을 고르지 않는다.

### Exp229: 상관관계 보정 합의

TRAIN OOF score 간 Spearman correlation을 이용해 서로 중복된 점수원의 합의 가중치를 낮춘다. Family 또는 TEST 결과를 사용하지 않는다.

### Exp230: Seed 및 kernel 안정성

사전 고정한 seed와 ROCKET kernel block에서 반복되는 Hard 후보를 측정한다. 결과는 stability count별로 모두 보고하며 TEST 성능으로 필요한 반복 횟수를 선택하지 않는다.

### Exp231: 정상 상태별 local calibration

TRAIN 정상 feature를 사전 고정 방식으로 군집화하고 가장 가까운 정상 submode의 calibration reference를 사용한다. 군집 수는 TRAIN-only stability 또는 사전 고정 grid로 보고하며 TEST 성능으로 선택하지 않는다.

### Exp232: Calibration reliability gate

TRAIN 수, attainable conformal resolution, OOF false-alert upper bound, score drift와 source stability를 이용해 `eligible`, `review-only`, `abstain`을 결정한다. 이 gate는 TEST 길이나 family 이름을 사용하지 않는다.

### Exp233: Conformal FDR sensitivity

한 dataset을 하나의 운영 단위로 보고 그 안의 후보 p-value에 사전 고정 FDR grid를 적용한다. Grid 전체 workload-performance를 보고하고 특정 FDR 값을 운영값으로 선택하지 않는다.

### Exp234: Precision-first rescue

Exp137이 놓친 영역에서 세 개 이상의 독립 TRAIN-calibrated 근거가 일치하는 후보만 research rescue lane에 추가한다. 기존 Hard alert와 rescue lane을 분리하고 rescue를 자동 경보로 합산하지 않는다.

### Exp235: Shadow export

Exp226~234에서 생성된 동결 prediction과 근거를 실제 shadow 운영에서 수집할 수 있는 형태로 내보낸다. 사용자 feedback이 없으므로 성능 실험이 아니며 운영 승격을 수행하지 않는다.

## 연결 실행기

`run_pipeline.py`는 다음 stage를 제공한다.

```text
audit -> cache -> baseline -> predict -> freeze -> evaluate -> reaggregate -> report
```

지원 명령:

```bash
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline --stage audit
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline --stage cache --workers 7
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline --stage predict --experiments 226-235
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline --stage evaluate --experiments 226-235
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline --stage report
```

각 stage는 완료 checkpoint와 input/output hash를 기록한다. 중단 후 완료된 dataset을 건너뛰고 재개할 수 있지만 hash 또는 config가 달라지면 기존 checkpoint를 재사용하지 않는다. 한 번에 하나의 실험만 active 상태로 실행하며 dashboard에는 active, completed, pending, failed 상태와 dataset 진행률을 기록한다.

## 산출물

대용량 cache와 결과는 다음에 저장한다.

```text
/Users/minho/Documents/Dataset/exp226_235_hard_alert_quality/
├── audit/
├── cache/
├── predictions/
├── results/
├── checkpoints/
├── logs/
└── manifest/
```

저장소에는 `outputs/exp226_235_hard_alert_quality/` 아래 요약 CSV, 독립 재집계 결과, invariance report, 결과 분석과 manifest를 보존한다. 기존 Exp137 및 Exp219~225 결과는 덮어쓰지 않는다.

## 평가 지표

각 실험은 다음 autonomous 지표를 분리해서 보고한다.

- Hard alerts, TP, FP, FN
- Micro Hard precision과 recall
- Mean dataset Hard F1
- Hard zero-F1 dataset 수
- 기존 FP 314 대비 변화
- 기존 TP 1,691 대비 보존율

Review/research lane은 candidate count, TP, FP, precision과 workload를 별도로 보고한다. Review를 포함한 combined F1은 human-assisted diagnostic으로만 표시하며 autonomous 성능과 합치지 않는다.

추가로 dataset별 paired bootstrap interval, exact-index Jaccard, lane transition, TRAIN 크기와 시계열 길이별 사후 진단을 제공한다. 크기·family 진단은 정책 선택에 사용하지 않는다.

## 테스트 전략

구현은 TDD로 진행한다.

1. DB label-free loader와 evaluation label loader 분리
2. TRAIN-only target length와 sanitize 동작
3. cache hash 및 config mismatch 거부
4. Exp137 baseline exact replay
5. 각 실험의 후보/lane 상호 배타성
6. prediction forbidden field 검사
7. label, anomaly position, family rename 불변성
8. dataset row order 불변성
9. 동일 seed output hash 재현성
10. independent reaggregation과 runner summary 일치

10개 dataset smoke가 모두 통과한 뒤에만 1,117개 full pipeline을 실행한다. 일부 dataset 실패를 성공으로 처리하지 않으며 실패 이유와 재개 조건을 checkpoint에 기록한다.

## 완료와 해석 제한

완료 기준은 1,117개 coverage, 오류 0, prediction hash 동결, 독립 재집계 일치, invariance test 통과, 실험별 summary와 manifest 생성이다.

모든 결과는 기존 TEST 데이터에 대한 retrospective counterfactual이다. 좋은 결과가 나와도 end-to-end strict TRAIN-only 개발 계보, 실제 설비 prospective 성능 또는 운영 threshold 확정을 주장하지 않는다. 새로운 운영 정책 선택은 잠금 benchmark 또는 신규 shadow feedback을 사용한 별도 사전등록 실험에서만 수행한다.
