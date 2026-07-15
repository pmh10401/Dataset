# Exp164 TRAIN-Normal Agreement Risk Audit Design

## 목적

Exp164는 Exp163에서 사용한 세 대안 점수원의 `2-of-3` 및 `3-of-3` 합의가 정상 TRAIN run에서 얼마나 자주 잘못된 일반 검토 후보를 만드는지 측정한다. TEST 값, TEST label, 실제 이상 위치, Exp160/163 candidate-zero 결과는 입력으로 사용하지 않는다.

이 실험은 성능 개선이나 운영 alpha 선택 실험이 아니다. 모든 결과는 TRAIN-normal calibration audit와 운영 위험 sensitivity로만 보고한다.

## 운영 해석

- 한 dataset은 하나의 가상 equipment/recipe/sensor/step 탐지 환경이다.
- dataset 내부의 TRAIN instance 하나는 하나의 정상 virtual wafer run trace로 취급한다.
- 각 pseudo-run은 정상으로 알려져 있으므로 합의 후보가 되면 `normal false review`로 집계한다.
- 이 가정은 공개 benchmark를 반도체 운영 형태로 해석한 연구 가정이며 실제 설비 prospective 증거가 아니다.

## 동결 population과 데이터 접근

- population: Exp162에서 검증한 1,117개 dataset 이름
- database: `/Users/minho/Documents/Dataset/univariate_ts.db`
- prediction/audit query: `instances.values_blob`, TRAIN split, instance index만 읽는다.
- TEST split과 label column은 SQL에서 읽지 않는다.
- family, dataset별 과거 TEST 성능, Exp160/163 lane은 사용하지 않는다.
- dataset 이름은 deterministic shapelet seed와 cache lookup key로만 사용하며 정책 분기에 사용하지 않는다.

## 교차적합 구조

`n_train < 5`는 완전 calibration abstention으로 유지한다.

- `5 <= n_train < 20`: deterministic leave-one-out
- `n_train >= 20`: deterministic shuffled 5-fold, seed `20260715`

각 fold에서 query pseudo-run은 feature fitting, target length, scaler, center/spread, threshold reference에서 완전히 제외한다.

1. fit TRAIN raw series 길이의 median으로 target length를 정하고 `[8, 2048]`로 제한한다.
2. fit/query series를 해당 길이로 정렬한다.
3. Exp162와 같은 세 feature source를 fit TRAIN으로 구성한다.
4. fit score와 query score를 계산한다.
5. 각 alpha에서 fit score의 `np.quantile(finite_fit_scores, 1-alpha, method="higher")`를 threshold로 사용한다.
6. `query_score > threshold`이면 해당 source가 pseudo-run을 후보로 지정한 것으로 기록한다.

원시 source score를 더하거나 평균하지 않는다. Fold 간 raw score도 직접 비교하지 않는다.

## 점수원과 합의

- 미분·스펙트럼 통계 점수
- 부분 형태 일치 점수
- 구간 분포 점수

동결 alpha:

`(0.005, 0.01, 0.02, 0.05)`

동결 variant:

- `train_audit_2of3`: 동일 pseudo-run을 두 source 이상이 지목
- `train_audit_3of3`: 동일 pseudo-run을 세 source 모두가 지목

이 실험은 일반 검토 위험만 감사한다. Hard 또는 Priority lane을 생성하거나 변경하지 않는다.

## 위험 sensitivity

다음 정상 false-review 목표를 모두 보고하고 하나를 선택하지 않는다.

- `0.005 per run`: 정상 run 100회당 0.5건
- `0.010 per run`: 정상 run 100회당 1건
- `0.020 per run`: 정상 run 100회당 2건

각 dataset/variant/alpha에 대해 관측 false-review rate와 one-sided 95% Wilson upper bound를 계산한다. Upper bound가 목표 이하인지 표시하지만, TEST 결과나 이번 audit 결과만으로 운영 policy를 확정하지 않는다.

## 출력 계약

각 dataset cache는 다음을 포함한다.

- pseudo-run 원래 TRAIN index
- fold index
- source별 query score
- source별 alpha threshold
- source별 candidate flag
- 2-of-3 및 3-of-3 flag
- TRAIN row count와 fold 방법
- source configuration hash
- `contains_test_values=false`
- `contains_test_labels=false`
- `contains_evaluation_metrics=false`

주요 집계:

- source별 false reviews / 100 normal runs
- variant별 false reviews / 100 normal runs
- dataset별 one-sided 95% Wilson upper bound
- 위험 목표별 eligible dataset 수와 abstention 수
- threshold 고유 개수와 해상도 제한
- source candidate flag pairwise phi correlation 및 Jaccard
- n_train 구간별 위험

## 실행과 복구

- 하나의 Exp164 process에서 dataset 단위 7 workers를 사용한다.
- BLAS/OpenMP inner thread는 worker당 1로 제한한다.
- dataset cache는 atomic write 후 SHA256 검증한다.
- checkpoint 재개 시 dataset, DB, population, source config, fold config hash가 모두 일치할 때만 재사용한다.
- 실패 또는 누락 dataset을 빈 결과로 바꾸지 않는다.
- 1,117개 coverage가 완성되지 않으면 실험을 성공 처리하지 않는다.

## 검증

- TEST table/label 접근을 금지하는 구조 테스트
- query index가 fit/reference에 포함되지 않는 테스트
- deterministic fold plan과 replay hash 테스트
- fold별 fit-derived target length 테스트
- strict `>` tie 처리와 `method="higher"` threshold 테스트
- 2-of-3/3-of-3 합의 테스트
- `n_train < 5` abstention 테스트
- Wilson upper bound 테스트
- cache tamper와 checkpoint config drift 테스트
- input order invariance와 7-worker/1-worker output hash 일치 테스트

## 해석 경계

Exp164가 낮은 false-review rate를 보이더라도 Exp163의 TEST TP/FP와 결합해 특정 alpha나 variant를 운영값으로 고르지 않는다. Exp164는 TRAIN-normal 내부 안정성을 보여줄 뿐 신규 설비 성능, 실제 사용자 workload, end-to-end configuration provenance 또는 prospective false-alarm 보장을 증명하지 않는다.
