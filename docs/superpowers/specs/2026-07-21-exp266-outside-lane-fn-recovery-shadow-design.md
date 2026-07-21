# Exp266 Outside-Lane FN Recovery Shadow Design

## 1. 목적

Exp266은 Exp253 자동 경보와 Exp265의 기존 검토 lane을 변경하지 않고,
현재 모든 lane 밖에 남아 있는 false negative 복구 가능성을 별도의
`복구 Shadow`와 `복구 우선 검토` lane에서 검증한다.

운영 목표는 자동 경보 수를 즉시 늘리는 것이 아니다. 기존 자동 경보의
오탐 수준과 사용자 신뢰를 보존하면서, 서로 다른 TRAIN-normal 기반
근거가 같은 virtual wafer-run instance를 지목하는 경우만 사용자 검토
대상으로 노출한다.

Exp266은 기존 TEST benchmark를 다시 사용하는 retrospective
counterfactual experiment다. 결과가 좋아도 자동 경보 승격, 실제 설비
성능, prospective validation 또는 end-to-end strict TRAIN-only 개발
계보를 주장하지 않는다.

## 2. 현재 기준과 변경 경계

현재 기준 lane은 Exp265 retrospective replay와 동일하게 고정한다.

- `Hard`: Exp253 자동 경보 2,253건
- `Shadow`: Exp264가 기존 Standard에서 분리한 2건
- `Priority`: Exp264 기준 31건
- `Standard`: Exp264 기준 806건
- 전체 기존 후보 union: 3,092건

Exp266은 위 네 lane의 candidate index와 lane을 byte-for-byte 보존한다.
새 후보는 반드시 기존 네 lane union 밖에 있어야 한다.

Exp264의 후향 결과에서 최신 보조 점수원 4개의 lane 밖 후보를 단순
합집합하면 437건 중 148 TP, 289 FP로 정밀도 33.9%였다. 따라서 점수원
합집합을 사용자 검토나 자동 경보로 직접 사용하지 않는다.

## 3. 운영 단위

- 한 `dataset`은 하나의 가상 equipment/recipe/sensor/step 탐지 환경이다.
- dataset 내부의 TRAIN 또는 TEST `instance` 하나는 virtual wafer-run
  trace 하나다.
- candidate index는 시계열 내부의 연속 시간 위치가 아니라 dataset 내부
  instance index다.

따라서 다음 정책을 사용하지 않는다.

- 인접 candidate index를 하나의 이상 event로 병합
- dataset의 TEST 길이에 비례한 top-N
- dataset당 임의의 최대 후보 수 K

실제 운영 검토량은 모델이 후보를 삭제하는 방식이 아니라 대시보드의
정렬, 필터, 사용자별 처리량 집계로 관리한다.

## 4. 두 개의 분리된 검증

### 4.1 Exp266-A: frozen retrospective feasibility replay

기존 동결 prediction artifact의 candidate index만 읽어 새로운 lane을
label-free로 재구성한다. Feature, score, threshold 또는 alpha를 다시
선택하지 않는다.

Prediction을 동결하고 SHA-256을 기록한 뒤에만 TEST label을 읽어 TP, FP,
precision, recall, F1을 사후 평가한다.

### 4.2 Exp266-B: TRAIN-normal combined-policy risk audit

동일한 3개/2개 근거 합의 규칙을 TRAIN 정상 pseudo-run에 재생한다. 각
점수원의 동결 cache에 저장된 교차적합 OOF score만 사용하며, cache
manifest가 query를 source fitting에서 제외한 계약을 증명해야 한다. 각
pseudo-run의 p-value를 계산할 때는 자신의 OOF score를 calibration
reference에서도 제외한다.

- `n_train < 5`: 복구 정책 calibration abstention
- `n_train >= 5`: 동결된 source별 deterministic OOF score 사용
- calibration: query를 제외한 나머지 finite OOF score의 empirical tail
  p-value
- audit seed: `20260721`; source fitting seed는 각 기존 artifact의 동결값
- 정상 검토 요청 위험 기준: virtual run당 0.01 이하
- 신뢰 구간: one-sided 95% Wilson upper bound

Source cache가 query-excluded OOF fitting을 증명하지 못하거나 TRAIN index
정렬과 길이가 서로 다르면 위험감사를 중단한다. 서로 다른 fold model에서
생성된 OOF score를 하나의 calibration reference로 사용하는 한계는 결과
보고서에 명시한다.

이 기준은 이전 TRAIN-normal risk policy의 운영 계약을 재사용하며 TEST
결과로 변경하지 않는다. Exp266-A 성능이 좋아도 Exp266-B가 위험 기준을
통과하지 못하면 prospective shadow exporter에 연결하지 않는다.

## 5. 독립 근거군

동일한 구현 계열의 여러 variant를 여러 표로 중복 계산하지 않는다.
다음 다섯 근거군만 사용한다.

1. `시간 랜덤 커널 근거`
   - Exp249 MiniROCKET 1,024 및 4,096 kernel seed-intersection
   - variant: `kernels_1024_seed_intersection`,
     `kernels_4096_seed_intersection`; alpha: `0.005`
   - 두 variant의 합집합을 하나의 kernel-family vote로 계산
2. `통계 근거`
   - Exp252 `combined_40`; alpha: `0.005`
3. `구조 근거`
   - Exp254 `structural_combined_440`; alpha: `0.005`
4. `변화점 근거`
   - Exp255 `change_point_combined_33`; alpha: `0.005`
5. `마스킹 복원 근거`
   - Exp257 `mask_025__seed_intersection`; alpha: `0.005`

MultiROCKET은 MiniROCKET과 별도 독립 표로 세지 않는다. 같은 random
convolution 계열이고, 현재 lane 밖 단독 근거로 충분히 검증되지 않았기
때문에 Exp266 vote에서 제외한다. Source 이름, artifact SHA-256, config,
seed, alpha, coverage와 eligibility를 manifest에 기록한다.

마스킹 복원 근거는 동결된 계약대로 `n_train >= 100`인 dataset에서만
사용할 수 있다. 의도된 미적격 source는 음성 vote로 위장하지 않고
`unavailable`로 기록한다. 사용 가능한 근거군이 2개 미만이면 복구
정책은 abstain한다.

## 6. 후보 생성과 routing

각 dataset에서 다음 집합을 먼저 계산한다.

```text
existing = Hard OR Shadow OR Priority OR Standard
outside_source_i = source_i_candidates MINUS existing
agreement_count(index) = number of independent available source groups
                       that selected the same instance index
```

Routing은 다음 순서로 고정한다.

1. 기존 lane 후보이면 기존 lane을 그대로 유지한다.
2. 기존 lane 밖이며 데이터 품질 검사가 TRAIN 정상 범위를 벗어나면
   `데이터 품질 검토`로 보낸다.
3. 기존 lane 밖이며 품질 검사를 통과하고 3개 이상 근거군이 동의하면
   `복구 Shadow`로 보낸다.
4. 기존 lane 밖이며 품질 검사를 통과하고 정확히 2개 근거군이 동의하면
   `복구 우선 검토`로 보낸다.
5. 근거가 0~1개이면 `알림 없음`으로 유지한다.

`복구 Shadow`, `복구 우선 검토`, `데이터 품질 검토`는 모두
human-assisted review lane이다. autonomous Hard 지표에 합산하지 않는다.
Priority review 또는 복구 Shadow를 자동 경보로 승격하지 않는다.

## 7. 후향 진단과 해석 제한

설계 검토 중 현재 artifact를 read-only로 조합한 진단에서는 기존 모든
lane 밖에서 다섯 근거군 중 3개 이상이 동의한 후보가 15건이었다. TEST
label을 사후 대조하면 13 TP, 2 FP, precision 86.7%였다.

이 수치는 구조의 연구 가능성을 보여줄 뿐 운영 성능이 아니다. 같은 TEST
benchmark를 반복 관찰한 뒤 작성된 설계이므로 다음에 사용할 수 없다.

- 자동 경보 승격 근거
- 특정 source 또는 alpha의 운영 최적성 주장
- 실제 설비 precision 예상
- 80% precision 보장

Exp266은 2개 합의와 3개 이상 합의를 모두 사전 고정해 보고한다. TEST
결과가 더 좋은 쪽을 primary 정책으로 다시 선택하지 않는다.

## 8. Prediction과 evaluation 분리

Prediction 인터페이스는 다음 정보만 받는다.

```python
prediction = predict_recovery_lanes(
    frozen_existing_lanes=frozen_existing_lanes,
    frozen_source_candidates=frozen_source_candidates,
    source_eligibility=source_eligibility,
    data_quality_state=data_quality_state,
    frozen_config=frozen_config,
)
```

Prediction은 다음 정보를 받거나 읽지 않는다.

- `y_test`
- 실제 이상 위치
- TP, FP, FN, precision, recall, F1
- family 이름 또는 과거 family TEST 성능
- dataset 이름에 따른 정책 분기
- TEST 길이 기반 threshold, budget 또는 gate

Evaluation은 정렬된 prediction file SHA-256을 확인한 뒤 별도 DB query로
TEST label을 읽는다.

## 9. 데이터 품질 처리

결측값 보정은 source score보다 먼저 수행하며 TRAIN에서 정한 길이와
정상 기준을 사용한다. 동결된 Exp263/264 품질 계약을 재사용한다.

- 짧은 내부 결측: PCHIP 기반 보정
- 긴 결측 또는 경계 결측: TRAIN 위치별 중앙값 기반 보정
- NaN/Inf 잔존: prediction 실패
- 결측 패턴이 TRAIN 정상 범위를 벗어남: 데이터 품질 검토

결측 품질 문제를 이상 근거 투표로 세지 않는다. 품질 문제 때문에
threshold를 완화하거나 복구 Shadow로 승격하지 않는다.

## 10. 오류와 중단 조건

다음 경우 전체 aggregation을 중단하고 blocker를 기록한다.

- 기존 lane이 1,117개 dataset을 완전하게 재현하지 못함
- 입력 artifact SHA-256 또는 config가 계약과 다름
- 필수 source coverage 누락을 빈 candidate로 처리해야 함
- candidate index가 instance 범위를 벗어남
- source variant를 독립 근거군으로 중복 집계함
- prediction 단계에서 TEST label 또는 평가 metric에 접근함
- 기존 Hard/Shadow/Priority/Standard index가 변경됨
- 일부 dataset 오류를 성공 또는 빈 결과로 처리함
- TRAIN pseudo-run이 자신의 calibration reference에 포함됨

네트워크나 대시보드 오류는 prediction을 삭제하지 않는다. 결과는 먼저
로컬 atomic file과 hash manifest에 저장한 뒤 Exp265 ledger로 전달한다.

## 11. 산출물

대용량 결과는 다음 경로에 저장한다.

```text
/Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow/
```

저장소 요약은 다음 경로에 보존한다.

```text
outputs/exp266_outside_lane_fn_recovery_shadow/
├── 00_contract_audit.json
├── 01_source_manifest.csv
├── 02_train_normal_risk_audit.csv
├── 03_frozen_predictions.jsonl
├── 04_prediction_manifest.json
├── 05_dataset_results.csv
├── 06_global_summary.csv
├── 07_source_agreement_diagnostics.csv
├── 08_invariance_report.md
├── 09_result_report.md
├── 10_independent_reaggregation.json
└── MANIFEST.csv
```

## 12. 평가 지표

Autonomous 지표는 Exp253 기준값을 그대로 표시하고 변경값을 0으로
검증한다.

- Hard alerts, TP, FP, FN
- micro Hard precision과 recall
- mean dataset Hard F1
- Hard zero-F1 dataset 수

Human-assisted 복구 지표는 별도로 보고한다.

- 복구 Shadow count, TP, FP, precision
- 복구 우선 검토 count, TP, FP, precision
- 데이터 품질 검토 count
- 복구된 고유 FN 수
- 후보가 추가된 dataset 수
- run당 review-request rate
- TRAIN-normal pseudo-run false-review rate와 Wilson upper bound
- source agreement 0/1/2/3/4/5 분포
- 각 source의 available/abstained dataset 수

`mean_combined_f1`을 계산할 경우 human-assisted diagnostic이라고 명시하고
autonomous Hard F1과 합치거나 대체하지 않는다.

## 13. 검증 전략

### 단위 테스트

- MiniROCKET 1,024/4,096 variant가 한 근거군으로만 계산됨
- 기존 lane union 밖의 후보만 복구 대상으로 허용됨
- 3개 이상/정확히 2개/0~1개 routing
- source unavailable과 negative vote 구분
- 데이터 품질 검토 우선순위
- lane 상호 배타성
- `n_train < 5` TRAIN-risk abstention

### 불변성 테스트

- TEST label permutation 후 prediction hash 불변
- anomaly 위치 변경 후 prediction 불변
- family 또는 dataset 표시 이름 변경 후 정책 결과 불변
- TEST instance 수 변경이 TRAIN-derived threshold와 risk parameter를 바꾸지 않음
- 입력 row 순서 변경 후 dataset별 prediction 불변
- 동일 입력과 seed로 재실행한 정렬 output hash 일치

### 통합 검증

- 10-dataset smoke 후 1,117-dataset full replay
- 기존 네 lane bit-for-bit 보존
- prediction에 label 또는 평가 metric 필드 없음
- candidate 범위 오류 0, lane overlap 0, 누락 dataset 0
- frozen prediction hash 확인 뒤에만 evaluation 실행
- 독립 재집계와 저장 summary 완전 일치
- Exp265 ledger 전송 실패 후 idempotent 재시도

## 14. 대시보드 표시

Exp265 화면에 다음을 별도 표시한다.

- 기존 자동 경보와 A/B/C 신뢰도
- 기존 Shadow/Priority/Standard
- 복구 Shadow
- 복구 우선 검토
- 데이터 품질 검토
- 각 후보에 동의한 독립 근거군과 score/threshold
- TRAIN-risk audit 상태와 config SHA-256

복구 후보는 기존 자동 경보와 색상, 명칭, 지표를 분리한다. 사용자가
복구 후보를 확인해도 자동 경보 수치가 증가한 것처럼 표시하지 않는다.

## 15. 완료 조건과 다음 단계

Exp266 완료 조건은 다음과 같다.

1. 1,117개 dataset coverage와 오류 0
2. 기존 Exp253/265 lane 완전 보존
3. Exp266-A prediction hash 동결과 독립 재집계 통과
4. Exp266-B TRAIN-normal risk audit 완료
5. label/family/TEST-length 불변성 테스트 통과
6. Hard와 review 지표 분리
7. 모든 입력, 코드, config, output SHA-256 기록

Exp266-B가 정상 false-review 위험 기준을 통과한 경우에만 동일한 동결
규칙을 신규 prospective feed의 `복구 Shadow`로 연결할 수 있다. 연결 후
신규 run과 사용자 feedback을 사전 고정된 기간 동안 수집한다.

Prospective 결과가 충분히 축적되더라도 자동 승격하지 않는다. 별도 실험
번호와 사전등록 문서에서 Exp253 후속 자동 경보 정책으로 병합할지
검토한다.
