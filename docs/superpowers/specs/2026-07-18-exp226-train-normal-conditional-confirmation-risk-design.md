# Exp226 TRAIN-normal Conditional Confirmation Risk Audit Design

## 목적

Exp225의 검토 후보 규칙을 TEST 라벨이나 TEST 성능 없이 TRAIN 정상 pseudo-run에 그대로 재생한다. 이 실험은 자동 경보를 만들거나 운영 정책을 선택하지 않는다. 다음 세 조건이 같은 TRAIN 정상 인스턴스에서 동시에 또는 부분적으로 발생하는 빈도만 측정한다.

1. 고정 사전 형상 후보: Exp219의 `fixed_dictionary_control`, `conservative_max`, alpha `0.005`
2. 시간 랜덤 커널 확인: Exp224의 동결 alpha `0.01`
3. 표준 정상 상태 확인: Exp224의 동결 alpha `0.01`

`dual-confirmed review`는 세 조건을 모두 만족한 정상 인스턴스이고, `single-confirmed review`는 고정 사전 후보이면서 두 확인 신호 중 정확히 하나만 만족한 정상 인스턴스다. 이 결과는 TRAIN 정상 조건부 오경보 및 검토 workload 진단이며 retrospective TEST 성능이 아니다.

## 검토한 접근

### A. 동결 캐시 정렬 replay (채택)

Exp215 캐시의 fold-local `normal_pvalues`와 Exp224 캐시의 두 OOF score를 사용한다. 각 실험의 서로 다른 deterministic fold plan을 독립적으로 재생해 값을 원래 TRAIN `instance_index`에 되돌린 뒤 같은 행끼리 결합한다.

장점은 feature 재계산 없이 기존 동결 산출물을 정확히 감사할 수 있고, TEST 데이터에 접근할 필요가 없다는 점이다. 위험은 행 정렬을 잘못 복원할 가능성이므로 캐시 길이, DB의 `ORDER BY instance_index`, fold별 소비 길이, 원본 manifest SHA-256을 모두 검증한다.

### B. 원시 TRAIN 전체 재계산

세 점수원을 원시 TRAIN에서 다시 cross-fit한다. 가장 강한 독립 재현이지만 251개 데이터셋 전체 feature 계산 비용이 크고, 이번 실험의 핵심인 동결 Exp225 규칙의 조건부 위험 감사보다 범위가 넓다. Exp226 replay가 불일치하거나 provenance 검증에 실패할 때만 후속 실험으로 사용한다.

### C. 기존 dataset 집계 수치 결합

기존 정상 오경보 합계만 결합하는 방식이다. 같은 정상 인스턴스에서 점수원이 겹쳤는지 알 수 없으므로 dual/single 위험을 계산할 수 없다. 사용하지 않는다.

## 데이터 계약

- 분석 모집단은 Exp219 primary prediction에 포함된 251개 데이터셋으로 고정한다.
- Exp219 prediction 및 manifest SHA-256을 검증한다.
- Exp215 cache는 `exp215_cache/<dataset>.npz`를 사용하며 `normal_pvalues`가 TRAIN 정상 행 수와 같아야 한다.
- Exp224 policy bundle과 raw replay report를 검증하고, 해당 데이터셋의 artifact SHA-256을 policy manifest와 대조한다.
- 두 캐시의 `n_train`은 서로 같아야 한다.
- 두 데이터 로더가 DB에서 TRAIN 행을 `instance_index` 순으로 읽는 코드 계약을 증빙한다.
- TEST series, TEST label, TEST anomaly position, family 이름, dataset별 과거 성능은 계산 입력으로 사용하지 않는다. Dataset 이름은 캐시 조회 key로만 사용한다.

## 정렬 및 계산

### 고정 사전 형상 후보

Exp215은 seed `20260716`, 5-fold plan으로 생성한 fold-local 정상 p-value를 fold 처리 순서로 이어 붙였다. 같은 fold plan을 재생하고 각 fold의 query index에 해당 p-value chunk를 배치해 원래 TRAIN 행 순서의 vector를 복원한다. 후보는 `p <= 0.005`다.

### 두 독립 확인 신호

Exp224 artifact의 `temporal_oof`와 `canonical_oof`는 원래 TRAIN 행 순서다. Exp224 seed `20260718`의 fold plan을 재생하고, 각 query score를 해당 fold의 fit score에 대해서만 empirical upper-tail p-value로 변환한다. 두 확인 hit는 각각 `p <= 0.01`이다.

### Lane

- dual: shape candidate AND temporal hit AND canonical hit
- single: shape candidate AND exactly one of temporal/canonical hit
- unconfirmed: shape candidate AND neither confirmation hit

세 lane은 상호 배타적이며 합집합은 shape candidate와 정확히 같아야 한다.

## 산출물

외부 대용량 결과는 `/Users/minho/Documents/Dataset/exp226_train_normal_conditional_risk/`에 저장한다.

- dataset별 TRAIN 정상 risk CSV
- 전체 summary CSV/JSON
- frozen TRAIN-only prediction JSONL과 manifest
- 실행 로그 및 input/code/output SHA-256 manifest

저장소에는 `outputs/exp226_validation/` 아래 요약 결과와 분석 문서를 보존한다.

핵심 지표는 전체 및 dataset별 다음 값이다.

- 정상 pseudo-run 수
- fixed-dictionary 후보 수와 rate
- dual/single/unconfirmed 수와 rate
- 100 정상 run당 workload
- 각 rate의 one-sided 95% Wilson upper bound
- 후보가 발생한 데이터셋 수
- dataset별 count의 median, p90, p95, max

Exp224의 `risk_target=0.005`는 참고선을 그대로 표시하지만, Exp226 결과로 lane 또는 threshold를 선택하지 않는다. Review lane에 대한 별도 운영 허용 기준이 사전 등록되지 않았으므로 운영 pass/fail을 선언하지 않는다.

## 오류 처리

다음 조건이면 결과를 부분 성공으로 기록하지 않고 실패한다.

- Exp219 primary coverage가 251개가 아님
- policy, replay, cache 또는 manifest hash 불일치
- Exp215/Exp224 TRAIN 길이 불일치
- fold plan이 모든 TRAIN index를 정확히 한 번 덮지 않음
- p-value가 유한 범위 `[0, 1]` 밖임
- lane overlap 또는 candidate partition 불일치
- 계산 경로에서 TEST label/evaluation loader 호출

## 테스트

테스트를 먼저 작성하고 다음 실패를 확인한 뒤 구현한다.

1. fold-concatenated Exp215 p-value를 원래 index로 정확히 복원
2. Exp224 fold-local p-value가 query 자신의 score를 reference에 포함하지 않음
3. dual/single/unconfirmed lane partition
4. TRAIN 길이 및 hash mismatch 거부
5. output payload에 TEST/evaluation field가 없음
6. 같은 입력과 seed에서 정렬된 output hash가 동일

단위 테스트 후 3개 데이터셋 smoke replay를 실행하고, 통과한 경우에만 251개 전체를 실행한다.

## 해석 제한

Exp226이 낮은 정상 확인률을 보여도 Exp225의 TEST precision을 운영 성능으로 확정하지 않는다. 반대로 낮은 TRAIN 정상 위험은 독립 신규 설비에서의 prospective 성능을 증명하지 않는다. Exp226은 현재 동결 규칙의 TRAIN-normal conditional risk와 workload를 확인하는 retrospective audit이며, 자동 경보 승격이나 TEST 기반 정책 선택은 후속 사전등록 없이는 수행하지 않는다.
