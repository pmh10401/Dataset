# Exp414 Missing-Aware Near-Repeat Qualification Design

작성일: 2026-07-28  
상태: 설계 승인, 구현 전 동결  
실험 모드: `TRAIN-only missing-aware near-repeat qualification`

## 1. 배경

Exp411은 정상 TRAIN과 TRAIN-only 합성 교란만으로 near-repeat 정상 규칙의
자격을 심사하도록 사전등록되었다. 1,117개 전체 실행은 186개 dataset을
완료한 뒤 `DodgerLoopDay_normal_1`의 비유한 정상 TRAIN 값 때문에
fail-closed로 중단되었다. 최종 Exp411 산출물은 게시되지 않았고 기존
Hard 2,218, Standard 806, Priority 31, exact-normal 35와 기존 prediction은
변경되지 않았다.

읽기 전용 TRAIN 전수 검사에서는 동결 DB와 모집단에서 다음 입력 품질
상태가 확인되었다.

| 항목 | 관측값 |
|---|---:|
| 모집단 | 1,117 datasets |
| 비유한값 영향 | 19 dataset identities |
| 영향 정상 TRAIN run | 164 |
| 비유한값 | 983 |
| 전체 값이 비유한 run | 0 |
| 영향 family | DodgerLoop, MelbournePedestrian |

DB의 해당 `datasets.has_missing` 값은 `false`이므로 메타데이터만으로 이
상태를 탐지할 수 없다. Exp411 사전등록은 비유한 TRAIN 값이 존재하면
실패하도록 명시했으므로 동일 실험 번호에서 입력을 보정해 재실행하지
않는다.

## 2. 목적과 주장 경계

Exp414의 목적은 Exp411의 거리, fold, quantile, 합성 교란과 안전성 gate를
바꾸지 않고 결정론적인 TRAIN-only 결측 복구만 추가하여 1,117개 전체
모집단에서 near-repeat 규칙을 다시 자격 심사하는 것이다.

Exp414는 다음을 주장하지 않는다.

- TEST false alarm 감소
- anomaly recall 또는 F1 개선
- 반도체 현장 성능
- 자동 정상 lane 승격
- Exp412/413 prediction 또는 감사 완료
- 운영 배포 가능성

결측 subgroup 결과는 진단으로만 보고한다. 그 결과를 본 뒤 quantile,
거리 view, 합의 수, threshold, 합성 교란 또는 review budget을 변경하지
않는다.

## 3. 실험 번호와 산출물 격리

- Exp411은 입력 계약 실패로 보존한다.
- Exp411 checkpoint는 읽기 전용 증거이며 수정하지 않는다.
- Exp412와 Exp413은 시작하지 않고 기존 예약 상태로 보존한다.
- Exp414는 별도 코드 패키지, checkpoint, benchmark와 출력 디렉터리를
  사용한다.
- Exp414가 자격을 통과한 뒤에만 Exp415 label-blind Shadow prediction을
  별도 사전등록할 수 있다.
- Exp415 prediction hash가 동결된 뒤에만 Exp416 독립 감사를 별도
  사전등록할 수 있다.

## 4. 동결 입력

Exp414는 Exp411과 동일한 다음 입력을 사용한다.

- Exp404 prediction manifest가 고정한 1,117개 `frozen_hard_lanes` key
- SHA256이 고정된 읽기 전용 SQLite DB
- 정상 TRAIN의 `instance_index`와 `values_blob`
- Exp407의 일곱 TRAIN-only 합성 교란 구현과 설정
- Exp411의 fold, 거리, quantile, gate와 seed 계약

SQL은 TEST split, `labels_blob`, 실제 이상 위치, TP/FP/FN/F1, 과거
family별 TEST 성능 또는 source ranking을 읽지 않는다. DB 파일과 blob은
수정하지 않는다.

현재 DB SHA256에 대해 1절의 `19 / 164 / 983 / 0` inventory를 고정한다.
영향 dataset identity를 정렬한 문자열 배열의 UTF-8 canonical JSON
(`ensure_ascii=false`, 공백 없는 separators)에 대한 SHA256은
`ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9`로
고정한다. 구현 후 첫 전체 실행 전에 별도의 read-only audit 명령으로 이
값들을 다시 검증한다.

## 5. 결측 복구 계약

각 정상 TRAIN blob은 먼저 원본 `<f4` bytes와 SHA256을 기록한 뒤 다음
순서로 처리한다.

1. `NaN`, `+Inf`, `-Inf`를 모두 결측 위치로 표시한다.
2. 모든 값이 유한하면 원본 bytes를 그대로 사용하며 복사나 재직렬화로
   identity를 바꾸지 않는다.
3. 내부 결측은 원래 시계열 좌표에서 양쪽 유한값 사이를 선형 보간한다.
4. 시작 결측은 첫 유한값, 끝 결측은 마지막 유한값으로 연장한다.
5. 계산 결과를 명시적으로 little-endian float32로 변환한다.
6. 전체 값이 비유한 run이 하나라도 있으면 값을 0으로 대체하거나 run을
   제거하지 않고 해당 dataset 전체를 `missing_support_abstain`으로
   처리한다.

길이는 변경하지 않는다. 길이 보간, padding, truncation과 tail
replacement를 사용하지 않는다. 복구에는 다른 TRAIN run, TEST 값,
family 통계 또는 label 정보를 사용하지 않는다.

각 run의 감사 기록은 다음을 포함한다.

- dataset과 `instance_index`
- 원본 blob SHA256
- 복구 후 `<f4` bytes SHA256
- 원본 길이
- `NaN`, `+Inf`, `-Inf` 및 전체 결측 위치 수
- 정렬된 결측 위치 목록의 canonical SHA256
- `unchanged`, `linear_and_edge_nearest` 또는 `all_missing_abstain`

원시 waveform 값과 TEST 정보는 감사 CSV에 기록하지 않는다.

## 6. Identity와 fold 안전성

결측 복구 후 bytes SHA256을 Exp414의 identity group key로 사용한다.
서로 다른 원본 blob이 복구 후 같은 waveform이 되면 같은 identity
group으로 묶어 fit과 query 양쪽으로 분리되지 않게 한다. `exact` tag도
복구 후 identity를 기준으로 판정한다.

비결측 run에서는 복구 후 bytes가 원본 bytes와 완전히 같아야 한다.
따라서 비결측 dataset의 identity group, fold, distance, tag와 qualification
결과는 Exp411과 동일해야 한다.

## 7. Exp411 연구 계약의 재사용

Exp414는 다음 Exp411 값을 그대로 유지한다.

- seed `411`
- identity group 수에 따른 abstain, leave-one-group-out와 5-fold 규칙
- 동일 길이 waveform만 비교하는 규칙
- `raw_rms`, `diff_rms`, `spectrum_rms`
- NumPy `method="lower"` quantile `0.01 / 0.05 / 0.10`
- `exact`, `near_qXX`, `unfamiliar`, support abstain tag 의미
- Exp407의 일곱 합성 교란
- TRAIN-only 안전성 gate
- distance tolerance `1e-6`

합성 교란은 복구가 끝난 정상 TRAIN waveform에 적용한다. 결측 여부별
결과는 진단 subgroup으로만 집계하며 global quantile 선택에는 사전등록된
전체 모집단 집계만 사용한다.

## 8. Qualification 판정

각 quantile은 다음을 모두 만족해야 한다.

- held-out 정상 query near 인정률 `>= 0.01`
- pooled synthetic false-normal rate `<= 0.001`
- 종류별 최악 synthetic false-normal rate `<= 0.005`
- family-cluster bootstrap 95% 상한 `<= 0.005`
- NumPy 정본과 Numba backend의 최대 거리 오차 `<= 1e-6`
- NumPy와 Numba의 tag와 선택 quantile 동일

여러 후보가 통과하면 가장 넓은 quantile 하나를 선택한다. 하나도
통과하지 않으면 `no_qualified_near_rule`로 종료한다. 결과를 본 뒤 후보를
추가하거나 재선택하지 않는다.

보고서는 전체, 비결측 1,098개, 결측 복구 19개, DodgerLoop,
MelbournePedestrian와 `missing_support_abstain` subgroup을 제공한다.
Subgroup 결과는 global 판정을 변경하지 않는다.

## 9. Exp411 Checkpoint의 제한적 이전

Exp411 checkpoint는 직접 재개하거나 수정하지 않는다. Exp414는 별도
checkpoint에 검증된 record만 복사할 수 있다. 각 dataset은 다음 조건을
모두 만족해야 이전할 수 있다.

- Exp411 checkpoint root와 dataset record의 schema와 hash 검증 통과
- DB, 모집단 manifest와 원본 TRAIN identity SHA256 일치
- 해당 dataset의 결측 복구 수가 정확히 0
- fold, 거리, quantile, perturbation, gate와 qualification 계약 SHA256
  일치
- backend audit가 성공 상태
- 고정된 대표 이전 표본을 Exp414에서 다시 계산했을 때 tag, count,
  quantile과 qualification 값 일치

이전된 행은 `execution_source=migrated_exp411_verified`, 새로 계산한 행은
`execution_source=exp414_computed`로 기록한다. 조건을 만족하지 않는
dataset만 새로 계산하며 전체 실행을 실패시키지 않는다. 그러나 검증되지
않은 record를 사용하거나 migration 검증 자체가 불완전하면 fail-closed로
종료한다.

기존 Exp411 완료분 186개는 모두 결측 최초 실패 지점 이전의 dataset이다.
실제 이전 가능 개수는 구현된 검증기가 결정하며 186개를 성공 조건으로
가정하지 않는다.

## 10. Backend와 실행 계약

NumPy CPU를 수치 정본으로 유지한다. Numba direct float64 RMS engine은
12 threads를 사용하고 중첩 수치 라이브러리는 1 thread로 제한한다.
MLX는 사용자 결정에 따라 제외한다. PyTorch와 MPS는 이 실험의 결측 복구
또는 거리 계산에 필요하지 않으므로 사용하지 않는다.

Exp414 benchmark는 기존 고정 small, medium, large TRAIN dataset과 결측
복구 dataset을 포함한다. 기존 세 dataset은 속도와 수치/tag parity를,
결측 dataset은 복구·identity·tag parity를 검증한다. 전체 실행 전 새
Exp414 Git commit, preprocessing contract SHA256와 execution contract
SHA256에 결합된 benchmark report가 필요하다.

## 11. Checkpoint와 게시

Exp414 checkpoint는 다음 provenance에 결합한다.

- Git commit
- DB SHA256와 snapshot artifact set
- population manifest SHA256
- Exp414 frozen config SHA256
- missing-repair contract와 inventory SHA256
- perturbation contract SHA256
- qualification와 acceleration contract SHA256
- dataset별 원본 및 복구 TRAIN identity SHA256
- Exp411 migration source provenance

dataset 단위 record와 progress는 원자적으로 쓴다. 최종 결과는 모든
1,117개 dataset의 `computed` 또는 검증된 `migrated` coverage가 확인된
뒤 존재하지 않는 새 디렉터리로 원자적으로 게시한다.

정본 출력은 다음 9개 파일이다.

- `exp414_frozen_config.json`
- `exp414_missing_repair_audit.csv`
- `exp414_dataset_qualification.csv`
- `exp414_quantile_summary.csv`
- `exp414_backend_assessment.json`
- `exp414_selected_rule.json`
- `exp414_summary.json`
- `EXP414_RESULT_REPORT_KO.md`
- `MANIFEST.json`

별도로 저장하는 benchmark report와 checkpoint는 정본 9개 파일에
포함하지 않되 summary provenance에 경로와 SHA256을 기록한다.

## 12. Fail-Closed 조건

다음 중 하나라도 발생하면 최종 결과를 게시하지 않는다.

- Git worktree가 dirty이거나 실행 중 commit 변경
- DB, manifest, perturbation 또는 코드 계약 SHA256 불일치
- SQLite quick check 실패 또는 WAL/SHM 출현
- 1,117 dataset, 128 family 또는 결측 inventory 불일치
- TEST split 또는 `labels_blob` 접근
- TRAIN label, blob width, index uniqueness 또는 row count 오류
- 비결측 run의 bytes 변경
- 복구 후 비유한값 잔존
- 전체 결측 run을 0으로 대체하거나 제거
- 복구 후 같은 identity가 fit/query 양쪽으로 분리
- Exp411 checkpoint를 직접 변경하거나 검증 없이 이전
- NumPy/Numba 수치, tag 또는 quantile parity 실패
- 전체 coverage 또는 9개 산출물 hash 검증 실패

실패 경로는 기존 lane, prediction, Exp411 checkpoint와 Exp411 실패
상태를 변경하지 않는다.

## 13. 테스트 전략

단위 테스트는 다음을 고정한다.

- 내부, 시작, 끝과 혼합 결측의 정확한 float32 결과
- `NaN`, `+Inf`, `-Inf`의 동일한 결측 취급
- 전체 결측 run의 dataset-level abstain
- 비결측 bytes와 SHA256 무변경
- 결측 위치 및 복구 결과 SHA256 결정성
- 복구 후 identity regrouping과 fold 격리
- SQL trace의 TEST와 `labels_blob` 미접근
- 현재 DB의 read-only `19 / 164 / 983 / 0` inventory
- 검증된 checkpoint 이전과 dataset별 재계산 fallback
- 비결측 대표 dataset의 Exp411 parity
- 결측 대표 dataset의 NumPy/Numba parity
- 중간 실패 시 최종 디렉터리 미생성
- 정확히 9개 정본 출력과 manifest hash

대표 smoke는 합성 dataset과 소수의 실제 정상 TRAIN dataset만 사용한다.
실제 DB의 1,117개 전체 실행은 smoke나 단위 테스트가 아니며, 구현·검토,
benchmark와 실행 전 검증이 모두 통과한 뒤 사용자의 별도 명시적 승인을
받아 시작한다.
