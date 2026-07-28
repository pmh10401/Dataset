# Exp411 TRAIN-only Near-Repeat 자격 심사 사전등록

작성일: 2026-07-28  
상태: 전체 실행 및 결과 확인 전 동결  
실험 모드: `TRAIN-only near-repeat qualification`

## 1. 목적과 운영 맥락

반도체 설비에서는 같은 equipment/recipe/sensor/step 조합이 안정적으로
운전될 때 정상 trace가 같거나 매우 비슷하게 반복되는 것이 바람직하다.
이 프로젝트는 이를 의도된 repeated-normal 구조로 해석한다. 반복 정상에
불필요한 경보를 줄여 사용자 신뢰를 높이는 것이 단일 mean F1 최대화보다
중요하다.

Exp290/291의 exact-normal guard는 정상 TRAIN과 byte-for-byte 같은 기존
Hard 35개를 정상 lane으로 분리했고, 사후 평가에서 FP 35개를 줄이면서 TP
손실은 0이었다. Exp411은 이 기준을 바로 완화하지 않는다. 정상 TRAIN끼리의
거리와 TRAIN에서만 생성한 합성 교란으로 near-repeat 후보 규칙의 자격만
심사한다.

Exp411은 TEST false alarm 감소, anomaly recall, 반도체 현장 성능 또는
운영 정책 개선을 주장하지 않는다. Exp412 prediction, Exp413 감사,
Hard/Standard/Priority 변경과 자동 정상 lane 승격은 범위 밖이다.

## 2. 동결 입력과 기준값

| 항목 | 동결값 |
|---|---|
| 실험 | `Exp411` |
| 데이터셋 수 | 1,117 |
| family 수 | 128 |
| 데이터셋 universe | Exp404 manifest의 `frozen_hard_lanes` key |
| prediction manifest SHA256 | `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8` |
| DB SHA256 | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |
| 허용 DB snapshot | main DB `db`만 허용, WAL/SHM 거부 |
| 현재 동결 lane | Hard 2,218 / Standard 806 / Priority 31 |
| 현재 exact-normal | 35 |
| seed | `411` |
| quantile 후보 | `0.01 / 0.05 / 0.10` |
| quantile 방법 | NumPy `method="lower"` |
| bootstrap | family cluster 2,000회 |
| distance tolerance | `1e-6` |

입력은 위 manifest가 고정한 dataset identity, 읽기 전용 SQLite의 정상
TRAIN `instance_index`와 `values_blob`, Exp407 TRAIN-only 합성 교란
구현과 설정 provenance뿐이다.

prediction 및 qualification에서는 TEST split을 읽지 않는다. TEST label,
`labels_blob`, 실제 이상 위치, TP/FP/FN/F1, 과거 family별 TEST 성능,
과거 family별 source ranking도 읽지 않는다.

## 3. Identity-grouped 교차적합

정상 TRAIN의 원본 float32 blob SHA256이 같은 instance는 하나의 identity
group으로 묶는다. 같은 group은 fit과 query로 갈라지지 않는다.

- 서로 다른 identity group이 5개 미만이면 dataset을 `abstain`
- 5-19개이면 deterministic leave-one-identity-group-out
- 20개 이상이면 deterministic 5-fold group split
- query와 길이가 같은 fit identity group이 5개 미만이면
  `length_support_abstain`
- 모든 순서와 split seed는 `411`

같은 길이 waveform만 비교한다. 보간과 tail replacement는 사용하지 않는다.

## 4. 동결 거리와 tag 규칙

각 fold의 fit TRAIN만 사용해 다음 세 view를 계산한다.

1. `raw_rms`: 원 waveform RMS 거리
2. `diff_rms`: 1차 차분 waveform RMS 거리
3. `spectrum_rms`: DC를 제외한 `rfft` magnitude RMS 거리

각 view는 fit TRAIN 전체 값의 MAD로 나눈다. MAD가 `1e-12` 이하이면 그
view는 사용할 수 없다. query 거리는 같은 길이의 다른 fit identity
group까지의 최소 거리다. reference도 자기 identity group 전체를 제외한
동일 계산으로 만든다.

quantile은 결과를 보기 전에 `0.01`, `0.05`, `0.10`으로 고정하고
`method="lower"`를 쓴다. `near_qXX`는 다음을 모두 만족해야 한다.

- exact byte match가 아님
- `raw_rms`가 해당 quantile 이하
- 사용 가능한 view가 둘 이상
- 사용 가능한 view 중 둘 이상이 해당 quantile 이하

exact byte match는 `exact`, 비교 근거가 부족하면 `abstain`, 그 밖은
`unfamiliar`로 기록한다.

## 5. TRAIN-only 안전성 gate

Exp407의 고정된 일곱 교란
`spike / drift / level_shift / flat_line / amplitude_scale /
frequency_warp / phase_order`를 held-out 정상 TRAIN query에 적용한다.
이 합성 교란을 near 정상으로 받아들이는 비율을
`synthetic_false_normal_rate`로 정의한다.

각 quantile은 다음을 모두 만족해야 한다.

- held-out 정상 query near 인정률 `>= 0.01`
- pooled synthetic false-normal rate `<= 0.001`
- 종류별 최악 synthetic false-normal rate `<= 0.005`
- family-cluster bootstrap 95% 상한 `<= 0.005`
- CPU 정본과 채택 backend의 거리 오차 `<= 1e-6`, tag와 선택 quantile 동일

여러 후보가 통과하면 가장 넓은 quantile 하나를 선택한다. 하나도 통과하지
않으면 `no_qualified_near_rule`로 종료한다. 이 경우 Exp412 prediction을
만들지 않으며, 결과를 보고 quantile·거리·view·합의 수·합성 교란을
재선택하지 않는다.

## 6. Backend 결정

NumPy CPU가 정본이자 항상 가능한 fallback이다. Numba CPU JIT와 PyTorch
CPU를 parity 후보로 평가한다. PyTorch MPS는 build와 device 가용성을
probe하고, 사용할 수 없으면 이유를 기록한 뒤 CPU로 닫힌다.

가속 backend는 같은 입력에서 최대 절대 오차 `<= 1e-6`, tag identity와
선택 quantile identity를 모두 만족하고 transfer·compile·synchronization을
포함해 NumPy CPU보다 최소 1.1배 빠를 때만 정본 후보가 될 수 있다.

사용자 결정에 따라 MLX는 구현·설치·import·benchmark 대상에서 제외하고
`mlx_status: excluded_by_user_decision`만 기록한다.

## 7. Fail-closed 조건

다음 중 하나라도 발생하면 전체 결과를 게시하지 않는다.

- Git worktree가 dirty이거나 실행 중 commit이 바뀜
- population manifest 또는 DB SHA256 불일치
- 데이터셋 1,117개 또는 family 128개 불일치
- SQLite `PRAGMA quick_check` 실패
- WAL/SHM sidecar 출현 또는 실행 중 DB snapshot 변화
- TRAIN에 `0` 이외 label, 잘못된 blob, 중복 index 또는 비유한값 존재
- SQL trace에 TEST 또는 `labels_blob` 접근 존재
- identity group이 fit/query 양쪽으로 분리됨
- 동결 seed, quantile, gate, perturbation 또는 거리 정의 변경
- 임시 artifact hash 검증이나 원자적 directory rename 실패

실패 경로는 현재 Hard 2,218, Standard 806, Priority 31, exact-normal 35,
기존 prediction, checkpoint와 report를 변경하지 않는다.

## 8. 정본 출력과 주장 경계

정본 출력은 다음 8개 파일이다.

- `exp411_frozen_config.json`
- `exp411_dataset_qualification.csv`
- `exp411_quantile_summary.csv`
- `exp411_backend_assessment.json`
- `exp411_selected_rule.json`
- `exp411_summary.json`
- `EXP411_RESULT_REPORT_KO.md`
- `MANIFEST.json`

Exp411은 정상 TRAIN과 TRAIN-only 합성 교란에 대한 자격 심사 결과만
주장한다. `qualified`도 운영 승격이 아니다. mean_combined_f1은
human-assisted 지표로만 다루고 Priority review와 Shadow 후보를 자동
경보로 승격하지 않는다. TEST 결과를 본 뒤 alpha, threshold, source,
budget을 재선택하지 않는다. 새 개선은 새 실험 번호와 사전등록으로
분리한다.

## 9. 실행 승인 게이트

단위 테스트와 대표 smoke는 32개 합성 dataset만 사용한다. 실제 DB의
1,117개 전체 Exp411 자격 심사는 테스트나 smoke의 일부가 아니다.

모든 단위·회귀 테스트와 대표 smoke가 통과한 뒤에도 사용자의 별도 명시적 승인이
있어야만 존재하지 않는 새 출력 디렉터리로 전체 실행한다.
Exp411 결과가 동결되기 전에는 Exp412/413을 시작하지 않는다.
