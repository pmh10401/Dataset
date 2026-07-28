# Exp410 동결 Hard Family 견고성 감사 사전등록

작성일: 2026-07-28  
상태: 구현 및 결과 확인 전 동결  
실험 모드: `read-only retrospective family robustness audit`

## 1. 목적

Exp410은 Exp403-408에서 동결한 자동 경보 Hard 결과를 바꾸지 않고,
대규모 one-vs-all family가 데이터셋 수만큼 가중되는 영향을 감사한다. 기존
데이터셋 평균과 family마다 한 표를 주는 평균을 비교해 현재 결과가
family 크기 불균형에도 견고한지 기술한다.

이 감사는 새 prediction을 만들거나 모델을 학습하지 않는다. threshold,
source, alpha, route, review budget, Hard/Standard/Priority lane을 선택하거나
변경하지 않는다. 새 반도체 설비 데이터, unseen family, prospective
validation 또는 배포 성능을 대신하지 않는다.

## 2. 동결 입력과 기준값

| 항목 | 동결값 |
|---|---|
| 실험 | `Exp410` |
| 데이터셋 수 | 1,117 |
| family 수 | 128 |
| family 크기 bucket 수 | `1 / 74 / 39 / 14` |
| Hard 경보 수 | 2,218 |
| Hard TP / FP / FN | `1,915 / 303 / 2,003` |
| 데이터셋 macro F1 정확값 | `0.6089825086931719` |
| 데이터셋 macro F1 6자리 표시 | `0.608983` |
| prediction SHA256 | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` |
| frozen label SHA256 | `b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4` |
| DB SHA256 | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |
| Exp291 lane SHA256 | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` |
| Exp292 lane SHA256 | `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b` |
| 부동소수점 허용오차 | `1e-12` |
| operational policy changed | `false` |
| deployment eligible | `false` |
| unseen family validated | `false` |
| prospective validated | `false` |

감사는 검증된 Exp291/292 lane publication과 manifest,
`exp404_hard_confidence.jsonl`, `exp404_prediction_manifest.json`,
`exp403_408_frozen_test_labels.csv`,
`exp403_408_evaluation_input_manifest.json`,
`exp403_408_evaluation_publication_manifest.json`, 읽기 전용 SQLite DB,
Exp410 설정과 코드만 소비한다. 모든 고정 SHA256과 두 evaluation
manifest의 결합을 값을 파싱하기 전에 검증한다.

Exp406-408 review-only 후보, checkpoint, TEST 기반 source ranking,
과거 family별 TEST 성능, 존재하지 않는 난이도 원본을 입력으로 사용하지
않는다.

## 3. 데이터셋·family 계약

family는 데이터셋 식별자의 유일한 `_normal_` 앞부분이다. 앞부분은
비어 있지 않고 suffix는 부호 있는 10진 정수여야 한다. 동결 universe에는
`_normal_-1` 식별자 5개가 존재한다. 현재 1,117개 데이터셋은 정확히
128개 family로 매핑되어야 한다.

각 데이터셋에서 frozen TEST label과 Hard index로 Hard 수, TP, FP, FN,
precision, recall, F1을 독립 재계산한다. 중복 identity, 범위 밖 index,
`0/1` 이외 label, 빈 label, anomaly가 없는 고정 데이터셋은 거부한다.
경보가 없으면 precision은 `null`이며, 양성은 있지만 TP가 없으면 F1은
0이다.

family별로 다음 두 관점을 모두 게시한다.

1. member 데이터셋 metric의 산술평균
2. family 안에서 TP/FP/FN을 합친 pooled metric

1번 mean-dataset family F1이 기본 family 점수이며, 128개 기본 점수의
산술평균이 family-macro F1이다.

## 4. Family 가중치 민감도와 영향도

family `f`의 mean-dataset F1을 `m_f`, 데이터셋 수를 `n_f`, cap을 `c`라
하면 다음 값을 계산한다.

```text
sum_f min(n_f, c) * m_f / sum_f min(n_f, c)
```

cap은 결과를 보기 전에 `1`, `3`, `5`, `10`, `unlimited`로 고정한다.
cap 1은 family-macro이고 unlimited는 데이터셋-macro와 대수적으로
동일해야 한다. 결과를 본 뒤 한 cap을 공식값으로 재선택하지 않으며,
분석을 위해 데이터셋을 버리거나 다시 표본 추출하지 않는다.

각 family를 하나씩 제외한 진단표에는 데이터셋-macro F1과 pooled Hard
precision의 변화, 제거되는 TP/FP/FN/경보 수, 데이터셋 점유율과 전체 FP
점유율을 기록한다. 이 표는 진단용이며 실제 공식 결과에서 family를
제외하는 근거가 아니다.

## 5. 동결 층화

다음 경계를 결과 확인 전에 고정한다.

- family 크기: `1`, `2-5`, `6-20`, `21+`
- normal TRAIN 수: `<20`, `20-49`, `50+`
- exact normal TRAIN/TEST overlap: `<50%`, `50-<90%`, `>=90%`
- TEST anomaly 수: `1`, `2-10`, `11+`

exact overlap은 같은 데이터셋의 normal TRAIN `values_blob` SHA256 집합에
TEST normal `values_blob`의 정확한 바이트 SHA256이 존재하는지를 뜻한다.
보간, 정규화, tolerance 또는 embedding을 사용하지 않는다.

데이터셋 층화도 먼저 각 family 안의 해당 데이터셋을 평균하고, 해당
데이터셋이 하나 이상 있는 family들을 동일 가중 평균한다. 모든 층화
행에는 family와 데이터셋 분모를 함께 기록한다.

난이도 원본은 복원 환경에 없으므로 추정하지 않고 정확히 다음을 기록한다.

```json
{
  "difficulty_stratification": {
    "status": "not_evaluated",
    "reason": "source_artifact_unavailable"
  }
}
```

## 6. Family 단위 bootstrap과 판정

bootstrap은 데이터셋이 아니라 128개 family를 복원추출한다. 한 family가
선택되면 그 family의 모든 데이터셋 행이 함께 이동한다.

- 반복 수: `2,000`
- seed: `410`
- 매 반복 계산값: family-macro F1, 포함된 family cluster 안의
  dataset-macro F1
- paired 차이: `delta_f1 = family_macro_f1 - dataset_macro_f1`
- 출력: 점추정값과 percentile 95% 구간

기술적 견고성 판정은 다음으로 고정한다.

- `robust`: 95% 구간 하한이 `-0.05` 이상
- `sensitive`: 95% 구간 하한이 `-0.05` 미만
- `severe_family_imbalance`: 점추정값이 `-0.10` 미만

`severe_family_imbalance`가 `sensitive`보다 우선한다. 이 판정은 운영
게이트가 아니며 Hard를 변경하거나 threshold를 자동 선택하지 않는다.

## 7. Fail-closed 조건

다음 중 하나라도 발생하면 결과를 게시하지 않는다.

- 게시 시점 Git worktree가 dirty임
- 고정 SHA256 또는 manifest artifact hash 불일치
- SQLite `PRAGMA quick_check` 결과가 `ok`가 아님
- 실행 전후 DB/WAL/SHM identity 변화
- 데이터셋 1,117개, family 128개 또는 family bucket
  `1 / 74 / 39 / 14` 불일치
- Hard 2,218개 또는 TP/FP/FN `1,915 / 303 / 2,003` 불일치
- 데이터셋-macro F1이 정확값에서 `1e-12`보다 크게 벗어나거나 6자리
  표시가 `0.608983`이 아님
- Exp291/292 lane parity 또는 Exp404 Hard identity parity 실패
- 잘못된 데이터셋 이름, label, index, 누락, 중복 또는 범위 오류
- 고정 cap, bucket, bootstrap, seed, margin 변경
- 완전한 임시 출력 작성·hash·원자적 rename 실패

실패 경로는 기존 lane, prediction, checkpoint, report를 변경하지 않는다.

## 8. 출력과 주장 경계

정본 출력은 다음과 같다.

- `exp410_frozen_config.json`
- `exp410_dataset_metrics.csv`
- `exp410_family_metrics.csv`
- `exp410_weight_sensitivity.csv`
- `exp410_family_influence.csv`
- `exp410_stratified_metrics.csv`
- `exp410_summary.json`
- `EXP410_RESULT_REPORT_KO.md`
- `MANIFEST.json`

Exp410은 dataset-macro와 family-macro 차이, 사전등록 cap 민감도,
family 영향도, 고정 층화 결과, 기술적 견고성 판정만 주장할 수 있다.
unseen family, prospective, 반도체 현장, deployment, 자동 경보 개선,
모델 우월성 또는 정책 변경을 주장할 수 없다.

prediction에서 TEST label, 실제 이상 위치, 과거 family별 TEST 성능을
사용하지 않는다. `mean_combined_f1`은 human-assisted 지표로만 다루며
Priority review와 Shadow 후보를 자동 경보로 올리지 않는다. tail
replacement를 쓰지 않고, TEST 결과를 본 뒤 alpha, threshold, source,
budget을 재선택하지 않는다. 개선안은 새 실험 번호와 새 사전등록으로
분리한다.

## 9. PyTorch·MLX·MPS 평가

검증된 현재 환경은 Python 3.14.6, NumPy 2.3.5, PyTorch 2.13.0,
PyTorch MPS built `true`, MPS available `false`, MLX installed `false`이다.

Exp410의 지배 작업은 SQLite streaming, 정확한 blob byte hashing,
CSV/JSON parsing, grouped counter, 128-cluster bootstrap이다. tensor 중심
작업이 아니며 MPS/MLX backend도 현재 사용할 수 없다. byte payload를
tensor backend로 옮기거나 MLX를 새로 설치하면 transfer/compile 및 의존성
비용만 늘고 정확한 byte 의미를 흐릴 수 있다.

따라서 CPU SQLite streaming, `hashlib`, NumPy를 정본 경로와 CPU
fallback으로 고정한다. 새 가속 package를 설치하지 않는다. 이후 모든
실험 설계는 PyTorch·MLX·MPS에 대해 tensor 적합성, backend 가용성,
transfer/compile overhead, 결정론·수치 parity, 정본 CPU fallback을
명시한다. 비CPU 경로는 동일 입력·seed에서 후보 identity가 동일하고
사전등록 tolerance를 만족하며 전체 transfer/synchronization 시간을
포함해 정본 경로보다 최소 1.1배 빠른 benchmark가 있을 때만 정본 승격을
검토한다.

## 10. 실행 승인 게이트

구현과 단위 테스트는 compact fixture를 사용한다. 대표 smoke는 singleton,
중간, 대형 family와 가능한 TRAIN/overlap/anomaly 층을 포함하는 고정된
32개 합성 데이터셋만 사용하며 실행·schema·원자적 게시를 확인한다.
smoke 결과로 최종 견고성 판정을 주장하지 않는다.

약 1.6GB SQLite DB를 읽는 전체 Exp410 감사는 구현 검증에 포함하지 않는다.
모든 단위·회귀 테스트와 대표 smoke가 통과한 뒤에도 사용자의 별도 명시적
승인이 있어야만 새롭고 존재하지 않는 Exp410 출력 디렉터리로 실행한다.
