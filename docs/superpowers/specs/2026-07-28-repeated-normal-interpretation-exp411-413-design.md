# 반복 정상 해석 정정 및 Exp411-413 설계

Date: 2026-07-28
Status: approved design, implementation not started
Audience: 연구 책임자, 후속 개발자, 미래 Codex 작업

## 1. 배경과 결정

현재 데이터의 정상 TRAIN/TEST 정확 중복은 단순한 데이터 결함으로만
해석하지 않는다. 이 프로젝트에서는 같은 설비·recipe·sensor·step 조합이
안정적으로 반복되는 반도체 운전 조건을 모사하는 의도적
`original repeated-normal` 시나리오로 해석한다.

쉬운 예로, 같은 recipe를 반복 실행한 정상 wafer trace가 어제와 오늘
같거나 매우 비슷하게 나오는 것은 바람직한 현상이다. 탐지기가 이런 반복
정상에 계속 경보를 내면 사용자는 경보를 신뢰하지 않게 된다. 따라서
정상 반복을 정확히 인식하여 false alarm을 억제하는 능력은 프로젝트의
핵심 목표와 맞는다.

이 해석은 다음 기존 근거와 일치한다.

- `docs/archive/task_history_20260709_before_compaction.md`는 정상 반복을
  실제 설비/recipe 운영에서 정상 패턴이 반복되는 상황을 흉내 낸 의도적
  구조로 기록한다.
- `docs/superpowers/specs/2026-07-15-exp164-train-normal-agreement-risk-audit-design.md`는
  데이터셋 하나를 가상 equipment/recipe/sensor/step 환경으로, 정상 TRAIN
  instance 하나를 정상 virtual wafer run trace로 해석한다.
- `docs/time_series_imaging_research_20260707.md`는 정상 설비의 recipe별 상태
  전이가 안정적이라는 연구 가정을 기록한다.
- Exp290은 TRAIN 정상과 byte-for-byte 같은 Hard 후보 35개를 자동 정상으로
  분리했고, 사후 평가에서 35개 모두 FP이며 제거된 TP는 0개였다.
- Exp291은 이 exact-normal guard를 현재 동결 routing에 포함했다.
- Exp410은 정상 정확 중복률이 `>=90%`인 층의 family-balanced F1이
  `0.649358068823`, `50-<90%` 층은 `0.415837057662`임을 보고했다.

다만 다음 두 한계를 함께 둔다.

1. UCR 기반 벤치마크의 같은 waveform은 실제 반도체 현장의 동일 wafer
   run이라는 뜻이 아니다.
2. 반복 정상 조건에서의 좋은 결과는 미관측 설비, 새 recipe 또는 실제
   현장에 대한 prospective 성능을 증명하지 않는다.

따라서 반복 정상은 제거할 오염이 아니라 주요 운영 조건이고,
`clean-balanced`는 원본을 대체하는 정본이 아니라 반복성이 약해졌을 때의
보조 스트레스·하한 검증으로 둔다.

## 2. 수정 범위

### 수정하는 문서

- `docs/PROJECT_RESEARCH_CONTEXT_KO.md`
  - 정상 중복을 단순 위험으로만 기술한 문장을 바로잡는다.
  - Exp290/291의 exact-normal 자동 정상 guard와 Exp410 결과를 현재 연구
    흐름에 추가한다.
  - `overlap-aware reporting`을 `repeatability-aware reporting`으로
    확장하고, 반복 정상의 운영상 가치와 일반화 주장 경계를 함께 적는다.
- `docs/REPEATED_NORMAL_INTERPRETATION_KO.md`
  - 반복 정상에 관한 공식 decision record를 새로 만든다.
  - 의도, 기존 근거, 허용되는 주장, 금지되는 주장, 후속 실험 연결을 한곳에
    기록한다.

### 보존하는 기록

다음은 당시 계약과 결과를 보존해야 하므로 소급 수정하지 않는다.

- `docs/EXP410_PREREGISTRATION_KO.md`
- `/Users/mac/Documents/Dataset/exp410_family_robustness/` 아래 결과와 manifest
- Exp290/291 사전등록, 설계, prediction 및 결과
- 기존 역사 문서와 이미 완료된 실험의 사전등록
- `docs/superpowers/specs/2026-07-28-project-research-context-design.md`

현재 공식 맥락 문서의 정정은 운영 lane, prediction, threshold, source,
alpha 또는 review budget을 변경하지 않는다.

## 3. Exp410 결과의 올바른 해석

Exp410의 `severe_family_imbalance`와 반복 정상의 가치는 서로 다른 질문이다.

- `severe_family_imbalance`는 큰 family가 dataset-macro 결과에 더 많은 표를
  가져가며 family별 성능 차이가 크다는 뜻이다.
- 높은 exact-overlap 층의 더 높은 F1은 반복 가능한 정상 조건에서 현재
  시스템이 상대적으로 강하다는 긍정적 관찰이다.
- 두 overlap 층은 family 구성, TRAIN 크기, anomaly 수가 동시에 다를 수
  있으므로 이 차이를 overlap 하나의 인과 효과라고 부르지 않는다.

쉬운 예로, 대형 family에 쉬운 반복 정상 문제가 많이 들어 있으면 전체
dataset 평균은 높아질 수 있다. 이때 “반복 정상을 잘 처리한다”와
“모든 family에 고르게 잘한다”는 동시에 참이 아닐 수 있다.

## 4. 후속 실험의 전체 구조

후속 실험은 exact-normal guard의 장점을 다시 발견하는 실험이 아니다.
이미 안전성이 확인된 exact equality에서 출발하여, 매우 유사한 정상까지
확장할 수 있는지를 세 단계로 검증한다.

```text
Exp411: 정상 TRAIN만으로 near-repeat 규칙을 자격 심사
   ↓ TRAIN-only gate를 통과한 규칙이 있을 때만
Exp412: 동결 규칙을 TEST 값에 label-blind 적용하고 Shadow로 사후 평가
   ↓ TP 손실 0과 사전등록 견고성 조건을 만족할 때만
Exp413: family·작은 표본·가속 parity를 독립 감사하고 다음 단계 여부 판정
```

Exp411-413 어디에서도 현재 Hard, Standard, Priority 또는 exact-normal lane을
변경하지 않는다. 자동 정책 변경은 세 실험의 결과가 모두 긍정적이어도
별도 Exp414 사전등록과 사용자 승인 없이는 수행하지 않는다.

## 5. Exp411 — TRAIN-only Near-Repeat 자격 심사

### 쉬운 의도

“완전히 같은 정상”보다 조금 달라도 같은 안정 운전으로 볼 수 있는 범위를
정답을 전혀 보지 않고 정한다.

예를 들어 같은 recipe의 압력 곡선이 센서 잡음 때문에 아주 조금 달라졌을
때, 이를 이상으로 보지 않을 수 있는지를 정상 TRAIN끼리의 비교와 합성
교란만으로 검사한다.

### 입력 경계

Exp411 prediction·qualification은 다음만 읽는다.

- 동결된 1,117개 dataset identity와 DB snapshot
- 각 dataset의 정상 TRAIN `instance_index`와 `values_blob`
- Exp407에서 동결한 TRAIN-only 합성 교란 정의와 provenance
- Exp290 exact-normal 정의

TEST split, TEST label, 실제 이상 위치, 과거 family별 TEST 성능, 기존
family별 source 순위는 읽지 않는다.

### Identity-grouped 교차적합

정상 TRAIN의 byte SHA256이 같은 instance는 하나의 identity group으로
묶는다. 같은 group의 복제본이 fit과 query 양쪽에 갈라져 들어가지 않도록
group 전체를 같은 fold에 둔다.

- 서로 다른 identity group이 5개 미만이면 `abstain`
- 5-19개이면 deterministic leave-one-identity-group-out
- 20개 이상이면 deterministic 5-fold group split
- seed는 `411`
- query와 같은 길이의 fit identity group이 5개 미만이면 해당 query를
  `length_support_abstain`으로 기록

### 고정 near-repeat 표현

같은 길이의 waveform끼리만 비교하며 길이 보간이나 tail replacement를
사용하지 않는다. 각 fold의 fit TRAIN만으로 scale을 계산하고 다음 세
거리를 만든다.

1. `raw_rms`: 원 waveform의 RMS 거리
2. `diff_rms`: 1차 차분 waveform의 RMS 거리
3. `spectrum_rms`: DC를 제외한 `rfft` magnitude의 RMS 거리

각 거리는 fit TRAIN 전체 값의 median absolute deviation으로 나눈다.
MAD가 `1e-12` 이하인 view는 사용할 수 없는 view로 표시한다. query의
거리값은 같은 길이의 다른 fit identity group까지의 최소 거리다.

fit identity group 각각에도 자기 group 전체를 제외한 동일 계산을 적용해
교차적합 reference를 만든다. 각 view의 낮은 거리 quantile은 결과를 보기
전에 `0.01`, `0.05`, `0.10`으로 고정하고 NumPy `method="lower"`를 쓴다.

candidate `near_qXX`는 다음을 모두 만족해야 한다.

- `raw_rms`가 해당 quantile 이하
- 사용 가능한 세 view 중 두 개 이상이 해당 quantile 이하
- 비교 가능한 view가 두 개 이상
- exact byte match가 아님

exact byte match는 기존 `exact`로 별도 기록하며 near 후보 수에 포함하지
않는다.

### TRAIN-only 안전성 gate

Exp407과 동일 provenance의 합성 교란을 held-out 정상 query에 적용한다.
near 규칙이 합성 교란까지 정상으로 받아들이는 비율을
`synthetic_false_normal_rate`로 정의한다.

각 quantile은 다음을 모두 만족해야 자격을 얻는다.

- 정상 held-out query의 near 인정률이 `1%` 이상
- pooled `synthetic_false_normal_rate <= 0.001`
- perturbation 종류별 `synthetic_false_normal_rate <= 0.005`
- family-cluster bootstrap 95% 상한이 `0.005` 이하
- CPU 1-worker와 canonical worker 설정의 tag SHA256이 동일

여러 quantile이 통과하면 가장 넓은 quantile을 선택한다. 이 선택은 오직
정상 TRAIN과 TRAIN에서 만든 합성 교란만 사용한다. 하나도 통과하지 않으면
`no_qualified_near_rule`로 종료하고 Exp412 prediction을 만들지 않는다.

### 결과와 주장

Exp411은 near-repeat 규칙의 TRAIN-only 자격만 주장한다. 실제 TEST false
alarm 감소, anomaly recall, 반도체 현장 성능 또는 정책 승격을 주장하지
않는다.

## 6. Exp412 — 동결 Near-Repeat Shadow 검증

### 쉬운 의도

Exp411에서 정답 없이 고른 단 하나의 규칙을 현재 TEST waveform에 적용한
뒤, 답안지를 나중에 열어 실제 이상을 정상으로 잘못 숨기지 않는지 확인한다.

예를 들어 현재 Hard 경보 10개 중 두 개가 TRAIN 정상과 매우 비슷하다고
판정되더라도 즉시 경보에서 빼지 않는다. 먼저 `near-repeat Shadow normal`
표시만 붙이고, prediction을 동결한 다음 두 개가 FP인지 TP인지 평가한다.

### Prediction 경계

prediction 단계는 다음만 읽는다.

- Exp411의 선택된 quantile, config hash와 qualification manifest
- Exp291의 동결 `automatic_hard_indices`
- Exp290의 동결 `automatic_normal_exact_indices`
- TRAIN/TEST `instance_index`와 `values_blob`

SQL과 prediction artifact는 TEST label, `labels_blob`, TP, FP, FN, F1,
과거 family별 TEST 성능을 포함하지 않는다.

모든 TEST instance에 `exact`, `near`, `unfamiliar`, `abstain` tag를 붙이되,
운영 검토 대상은 다음 counterfactual 집합으로만 기록한다.

```text
near_repeat_shadow_normal
  = automatic_hard_indices
    ∩ near_indices
    - automatic_normal_exact_indices
```

이 집합은 기존 Hard에서 실제로 제거하지 않는다. Standard, Priority,
Shadow, exact-normal lane도 그대로 보존한다.

### 동결 후 평가

prediction JSONL과 manifest SHA256을 동결한 뒤 별도 evaluator가 TEST
label을 읽는다. 다음을 모두 보고 하나의 variant를 재선택하지 않는다.

- near-repeat Shadow 수, TP, FP
- counterfactual Hard의 TP, FP, FN, precision, recall, dataset-macro F1
- family-macro F1과 family별 제거 TP/FP
- TEST anomaly 수와 TRAIN identity-group 수 층화
- Exp410 exact-overlap bucket별 결과
- family-cluster bootstrap 2,000회, seed `412`

Exp413으로 진행할 최소 조건은 다음과 같다.

- `near_repeat_shadow_normal`에서 제거되는 TP가 정확히 0
- 제거되는 FP가 1개 이상
- 기존 Hard, review 및 exact-normal lane 파일의 byte hash 불변
- 1,117개 coverage, index 범위, lane overlap, manifest 검증 오류 0

조건을 만족하지 못하면 near-repeat 자동 정상 확장안은 기각한다. 결과를
보고 quantile, distance, view, 합의 수를 변경하지 않는다.

## 7. Exp413 — Family 견고성 및 재현성 감사

### 쉬운 의도

Exp412가 좋아 보이더라도 특정 family 몇 개에서만 우연히 성공했는지
확인한다.

예를 들어 FP 10개를 줄였더라도 9개가 하나의 대형 family에서만 나왔다면
전체 설비 유형에 고르게 유용하다고 말할 수 없다. 반대로 여러 family에서
TP 손실 없이 조금씩 FP를 줄였다면 더 강한 후속 근거가 된다.

### 감사 범위

Exp413은 Exp412의 단 하나의 동결 prediction만 읽으며 새 threshold나
variant를 만들지 않는다.

- family-macro와 dataset-macro 효과 비교
- leave-one-family-out 영향도
- singleton, `2-5`, `6-20`, `21+` family 크기 층화
- TEST anomaly `1`, `2-10`, `11+` 층화
- TRAIN identity group `<5`, `5-19`, `20+` 층화
- exact-overlap `50-<90%`, `>=90%` 층화
- family-cluster bootstrap 2,000회, seed `413`
- CPU와 채택된 가속 backend의 tag/index parity 재검증

Exp413은 `promotion_research_worthy` 또는 `rejected`만 판정한다.
`promotion_research_worthy`도 운영 승격이 아니며 다음 조건을 모두 뜻한다.

- Exp412의 TP 손실 0 유지
- 감소 FP가 둘 이상의 family에 존재
- 어느 leave-one-family-out 결과에서도 TP 손실이 새로 나타나지 않음
- 모든 parity, provenance, coverage, 불변성 검사 통과

실제 자동 정상 lane 변경은 Exp414의 별도 사전등록, 구현 계획, 독립 검토와
사용자 실행 승인이 필요하다.

## 8. Numba·PyTorch·MPS 설계와 MLX 제외 결정

모든 실험은 CPU 정본 fallback을 먼저 제공한다.

사용자의 2026-07-28 명시적 결정에 따라 Exp411 이후 이 연구 흐름에서는
MLX를 구현, 설치, benchmark 또는 정본 후보로 고려하지 않는다. manifest의
가속 평가에는 `mlx_status: excluded_by_user_decision`만 기록한다.

### Exp411

동일 길이 waveform의 batched RMS와 `rfft` 거리 행렬은 tensor 적합성이
있다. NumPy CPU를 기준 구현으로 만들고 다음 경로를 benchmark한다.

- Numba CPU JIT
- PyTorch CPU
- PyTorch MPS

가속 경로는 동일 input·fold·seed에서 모든 exact/near/abstain tag와 선택
quantile이 CPU와 완전히 같아야 한다. 거리 수치의 최대 절대차는
`1e-6`, quantile 경계에서 차이가 나면 CPU를 유지한다.

### Exp412

TRAIN–TEST batched distance와 `rfft`가 tensor 적합하다. Exp411에서 parity를
통과한 backend만 후보로 사용한다. candidate index가 하나라도 다르면 CPU로
fail closed한다.

### Exp413

CSV/JSON 검증, grouped 집계, bootstrap이 지배 작업이므로 CPU/NumPy가
정본이다. GPU 사용을 강제하지 않는다.

가속 경로가 정본이 되려면 Numba JIT compile 또는 device transfer,
compile, synchronization을 포함한 동일 stage benchmark에서 NumPy
CPU보다 최소 `1.1x` 빨라야 한다. Numba, llvmlite, PyTorch, macOS, device
및 backend version과 benchmark 결과를 manifest에 기록한다. 한 번에
하나의 MPS heavy process만 실행한다.

## 9. 공통 불변 원칙

- prediction에서 TEST label, 실제 이상 위치, 과거 family별 TEST 성능을
  사용하지 않는다.
- `mean_combined_f1`은 human-assisted 지표로만 취급한다.
- Priority, Standard 및 Shadow 후보를 자동 경보로 조용히 승격하지 않는다.
- tail replacement를 사용하지 않는다.
- TEST 결과를 보고 distance, quantile, view, 합의 수, threshold, source,
  alpha 또는 budget을 재선택하지 않는다.
- 현재 Hard 2,218, Standard 806, Priority 31과 exact-normal 35를
  Exp411-413에서 변경하지 않는다.
- 새 개선은 새 실험 번호와 사전등록으로 분리한다.
- DB, prediction, manifest, code와 config SHA256을 검증하며 기존 산출물을
  덮어쓰지 않는다.
- full run은 단위 테스트와 작은 smoke가 통과해도 사용자의 별도 명시적
  승인 전에는 시작하지 않는다.
- 상세 파생 결과와 민감한 로컬 artifact는 공개 GitHub에 게시하지 않는다.

## 10. 구현 단위와 문서 산출물

설계 승인 후 구현 계획은 다음 단위로 분리한다.

1. 공식 맥락 정정과 반복 정상 decision record
2. Exp411 사전등록·runbook·TRAIN-only qualification 구현
3. Exp411 전체 실행 결과를 동결한 뒤 Exp412 실행 가능 여부 판정
4. Exp412 사전등록·runbook·Shadow prediction/evaluation 구현
5. Exp412 결과 gate 통과 시 Exp413 사전등록·runbook·감사 구현

예정 저장소 문서는 다음과 같다.

- `docs/REPEATED_NORMAL_INTERPRETATION_KO.md`
- `docs/EXP411_PREREGISTRATION_KO.md`
- `docs/EXP411_RUNBOOK_KO.md`
- gate를 통과한 경우에만 `docs/EXP412_PREREGISTRATION_KO.md`
- gate를 통과한 경우에만 `docs/EXP412_RUNBOOK_KO.md`
- Exp412 gate를 통과한 경우에만 `docs/EXP413_PREREGISTRATION_KO.md`
- Exp412 gate를 통과한 경우에만 `docs/EXP413_RUNBOOK_KO.md`

Exp412/413의 이름과 목적은 이 umbrella 설계에서 고정하지만, upstream
manifest SHA256과 실제 선택 quantile은 직전 실험 결과가 동결된 뒤 해당
사전등록에 정확히 복사한다. 결과를 보고 알고리즘을 바꾸는 것은 허용하지
않는다.

## 11. 완료 기준

- 현재 공식 맥락 문서가 반복 정상을 의도적 운전 조건으로 명시한다.
- Exp290/291의 exact-normal 35건, TP 손실 0, FP 35 근거가 연결된다.
- Exp410의 high-overlap 결과를 긍정적으로 설명하되 인과·현장 일반화
  주장을 하지 않는다.
- clean-balanced를 보조 스트레스·하한 검증으로 명시한다.
- Exp411-413 각각의 쉬운 목적, 입력 경계, gate와 정책 불변성이 문서화된다.
- Numba·PyTorch·MPS 적합성, parity, 속도 gate, NumPy CPU fallback과
  사용자의 MLX 제외 결정이 명시된다.
- Exp410과 이전 실험의 사전등록·결과 artifact는 변경되지 않는다.
