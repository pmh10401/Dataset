# Exp441 기존 Hard 미탐 데이터셋 단발 복구 설계

## 한눈에 보는 결론

Exp441은 새 모델을 다시 학습하는 실험이 아니다. Exp440이 이미 정상 TRAIN만으로
만들어 둔 저주파 임시 후보를 다시 계산하지 않고, 그중 **기존 Hard가 경보를 하나도
내지 못한 데이터셋에서 가장 강한 후보 한 개만** 고른다. 전체 추가 후보는 최대
30개다.

쉽게 말하면 다음과 같다.

```text
기존 Hard가 이미 찾은 데이터셋       -> 건드리지 않음
기존 Hard가 하나도 찾지 못한 데이터셋 -> Exp440 후보 중 가장 강한 1개만 검토
전체 추가 후보                         -> 최대 30개
```

이 구조는 기존에 잘하던 데이터셋의 점수를 떨어뜨리지 않으면서, 완전히 놓친
데이터셋을 조심스럽게 복구하는 데 목적이 있다.

## 연구 질문

Exp440의 동결된 저주파 후보를 아주 보수적으로 줄이면, Exp404 Hard가 완전히
놓친 데이터셋에서 새 이상을 찾으면서 다음 세 가지를 동시에 지킬 수 있는가?

1. 기존 강한 데이터셋의 성능을 떨어뜨리지 않는다.
2. 추가 오탐을 최대 30개 안에 묶는다.
3. 전체 1,117개 데이터셋에서 이미 관찰된 micro/macro F1 개선을 같은 동결 규칙으로
   재현하고, 그 결과가 사후 선택된 설명값임을 분명히 남긴다.

## 왜 Exp440을 그대로 승격하지 않는가

Exp440은 1,117개 데이터셋을 오류 없이 끝냈고 prediction SHA256도 동결했다.
그러나 사후 평가에서 후보 100개 중 TP 40, FP 60이었다. micro F1은
`+0.0028193532` 증가했지만 bootstrap 95% 신뢰구간 하한이
`-0.0008748177`이었고, 강한 데이터셋의 최대 F1 손실은 `0.5`였다. 따라서
Exp440 전체 후보를 Hard에 합치는 방법은 승자가 아니다.

Exp440의 문제는 저주파 신호가 전혀 쓸모없다는 것이 아니라, 너무 넓은 곳에
후보를 추가했다는 것이다. Exp441은 점수를 새로 만들지 않고 선택 범위를 안전한
구역으로 줄인다.

## 검토한 방법

### A. Hard 미탐 데이터셋에서 한 개만 복구 — 선택

Hard 좌표가 하나도 없는 데이터셋만 허용하고, 데이터셋마다 가장 강한 후보 한
개만 남긴 뒤 전역 상위 30개를 선택한다.

장점:

- 기존 Hard가 잘하던 데이터셋은 전혀 건드리지 않는다.
- 데이터셋당 한 후보라서 한 곳에 오탐이 몰리지 않는다.
- 30개 예산 때문에 추가 FP는 최악에도 30개를 넘지 않는다.
- Exp440 전체 점수 계산을 다시 하지 않아 빠르고 재현하기 쉽다.

한계:

- Hard가 일부만 맞힌 데이터셋의 추가 이상은 복구하지 않는다.
- 같은 TEST 집단을 여러 번 분석한 뒤 만든 후속 규칙이므로 독립 검증이 아니다.

### B. 모든 데이터셋에서 심각도 순으로 30개 선택

탐색 결과 후보 정밀도는 높았지만, Hard가 이미 잘하던 데이터셋에도 후보가 들어가
강한 데이터셋 손실을 구조적으로 막지 못했다. 이번 우선순위인 안전성에 맞지 않아
선택하지 않는다.

### C. 새 주파수 대역 또는 새 모델로 독립 확인

과학적으로 중요한 다음 연구다. 다만 구현과 전체 점수 계산이 다시 필요하고 성공
여부가 불확실하다. Exp441의 안전한 선택기를 먼저 검증한 뒤 별도 Exp442에서
진행한다.

## 과학적 해석의 한계

Exp441 예측 코드는 TEST label, 실제 이상 위치, 과거 family별 TEST 성능을 읽지
않는다. 그러나 이 선택 전략 자체는 Exp440 사후 평가를 본 뒤 정했다. 따라서
다음 값을 모든 문서와 산출물에 명시한다.

```text
adaptive_followup = true
independent_validation = false
external_generalization_supported = false
official_run_purpose = deterministic_retrospective_reproduction
confirmatory_statistical_claim = false
bootstrap_interpretation = descriptive_post_selection
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
```

Exp441이 아래 판정표를 모두 통과해도 새 내부 챔피언이나 통계적으로 검증된 개선으로
승격하지 않는다. Exp404 Hard/Exp416 감사본이 계속 검증 기준이다. Exp441이 줄 수
있는 결론은 "동결된 선택 규칙과 이미 본 결과가 독립된 label 차단 실행에서도
정확히 재현되었다"까지다. 새 설비나 보지 않은 외부 데이터에서도 좋아진다고
주장하지 않으며 자동 경보로 승격하지 않는다.

## 동결 신원과 입력

```text
experiment_id = Exp441
run_identity = exp441-hard-empty-single-shot-recovery-antigravity-20260808
dataset_count = 1117
run_root = /Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
hard_prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
hard_manifest_sha256 = 41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8
hard_coordinate_identity_sha256 = 858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322
exp440_prediction_sha256 = 06d905d9d6f96fa67704111b984ed87b74229e13fd0690401db67a4a0fdd79af
exp440_manifest_sha256 = 3778a0e8770e06f4be87d1b87281c858518169c377ee6c53b78f32ce01d39e85
exp440_artifact_map_sha256 = 7a083fa8f1287205d3469e36f2f88b95f72adecd90b3b79167aad673b3439a1a
exp440_checkpoint_context_sha256 = d598a159d798f0f1dbdcbf684ef7994c89c982b60bce62b877a82c7e04ba7dfc
exp440_checkpoint_state_sha256 = 486235b70856247912a02c115748e22477ea11224d3a50de088187903da56268
exp440_checkpoint_retained_sha256 = c31c0414dced3f463d187e8e17adf84a50eb0c7cf86a6e8399f9074fda94ccf2
candidate_budget = 30
candidate_per_dataset = 1
bootstrap_seed = 44120260808
bootstrap_samples = 20000
bootstrap_quantile_method = linear
bootstrap_numpy_version = 2.4.6
bootstrap_bit_generator = PCG64
bootstrap_index_dtype = int64
bootstrap_endpoint = false
reproduction_float_abs_tolerance = 1e-15
reproduction_float_rel_tolerance = 0.0
```

고정 경로 허용 목록은 다음과 같다.

```text
database = /Users/mac/Documents/Dataset/univariate_ts.db
exp440_checkpoint = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity
exp440_context = $exp440_checkpoint/context_antigravity.json
exp440_state = $exp440_checkpoint/state_antigravity.json
exp440_records = $exp440_checkpoint/records_antigravity
exp440_retained_inventory = $exp440_checkpoint/retained_candidates_antigravity.json
exp440_publication = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/publication_antigravity
hard_prediction = /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl
hard_manifest = /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json
forbidden_exp440_evaluation_root = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/evaluation_antigravity
forbidden_exp440_evaluation_module = experiments/exp440_train_only_low_frequency_precision_gate_antigravity/evaluation_antigravity.py
```

Exp440 publication은 `artifact_sha256_antigravity.json`을 포함한 정확히 10개 일반
파일이어야 하며, artifact map이 선언한 나머지 9개 파일을 모두 descriptor로 읽어
해시를 다시 확인한다. checkpoint는 context, state, 1,117개 record,
`retained_candidates_antigravity.json`만 허용한다. retained 파일은 위 SHA로 무결성만
확인하고 선택에는 사용하지 않는다. 이 정확한 inventory 밖의 파일, symlink, FIFO,
디렉터리는 실패다.

예측 단계에서 허용하는 입력은 다음뿐이다.

- Exp440의 동결 checkpoint context, state, 1,117개 record와 무결성 전용 retained inventory
- Exp440의 동결 publication 10개 파일 전체; 선택 의미에는 context/state/record와
  publication prediction만 사용
- Exp404 Hard prediction manifest와 2,218개 좌표
- readiness가 검증한 clean reviewed implementation commit의 Exp441 고정 설정과
  코드 신원

예측 단계는 SQLite 데이터베이스를 열지 않는다. 따라서 TRAIN/TEST 값과 TEST label
모두 읽지 않는다. 데이터베이스는 prediction SHA가 동결된 뒤 명시적인 평가 명령에서
TEST `instance_index,label`만 읽을 때 사용한다.

Exp440 evaluation 디렉터리는 Exp441 예측 프로세스의 경로 허용 목록에 들어가지
않는다. 모듈 import, 문자열 경로, 파일 열기 모두 금지한다.

## 동결 선택 규칙

### 1. 원본 확인

readiness는 먼저 다음을 모두 확인한다.

1. Exp440 context와 state의 SHA256이 위 값과 같다.
2. context에는 정렬되고 중복 없는 1,117개 데이터셋이 있다.
3. state의 `completed`는 1,117이고 record digest도 정확히 1,117개다.
4. `context["datasets"][sequence]`와
   `records_antigravity/{sequence:04d}_record_antigravity.json`의 dataset이
   정확히 대응하며 sequence는 0부터 1116까지 빈틈이 없다.
5. 각 record는 일반 파일이며 state에 기록된 digest와 실제 바이트가 같다.
6. 모든 record가 Exp440의 정확한 계약으로 다시 파싱된다.
7. 각 provisional candidate와 Hard 좌표의 `instance_index`가 해당 record의
   `test_row_count` 범위 안에 있는지 별도로 확인한다. Exp440 parser 자체가 이 상한을
   보장한다고 가정하지 않는다.
8. Exp440 publication의 정확한 10-file inventory와 모든 artifact-map hash를 다시 검증한다.
9. Hard manifest와 2,218개 좌표를 SHA256, 구조, canonical coordinate identity로
   다시 검증한다.
10. Exp440 evaluation 경로와 TEST label은 열지 않았고 label access count는 0이다.

현재 동결 원본에서 예상되는 label-free 구조 수는 다음과 같다. 이 수가 다르면
선택하지 않고 실패한다.

```text
Exp440 provisional candidates = 840
outside-Hard provisional candidates = 394
Hard-empty eligible provisional candidates = 73
eligible datasets with candidates = 49
retained candidates = 30
```

이 값은 성능 점수가 아니라 동결 원본이 바뀌지 않았는지 확인하는 수다.

### 2. Hard 미탐 데이터셋만 허용

전체 1,117개 데이터셋 각각에 대해 frozen Hard 좌표 수를 센다. 좌표 수가 정확히
0인 데이터셋만 선택 대상이다. Hard 좌표가 하나라도 있으면 Exp441은 그 데이터셋의
모든 후보를 버린다.

이 규칙은 TEST label이나 과거 성능표가 아니라 이미 동결된 label-free Hard 출력만
사용한다. 따라서 다음 출처 표시를 publication에 남긴다.

```text
uses_frozen_test_candidate_empty_gate = true
uses_test_labels_for_prediction = false
```

### 3. 데이터셋마다 한 후보만 선택

Exp440 provisional candidate는 이미 두 개 이상의 안정적인 저주파 깊이에서 문턱을
넘은 좌표다. `agreement_count`는 이 자격 확인에만 쓰고 순위에는 쓰지 않는다.

같은 데이터셋 안에서는 아래 순서로 첫 후보 하나를 고른다.

1. `second_excess_ratio` 큰 값 우선
2. `second_empirical_rank` 작은 값 우선
3. `instance_index` 작은 값 우선

### 4. 전역 30개 선택

데이터셋별 우승 후보를 아래 순서로 정렬해 처음 30개만 남긴다.

1. `second_excess_ratio` 큰 값 우선
2. `second_empirical_rank` 작은 값 우선
3. dataset 이름 오름차순
4. `instance_index` 오름차순

모든 실수는 반올림하거나 문자열로 바꾸지 않고 원본 Python float 값으로 비교한다.
dataset 문자열은 대소문자를 구분하는 Python 기본 순서로 비교한다. 완전히 같은
좌표가 두 번 나오면 tie-break하지 않고 source 오류로 실패한다.

임계값, 저주파 깊이, Exp440 점수, alpha, source, dataset 이름별 예산은 바꾸지
않는다. family별 과거 TEST 성능, 수동 선택, tail replacement도 사용하지 않는다.

## 이 선택기가 구조적으로 보호하는 것

Hard가 아무 좌표도 고르지 않은 데이터셋의 baseline F1은 이 프로젝트의 고정 정의상
0이다.

- 추가 후보가 FP이면 그 데이터셋 F1은 0으로 그대로다.
- 추가 후보가 TP이면 그 데이터셋 F1은 0보다 커진다.

따라서 선택 규칙이 정확히 지켜지면 dataset macro F1은 내려갈 수 없고, baseline
F1이 0.8 이상인 강한 데이터셋도 건드릴 수 없다. 후보가 최대 30개이므로 추가 FP도
30개를 넘을 수 없다. micro F1과 신뢰구간은 TP/FP 비율에 따라 달라지므로 사후
평가로 반드시 확인한다.

## 탐색 replay의 위치

설계 전에 같은 동결 자료를 읽기 전용으로 확인한 결과는 다음과 같다.

```text
retained = 30
TP = 15
FP = 15
candidate precision = 0.50
rescued datasets = 15
regressed datasets = 0
baseline confusion = TP 1915 / FP 303 / FN 2003 / TN 191679
Hard union Exp441 confusion = TP 1930 / FP 318 / FN 1988 / TN 191664
baseline micro F1 = 0.6241851368970013
Hard union Exp441 micro F1 = 0.6260136230943887
micro F1 delta = +0.0018284861973874023
micro paired-bootstrap 95% CI = [0.00005373870255999115, 0.003814109586073529]
macro F1 delta = +0.012444046553267682
macro paired-bootstrap 95% CI = [0.006535362578334826, 0.019158460161145925]
maximum strong-dataset loss = 0
```

이는 구현 가능성을 확인한 **사후 탐색값**이다. 정확한 선택 규칙과 TP/FP/F1 및
bootstrap 구간을 모두 같은 TEST label에서 이미 얻었으므로, Exp441 공식 실행은
새 확인 실험이 아니라 이 결과의 결정론적 retrospective reproduction이다. 새
prediction freeze는 runtime label 누수를 막고 provenance를 동결하는 데 필요하지만,
이미 발생한 사후 선택을 되돌리거나 bootstrap을 confirmatory interval로 바꾸지는
못한다. 공식 평가가 같은 구간을 다시 내더라도 이는 descriptive/post-selection
interval이며 통계적 일반화 증거가 아니다.

## 실행 구조

Exp441은 Exp440 디렉터리와 산출물을 수정하지 않는 별도 패키지, wrapper, 문서,
run root를 사용한다. 주요 단위는 다음과 같다.

- source verifier: Exp440 checkpoint/publication과 Hard를 읽기 전용으로 검증
- candidate loader: 1,117개 record에서 provisional candidate를 정확히 복원
- pure selector: Hard-empty, 데이터셋당 1개, 전역 30개 규칙만 수행
- progress writer: PID, 현재 record, 단계, 처리 수, 속도, 경과시간, ETA, 오류 기록
- publisher: prediction과 근거를 create-only 임시 디렉터리에 쓰고 원자적으로 동결
- inspector: 모든 해시와 선택 규칙을 재계산하되 파일을 바꾸지 않음
- evaluator: prediction 동결 뒤에만 TEST label을 읽는 별도 lazy-import 모듈

CLI는 고정 경로와 고정 규칙만 제공한다.

```text
readiness
new
inspect
evaluate
```

경로, source, 예산, 순위, seed, bootstrap 횟수, backend를 바꾸는 옵션은 제공하지
않는다. 작업이 짧고 source를 다시 계산하지 않으므로 `resume` 명령은 만들지 않는다.
각 `new`는 `$RUN_ROOT/attempts_antigravity/attempt_{sequence:04d}_antigravity`에서
가장 작은 미사용 sequence를 고른다. 중단되면 그 attempt와 미완성 staging을 실패
증거로 보존하고, 다음 `new`가 새 sequence에서 처음부터 다시 검증한다. 성공한
attempt 하나만 고정 `$RUN_ROOT/publication_antigravity`로 no-clobber 원자 승격할 수
있다.

publication은 다음 의미 파일을 create-only로 만든다.

```text
prediction_antigravity.jsonl
candidate_ranking_antigravity.jsonl
source_audit_antigravity.json
manifest_antigravity.json
status_antigravity.json
artifact_sha256_antigravity.json
```

모든 파일은 canonical JSON 또는 JSONL을 사용한다. prediction은 dataset,
instance_index와 label-free 근거만 담고 label, 실제 이상 위치, confusion, F1 같은
평가 필드를 담지 않는다. manifest는 source SHA, 선택 규칙, 1,117 population,
후보 수, label access count 0, review-only 상태를 묶는다. artifact map은 자기 자신만
제외하고 나머지 publication 파일을 모두 해시한다.

publication의 `manifest_antigravity.json`, `status_antigravity.json`과 evaluation의
`evaluation_antigravity.json`, `bootstrap_antigravity.json`,
`decision_antigravity.json`에는 아래 9개 governance 값을 flat top-level 필드로 모두
기록한다. 하나라도 빠지거나 추가된 부분 묶음, 중첩 표현, truthy 대체값은 허용하지
않는다. JSONL 의미 행, source audit, confusion, telemetry와 artifact map은 이 값을
반복하지 않고 governing manifest/decision과 hash graph로 결속한다. 같은 flat block은
사전등록, runbook, 구현 보고서, Exp441 control records, config payload, readiness,
terminal selection/failure evidence에도 적용한다.

```text
official_run_purpose = deterministic_retrospective_reproduction
bootstrap_interpretation = descriptive_post_selection
adaptive_followup = true
independent_validation = false
external_generalization_supported = false
confirmatory_statistical_claim = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
```

## PyTorch, MLX, MPS 판단

Exp441의 핵심은 JSON record 검증, SHA256, 문자열 필터, 짧은 정렬이다. 따라서
`tensor_eligible = false`다. PyTorch, MLX, MPS가 잘하는 큰 행렬 연산이나 신경망
학습이 없고, 문자열·파일 해시는 tensor 연산으로 같은 알고리즘을 표현할 수 없다.
장치 전송과 compile/synchronization 비용만 추가된다. 현재 설계 확인 환경의 backend
상태는 다음과 같다.

```text
Python = 3.14.6
PyTorch = available, version 2.13.0
PyTorch MPS = built true, available true
MLX = available, version 0.32.0, default device GPU 0
mlx-metal = available, version 0.32.0
tensor_eligible = false
acceleration_benchmark_status = not_run_inapplicable
canonical_backend = cpu
```

readiness는 실행 시점의 버전, device, availability를 다시 기록한다. 현재는 적용성
첫 단계에서 탈락하므로 의미 없는 속도 숫자를 만들기 위한 tensor rewrite나
microbenchmark를 하지 않는다. 향후 실제 tensor stage가 추가된다면 Exp441 범위가
아니며, 동일 입력에서 prediction 바이트와 후보 좌표가 완전히 같고 모든 출력이
유한하며 transfer/compile/synchronization을 포함해 CPU보다 최소 1.1배 빠른 전체-stage
benchmark를 통과해야만 accelerated path가 canonical이 될 수 있다.

공식 검증은 이 Mac의 논리 코어 수에서 하나를 뺀 11 CPU worker를 사용해 1,117개
record의 바이트 해시와 파싱을 병렬 확인한다. 이는 AGENTS의
`max(1, logical CPUs - 1)` 정책과 사용자의 공식 실행 지시를 함께 따른다. readiness는
감지한 논리 CPU 수와 worker 수를 provenance에 기록하고, 이 Mac에서 worker가 11이
아니면 실행하지 않는다. 부모 프로세스가 정렬된 순서로 결과를 합치고 publication은
부모 하나만 쓴다. worker 안의 수치 스레드는 1개로 제한한다. manifest에는 PyTorch,
MLX, MPS가 `inapplicable_no_tensor_stage`로 제외된 이유와 benchmark 미실행 상태를
기록한다.

이 worker 정책은 현재 worktree 루트의 가장 가까운 `AGENTS.md`가 정한 정책이다.
상위 checkout의 오래된 `up to seven` 문구보다 가까운 worktree 규칙과 사용자의 후속
`항상 max - 1` 지시가 우선한다. 구현에는 임의의 7-worker fallback이나 CLI override를
두지 않는다. 단, 11은 알고리즘 상수가 아니라 현재 12 logical CPU topology에서 나온
readiness 결과다.

## 진행 로그

프로세스가 직접 다음을 원자적 JSON과 text log에 쓴다. Codex가 반복 감시하지
않는다.

- PID와 활성 여부
- 현재 검증 중인 데이터셋/record 이름
- source record의 TEST 행 수와, 후보가 있을 때 record에 들어 있는 시계열 길이
- 현재 단계와 backend
- 완료/전체, import/success/error 수
- provisional/outside-Hard/eligible/selected 후보 수
- 처리 속도, 경과시간, ETA
- source checkpoint 위치와 최신 오류

selector-only 실행이라 정상적으로는 짧게 끝나지만, 동일한 관찰 가능성 계약을
유지한다. Exp440 record에는 전체 TRAIN 행 수가 없고 후보 없는 record에는
`raw_length`도 없다. 이때 `train_row_count`와 `raw_length`는 값을 추측하지 않고
`null`로 기록하며 `dimension_status = "not_available_in_frozen_selector_source"`를
함께 남긴다. 원시 DB를 다시 열어 로그용 정보를 보충하지 않는다.

## 사후 재현 평가와 기술적 판정표

평가는 publication을 완전히 재검증하고 prediction SHA256을 고정한 뒤에만 SQLite를
하나의 descriptor-bound read-only connection으로 연다. 연결을 여는 과정에서 DB
descriptor SHA가 확인된 뒤 `PRAGMA quick_check`를 정확히 한 번 실행하고 결과가 오직
`ok`인지 확인한다. 그다음 Exp440 context tuple의 1,117개 dataset만 순서대로
parameterized query하여 TEST `instance_index,label`을 dataset당 정확히 한 번 읽는다.
DB 전체 dataset을 열거하거나 전체 TEST row를 bulk SELECT하지 않는다. 현재 DB의
범위 밖 `CornellWhaleChallenge`, `Wafer_normal_1`을 포함한 어떤 context 밖 dataset도
label query parameter가 될 수 없다. Frozen Hard와 `Hard union Exp441`을 이 동결된
1,117개에서만 비교한다. 이 순서는 runtime label 누수를 막지만, 이미 TEST 결과를
보고 선택한 규칙을 독립 검증으로 바꾸지는 않는다.

다음 항목은 통계적 성공 기준이나 챔피언 승격 gate가 아니라, 공식 구현이 설계 때
이미 관찰한 결과와 같은 정책을 재현했는지 확인하는 기술적·설명용 판정표다.

- 전체 데이터셋 수가 정확히 1,117개
- Exp441 후보 수가 30개 이하이고 데이터셋당 1개 이하
- 모든 후보가 Hard 좌표 0개인 데이터셋에만 존재
- 기존에 놓친 데이터셋을 10개 이상 복구
- 추가 FP가 30개 이하
- 후보 precision이 0.35 이상
- micro F1 delta가 0보다 큼
- dataset-balanced macro F1 delta가 0보다 큼
- 20,000회 paired dataset bootstrap의 micro/macro 95% CI 하한이 모두 0보다 큼
- baseline F1 0.8 이상 데이터셋의 최대 손실이 0.2 이하
- prediction 이전 label access count가 정확히 0

bootstrap은 1,117개 dataset cluster를 replacement 방식으로 paired 재표본화하며
다음 구현을 그대로 사용한다. 비슷한 식이나 다른 난수 API로 바꾸지 않는다.

1. dataset 순서는 SHA256이 동결된 Exp440
   `context_antigravity.json["datasets"]`의 tuple 순서 그대로다. 별도 정렬,
   set 순회, publication summary 순서는 사용하지 않는다.
2. 각 dataset 행은
   `[baseline_tp, baseline_fp, baseline_fn, baseline_tn, union_tp, union_fp,
   union_fn, union_tn, dataset_f1_delta]` 순서의 `np.float64` 배열이다.
3. F1은 기존 프로젝트 구현과 같은 아래 precision/recall 조화평균식만 사용한다.
   직접식 `2TP / (2TP + FP + FN)`은 수학적으로 같지만 마지막 bit가 달라질 수
   있으므로 공식 재현 계산에는 사용하지 않는다.

```python
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

4. 공식 환경은 NumPy `2.4.6`으로 고정한다. readiness는 버전이 다르면 평가를
   시작하기 전에 실패한다.
5. 난수 생성과 각 반복의 추출 호출은 정확히 다음과 같다. 20,000개 반복마다
   `integers`를 정확히 한 번 호출한다.

```python
rng = np.random.Generator(np.random.PCG64(44120260808))
for repetition in range(20000):
    indices = rng.integers(
        low=0,
        high=1117,
        size=1117,
        dtype=np.int64,
        endpoint=False,
    )
    sample = rows[indices]
```

6. 각 sample의 micro confusion은 앞의 네 열과 다음 네 열을 각각
   `sum(axis=0, dtype=np.float64)`로 합친 뒤 위 F1 식의 차이를 구한다. macro는
   `sample[:, 8].mean(dtype=np.float64)`다.
7. 구간은 정확히 다음 호출로 계산한다.

```python
percentiles = np.asarray([0.025, 0.975], dtype=np.float64)
ci95 = np.quantile(samples, percentiles, method="linear")
```

이 규칙으로 기대하는 aggregate와 point/bootstrap 값은 위 탐색 replay 블록의
full-precision 숫자다. 감사용 IEEE-754 신원은 다음과 같다.

```text
micro_delta.hex = 0x1.df53a1ad05800p-10
macro_delta.hex = 0x1.97c43a7cf3ab8p-7
micro_ci_lower.hex = 0x1.c2cafc29c4668p-15
micro_ci_upper.hex = 0x1.f3ec47deffdf7p-9
macro_ci_lower.hex = 0x1.ac4d308a9c7cep-8
macro_ci_upper.hex = 0x1.39e467f55da52p-6
```

후보 좌표, dataset 순서, 구조 수, confusion 정수, TP/FP, rescue/regression 수는
정확히 같아야 한다. point와 CI의 유한 float 여섯 개는 각 기대값에 대해
`math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)`를 모두 만족해야
같다고 판정한다. tolerance를 반올림에 사용하지 않고 상태 판정에만 사용한다.
NaN 또는 무한대는 engineering failure다. 산출물에는
`bootstrap_interpretation = "descriptive_post_selection"`과
`confirmatory_statistical_claim = false`를 함께 기록한다.

source hash, population, 선택 규칙, 후보 수, confusion 정수 또는 label 차단선이 다르면
`engineering_reproduction_failed`로 기록한다. 구조는 같지만 사후 지표가 위 탐색
replay의 full-precision 값에서 위 절대 허용오차를 벗어나면
`retrospective_reproduction_discrepancy`로 기록하고 원인을 감사한다. 구조와 지표가
모두 재현되면 `retrospective_reproduction_complete`로 기록한다.
세 상태 모두 `scientific_winner_selected = false`, review-only,
operational-alert-ineligible이며 CHAMPION_LEDGER를 Exp441로 바꾸지 않는다. 같은
실험에서 예산, 순위, source를 바꾸지 않으며 변경은 새 실험 번호와 새 사전등록으로만
한다.

## 오류 처리와 불변성

- source hash, population, record digest, schema가 하나라도 다르면 label을 열기 전에
  실패한다.
- symlink, 비정규 파일, 누락/추가 record, 중복 좌표, 범위를 벗어난 instance index는
  실패한다.
- readiness와 new 사이에 clean HEAD 또는 source 바이트가 바뀌면 실패한다.
- 기존 publication/evaluation은 덮어쓰거나 채택하지 않는다. 완전한 byte replay가
  되는 동일 산출물만 inspect할 수 있다.
- prediction 동결 뒤 source가 바뀌면 evaluation을 만들지 않는다.
- 실패 root와 staging은 원인을 확인할 수 있게 보존한다.

## 검증 계획

구현은 test-first로 진행하며 최소 다음을 확인한다.

- Hard 좌표가 0개인 데이터셋만 통과
- Hard 좌표가 하나라도 있으면 모든 Exp441 후보 차단
- 같은 데이터셋 후보 여러 개 중 정확한 tie-break로 하나만 선택
- `agreement_count`가 순위에 영향을 주지 않음
- 전역 순위와 30개 절단이 입력 순서와 worker 완료 순서에 무관함
- 1,117 record와 840/394/73/49/30 label-free 구조 수 재현
- source 파일 하나의 바이트 변경, 누락, 추가, symlink를 모두 거부
- candidate와 Hard 좌표를 각 record의 `test_row_count`에 대해 label 전에 범위 검증
- `readiness`, `new`, `inspect` 각각에서 SQLite connect와 label-capable import를 강제로
  실패시키고 TEST label 및 Exp440 evaluation 경로 접근이 전혀 없음을 증명
- 11-worker 결과와 1-worker reference의 prediction 바이트 완전 일치
- 진행 JSON의 PID, 현재 데이터셋, 크기, 단계, backend, 완료 수, 속도, ETA, 오류
- create-only publication, 모든 artifact tamper, 동시 destination 생성을 거부
- prediction freeze 전 evaluate 금지, DB hash -> quick_check -> context-scoped 1,117 query의
  정확한 post-freeze 순서, context 밖 2개 dataset label 미조회
- 손으로 계산 가능한 TP/FP/FN/TN, rescue, macro, strong-loss, descriptive bootstrap
  판정표와 세 reproduction 상태
- 실제 Exp440/Hard artifact를 이용한 production-shaped label-free replay

공식 실행 전에 adversarial production-path suite가 모두 통과해야 한다. fixture smoke만
통과한 상태에서 곧바로 공식 prediction을 만들지 않는다.

## 통제 파일과 코드맵

Exp441 구현 전에 한 번의 원자적 통제 단계에서 다음 사실을 현재 저장소와 맞춘다.

- Exp440: 1,117/1,117, 오류 0, prediction 동결, 평가 완료, winner gate 실패
- FAILURE_LEDGER: Exp440의 FP, micro CI, strong-loss 실패 원인
- CHAMPION_LEDGER: Exp404 Hard/Exp416 감사본이 아직 검증 기준
- RESEARCH_STATE와 EXPERIMENT_CATALOG: Exp441 사전등록 단계로 전환

Exp441은 별도 pipeline 경계를 만들되 코드맵의 20개 primary node 제한 안에서 기존
frequency research 그룹 아래에 묶는다. runtime이 frozen Exp440 checkpoint와 Hard를
읽고, publication을 쓰며, evaluation이 prediction freeze 뒤에만 DB label을 읽는
흐름을 source symbol 근거와 함께 표시한다. `codemap.html`, `codemap.json`,
`codemap.lock`은 항상 함께 갱신한다.

## 구현 범위 밖

- Exp440 점수 재계산 또는 threshold 재조정
- 대형 TRAIN 그룹의 Exp440 coverage 확대
- 새 주파수 대역, CR-NN, HydraAD, PhysDiff 모델 추가
- 자동 경보 승격
- 외부 데이터 일반화 주장
- Exp440/Exp404 기존 산출물 수정

이 항목들은 Exp441 결과를 본 뒤 각자 새 실험 번호로 분리한다.
