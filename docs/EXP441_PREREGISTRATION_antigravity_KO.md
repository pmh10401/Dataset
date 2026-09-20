# Exp441 Hard-empty 단발 복구 선택기 사전등록

## 사전등록 원본과 무결성

이 문서는 검토된 Exp441 설계 문서의 같은 커밋 내용을 그대로 읽고 동결한
통제 문서다. 복사하기 전에 다음 SHA256을 확인했으며, 이후 이 설계 파일의
바이트가 바뀌면 Exp441을 실행하지 않고 새 실험 번호와 새 사전등록을 만든다.

```text
design_source = docs/superpowers/specs/2026-08-08-exp441-hard-empty-single-shot-recovery-antigravity-design.md
design_source_sha256 = c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da
sha256_verification = passed_before_preregistration_copy
```

## 연구 질문과 해석

Exp441은 새 모델을 학습하거나 Exp440 점수를 다시 계산하는 실험이 아니다.
동결된 Exp440 provisional 후보와 Exp404 Hard 좌표를 읽고, Hard 좌표가 하나도
없는 데이터셋에서만 가장 강한 후보 하나를 고른다. 데이터셋별 후보를 하나로
줄인 뒤 전역 최대 30개로 제한한다. 목적은 이미 Hard가 경보를 낸 데이터셋을
건드리지 않고 완전히 놓친 데이터셋을 보수적으로 복구하는 선택 규칙을
결정론적으로 재현하는 것이다.

이 규칙은 Exp440의 TEST 결과를 본 뒤 정해진 후속 규칙이다. 따라서 실행이
모든 기술적 판정표를 통과해도 독립 검증, 확인적 통계 주장, 외부 일반화,
과학적 승자 또는 운영 경보를 뜻하지 않는다. Exp404 Hard와 Exp416 감사본은
계속 검증 기준이며 Exp441은 챔피언 원장에 승격되지 않는다.

## 동결 신원과 원본

```text
experiment_id = Exp441
run_identity = exp441-hard-empty-single-shot-recovery-antigravity-20260808
dataset_count = 1117
run_root = /Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
hard_prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
hard_manifest_sha256 = 41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8
hard_coordinate_identity_sha256 = 858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322
selected_coordinate_identity_sha256 = db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048
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

예측 단계의 고정 입력 allowlist는 다음과 같다.

```text
database = /Users/mac/Documents/Dataset/univariate_ts.db
exp440_checkpoint = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity
exp440_context = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/context_antigravity.json
exp440_state = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/state_antigravity.json
exp440_records = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/records_antigravity
exp440_retained_inventory = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/checkpoint_antigravity/retained_candidates_antigravity.json
exp440_publication = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/publication_antigravity
hard_prediction = /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl
hard_manifest = /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json
forbidden_exp440_evaluation_root = /Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/evaluation_antigravity
forbidden_exp440_evaluation_module = experiments/exp440_train_only_low_frequency_precision_gate_antigravity/evaluation_antigravity.py
```

Exp440 publication은 artifact map이 선언한 정확히 10개 일반 파일을 모두
descriptor로 읽어 해시를 다시 확인한다. checkpoint는 context, state, 1,117개
record와 retained inventory만 허용한다. 경로 allowlist 밖의 파일, symlink, FIFO,
추가 record 또는 누락 record는 실패다. Exp440 evaluation 디렉터리와 evaluator
모듈은 prediction 과정에서 문자열·import·파일 읽기 모두 금지한다.

Hard 좌표와 Exp441 선택 좌표의 identity 직렬화는 서로 바꾸지 않는다. Hard는
좌표 정렬 compact JSON 배열(끝 newline 없음), Exp441은 좌표 정렬 compact JSON
object JSONL(행마다 newline 하나)이며 위의 두 SHA가 각각 일치해야 한다.

## Global Constraints

### 정보 경계와 선택 규칙

1. prediction은 Exp440 동결 checkpoint/publication과 Exp404 Hard manifest/좌표만
   읽는다. TEST label, 실제 이상 위치, 과거 family별 TEST 성능, Exp440 평가
   산출물은 읽지 않는다.
2. Hard 좌표가 하나라도 있는 데이터셋은 모든 Exp441 후보를 차단한다.
3. Hard 좌표가 0개인 데이터셋에서 Exp440 provisional 후보의 deterministic
   최고 순위 하나만 남긴다. 동률은 동결된 score, evidence, dataset, index
   순서를 사용하며 worker 완료 순서나 입력 순서에 의존하지 않는다.
4. 전역 후보 예산은 30이고 데이터셋당 후보는 최대 1개다. 후보를 채우기 위한
   padding은 하지 않는다.
5. prediction SHA256과 publication이 먼저 동결된 뒤에만 evaluate가 SQLite를
   descriptor-bound read-only connection으로 열 수 있다. 그때도 frozen context의
   정확한 1,117개 dataset을 parameterized query로 dataset당 한 번 읽는다.
6. source hash, population, coordinate identity, candidate/range/schema가 하나라도
   다르면 label을 열기 전에 fail closed한다. prediction 이전 label access count는
   정확히 0이어야 한다.

### 비승격 해석 블록

아래 블록은 Exp441 사전등록, 실행 안내서, 원장 control record와 향후 governing
artifact에 모두 같은 flat key/type/value로 기록한다. 중첩 object, 다른 alias,
truthy 대체값은 허용하지 않는다.

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

### 평가와 재현

평가 지표와 20,000회 paired dataset bootstrap은 이미 관찰된 사후 선택 결과를
같은 동결 규칙으로 재현하는 기술적·설명용 판정표다. NumPy 2.4.6 PCG64,
int64 sampling shape, context dataset tuple 순서, 프로젝트 F1 식, linear quantile,
full precision expected values와 `abs_tol=1e-15`, `rel_tol=0.0`을 바꾸지 않는다.
NaN·infinity는 engineering failure다. 결과 상태는 다음 셋 중 하나로만 기록한다.

```text
engineering_reproduction_failed
retrospective_reproduction_discrepancy
retrospective_reproduction_complete
```

어느 상태에서도 `scientific_winner_selected = false`, `review_only = true`,
`operational_alert_eligible = false`를 유지한다. 성공 판정표는 1,117개 dataset,
최대 30 후보, dataset당 1개, Hard-empty만 허용, 10개 이상 복구, 추가 FP 30 이하,
precision 0.35 이상, micro/macro F1 및 paired CI 하한 양수, baseline F1 0.8 이상
데이터셋 최대 손실 0.2 이하, prediction 이전 label access 0이다. 이 항목들은
챔피언 승격 기준이 아니다.

### CPU와 가속 판단

Exp441은 JSON record 검증, SHA256, 문자열 필터, 짧은 정렬만 수행하므로
`tensor_eligible = false`, canonical backend는 CPU다. PyTorch, MLX, MPS는 큰
tensor stage가 없어 `inapplicable_no_tensor_stage`로 기록하며 acceleration
benchmark는 실행하지 않는다. 향후 tensor stage가 별도 실험에 추가되면 동일 입력
parity, finite output, 후보 identity 일치, transfer/compile/synchronization 포함
CPU 대비 1.1x speed gate를 모두 통과해야 한다. 공식 CPU 실행 worker 수는
`max(1, logical_cpus - 1)`이며 현재 topology에서 11, worker 내부 수치 thread는
1이다. MLX/MPS-heavy process는 동시에 하나만 허용한다.

## 산출물과 불변성

prediction publication은 canonical JSON/JSONL create-only 파일로 만든다. label,
confusion, F1을 prediction 데이터에 넣지 않고, manifest/status에만 위 governance
block을 flat하게 둔다. evaluation은 prediction freeze 후 별도 lazy evaluator에서
만든다. 기존 Exp440, Exp404, Exp416 산출물과 실패 attempt는 수정·삭제·덮어쓰기
하지 않는다. 실패 attempt와 staging evidence는 다음 `new`가 새 sequence에서
처음부터 재검증할 수 있도록 보존한다.

## 승인 상태

이 사전등록은 reviewed design SHA에 묶인 selector-only retrospective reproduction
계획이다. 현재 control state는 `design_approved_implementation_planned`이며,
공식 prediction/evaluation이 아직 생성되지 않았다. 이 문서의 규칙·identity·경로
또는 governance block을 바꾸려면 Exp442 이후의 새 실험 번호와 새 사전등록이
필요하다.
