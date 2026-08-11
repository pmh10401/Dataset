# Exp448 Exp447 TEXT label 경계 수리 설계

## 1. 한 문장 설명

Exp448은 Exp447의 유일한 사후 평가가 SQLite TEST label의 실제 저장 형식인 TEXT를
정수라고 잘못 가정해 실패한 공학적 경계만 수리한다. 예측과 탐지 방법은 전혀 바꾸지
않고, 새로 동결한 Exp448 경계에서만 `'0'`과 `'1'`을 정수 label로 정확히 변환하여
독립적으로 설명용 평가한다.

## 2. 고정 신원과 전역 제약

```text
experiment_id = Exp448
artifact_schema = exp448
kind = exp447_text_label_boundary_repair
run_identity = exp448-exp447-text-label-boundary-repair-antigravity-20260812
run_root = /Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812
package = experiments/exp448_exp447_text_label_boundary_repair_antigravity
launcher = scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py
interpreter = /Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python
```

다음 제약은 설계, 구현, 공식 실행과 결과 해석 전체에 적용한다.

- 새로 만드는 모든 파일명에는 `antigravity`가 포함되며 `__init__.py`는 만들지 않는다.
- 이것은 engineering-only post-freeze evaluation repair다. prediction, candidate, score,
  threshold, source, lane, budget, method, backend, training을 변경하지 않는다.
- Exp447 failure truth의 reviewed commit은
  `7e85d5c7948f8592db80f958d746f31138c67bfb`이고, Exp447 공식 시도의 reviewed commit은
  `33e7dc9c18c179a64bc0020b0bb939c5e4caa394`이다. readiness는 1회 exit 0,
  evaluate는 1회 exit 1, inspect는 0회였으며 Exp447은 절대 다시 실행하지 않는다.
- Exp448 CLI는 reviewed commit을 받는 `readiness`, `evaluate`, `inspect`만 정확히
  제공한다. `new`, `resume`, `retry`, override, 약어는 없다.
- 패키지와 의존성은 바꾸지 않는다. MLX/MPS, worker, training은 쓰지 않는다.
- evaluator는 network나 dashboard를 import하지 않는다. readiness와 inspect에는
  label-capable SQLite가 없다.
- 공식 명령이 하나라도 nonzero이면 Exp448은 영구 중단하며 재시도하지 않는다. 다음
  행동은 별도 새 실험이다.
- map은 함께 갱신하며, 구현은 Terra, 최종 검토는 fresh Sol이 맡는다.

설계·plan·preregistration·runbook의 SHA와 reviewed commit은 해당 bytes가 실제로
생기기 전에는 unknown이다. 이 설계는 알 수 없는 해시를 만들어 기록하지 않는다.

## 3. Exp447 실패 사실과 불변 증거

Exp447의 immutable failure audit은 다음 regular single-link 파일이다.

```text
path = /Users/mac/Documents/Dataset/evaluation_audit_89aff47cdd7c43cd919a34f81982bc8b_antigravity.json
sha256 = 9f8f7d3d9b92670988ca1d0bb689711d12ef1536777881f18991143a090a2b97
inode = 19812443
nlink = 1
schema = exp447_evaluation_audit
phase = label_snapshot
query_attempts = 1
query_completions = 0
label_rows = 0
label_session_closed = true
label_session_state = closed
destination_state = absent
error_type = ValueError
error = TEST label row is invalid
```

Exp447 run root
`/Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811`
는 absent/non-symlink다. Exp447 result, result hash, promotion, inspect, dashboard는 없으며,
이 partial failure는 재사용하거나 다시 해석하지 않는다.

입력 Exp446 publication의 고정 값은 다음과 같다.

```text
prediction_sha256 = 602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100
manifest_sha256 = bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11
artifact_map_sha256 = d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d
datasets = 1117
source = 30
stable = 26
unstable = 4
prediction_label_access = 0
```

Exp404 Hard prediction SHA는
`437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`이며 좌표 수는
2,218개다. DB SHA는
`8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`, inode는
1048138, 공식 nlink는 1이다.

## 4. 왜 새 controlled fork가 필요한가

Exp447은 label 값이 `int`라고 요구해 첫 TEST query 직후 실패했다. 그 실패 사실은
보존하고 Exp447 코드나 lifecycle을 고치지 않는다. Exp448은 새 번호, 새 root, 새 audit,
새 reviewed commit으로 같은 동결 예측을 평가하는 controlled fork다. 바뀌는 것은 raw
SQLite TEXT를 evaluator가 받는 0/1 정수로 옮기는 경계뿐이다.

Exp447 package+tests와 launcher의 전체 bytes는 `7e85d5c` 기준과 같아야 한다. 비교하는
Exp447 source hashes는 다음과 같다.

```text
artifacts = 9527611039e0a6a35c689f86a473e6fc178af6471effddcb6937932c54402cd1
bootstrap = 4c8c89e50c01ff492f7547ee6324998318503217f6b6a5546123d4186b5f423e
cli = fcd3a781b7f5495c58ce464ce864fd0151073b9488a6cea2cb2ad08c311414f6
config = e5caacb2f0f163b071b2b4c1f72ace7c16163ef275585492a47a8b2f840c66c0
contracts = ac878f045da986372888ba5c3f56bf7b82e807fc8b25c2e80656771cc1a58912
evaluation = dd718da876fd53792dc1911d34f3213ed1645705c94a557689222cace104d9cc
publication_identity = af2f9f280a70662b4d924058f797d954919c6efb6afc33d705f527ab47b73f1c
readiness = 1e51699b69205aa281ccab57c4670dd5e7f7c8d6c466aa90f1db71efea8937ba
replay = bdcf996e476fa24af5cb83e562372239d8bb3f84ff3eeefd466337d4679f3b9d
sources = c8ff6995ad514532f009983d61322ee06e4332001447fc3287238eb96c2fea2c
launcher = 1497a781eeaddb5309878e92e4b13552addd82c9f4224858533845ab76ea2309
```

controlled-fork source allowlist는 Exp447 이름과 같은 정확히 열 개 모듈
`artifacts`, `bootstrap`, `cli`, `config`, `contracts`, `evaluation`,
`publication_identity`, `readiness`, `replay`, `sources`와 launcher다. 테스트는 같은
열 개 이름과 `test_controlled_fork_differential_antigravity.py`만 허용한다.

허용 delta는 다음 다섯 가지뿐이다.

1. A: Exp448 identity, schema, root, 문서와 audit 이름
2. B: Exp447 audit verifier와 Exp447 root absent 검증
3. C: DB schema metadata가 `instance_index INTEGER`, `label TEXT`임을 확인하는 것
   (값 개수는 확인하지 않음)
4. D: 아래 exact TEXT converter
5. E: 이로 인해 생기는 commit과 artifact hash

과학 방법, bootstrap, 지표, 진단, leaves, 방법 순서, query/close/recheck,
create-only, replay, governance의 의미는 모두 그대로여야 한다.

## 5. 유일한 label 경계와 exact converter

label은 exact Exp447 failure verify, 서로 같은 Exp446 capture 두 번, Exp404 validation이
모두 성공한 뒤에만 닿을 수 있다. 고정 dataset 순서로 1,117 query, 총 195,900행만 읽는다.

각 row에서 `instance_index`는 exact `int`여야 하고, raw label은 exact `str`이면서
오직 `'0'` 또는 `'1'`만 허용한다. converter는 정확히 다음이다.

```python
0 if raw_label == '0' else 1
```

int, bool, float, bytes, whitespace가 섞인 문자열, null, 그 밖의 문자열은 모두 거절한다.
raw string은 evaluator와 artifacts에 절대로 도달하지 않는다. 이 변환은 prediction,
readiness, routing, tuning에 쓰이지 않는다.

사후 forensic whole-DB 사실인 raw `'0'=221284`, `'1'=4516`, integer index/TEXT label은
post-freeze storage evidence일 뿐이며, dataset 이름·위치·개별 label을 포함하지 않는다.
이 사실을 prediction/readiness/routing/tuning에 query하거나 사용하지 않는다.

## 6. 평가 방식과 불변 결과 계약

다음 순서를 정확히 유지한다.

1. `hard_baseline`
2. `hard_plus_source_30`
3. `hard_plus_stable_26` (주 비교)
4. `hard_plus_unstable_4`

진단은 `rescued`, `false_alarm`, `regressed`, `strong_regression`이며 방법 선택에 쓰지
않는다. 혼동행렬 기준은 다음으로 고정한다.

```text
hard_baseline = [1915, 303, 2003, 191679]
hard_plus_source_30 = [1930, 318, 1988, 191664]
Exp442_evaluation_sha256 = f432ffdf530b2e32bc7a62fec6c7d07cff2eff9f3049489b5e03136b445a33cf
Exp442_artifact_map_sha256 = e1155acf05e00935710243e875eaeb2d13047d6236d026ee93fe654fc33c670b
```

Bootstrap도 수리 대상이 아니다.

```text
Generator = PCG64(44720260811)
samples = 20000
indices = int64
endpoint = false
percentile = linear 2.5/97.5
draw_sha256 = f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e
```

결과 root의 leaves는 정확히 다음 열한 개다.

```text
evaluation_antigravity.json
dataset_metrics_antigravity.jsonl
candidate_transitions_antigravity.jsonl
rescued_datasets_antigravity.jsonl
false_alarm_datasets_antigravity.jsonl
regressed_datasets_antigravity.jsonl
strong_regressions_antigravity.jsonl
bootstrap_antigravity.json
confusion_antigravity.json
decision_antigravity.json
artifact_sha256_antigravity.json
```

결과 상태는 항상 `descriptive_complete_no_promotion`이다. success criterion은 `null`이고,
winner, champion, pending candidate, operational alert, promotion, reselection은 없다.

## 7. differential 검증

controlled fork임을 다음으로 증명한다.

- canonical profile은 identity key를 제외하면 정확히 같아야 한다.
- paired module AST를 normalize해 named allowlist node 외 차이가 없음을 확인한다.
- constants, leaf order, method order, query/close/recheck와 governance 값을 exact 비교한다.
- synthetic 1,117 dataset TEXT runtime Exp448과 integer Exp447 pure evaluator가
  identity/hash를 제외하고 같은 결과를 내야 한다.
- 정확한 rejection matrix, failure parity, import firewall을 검사한다.
- 테스트에는 실제 label을 넣지 않는다.

## 8. 구현 작업 여섯 개

1. governance와 versioned code map을 만들고 Exp447 failure truth·controlled fork
   boundary·official one-shot을 문서화한다.
2. fork foundations와 differential verifier를 만들고 identity key를 제외한 canonical
   profile 및 normalized AST/constant parity를 테스트한다.
3. predecessor/source/readiness를 구현한다. Exp447 failure audit과 absent root를 확인하고,
   두 Exp446 capture와 Exp404 검증 후 schema metadata와 exact TEXT converter를 적용한다.
4. evaluator/replay/audit parity를 구현한다. 생성물·bootstrap·diagnostic·create-only와
   label close/failure audit 의미를 Exp447과 같게 유지한다.
5. CLI/launcher, 전체 검증, fresh Sol 최종 검토를 완료한다.
6. official `readiness` 1회 → `evaluate` 1회 → `inspect` 1회 → independent verify →
   truth/map → final review 순서로 실행한다. 성공 뒤에만 dashboard boundary를 연다.

## 9. dashboard presentation boundary

owner-only 기존 site는 Exp448 성공과 truth closeout, 그리고 site repository clean 상태
뒤에만 사용할 수 있다. live status는 exact 12 fields이고 Exp448 `total`은 1117이다.
완료 payload schema는 `exp448.public-evaluation.v1`이며 aggregate만 담는다.

raw label, index, dataset name/list, per-dataset row, hash, local path, log, DB, checkpoint,
secret은 dashboard로 절대 보내지 않는다. dashboard 실패는 과학 결과를 바꾸지 않고
재실행·retry를 허용하지 않는다.

## 10. 공식 실행 gate

공식 실행 전 clean candidate commit, fresh Sol `ship`, DB SHA/inode/nlink, Exp447 audit
identity, Exp447 root absence, Exp446/Exp404 bytes, Exp448 root absence를 확인한다. 그 뒤
정확한 순서로 한 번씩만 실행한다. 결과가 무엇이든 자동 승격하지 않으며, 사람이
혼동행렬·신규 탐지·false alarm·기존 강한 dataset 악화·bootstrap을 함께 해석한다.
