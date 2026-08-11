# Exp447 Exp446 동결 예측 사후 성능평가 설계

## 1. 한 문장 설명

Exp447은 Exp446가 이미 동결한 예측을 전혀 바꾸지 않고 TEST label을 사후에 한 번
읽어, 기존 Hard 기준보다 실제 이상을 더 찾는지와 오경보가 얼마나 늘어나는지를
1,117개 전체 데이터셋에서 설명적으로 평가하는 독립 실험이다.

## 2. 목적과 범위

Exp446의 공식 CPU/11 실행과 create-only publication은 완료됐다. prediction 단계의
TEST-label 접근은 0이며 source 후보 30개는 `stable_review` 26개와
`unstable_review` 4개로 동결됐다. Exp447의 질문은 하나다.

> 동결된 30개 후보, 특히 안정 후보 26개를 기존 Hard 탐지에 더하면 놓친 이상을
> 얼마나 새로 찾고 오경보와 기존 강한 데이터셋의 악화가 얼마나 생기는가?

Exp447은 score, feature, threshold, source, 후보 좌표, lane, budget 또는 backend를
바꾸지 않는다. 결과를 보고 같은 Exp447 안에서 후보를 빼거나 더하지 않는다.
자동 champion, 운영 경보 승격, scientific winner 선택도 하지 않는다.

## 3. 고정 실험 신원

```text
experiment_id = Exp447
artifact_schema = exp447
run_identity = exp447-exp446-postfreeze-performance-evaluation-antigravity-20260811
run_root = /Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811
source = immutable Exp446 publication
dataset_count = 1117
test_label_rows = 195900
source_candidate_count = 30
stable_candidate_count = 26
unstable_candidate_count = 4
bootstrap_seed = 44720260811
bootstrap_samples = 20000
review_only = true
operational_alert_eligible = false
scientific_winner_selected = false
```

공식 명령은 다음 세 개뿐이다.

```text
readiness --reviewed-commit REVIEWED_COMMIT
evaluate --reviewed-commit REVIEWED_COMMIT
inspect
```

`new`, `resume`, `retry`, 임의 root·hash·source·lane·threshold·budget 인자는 없다.
Exp446 CLI, 공식 root, publication과 prediction bytes는 수정하지 않는다.

## 4. 동결 입력과 기준값

Exp447 readiness는 경로 이름을 신뢰하지 않고 exact inventory와 canonical bytes를
descriptor-relative no-follow 방식으로 검증한다.

```text
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
expected_database_inode = 1048138
expected_database_nlink = 1
exp404_hard_prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
exp446_prediction_sha256 = 602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100
exp446_manifest_file_sha256 = bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11
exp446_artifact_map_file_sha256 = d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d
exp446_inspect_output_sha256 = 937b46960ed0b611fc9d3c74bffa7b494ff306af6d78a92129217b0da691fdf0
baseline_confusion_tp_fp_fn_tn = 1915, 303, 2003, 191679
```

Exp446 publication은 dataset 1,117개, source 30개, stable 26개, unstable 4개,
prediction label-access count 0을 다시 증명해야 한다. Exp404 Hard prediction은 기존
baseline을 구성하는 유일한 기본 예측이다. Exp442의 역사적 30-candidate union 결과는
Exp447의 `Hard + source 30` 재현 결과와 일치 여부만 비교하며 선택 규칙으로 쓰지 않는다.

## 5. TEST label 방화벽과 평가 순서

TEST label은 다음 순서가 모두 끝난 뒤에만 열 수 있다.

1. operator가 제공한 40-hex reviewed commit이 현재 clean HEAD와 같은지 확인한다.
2. Exp447 root가 없고 symlink도 아닌지 확인한다.
3. Exp446 publication의 root, exact leaf inventory, manifest, artifact map, prediction
   SHA, dataset 순서, 30개 좌표와 26/4 lane partition을 검증한다.
4. Exp404 Hard prediction의 bytes와 SHA를 검증한다.
5. Exp446 publication을 독립적으로 한 번 더 열어 첫 검증과 root inode, leaf bytes,
   dataset 순서, 후보·lane 순서가 같은지 확인한다.
6. 위 두 검증이 일치할 때만 SQLite를 immutable read-only로 열고 고정 dataset 순서대로
   TEST `(instance_index, label)`을 정확히 1,117회, 총 195,900행 읽는다.
7. label session과 SQLite authorizer를 닫고 descriptor와 source pin이 해제됐는지,
   원본 DB가 같은 inode·SHA·single-link인지 확인한다.
8. Exp446 publication을 label close 뒤, 계산 뒤, commit 직전에 다시 확인한다.
9. Exp447 결과를 새 root에 create-only로 기록한다.
10. `inspect`는 SQLite, evaluator label route와 worker를 열지 않고 동결된 결과 bytes만
    재계산한다.

Prediction 코드, Exp446 `readiness/new/inspect`, publication inspection에는 SQLite
label query나 evaluator import가 들어가지 않는다. Exp447 evaluator 이외의 경로에서
TEST label 접근이 한 번이라도 발생하면 실패한다.

## 6. 비교 대상

같은 1,117개 dataset과 같은 195,900개 TEST label snapshot으로 다음 네 결과를 만든다.

1. `hard_baseline`: Exp404 Hard만 사용
2. `hard_plus_source_30`: Hard와 Exp446 source 30개 합집합
3. `hard_plus_stable_26`: Hard와 stable 26개 합집합 — 주 비교 대상
4. `hard_plus_unstable_4`: Hard와 unstable 4개 합집합 — 설명용

후보 좌표가 이미 Hard에 포함돼 있으면 중복 카운트하지 않는다. 각 dataset의 TEST
index는 0부터 연속이어야 하며 label은 exact integer 0 또는 1이어야 한다. dataset
순서, label 행 수, 후보 lane 보존 법칙이 어긋나면 결과를 만들지 않는다.

## 7. 평가 지표

각 비교 대상에 대해 dataset별 및 전체 수치를 함께 기록한다.

- TP, FP, FN, TN 혼동행렬
- precision, recall, F1, false-positive rate
- dataset별 F1과 1,117개 dataset macro F1
- Hard가 놓쳤지만 새 후보가 찾은 이상 개수와 dataset 목록
- 새로 추가된 false alarm 개수와 dataset 목록
- Hard 대비 F1이 낮아진 dataset과 감소폭
- Hard F1이 0.8 이상이었던 dataset 중 악화된 strong regression과 최대 감소폭
- stable 26개와 unstable 4개 각각의 TP/FP 및 기존 Hard 중복 여부
- Exp442 역사적 source-30 결과와 `hard_plus_source_30`의 exact confusion 일치 여부

`rescued dataset`은 Hard보다 TP가 증가한 dataset이다. `regressed dataset`은
`hard_plus_stable_26`의 dataset F1이 Hard보다 낮은 dataset이다. `strong regression`은
Hard dataset F1이 0.8 이상이면서 stable 결과가 더 낮은 경우다. 빈 양성 dataset의
F1은 프로젝트의 기존 `2TP / (2TP + FP + FN)` 정의에 따라 분모가 0이면 0으로 둔다.

## 8. paired bootstrap

dataset을 통째로 짝지어 복원추출하는 20,000회 bootstrap을 사용한다. NumPy
`Generator(PCG64(44720260811))`, int64 indices, `endpoint=False`, linear 2.5/97.5
percentile을 고정한다.

다음 세 비교의 micro·macro F1 delta와 95% 구간을 기록한다.

1. `hard_plus_source_30 - hard_baseline`
2. `hard_plus_stable_26 - hard_baseline`
3. `hard_plus_stable_26 - hard_plus_source_30`

구간은 불확실성을 설명하는 자료일 뿐 합격선이나 후보 선택 규칙이 아니다.

## 9. artifact와 감사 기록

Exp447 root에는 최소 다음 create-only 파일을 저장한다. 모든 새 파일명에는
`antigravity`가 포함되고 JSON/JSONL은 canonical bytes와 고정 순서를 사용한다.

- `evaluation_antigravity.json`
- `dataset_metrics_antigravity.jsonl`
- `candidate_transitions_antigravity.jsonl`
- `rescued_datasets_antigravity.jsonl`
- `false_alarm_datasets_antigravity.jsonl`
- `regressed_datasets_antigravity.jsonl`
- `strong_regressions_antigravity.jsonl`
- `bootstrap_antigravity.json`
- `confusion_antigravity.json`
- `decision_antigravity.json`
- `artifact_sha256_antigravity.json`

평가 전 실패는 Exp447 root를 만들지 않는다. label boundary 진입 뒤 실패하면 고유한
create-only audit에 phase, publication identity, query attempts/completions, 읽은 행 수,
session close 상태, destination 상태와 원래 오류를 기록한다. destination 쓰기 중 실패한
partial root는 증거이며 수정하거나 재사용하지 않는다. 같은 root에서 retry하지 않고
필요하면 다음 실험 번호로 분리한다.

## 10. 결과 해석

성공 기준은 사전등록하지 않는다. 결과 상태는 coherent evaluation이면 항상 다음이다.

```text
evaluation_status = descriptive_complete_no_promotion
scientific_success_criterion = none
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
test_informed_reselection_allowed = false
automatic_champion_promotion = false
```

평균 F1 하나만으로 결론을 내리지 않는다. 신규 탐지, 추가 false alarm, 기존 강한
dataset 악화, 전체 paired 성능과 bootstrap 구간을 함께 보여준다. 결과가 좋아도
Exp447 안에서 threshold, source, lane 또는 budget을 다시 고르지 않으며 champion과
운영 경보 승격은 별도 논의와 별도 사전등록이 필요하다.

## 11. 구현 경계와 자원

Exp447은 새로운 score source나 학습·추론 stage가 아니므로 MLX/MPS backend 선택을
다시 하지 않는다. 성능 계산은 NumPy 벡터 연산과 순차적인 read-only label snapshot을
사용한다. DB label session은 하나만 열고 dataset 순서를 보존한다. CPU worker 정책은
새 과학 계산에 적용되지 않지만, 병렬화가 필요한 순수 사후 계산을 추가한다면 현재
Mac의 `max(1, logical_cpu_count - 1) = 11`을 넘지 않고 결과 bytes가 순차 경로와
동일해야 한다.

Exp447 package는 Exp446 CLI를 확장하지 않는 얇은 독립 경계로 만든다. Exp444의 검증된
evaluation·bootstrap 원리를 선택적으로 재사용하되 Exp447 profile, schema, fixed root,
frozen hashes와 artifact inventory를 독립적으로 닫는다.

## 12. 검증과 공식 실행 gate

구현은 TDD로 다음을 자동 검증한다.

- Exp446 prediction·manifest·artifact·leaf·lane tamper 차단
- 두 번의 동일한 publication inspection 전 label open 차단
- Exp404 Hard hash·population·baseline confusion 고정
- label query 정확히 1,117회, 195,900행과 dataset 순서 보존
- label exact type, contiguous index, close failure와 DB pin 정리
- 30 = 26 + 4 lane conservation과 TP/FP delta conservation
- confusion, precision/recall/F1/FPR, dataset macro 계산
- rescued, false alarm, regression, strong regression의 exact 정의
- 세 paired bootstrap의 seed·draw SHA·percentile 재현
- 계산 후 publication drift와 commit 직전 drift 차단
- create-only complete/partial/audit와 immutable replay
- `readiness/evaluate/inspect`만 허용하는 CLI와 label-free inspect import firewall
- Exp444, Exp445, Exp446, Exp447 영향 회귀
- versioned `codemap_exp447_antigravity.html/json/lock` 동시 검증

공식 실행은 exact clean candidate commit에 대한 fresh Sol High 최종 검토가 `ship`이고,
DB SHA/inode/nlink, Exp446 root/leaf/hash, Exp447 root absence가 모두 맞을 때만 허용한다.
공식 순서는 `readiness -> evaluate -> inspect`이며 evaluate는 single-use다. 완료 후
수치·혼동행렬·dataset 전환·bootstrap을 쉬운 한국어와 차트로 보고하고 자동 승격 없이
논의를 위해 멈춘다.

## 13. private dashboard presentation boundary

Exp447 evaluator는 local/deterministic이며 network 또는 dashboard code를 import하지 않는다.
immutable Exp447 evaluation과 label-free inspect가 모두 성공한 뒤에만 별도 exporter가
sanitized aggregate data를 만들 수 있다. exporter는 raw TEST label, instance/anomaly index,
dataset name, local path, hash, log, checkpoint, DB name, secret 또는 per-dataset row를 절대
export하지 않는다.

private owner-only dashboard는
`https://timeseries-research-observatory.pmh10401.chatgpt.site`이며, separate site repository
`/Users/mac/.codex/visualizations/2026/08/10/019fedc4-df12-73b2-9812-a5f8f387ac78/research-observatory-antigravity-site`
만 presentation consumer다. Live status는 evaluator와 분리되어 existing exact
12-field `PublicExperimentStatus` allowlist만 사용한다:
`experiment_id`, `display_name`, `status`, `stage`, `completed`, `total`, `success`, `errors`,
`backend`, `throughput`, `eta_seconds`, `updated_at`. Exp447 `total`은 1117이다.

completed payload schema는 `exp447.public-evaluation.v1`이고 다음만 포함한다:
`experiment_id`, `generated_at`, `dataset_count`, `decision`; aggregate comparison four rows
(`hard_baseline`, `hard_plus_source_30`, `hard_plus_stable_26`, `hard_plus_unstable_4`) 각각의
TP/FP/FN/TN/precision/recall/F1/FPR; aggregate diagnostic counts
(`rescued`, `added_false_alarm`, `regressed`, `strong_regressed`); 및 three paired-bootstrap
aggregate delta/CI rows. Dataset list를 포함하지 않는다.

dashboard는 scientific or promotion gate가 아니라 immutable truth closeout 이후의 별도
presentation boundary다. site repository가 clean 상태가 되기 전에는 변경·build·commit·deploy를
하지 않는다. dashboard parser/UI/build/deploy failure는 Exp447 scientific artifact를 바꾸지
않고 evaluation 재실행 또는 retry를 허용하지 않는다. 완료된 site work는 owner-only access를
보존하고 existing site의 saved version을 같은 private site에 deploy한다.
