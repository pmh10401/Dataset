# Exp446 Exp445 publication materialization 복구 사전등록

## 목적과 고정 신원

Exp446는 Exp445의 과학 계산을 바꾸지 않고 publication 직전 frozen JSON handoff의
중첩 tuple을 canonical JSON list로 materialize하지 않은 engineering 결함만 복구하는
label-free, non-promoted 실험이다. 새 score source, threshold, mask, view, lane, backend,
성능 선택 또는 scientific winner를 만들지 않는다.

```text
experiment_id = Exp446
artifact_schema = exp446
run_identity = exp446-exp445-publication-materialization-repair-antigravity-20260811
run_root = /Users/mac/Documents/Dataset/exp446_exp445_publication_materialization_repair_antigravity_20260811
attempt = attempt_0000_antigravity
predecessor = immutable Exp445 failed attempt_0000_antigravity
backend = cpu
logical_cpu_count = 12
outer_workers = max(1, logical_cpu_count - 1) = 11
inner_numerical_threads = 1
label_access_count = 0
design_source_sha256 = 3fce139171efbfc0a85fffeb44ea201116053efbe76e5a190e7198a3bdf3d281
```

## 불변 Exp445 failure authority

Exp445 공식 root는
`/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810`이며,
`attempt_0000_antigravity`는 complete/inactive다. 계산 `30/30`, runtime error `0`,
label access `0`, CPU outer `11`/inner `1`, elapsed `913.3950564998668`, abstention `4`와
ordered record-prefix aggregate SHA256
`af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`는 immutable authority다.
publication과 evaluation은 absent이며 failure text는 정확히
`Exp444 original-control replay is invalid`다. 이는 failed scientific result가 아니라
immutable failed-publication predecessor다.

```text
checkpoint context = e7ea9465327ae4b71dfe06c8a83f7c3eb77e4d1445b0598da3de6afa4a5a53c3
checkpoint state   = b8a4d6f1bb1cb30680f4244c7e025c43367a41ed90518b6e49dc4d09e890ede4
progress           = 8d34590450e2fcd2e21dcc6f03454ac7d4cd9fc1d15e1ecb25fc858c20615eb5
PID                = e0f2d8ecc65008decdb8ba8292c7204c377bebeccb8ab564bb225bae307b3fbf
publication failure= 9c58eddaa152bde221a5b7673be3d413204ff3dfe9fd2b4262e6d9065b2c6db0
```

## 동결된 변경과 parity gate

유일한 변경은 encoding boundary에서 `authority.control_rows`를 recursive JSON
materialization하여 parser와 canonical control-replay bytes에 같은 materialized rows를
전달하는 것이다. tuple을 parser가 직접 허용하도록 계약을 느슨하게 바꾸지 않는다.

```text
only change = recursive JSON materialization at the encoding boundary
record parity = 30 index-wise wrapper SHA equal to Exp445 and Exp444
aggregate parity = af24dd48...abab034
commands = readiness, new, inspect
labels = 0
backend = cpu / outer 11 / inner 1
```

Exp446는 Exp445/Exp444와 같은 30 coordinates를 CPU/11로 처음부터 재계산한다. 각 wrapper
SHA가 index-wise 같고 aggregate가 같을 때만 create-only publication을 허용한다. Exp445와
Exp444 root에는 write, rename, unlink, retry, resume, publication 또는 evaluation을 하지 않는다.

## 금지와 실패 보존

operator surface는 `readiness --reviewed-commit REVIEWED_COMMIT`,
`new --reviewed-commit REVIEWED_COMMIT`, `inspect`뿐이다. `resume`, `retry`, `evaluate`는
없다. TEST label-capable evaluator와 SQLite label query 경로는 package/launcher에서
import하지 않는다.

root allocation 전 failure는 mutation-free로 root를 만들지 않는다. allocation 뒤
record, materialization, parity 또는 publication failure는 새 Exp446 root에 create-only
evidence로 보존하고 같은 root에서 재시도하지 않는다. prediction, publication,
evaluation, promotion, champion hashes는 현재 모두 null이다.

```text
scientific_success_criterion = none
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
test_informed_reselection_allowed = false
```
