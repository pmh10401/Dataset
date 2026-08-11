# Exp447 Exp446 동결 예측 사후 성능평가 실행 안내서

## 현재 상태

Exp447은 `design_approved_implementation_planned`다. reviewed implementation과 fresh
Sol High `ship` 검토가 없으므로 아래 명령은 고정된 미래 operator surface일 뿐 지금은
unavailable이다. 이 문서는 TEST label 접근, DB 열기, Exp446 root 변경 또는 evaluation
실행을 허가하지 않는다.

## 고정된 미래 명령

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py evaluate --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py inspect
```

`REVIEWED_COMMIT`은 clean current HEAD와 정확히 같은 operator-supplied 40-hex commit이다.
`new`, `resume`, `retry`, 임의 root/hash/source/lane/threshold/budget override는 없다.

## 미래 실행 순서와 불변 경계

허가된 구현 이후의 유일한 순서는 `readiness -> evaluate -> inspect`다. `readiness`는
Exp447 root absence, clean reviewed HEAD, immutable Exp446/Exp404 bytes, database SHA/inode/nlink,
두 번의 동일한 Exp446 inspection 및 frozen population을 검증한다. `evaluate`만 ordered
1,117-query/195,900-row TEST-label snapshot을 lazy read-only로 열 수 있으며 single-use다.
`inspect`는 frozen result bytes만 읽고 SQLite, evaluator label route 및 worker를 열지 않는다.

Exp446 root, CLI, publication, manifest, prediction 및 artifacts는 read-only다. coherent
evaluation도 descriptive-only이며 champion, promotion, scientific winner, operational alert,
same-experiment reselection을 허용하지 않는다.

## Private dashboard handoff (future-only)

Immutable evaluation과 label-free inspect의 independent truth closeout 뒤에만, clean separate
site repository가 private owner-only observatory에 aggregate-only payload를 제시할 수 있다.
evaluator는 network/dashboard code를 import하지 않고, exporter는
`exp447.public-evaluation.v1`의 four aggregate comparisons, aggregate diagnostic counts, three
paired-bootstrap delta/CI rows 및 decision만 제공한다. Raw TEST label, index, dataset name/list,
local path, hash, log, checkpoint, DB name, secret 및 per-dataset row는 제공하지 않는다.

Live status는 scientific evaluator와 별도이며 exact 12-field `PublicExperimentStatus`
allowlist만 사용한다: `experiment_id`, `display_name`, `status`, `stage`, `completed`, `total`,
`success`, `errors`, `backend`, `throughput`, `eta_seconds`, `updated_at`; Exp447 `total=1117`이다.
Dashboard failure는 scientific artifact를 바꾸거나 evaluate retry/reevaluation을 허용하지 않는다.
