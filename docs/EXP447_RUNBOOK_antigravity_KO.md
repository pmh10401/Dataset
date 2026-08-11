# Exp447 Exp446 동결 예측 사후 성능평가 실행 안내서

## 현재 상태

Exp447은 `official_readiness_passed_evaluation_failed_immutable_no_retry`다. fresh Sol High
`ship` 검토 뒤 readiness는 정확히 한 번 성공했지만 evaluate는 정확히 한 번
`ValueError: TEST label row is invalid`로 실패했다. Exp447 root는 생성되지 않았고 sibling
failure audit만 남았다. 이 문서는 TEST label 접근, DB 열기, Exp446 root 변경 또는 Exp447
명령 재실행을 허가하지 않는다.

## 실행된 명령과 영구 중단

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py readiness --reviewed-commit 33e7dc9c18c179a64bc0020b0bb939c5e4caa394  # executed once, exit 0
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py evaluate --reviewed-commit 33e7dc9c18c179a64bc0020b0bb939c5e4caa394   # executed once, exit 1; no retry
```

`inspect`는 실행하지 않았으며, 실패 뒤 실행할 수 없다. `new`, `resume`, `retry`, 임의
root/hash/source/lane/threshold/budget override는 처음부터 없었고, 이제 모든 Exp447 lifecycle
명령은 영구 금지다.

## 실패 경계와 보존 증거

Readiness는 Exp447 root absence, clean reviewed HEAD, immutable Exp446/Exp404 bytes, database
SHA/inode/nlink, 두 번의 동일한 Exp446 inspection 및 frozen population을 통과했다. 이어진
단 한 번의 evaluate는 label snapshot에서 실패했다. audit은 label session이 열리고 닫혔으며
`query_attempts=1`, `query_completions=0`, `label_rows=0`임을 기록한다. 결과 root는 여전히
absent/non-symlink이므로 evaluation artifact와 source-free inspect 대상은 없다.

실패 audit 경로는 다음이며 SHA-256은
`9f8f7d3d9b92670988ca1d0bb689711d12ef1536777881f18991143a090a2b97`이다.

```text
/Users/mac/Documents/Dataset/evaluation_audit_89aff47cdd7c43cd919a34f81982bc8b_antigravity.json
```

2026-08-12의 한 번의 사후 forensic aggregate SQL 확인은 raw DB label storage가 exact TEXT
`'0'=221284`, `'1'=4516`이고 `instance_index`가 integer임을 기록했다. 이는 prediction,
readiness 또는 Exp447 재시도 근거가 아니며 dataset 이름·위치·개별 label은 사용하지 않았다.

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
Exp447에는 dashboard로 제시할 immutable evaluation artifact 자체가 없다. 다음 연구 작업은
Exp448의 새 설계와 사전등록뿐이며 Exp447 command는 다시 실행하지 않는다.
