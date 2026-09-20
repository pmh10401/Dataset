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

## 닫힌 dashboard 경계

Exp447은 `exp447.public-evaluation.v1` 또는 다른 dashboard payload를 절대 만들 수 없다.
유일한 evaluate가 valid TEST label row와 immutable result artifact 이전에 실패했고 inspect도
영구 금지이기 때문이다. 사이트 repository·dashboard·exporter는 이 실패를 다시 해석하거나
표시할 권한이 없다. dashboard에 도달할 수 있는 것은 별도 Exp448의 새 설계·사전등록·독립
immutable truth closeout을 통과한 근거뿐이며, Exp447의 partial failure를 재사용할 수 없다.
