# Exp448 Exp447 TEXT label 경계 수리 사전등록

```text
approved_design_sha256 = ed2b6e5e8b8ee63ab41a15e54f679147f7e14aeca601f4d62f694ca104ed3ce4
approved_plan_sha256 = b79b3b7fc43aae7b15e3c7c52e648dd4404d50930f5bd71ce106c244d21fe934
```

## 목적

Exp448은 Exp447의 사후 evaluation이 exact TEXT TEST label을 int로 잘못 검사해 실패한
공학적 경계만 수리하는 독립 controlled fork다. prediction, candidate, score, threshold,
source, lane, budget, method, backend, training은 변경하지 않는다.

## 동결 신원과 predecessor

```text
experiment = Exp448
schema = exp448
kind = exp447_text_label_boundary_repair
run_identity = exp448-exp447-text-label-boundary-repair-antigravity-20260812
root = /Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812
```

Exp447 failure truth commit은 `7e85d5c7948f8592db80f958d746f31138c67bfb`, official attempt
commit은 `33e7dc9c18c179a64bc0020b0bb939c5e4caa394`다. readiness 1회 exit 0, evaluate
1회 exit 1, inspect 0회라는 Exp447 lifecycle은 immutable이다. audit
`/Users/mac/Documents/Dataset/evaluation_audit_89aff47cdd7c43cd919a34f81982bc8b_antigravity.json`
의 SHA `9f8f7d3d9b92670988ca1d0bb689711d12ef1536777881f18991143a090a2b97`, inode 19812443,
nlink 1, `exp447_evaluation_audit`, `label_snapshot`, attempts/completions/rows 1/0/0,
closed state, absent destination, exact `ValueError: TEST label row is invalid`을 verify한다.
Exp447 root는 absent/non-symlink이며 Exp447 retry, inspect, result, promotion, dashboard는 없다.

## label boundary

Exp446 prediction/manifest/artifact map SHA와 Exp404 Hard SHA, DB SHA/inode/nlink 및
1117 dataset, 195900 row, 30/26/4 lane을 design의 exact values로 검사한다. exact Exp447
failure verify, two byte/root-identical Exp446 captures, Exp404 validation 뒤에만 1117 ordered
query를 연다. `instance_index`는 exact int, `raw_label`은 exact str `'0'` 또는 `'1'`만 허용한다.

```python
0 if raw_label == '0' else 1
```

int/bool/float/bytes/whitespace/null/other string은 실패한다. raw label은 evaluator,
artifact, prediction, readiness, routing, tuning에 닿지 않는다. forensic whole-DB `'0'=221284`,
`'1'=4516`은 post-freeze storage evidence일 뿐이다.

## 평가와 해석

순서는 `hard_baseline`, `hard_plus_source_30`, `hard_plus_stable_26`(primary),
`hard_plus_unstable_4`로 고정한다. diagnostics는 rescued, false_alarm, regressed,
strong_regression이다. baseline `[1915,303,2003,191679]`, source30
`[1930,318,1988,191664]`, and PCG64(44720260811), 20000, int64, endpoint false,
linear 2.5/97.5, draw SHA
`f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e`를 그대로 쓴다.

결과는 항상 `descriptive_complete_no_promotion`이다. scientific success criterion은 null이며
winner/champion/pending/operational/promotion/reselection은 금지다.

## 운영 경계

새 파일명은 모두 `antigravity`를 포함하고 `__init__.py`는 없다. MLX/MPS/workers/training을
쓰지 않는다. CLI는 reviewed commit의 `readiness`, `evaluate`, `inspect`만 제공하고 official
nonzero면 영구 중단한다. evaluator는 network/dashboard import가 없으며, dashboard는 성공한
immutable truth closeout과 clean owner-only site 뒤의 sanitized aggregate-only
`exp448.public-evaluation.v1` consumer일 뿐이다. raw labels/indices/dataset names/lists/
per-dataset rows/hashes/paths/logs/DB/checkpoints/secrets은 export하지 않는다.
