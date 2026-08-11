# Exp447 Exp446 동결 예측 사후 성능평가 사전등록

## 목적과 고정 범위

Exp447은 immutable Exp446 publication의 30개 동결 후보와 Exp404 Hard 기준을
사후에 설명적으로 비교하는 evaluation-only 실험이다. Exp446 prediction bytes, source,
feature, score, threshold, lane, budget, backend, CLI, root, publication 및 manifest를
바꾸지 않는다. 결과를 보고 같은 Exp447에서 후보를 더하거나 빼지 않으며, champion,
scientific winner, operational alert 또는 promotion을 만들지 않는다.

## Global Constraints (verbatim)

- Every newly created filename contains `antigravity`; do not create `__init__.py`.
- Experiment identity is `Exp447`; schema is `exp447`; run identity is `exp447-exp446-postfreeze-performance-evaluation-antigravity-20260811`.
- Run root is `/Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811` and must be absent before official readiness.
- Approved design path is `docs/superpowers/specs/2026-08-11-exp447-exp446-postfreeze-performance-evaluation-antigravity-design.md` with SHA256 `043009941619335b658af99911b1463f66ea73e93d600fe1cfb884391afea868`.
- Exp446 prediction SHA is `602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100`; manifest file SHA is `bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11`; artifact-map file SHA is `d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d`.
- Exp446 population is exactly 1,117 datasets, 30 source coordinates, 26 stable coordinates, 4 unstable coordinates, and prediction label-access count 0.
- Exp404 Hard prediction SHA is `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`.
- Database SHA is `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`; expected inode is `1048138`; official preflight requires link count 1.
- TEST labels become reachable only after two byte/root-identical Exp446 inspections and Exp404 validation. Exactly 1,117 ordered queries and 195,900 `(instance_index,label)` rows are allowed.
- Frozen baseline confusion in `[TP,FP,FN,TN]` order is `[1915,303,2003,191679]`.
- Bootstrap uses NumPy `Generator(PCG64(44720260811))`, exactly 20,000 paired dataset draws, int64 indices, `endpoint=False`, and linear 2.5/97.5 percentiles.
- Evaluation is always `descriptive_complete_no_promotion`; there is no success threshold, scientific winner, same-experiment reselection, champion update, or operational-alert promotion.
- Exp446 root, CLI, publication, prediction, manifest, and artifacts are read-only. Do not add an evaluate command to Exp446.
- Exp447 exposes exactly `readiness --reviewed-commit`, `evaluate --reviewed-commit`, and `inspect`; no root/hash/source/lane/threshold/budget overrides, `new`, `resume`, or `retry`.
- The isolated interpreter is `/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python`; do not alter its package set.
- Exp447 is an evaluation-only experiment, not a score source; no MLX/MPS benchmark or training is performed.
- Compare `docs/codemap/codemap_exp446_antigravity.lock` with the current repository before the first source edit. Use `docs/codemap/codemap_exp446_antigravity.json` to identify callers, effects, and tests. Generate `docs/codemap/codemap_exp447_antigravity.html`, `.json`, and `.lock` together whenever the new boundary changes.
- Native implementation uses `sol_advisor_terra_implementer`; design, commitment-boundary review, and final review use a fresh `sol_advisor_sol_reviewer`.
- The private owner-only dashboard is `https://timeseries-research-observatory.pmh10401.chatgpt.site`; its separate repository is `/Users/mac/.codex/visualizations/2026/08/10/019fedc4-df12-73b2-9812-a5f8f387ac78/research-observatory-antigravity-site`. Do not touch it until immutable Exp447 evaluation plus label-free inspect have succeeded and that repository is clean.
- The local/deterministic evaluator imports no network/dashboard code. A separate exporter may emit only sanitized aggregate data after truth closeout: never raw TEST labels, instance/anomaly indices, dataset names, local paths, hashes, logs, checkpoints, DB names, secrets, or per-dataset rows.
- Live status is a separate non-scientific sync and uses only the existing exact 12-field `PublicExperimentStatus` allowlist: experiment_id, display_name, status, stage, completed, total, success, errors, backend, throughput, eta_seconds, updated_at. Exp447 total is 1117.
- The completed dashboard payload is `exp447.public-evaluation.v1`: experiment_id, generated_at, dataset_count, decision, four aggregate comparison rows (hard baseline, hard+source30, hard+stable26, hard+unstable4; each TP/FP/FN/TN/precision/recall/F1/FPR), aggregate diagnostic counts (rescued, added-false-alarm, regressed, strong-regressed), and three paired-bootstrap aggregate delta/CI rows. It contains no dataset lists.
- Dashboard work is a post-evaluation presentation boundary, not a scientific or promotion gate. Dashboard failure never changes immutable scientific artifacts and never permits evaluation retry or reevaluation.

## 동결 평가 계약

비교 대상은 `hard_baseline`, `hard_plus_source_30`, `hard_plus_stable_26`,
`hard_plus_unstable_4`다. dataset별 및 전체 TP/FP/FN/TN, precision, recall, F1,
false-positive rate, macro F1, rescue/false-alarm/regression/strong-regression 전환과 세
paired-bootstrap interval을 기록한다. 이는 결과 설명용이며 합격선이나 선택 규칙이 아니다.

```text
scientific_success_criterion = none
evaluation_status = descriptive_complete_no_promotion
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
test_informed_reselection_allowed = false
automatic_champion_promotion = false
```

구현과 fresh Sol High `ship` 검토 전에는 official command, TEST label access,
evaluation artifact, champion change 및 promotion이 모두 unavailable이다.

## Private dashboard presentation boundary

```text
approved_design_sha256 = 043009941619335b658af99911b1463f66ea73e93d600fe1cfb884391afea868
approved_plan_sha256 = d137020a319ec004a6054744a215ffbce9ecef4b6bfb3f2bab5f6f71f0a921d3
```

The dashboard may consume only the future immutable aggregate export after evaluation and
label-free inspect have both succeeded. It is private owner-only, separate from the evaluator,
and never a scientific/promotion gate. The `exp447.public-evaluation.v1` payload is limited to
the fixed aggregate schema; raw labels, indices, dataset names/lists, local paths, hashes, logs,
checkpoints, DB names, secrets, and per-dataset rows are forbidden. Its separate live status uses
only the exact `PublicExperimentStatus` 12-field allowlist and `total=1117`.
