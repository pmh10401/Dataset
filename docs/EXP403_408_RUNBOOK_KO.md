# Exp403-408 실행 Runbook

## 목적

Exp403-408은 동결된 automatic Hard를 유지한 채, Exp405의 반사실적 하향
가능성과 Exp406-408의 review-only Shadow 후보를 retrospective로 평가한다.

## 실행 순서

1. 모든 prediction과 evaluation publication을 동결한다.
2. `report` stage가 `exp405_demotion_dataset_metrics.csv`와
   `exp408_variant_metrics.csv`를 포함한 평가 산출물을 확인한다.
3. 독립 재집계가 prediction SHA256, Hard lane parity, CSV/JSONL coverage,
   정수 exact match, float 1e-12 tolerance를 통과해야 한다.
4. 그 다음에만 Korean result report와 `MANIFEST.json`/`MANIFEST.md`를 발행한다.

## 해석 규칙

- automatic Hard는 변경하지 않는다.
- Exp405는 counterfactual retrospective 진단이다. production 변경이나 배포
  검증 근거가 아니다.
- Exp406-408은 human-assisted review-only Shadow다. Priority/Standard/Hard를
  자동으로 승격하거나 대체하지 않는다.
- human-assisted review 지표는 autonomous Hard 지표와 분리한다.
- prospective validation 또는 deployment validation을 주장하지 않는다.

## 필수 산출물

- `exp404_prediction_manifest.json`, `exp404_hard_confidence.jsonl`
- `exp405_demotion_evaluation.json`, `exp405_demotion_dataset_metrics.csv`,
  `exp405_demotion_candidate_evaluation.csv`
- `exp403_feature_audit.jsonl`
- `exp408_consensus_results.jsonl`, `exp408_variant_metrics.csv`,
  `exp408_variant_candidate_evaluation.csv`
- `EXP403_408_RESULT_REPORT_KO.md`, `MANIFEST.json`, `MANIFEST.md`

## 중단 조건

다음 중 하나라도 발생하면 report를 발행하지 않는다.

- prediction SHA256 또는 frozen Hard lane parity 불일치
- 모든 integer metric의 불일치
- 모든 float metric의 차이가 `1e-12` 초과
- CSV/JSONL의 dataset, variant, alpha, candidate coverage 누락
- Exp405 production 변경 또는 Exp406-408 prospective 검증을 암시하는 주장
