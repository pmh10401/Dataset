# Exp403-408 실행 Runbook

## 목적

Exp403-408은 동결된 automatic Hard를 유지한 채, Exp405의 반사실적 하향
가능성과 Exp406-408의 review-only Shadow 후보를 retrospective로 평가한다.

## 실행 순서

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py \
  --stage all \
  --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
```

재개 시에도 같은 명령을 사용한다. `report`만 다시 발행하려면 다음을 사용한다.

```bash
python3 scratch/run_exp403_408_feature_ml_validation.py \
  --stage report \
  --output-dir /Users/minho/Documents/Dataset/exp403_408_feature_ml_validation
```

1. `consensus-evaluate` 완료 시 prediction/provenance/audit/consensus와 report 직전
   DB+WAL+SHM/lane 해시를 `exp403_408_evaluation_input_manifest.json`으로 동결한다.
2. `report` stage는 이 manifest와 외부 pinned prediction/provenance를 검증한 뒤에만
   TEST label DB를 읽고, 그 뒤에 `exp405_demotion_dataset_metrics.csv`,
   `exp405_demotion_candidate_evaluation.csv`, `exp408_variant_metrics.csv`,
   `exp408_variant_candidate_evaluation.csv` 평가 CSV를 생성하고 publication manifest로 동결한다.
3. 독립 재집계는 동결된 평가 CSV와 frozen label publication을 다시 읽어 prediction
   SHA256, Hard lane parity, CSV/JSONL coverage,
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
- `exp403_feature_audit.jsonl`, `exp403_408_frozen_test_labels.csv`
- `exp408_consensus_results.jsonl`, `exp408_variant_metrics.csv`,
  `exp408_variant_candidate_evaluation.csv`
- `exp403_408_evaluation_input_manifest.json`,
  `exp403_408_evaluation_publication_manifest.json`
- `EXP403_408_RESULT_REPORT_KO.md`, `MANIFEST.json`, `MANIFEST.md`

## 중단 조건

다음 중 하나라도 발생하면 report를 발행하지 않는다.

- prediction SHA256 또는 frozen Hard lane parity 불일치
- 모든 integer metric의 불일치
- 모든 float metric의 차이가 `1e-12` 초과
- CSV/JSONL의 dataset, variant, alpha, candidate coverage 누락
- Exp405 production 변경 또는 Exp406-408 prospective 검증을 암시하는 주장

## Task 13 사후 실행 메타데이터 (2026-07-27)

- representative smoke는 `/Users/minho/Documents/Dataset/exp403_408_feature_ml_validation_smoke`에서
  20개 dataset으로 완료했다. TRAIN eligibility coverage는 abstain/robust_distance/
  classical_committee/learned_methods 각각 5개였고 stage error는 0이었다.
- smoke의 Exp408 all-source abstention은 5개였으며, 각 artifact는 Hard/Standard/
  Priority를 보존하고 5개 variant의 모든 alpha 후보를 0개로 기록했다.
- Exp406 CPU/MPS parity smoke는 `CricketX_normal_1` (`n_train=52`)에서 CPU 2회,
  MPS 2회 실행했다. raw rank는 모두 일치했고 최대 raw-score 차이는
  `2.384185791015625e-07` (`<= 1e-5`)이었다. publication full run은 frozen CPU
  canonical path를 유지했다.
- full sequential run은 verified frozen universe 1,117 dataset으로 시작했다.
  실행 중인 상태와 최종 검증 값은 Task 13 report에 기록한다.
