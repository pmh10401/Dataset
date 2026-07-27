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
- full sequential run은 verified frozen universe 1,117 dataset으로 완료했다.
  모든 data stage checkpoint와 audit/consensus JSONL은 각각 1,117개이고 stage
  error는 0이다. Exp406/407은 `n_train_below_50` explicit abstention을 각각
  616개 기록했고, learned completed dataset은 source별 501개다.
- final frozen lane total은 Hard `2218`, Standard `806`, Priority `31`이다.
  forbidden prediction field는 0개, independent reaggregation mismatch는 0개다.
  read-only sorted Exp404 prediction SHA256 재검증은
  `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`로
  manifest와 두 번 모두 정확히 일치했다.
- full command의 `/usr/bin/time -p` wall runtime은 `6832.25s`였다. MPS parity
  조건은 통과했지만 CPU보다 느려 full publication은 CPU canonical path를 유지했다.
- full input hash는 dataset별로 streaming 처리한다. CPU stage는 stale dataset을
  hash 직후 worker에 제출하고 parent만 checkpoint/status를 기록한다. 따라서
  전체 hash 사전계산 때문에 checkpoint가 지연되지 않는다.

## Task 13 Fix Round 1 사후 메타데이터 (2026-07-27)

- final representative smoke run ID는 `5c61c5c193604deb94a104e37b40f694`이다.
  지정 smoke output만 reset했고, 각 data stage checkpoint/audit/consensus는 정확히
  20개이며 TRAIN tier는 abstain/robust_distance/classical_committee/learned_methods
  각각 5개다. error는 0, cross-stage dataset ID/n_train/eligibility parity와
  independent reaggregation은 모두 통과했다.
- smoke reset entrypoint wrapper는 지정 output의 `checkpoints/`와 `status.json`만
  삭제한다. immutable publication은 유지되며, stale checkpoint가 남지 않는
  regression으로 검증했다.
- representative selector는 normal TRAIN row만 읽는 canonical SQLite query를 쓴다.
  TEST value/metadata/query/order에 의존하지 않는다.
- full consensus valid all-source-abstention은 `162`개이며 모든 FeatureBlock,
  SSL, synthetic status/reason을 기록한다. consensus SHA256은
  `c4016db7a3c17d130b5bd8796e0139325cb63b0645bfe3f0ca7b96c825339d23`이다.
- frozen MPS benchmark artifact는 `task13_exp406_mps_benchmark.json` (SHA256
  `81d606ac6210f3b2edf03ec29a688b65623390d4e036336bc938e5ab8f417249`)이다.
  `CricketX_normal_1`, seed `20260727`에서 rank는 모두 일치하고 최대 raw-score
  delta는 `2.384185791015625e-07`이다. CPU mean `0.666162s`, MPS mean
  `2.181010s`이므로 parity는 통과했지만 MPS가 느려 CPU를 유지했다. full manifest는
  benchmark hash와 정확한 full/report command를 기록한다.
- final full report refresh run ID는 `95a8449547124c0f9a97feca9aac725a`이며
  verified full upstream checkpoint만 읽어 완료했다.
