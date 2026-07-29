# Exp417 인과적 다중-수평선 Shadow 사전등록

## 목적과 경계

Exp417은 정상 TRAIN 파형만으로 작은 인과적 다중-수평선 모델을 학습해,
지속적으로 예측 불가능한 계열을 `causal_multihorizon_shadow` 검토 후보로
표시하는 연구 전용 Shadow 점수원이다. Hard, Standard, Priority,
automatic-normal lane은 수정·대체·승격하지 않는다. 후보가 0개인 깨끗한
abstention도 유효한 연구 결과다.

Exp417은 TEST label, 실제 anomaly 위치, 과거 TEST 또는 family 성능,
사람 검토 결과를 모델 적합, 정규화, 보정, eligibility, threshold, routing,
workload gate에 사용하지 않는다. TEST values는 정상 TRAIN contract와 동결
configuration을 검증한 뒤에만 score할 수 있다. 이 패키지는 동결된 TEST label
artifact를 import, open, hash, stat, query, 또는 이름으로 참조하지 않으며, TEST
SQL은 values와 instance identity만 선택한다.

## 동결된 입력과 식별성

- population은 정확히 1,117 datasets이다.
- database SHA256은
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`이다.
- lane totals는 Hard 2,218, Standard 806, Priority 31, automatic-normal exact 35다.
- Exp291 prediction/manifest SHA256은 각각
  `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848`,
  `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e`다.
- Exp292 prediction/manifest SHA256은 각각
  `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b`,
  `0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0`다.
- 실행 전 database, population, 네 lane artifact, config, code commit과
  Python/NumPy/Numba/PyTorch/platform version을 동결·재검증한다.
- Exp414의 exact-normal TRAIN guard는 필수다. near-repeat guard는
  `not_available`로 기록하고 적용하지 않는다.

## 금지와 fail-closed 규칙

TEST-layout rule, tail replacement, padding, truncation, resampling,
dataset/family-specific routing, 또는 top-N trimming은 금지다. qualification 또는
workload gate 실패 시 threshold, horizon, seed, source, budget을 완화하지 않고
abstain 또는 official publication 거부로 끝낸다. Exp418 이후 rule 변경은 새
experiment 번호와 새 preregistration을 요구한다.

생성 prediction, checkpoint, metric, dataset-derived result는 로컬에만 남기고
GitHub에 export하지 않는다. GitHub에는 code, preregistration, runbook,
synthetic test evidence만 가능하다. 이 구현 계획은 전체 1,117-dataset run을
포함하지 않으며, full run에는 사용자의 **별도 명시적 승인**이 필요하다.

## TRAIN-only 전처리와 fold

계열은 관측된 정확한 길이로만 group하며 resample, pad, truncate하지 않는다.
세 distinct usable horizon이 없거나 정상 TRAIN이 50개 미만, Exp414의
deterministic repair 뒤 non-finite가 남거나, raw/difference robust scale이
non-finite 또는 `<=1e-12`이면 group은 abstain한다. 50–99개는 diagnostic OOF만,
100개 이상만 candidate eligibility를 가질 수 있다.

두 input channel은 (1) group-level 정상 TRAIN median/MAD robust scale으로
표준화한 raw waveform, (2) 자체 normal TRAIN median/MAD scale로 표준화한
first difference다. per-series normalization은 사용하지 않는다. canonical
waveform bytes가 같은 normal TRAIN은 같은 identity group으로 묶고, identity
hash 기반 결정론적 5-fold OOF에서 fold를 가로지르지 않는다. 각 OOF training
side의 early-stopping split도 identity hash로 정하며 양쪽에 최소 다섯 identity
group이 없으면 abstain한다.

## 동결된 모델과 점수

길이 `L`의 horizon은 정확히 `1`, `ceil(0.02*L)`, `ceil(0.10*L)`다. 모델은
2 input channel, 3 causal dilated 1D convolution block, 32 hidden channel,
kernel 5, dilation 1/2/4, 세 output head를 사용한다. equal-weight Huber loss,
AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch `min(32,n_train)`, 최대 100 epoch,
patience 10으로 고정한다. model seed는 `41701`, `41702`, `41703`이다.

각 horizon에서 valid causal residual의 absolute value를 normal OOF median과
robust scale로 정규화한다. `Q_h`는 series normalized residual 95th percentile,
point threshold는 normal OOF 99th percentile, `R_h`는 최장 연속 exceedance run /
valid position 수다. `S_h=Q_h+2R_h`이고 empirical p-value는
`(1+count(S_oof>=S_test))/(1+n_oof)`다. 한 seed 후보는 세 horizon 중 두 개 이상이
`p<=0.01`이고 각 agreeing horizon `R>=0.05`일 때만 가능하다. official candidate는
세 model seed가 동일하게 결정하고 exact-normal guard, frozen lane exclusion,
qualification/workload gate를 모두 통과해야 한다.

## TRAIN-only qualification과 workload

normal OOF candidate rate `<=0.01`, 95% Wilson upper `<=0.05`, 100% seed agreement,
finite score/parameter/loss/metric, published model에서 backend fallback 없음,
pooled primary synthetic sensitivity `>=0.70`, 각 primary type `>=0.50`을 모두
요구한다. primary perturbation은 persistent level shift, gradual drift,
persistent amplitude scaling, persistent period change다. secondary isolated spike는
diagnostic일 뿐 qualification 조건이 아니다.

onset은 valid length의 40%, duration은 남은 60%, level/drift endpoint는
`3.0` normal TRAIN robust scales, amplitude는 centered waveform `1.5`배, period
warp은 factor `1.25`, secondary spike는 1% length에 `5.0` scales로 고정한다.
perturbation seed는 `41711`, `41712`, `41713`이고 모든 eligible normal OOF series에
각 primary perturbation을 seed당 한 번 적용한다.

unlabeled candidate count만으로 workload를 계산한다. dataset 새 Shadow rate가
`>0.05`면 전체 dataset은 `excessive_dataset_workload_abstain`이고 후보를 보존하지
않는다. lane exclusion과 dataset abstention 뒤 global 새 Shadow rate가 `>0.01`이면
official eight-file bundle을 publish하지 않고 `global_workload_gate_failed`를
기록한다. 모든 dataset은 정확히 qualified, abstained, failed 중 하나로 accounting한다.

## Backend와 resume

canonical backend는 deterministic PyTorch CPU다. NumPy/Numba는 deterministic score
aggregation 등에만 쓰며 NumPy parity 이후에만 Numba를 쓴다. MLX는 명시적으로
제외한다. MPS는 same seeds/input, identical candidate coordinates, normalized max
error `<=1e-4`, Spearman `>=0.999`, finite/no fallback, transfer와 synchronization을
포함한 speedup `>=1.1`을 모두 통과한 TRAIN-only benchmark일 때만 채택 가능하다.
현재 환경은 `torch 2.13.0`, `numpy 2.3.5`, `numba 0.66.0`, MPS built but unavailable이며
CPU가 canonical이다. MPS-heavy process는 한 번에 하나만 실행한다.

checkpoint 재사용은 Git commit, database/population/lane/config identity,
preprocessing/length-group/fold/model/seed/runtime/backend/synthetic contract가 모두
같을 때만 가능하다. mismatch, truncation, duplicate record, invalid type,
non-finite, failed parity, unexplained fallback은 fail-closed로 reject한다.

## 공식 산출물과 Exp418 분리

global qualification과 workload gate를 통과했을 때만 새 no-clobber directory에
원자적으로 정확히 다음 여덟 파일을 publish한다.

1. `exp417_shadow_predictions.jsonl`
2. `exp417_dataset_qualification.csv`
3. `exp417_oof_normal_metrics.csv`
4. `exp417_synthetic_safety.csv`
5. `exp417_backend_benchmark.json`
6. `exp417_frozen_config.json`
7. `EXP417_RESULT_REPORT_KO.md`
8. `MANIFEST.json`

prediction row에는 dataset/candidate index, length group, 세 TRAIN-calibrated score와
p-value, persistence, seed consensus, lane exclusion, `contains_test_labels=false`,
`contains_evaluation_metrics=false`만 담는다. TP, FP, FN, TN, precision, recall,
F1, anomaly label/location, human-reviewed result는 담지 않는다.

Exp418은 Exp417 prediction과 manifest가 동결된 뒤에만 만드는 별도 retrospective
audit다. Shadow TP/FP, precision, concentration, overlap/workload diagnostic은
보고할 수 있지만 Exp417 horizon, architecture, p-value, persistence, seed,
perturbation, eligibility, budget을 바꾸거나 lane을 승격·재작성할 수 없다.
불리한 Exp418 결과도 Exp417을 frozen research evidence로 남긴다.
