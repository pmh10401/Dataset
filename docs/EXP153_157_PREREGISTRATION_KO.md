# Exp153–157 사전등록

## 기준선

- 기준 정책: `experiment_152_virtual_run_conformal_policy_v2`
- 데이터셋: Exp137 기준 1,117개
- source: ROCKET Exp40, Exp55, Exp56, family-independent Exp84
- 후보 조건: 최소 2개 source 동의
- alpha grid: `0.005, 0.01, 0.02, 0.05`
- Block-B legacy rate: 1.5%
- Block-C legacy rate: 1.0%

## 공통 금지 사항

- TEST label, anomaly 위치, TEST anomaly 수로 정책을 변경하지 않는다.
- TEST precision/F1을 보고 alpha·pooling·fusion·confirmation·risk target을 선택하지 않는다.
- TEST-length top-N budget을 재도입하지 않는다.
- family 이름 또는 과거 family별 TEST 성능으로 pooling/source/routing을 바꾸지 않는다.
- Priority review를 autonomous Hard alert로 승격하지 않는다.
- Exp152와 기존 결과를 덮어쓰지 않는다.

## Exp153 — TRAIN-normal calibration validity audit

각 source의 TRAIN OOF score를 사용한다. 각 OOF score는 자신의 empirical reference에서 제외한 뒤 pseudo-normal p-value를 계산한다. Block-B/C도 TRAIN OOF score의 leave-one-out threshold로 audit한다.

모든 pseudo-run은 정상이다. 따라서 Hard alert는 false alert로 집계한다.

주요 출력:

- 정상 pseudo-run 수
- Hard false alerts / 100 normal runs
- Standard/Priority review / 100 normal runs
- dataset별 false-alert rate
- alpha별 abstention 및 resolution 제한

## Exp154 — pooled/hierarchical conformal

네 variant를 모두 보고하며 선택하지 않는다.

```text
local_baseline
leave-one-dataset-out global robust-z pool
TRAIN-only group pool
local → group → global hybrid fallback
```

그룹은 다음 TRAIN-only 정보만 사용한다.

- TRAIN instance 수 bin: 5–19, 20–49, 50–99, 100+
- TRAIN target length quartile

Family와 TEST 통계는 사용하지 않는다. Target dataset의 TRAIN score는 pooled reference에서 제외한다. Pool reference 최소 크기는 200으로 고정한다.

## Exp155 — source independence 및 continuous fusion

네 source의 TRAIN OOF Spearman correlation을 감사한다.

세 variant를 모두 보고한다.

```text
discrete_2of4
top2_mean_meta
corr_weighted_top2_meta
```

연속형 meta-score는 TRAIN OOF score로 calibration하며 TEST label을 사용하지 않는다.

## Exp156 — confirmation calibration

Exp152의 frozen candidate universe를 유지하고 confirmation만 비교한다.

```text
legacy_countcap: Block-B 1.5%, Block-C 1.0%
conformal_strict: Block-B 0.01, Block-C 0.02
conformal_balanced: Block-B 0.02, Block-C 0.05
```

세 variant를 모두 sensitivity로 보고하고 선택하지 않는다.

## Exp157 — risk-controlled alpha

Exp153 dataset별 pseudo-normal audit로 Wilson one-sided 95% upper bound를 계산한다.

정상 Hard false alert target:

```text
0.005 per run = 0.5 / 100 runs
0.010 per run = 1.0 / 100 runs
0.020 per run = 2.0 / 100 runs
```

각 target에서 upper bound를 만족하는 가장 큰 사전등록 alpha를 선택한다. 만족하는 alpha가 없으면 dataset-level abstention한다. TEST label은 선택에 사용하지 않는다.

## 해석 제한

- Cross-dataset pooled calibration은 동일한 공개 벤치마크 내 연구 가정이다.
- Cross-fitted empirical conformal audit이며 실제 설비 false-alarm 보장을 의미하지 않는다.
- Exp152 configuration provenance D2는 별도 미해결 과제다.
- 신규 잠금 데이터 검증 전에는 일반화 또는 prospective 성능을 주장하지 않는다.
