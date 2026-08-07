# Exp438 주파수·복잡도 근거 사전등록

이 문서는 Exp438의 과학 규칙을 실행 전에 고정한다. Exp438이 묻는 질문은 하나다.

> 정상 TRAIN으로만 고정한 고주파·저주파 분해와 진폭 민감 복잡도가 Exp437의
> raw·gradient·ACF와 다른 이상 근거를 찾는가?

Exp438은 PhysDiff 논문의 아이디어에서 출발하지만 diffusion 모델이나 PhysDiff
전체를 재현하지 않는다. 단변량 1,117개 데이터셋에 맞춘 별도 review-only
실험이며, 세 결과 채널을 합치거나 가장 좋은 채널을 고르지 않는다.

## 동결 신원과 격리

```text
Experiment: Exp438
Run identity: exp438-frequency-complexity-evidence-antigravity-20260807
Branch: codex/exp438-frequency-complexity-evidence
Worktree: /Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence
Run parent: /Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807
Dataset population: 1,117 sorted unique datasets
Mode: review_only=true
Operational alert eligible: false
Prediction TEST-label access: exactly zero
```

Exp438 prediction 코드는 Exp437의 코드, 실행 폴더, checkpoint, progress,
prediction 또는 publication을 수정·정리·재개하지 않는다. Exp437 결과는 두
실험의 prediction이 각각 동결된 뒤 읽기 전용 비교 근거로만 사용한다.

2026-08-07 사용자 승인에 따라 운영 gate만 보완했다. 별도 운영 런처가 정확한
Exp437 clean commit의 기존 공개 runtime API로 stale-PID checkpoint를 복구할 수
있다. 복구된 Exp437가 `running`, CPU, 실제 PID 생존, heartbeat 300초 이내,
오류 0, TEST-label 접근 0일 때 Exp438 readiness와 공식 prediction을 병행할 수
있다. terminal/inactive Exp437도 계속 허용한다. 죽은 PID, 오래된 heartbeat,
MLX/MPS predecessor, 오류 또는 카운터 후퇴는 Exp438 root 생성 전에 거부한다.
이 보완은 Exp438의 feature, score, threshold, budget, backend 선택 규칙 또는
정보 방화벽을 바꾸지 않는다.

## 정보 방화벽

Prediction 단계가 읽을 수 있는 정보는 다음뿐이다.

- DB `/Users/mac/Documents/Dataset/univariate_ts.db`
- DB SHA256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- 정상 TRAIN의 raw 값과 identity
- TEST raw 값, dataset/instance identity와 raw length
- 동결된 1,117 dataset population과 네 frozen lane 좌표

Prediction, manifest와 artifact hash가 동결되기 전에는 TEST label,
`labels_blob`, 실제 이상 위치, TEST metric/F1/confusion, family별 과거 TEST
성능, human-assisted 지표를 SELECT·open·전달·접근하지 않는다. TEST 결과를 보고
depth, feature, score source, threshold, alpha, budget, route, backend 또는 fusion을
다시 고르지 않는다. Prediction label access count는 정확히 0이어야 한다.

## 정확한 길이 입력 규칙

행은 `(dataset, raw_length)`로 묶는다. padding, truncation, interpolation,
resampling, tail replacement, sibling 값 대체는 모두 금지한다.

- little-endian float32 값을 원래 identity와 길이 그대로 사용한다.
- 최소 raw length는 16이다.
- 같은 길이의 finite 정상 TRAIN 행이 최소 4개 있어야 한다.
- nonfinite 정상 TRAIN 행은 행 전체를 제외하고 dataset, instance, length,
  nonfinite 수·위치와 원본 values-blob SHA256을 audit에 남긴다.
- nonfinite TEST 행은 그 행만 `nonfinite_test`로 판단 보류한다.
- finite 정상 TRAIN이 4개 미만이면 `insufficient_finite_train_rows`로 판단
  보류한다.
- missing exact length, unsupported length, zero-scale channel, corrupt input,
  nonfinite output은 사전등록된 방식으로 판단 보류하거나 fail closed한다.
- 어떤 경우에도 값을 고치거나 만들어내지 않는다.

## 프로젝트 자체 주파수 분해

길이 `n`의 최대 depth는 다음과 같다.

```text
max_depth(n) = min(5, max(1, floor(log2(n)) - 3))
depths = 1, ..., max_depth(n)
```

depth `d`에서 real FFT를 계산한다.

```text
nyquist_bin = floor(n / 2)
cutoff_bin = max(1, floor(nyquist_bin / 2**d))
```

low는 FFT bin `0..cutoff_bin`을 포함해 남기고 그 위를 0으로 만든 뒤 길이 `n`의
inverse real FFT로 얻는다. high는 `original - low`다. 두 결과는 contiguous
float32로 고정한다. 모든 값은 finite이고
`max(abs(original - (low + high))) <= 1e-5`여야 한다.

이 분해는 프로젝트 자체의 투명한 FFT split이다. AFD 또는 MAFD라고 부르지
않으며 외부 AFD 구현을 복사·수정·import·vendor하지 않는다.

## 진폭 민감 순열 엔트로피(ASPE)

embedding dimension은 3, delay는 `(1, 2, 4)`로 고정한다.

1. 전체 입력을 `[0, 1]`로 min-max normalize한다.
2. 각 delay로 dimension-3 window를 만든다.
3. stable ascending sort로 6개 ordinal pattern 중 하나를 정한다. 같은 값은 원래
   위치 순서를 유지한다.
4. 각 window의 weight는 `sample_std(window) / mean(window)`다. `0 / 0` weight는
   0으로 처리한다.
5. pattern별 finite weight 합을 전체 finite weight로 나눈다.
6. `-sum(p * log2(p)) / log2(3!)`를 반환한다.

입력 peak-to-peak 또는 전체 finite weight가 `<=1e-12`이면 정확히 `0.0`과
`flat_or_zero_weight=true`를 기록한다. 그 밖의 nonfinite 중간값은 fail
closed한다. 계산은 마지막 feature vector를 contiguous float32로 만들기 전까지
float64다.

기준 출처는 `Brioal/ASPE` commit
`f17cf8d796c4d25bc1b5ace0e9a7ef7be2944a89`, `entropy/aspe.m::aspe`,
Apache-2.0이다. Python 구현은 출처와 상수 신호 확장을 명시하며 MATLAB bitwise
동일성이나 PhysDiff 전체 호환성을 주장하지 않는다.

## TRAIN-only depth 선택

각 TRAIN 행과 depth에서 high component의 delay 1, 2, 4 ASPE 중앙값을 `A[d]`로
둔다.

```text
처음으로 d >= 2이고 A[d] >= A[d - 1] - 1e-6이면 d - 1 선택
그런 d가 없으면 max_depth 선택
max_depth == 1이면 1 선택
```

최종 TEST 모델 depth는 finite TRAIN 행 preferred depth의 lower median이다.
TRAIN OOF calibration에서는 pseudo-test TRAIN 행을 완전히 빼고 depth, transform,
reference와 threshold를 다시 만든다. 자기 자신의 점수 결정에 자기 값이 영향을
주면 안 된다.

## 서로 합치지 않는 세 점수 채널

동결 depth에서 각 finite 행은 다음 세 채널을 만든다.

1. `high_frequency`: high waveform
2. `low_frequency`: low waveform
3. `amplitude_complexity`: `(original, low, high) x delay (1, 2, 4)`의 ASPE 9개

high/low waveform은 채널별 TRAIN matrix 전체에서 scalar median과
`1.4826 * MAD`를 사용한다. scale `<=1e-12`이면 unavailable이다. ASPE 9개 열은
각 열의 TRAIN median과 `max(1.4826 * MAD, 1e-6)`를 쓴다. flat flag는 audit
정보일 뿐 distance 차원에 넣지 않는다.

각 채널 score는 변환된 정상 TRAIN reference까지의 RMS distance 중 가장 작은
3개의 중앙값이다. TRAIN calibration은 leave-one-row-out이다. threshold는 finite
TRAIN OOF score의 `q=0.99`, `method='higher'`이고 TEST 이상 판정은 strict
`score > threshold`다.

세 abnormal set은 끝까지 따로 둔다. Exp438 내부에서 OR, AND, weighted fusion,
majority vote, best-channel selection, channel별 alpha 선택을 하지 않는다.
recovery candidate는 해당 채널에서 abnormal이면서 네 frozen lane 모두의
밖인 좌표다. 기존 lane은 바꾸지 않는다.

- 채널별 outside-lane budget: 10,000
- 세 채널 합산 budget: 30,000
- budget 초과: prediction 전체 실패
- top-N 절단이나 threshold 변경: 금지

## backend와 자원 규칙

정본은 NumPy CPU다. 공식 CPU worker는
`max(1, logical_cpu_count - 1)`이며 이 12-logical-CPU Mac에서는 11개다.
worker마다 numerical-library thread는 1개다.

- smoke seed: `43801`
- batch sizes: `(16, 32, 64, 128, 256)`
- warm-up: 2회
- paired timed runs: 5회
- score tolerance: `1e-5`
- minimum speedup: `1.1x`

Numba는 ASPE loop만, MLX/MPS는 frozen algorithm을 바꾸지 않는 batched FFT 또는
dense distance stage만 후보가 될 수 있다. 같은 입력·seed에서 finite output,
모든 사전등록 alpha의 candidate identity, score tolerance, transfer와 synchronize를
포함한 1.1x speed gate를 모두 통과해야 한다. Numba timing은 compile cost를
사전등록 smoke workload에 나누어 포함한다. 한 번에 accelerator-heavy process는
하나뿐이며 채택된 MLX/MPS와 outer process pool을 동시에 쓰지 않는다. 실패하면
CPU fallback이 정상 결과다. readiness가 실제 backend, version, device, timing,
parity, batch, 선택·fallback 이유를 기록하기 전에는 어떤 backend도 채택됐다고
기록하지 않는다.

## 실행·checkpoint·publication

CLI는 `readiness`, `new`, `resume`, `inspect`, `evaluate` 다섯 명령만 제공한다.
경로, worker, backend, seed, depth, threshold, alpha, lane, budget override는 없다.

`new`는 clean reviewed HEAD와 정확히 검증된 readiness artifact가 필요하다.
`resume`은 frozen identity/config/input/progress/checkpoint가 모두 맞는 interrupted
root에만 허용한다. failed root는 immutable하며 삭제·덮어쓰기·정리·resume하지
않는다.

프로세스가 PID, heartbeat, 현재 dataset, TRAIN/TEST 행 수·길이·크기, stage,
backend, completed/total, success/abstention/error, 처리속도, 경과시간, ETA,
checkpoint, 채널별 후보 수와 최신 오류를 JSON·JSONL·text log에 직접 쓴다.
외부 Codex polling은 필요하지 않다.

publication에는 정확히 1,117 unique completed datasets, missing/extra/failed 0,
inactive PID, `current_dataset=null`, budget 준수, prediction label access 0,
checkpoint/event/summary/prediction/manifest/artifact hash 일치가 필요하다.

## 동결 후 평가

모든 prediction hash가 동결된 뒤에만 descriptor-bound read-only SQLite connection
하나로 TEST `instance_index,label`을 읽는다. 세 채널은 따로 평가하고 Exp438
내부 winner를 고르지 않는다.

채널별로 full/outside-lane 후보 수, TP/FP/FN/TN, precision, recall,
specificity, FPR, F1, 새로 찾은 dataset, 아직 놓친 dataset, 회귀 dataset,
Exp404 Hard와의 counterfactual union, paired micro/macro F1 delta를 기록한다.
dataset-cluster bootstrap은 20,000회, seed `43820260807`, percentile
2.5%/97.5%다. 가능한 경우 Exp437 prototype·gradient·ACF TRAIN OOF score와
Spearman 및 aligned count, label-free candidate overlap도 기록한다.

두 판정은 별개다.

```text
direct_improvement_pass:
  outside-lane candidates <= 100
  rescued datasets >= 10
  added false positives <= 30
  paired micro and macro F1 point delta > 0
  both 95% CI lower bounds > 0

integration_evidence_pass:
  outside-lane true positives >= 25
  rescued datasets >= 10
  outside-lane precision >= 0.35
  each available Exp437 TRAIN OOF comparison has aligned >= 20
  and abs(Spearman) < 0.90
```

상관관계가 없거나 aligned 수가 20 미만이면 독립성 gate를 통과할 수 없다.
integration evidence 통과도 새 통합 실험을 논의·사전등록할 근거일 뿐 자동 통합,
lane 변경 또는 경보 승격이 아니다.

## 명시적 비목표

- PhysDiff diffusion reconstruction 재현 또는 호환성 주장
- dataset 이름이나 과거 family TEST 성능에 따른 routing
- Exp438 TEST 결과로 best channel 선택
- Exp437 또는 frozen lane과의 내부 fusion
- Exp437 파일·프로세스·checkpoint·prediction 수정
- Exp437 종료와 방향 논의 전 공식 Exp438 실행
- review-only evidence의 자동 경보 승격
