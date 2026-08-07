# Exp439 크기 대응 FFT 복원 검증 사전등록

## 질문

Exp439는 새로운 탐지 아이디어를 시험하지 않는다. Exp438의 유효한 과학 규칙을
그대로 유지하면서, 큰 유한값에서 발생한 정상적인 float32 반올림을 고정 절대
오차 기준이 잘못 거부한 실행 문제만 고친다.

## 동결 신원

```text
Experiment: Exp439
Run identity: exp439-scale-aware-reconstruction-repair-antigravity-20260807
Branch: codex/exp439-scale-aware-reconstruction-antigravity
Run parent: /Users/mac/Documents/Dataset/exp439_scale_aware_reconstruction_repair_antigravity_20260807
Dataset population: 1,117 sorted unique datasets
Review only: true
Operational alert eligible: false
Prediction TEST-label access: exactly zero
CPU workers: max(logical CPUs - 1) = 11
```

## 실패한 선행 실험

Exp438은 reviewed commit
`d1da286e9c74c64abe76e0c255c86c30141813bd`에서 89/1,117개를 commit한 뒤
`Chinatown_normal_1` TRAIN 분해 중 다음 오류로 종료됐다.

```text
ValueError: decomposition reconstruction tolerance failed
```

해당 TRAIN 값은 모두 finite이며 결측치가 아니다. 값의 절댓값이 약 1,700까지
커서 float32 component 반올림 오차가 최대 `6.103515625e-05`였지만 Exp438의
고정 기준은 `1e-5`였다. Exp438 root, readiness, checkpoint, output과 89개
record는 실패 증거로 수정·삭제·재개하지 않는다.

## 정보 방화벽

Prediction에서 읽을 수 있는 정보는 정상 TRAIN raw 값, TEST raw 값과 identity,
raw length, 동결 lane, 동결 source manifest, 검증된 Exp438 선행 record뿐이다.
TEST label, 실제 이상 위치, 과거 family별 TEST 성능과 사후 평가는 prediction
hash가 동결되기 전에 읽지 않는다. TEST 결과를 보고 depth, feature, score,
threshold, budget, route 또는 backend를 다시 선택하지 않는다.

## 결측치 정책

- TRAIN 행에 NaN, positive infinity 또는 negative infinity가 하나라도 있으면
  그 행 전체를 학습에서 제외한다.
- finite sibling TRAIN 행은 계속 사용한다.
- 값을 채우거나 보간하지 않는다.
- 같은 정확한 길이의 finite TRAIN 행이 4개 미만이면 해당 group을 unavailable로
  처리한다.
- nonfinite TEST 행은 해당 행만 `nonfinite_test`로 판단 보류한다.

## 동결된 분해와 단 하나의 수정

FFT bin, cutoff, inverse FFT, high 계산과 float32 cast는 Exp438과 같다.

```text
nyquist_bin = floor(n / 2)
cutoff_bin = max(1, floor(nyquist_bin / 2**depth))
low = irfft(FFT bins 0..cutoff_bin)
high = original - low
stored low/high = contiguous float32
```

복원 검증만 다음으로 바꾼다.

```text
peak = max(1, max(abs(original)), max(abs(low)), max(abs(high)))
one_ulp = spacing(float32(peak))
tolerance = max(1e-5, one_ulp)
error = max(abs(original - (low + high)))
accept iff finite(error) and error <= tolerance
```

작은 정규화 신호에는 기존 `1e-5`가 유지된다. 큰 신호에는 float32 한 칸만
허용한다. 한 칸보다 큰 손상은 계속 fail closed한다.

## 바뀌지 않는 과학 규칙

- 최소 raw length 16, 최소 finite TRAIN 행 4
- 최대 depth `min(5, max(1, floor(log2(n)) - 3))`
- ASPE embedding dimension 3, delays `(1, 2, 4)`
- nearest-normal `k=3`
- TRAIN OOF threshold `q=0.99`, method `higher`
- TEST 판정 `score > threshold`
- high-frequency, low-frequency, amplitude-complexity 세 채널 분리
- 채널별 candidate budget 10,000, 합계 30,000
- frozen lane 밖의 후보만 review evidence
- 내부 fusion, best-channel 선택, top-N 절단 금지

## Exp438 89개 record 가져오기

가져오기는 전부 또는 0개다. 다음이 모두 일치할 때만 89개를 순서대로 가져온다.

- Exp438 reviewed commit, readiness, context와 database/source SHA
- failed status와 exact `ValueError`
- completed record 수 89, 마지막 dataset `BeetleFly_normal_1`
- dataset order prefix와 모든 record SHA256
- prediction label access 0
- old `1e-5` guard를 통과한 입력에서 Exp438/Exp439 component·score·candidate
  bytes가 동일하다는 테스트

한 조건이라도 다르면 부분 가져오기 없이 dataset 0부터 다시 계산한다. 가져온
source path와 SHA256은 `import_manifest_antigravity.json`에 기록한다.

## backend와 실행

정본은 NumPy CPU다. Numba, MLX, MPS는 같은 입력과 seed에서 finite output,
candidate identity, 사전등록 tolerance와 end-to-end 1.1x speedup을 모두 통과할
때만 채택한다. 실패하면 CPU fallback이 정상이다. 한 번에 MLX/MPS-heavy
process는 하나뿐이다.

## 실행 최적화 허용 범위

동일 TRAIN 행의 preferred depth와 동일 depth decomposition feature는 캐시해
leave-one-out fold에서 재사용할 수 있다. 단, 각 fold는 held-out 행을 제외한
lower-median depth와 robust transform을 그대로 계산해야 하며 최종 score,
threshold, 후보, lane, budget은 기존 계산과 byte parity를 만족해야 한다.

CPU scheduler는 frozen commit 순서를 유지하면서 느린 head 뒤에서 끝난 결과를
제한된 reorder window에 보관하고 다음 작업을 dispatch할 수 있다. 이 변경은
dataset order, prediction 내용, TEST-label boundary를 바꾸지 않는다.

## checkpoint와 로그

프로세스가 PID, current dataset, TRAIN/TEST 크기와 길이, stage, backend,
completed/total, import/recompute/success/abstention/error, 처리속도, 경과시간,
ETA, checkpoint와 latest error를 atomic JSON, JSONL, text log에 직접 쓴다.
worker 오류에는 dataset 이름, 원래 exception type과 message가 반드시 남는다.

Failed root는 resume하지 않는다. Interrupted root만 exact identity와 ordered
prefix가 일치할 때 resume한다.

## 동결 후 평가

1,117개 prediction과 publication hash가 동결된 뒤에만 explicit `evaluate`가 TEST
label을 읽는다. 세 채널별 confusion, F1, false alarm, 새로 찾은 dataset, 기존
강한 dataset 악화, paired micro/macro delta와 dataset bootstrap CI를 보고한다.
평가 결과로 Exp439 안의 parameter를 바꾸지 않는다.
