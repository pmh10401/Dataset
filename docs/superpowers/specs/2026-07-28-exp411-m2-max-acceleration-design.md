# Exp411 M2 Max 12-Core 실행 가속 설계

작성일: 2026-07-28
상태: 사용자 승인 완료, 구현 전 동결
대상 브랜치: `codex/exp411-413-near-repeat-normal-validation`
대상 PR: `#16`

## 1. 목적

Exp411의 TRAIN-only near-repeat 자격 심사 의미와 정본 출력 계약을 바꾸지
않고, 현재 MacBook Pro의 Apple M2 Max 12코어와 32GB 메모리를 활용해 전체
1,117개 dataset 실행 시간을 줄인다.

현재 구현은 dataset을 순차 처리하면서 거리 계산도 사실상 단일 코어에서
실행한다. 또한 identity group이 20개 이상인 5-fold dataset에서 비슷한
pairwise 거리를 fold마다 다시 계산한다. 최대 동결 dataset인
`StarLightCurves_normal_3`은 정상 TRAIN run과 서로 다른 identity group이
각각 4,261개이고 길이는 1,024이므로, fold 중복과 단일 코어 계산이 전체
실행의 지배 병목이다.

가속은 실행 공학 변경이다. 거리 view, fold, quantile, gate, seed, Exp407
perturbation, TEST 접근 금지, 운영 lane 불변성과 최종 8-file publication
계약은 변경하지 않는다. 가속 결과가 수치·tag 계약을 만족하지 못하면
Exp411에서 채택하지 않고 NumPy 정본으로 닫는다.

## 2. 사용자 결정

- 거리 허용오차는 NumPy direct RMS 대비 `1e-6`이다.
- 성능·효율 코어를 포함한 12개 CPU core를 모두 사용한다.
- Numba 12-thread direct RMS, fold 거리 재사용, dataset checkpoint를
  결합한 접근을 채택한다.
- 실행 중 Mac의 반응성 저하와 높은 CPU 사용률을 허용한다.
- MLX는 이전 사용자 결정에 따라 구현·설치·benchmark 대상에서 제외한다.
- PyTorch MPS가 현재 unavailable이므로 canonical 실행에 사용하지 않는다.

## 3. 범위 밖

- TEST split, TEST label, `labels_blob`, 실제 이상 위치 접근
- 과거 family별 TEST 성능을 사용한 backend·threshold·source 선택
- 거리 view, quantile `0.01 / 0.05 / 0.10`, gate 또는 fold 변경
- tail replacement
- Hard 2,218 / Standard 806 / Priority 31 또는 exact-normal 35 변경
- Exp412 prediction 생성
- full 결과 artifact 또는 상세 로컬 metric의 GitHub 게시

## 4. 실행 구조

### 4.1 Dataset 순차·내부 병렬

dataset은 하나씩 SQLite에서 읽고 qualification 직후 waveform을 해제한다.
서로 다른 큰 dataset을 동시에 메모리에 올리지 않는다. 각 dataset 안의
거리 계산은 Numba `parallel=True` kernel이 12개 thread를 사용한다.

이 구조는 최대 dataset의 계산에 12코어를 집중하면서도 32GB 메모리를
넘는 process-level 중복 배열을 방지한다. nested BLAS/OpenMP thread는
1개로 제한해 Numba thread와의 oversubscription을 막는다.

### 4.2 Direct RMS kernel

Numba kernel은 pair마다 다음 값을 float64로 직접 계산한다.

```text
sqrt(sum_k ((query[k] - fit[k]) ** 2) / length)
```

Gram identity `||q||² + ||f||² - 2q·f`는 큰 공통 offset에서 상쇄 오차를
만들 수 있으므로 사용하지 않는다. pair 하나 안의 column 합산 순서는
고정하고, thread는 서로 다른 pair만 나눠 계산한다.

전체 `query × fit × length` 3차원 tensor를 만들지 않는다. full identity
pairwise 결과는 float64 2차원 거리 행렬로 저장한다. identity 4,261개인
경우 view 하나의 거리 행렬은 약 145MB이다.

### 4.3 View별 fold 재사용

`raw_rms`, `diff_rms`, `spectrum_rms` view는 하나씩 처리한다.

1. 전체 identity representative의 pairwise 거리 행렬을 한 번 계산한다.
2. 각 fold의 fit/query index로 self-excluded reference와 normal query
   minimum을 추출한다.
3. 해당 view의 fold threshold와 normal distance를 작은 배열로 보존한다.
4. 큰 pairwise 행렬을 해제한 뒤 다음 view로 이동한다.

따라서 5-fold가 동일 거리 계산을 반복하지 않는다. 세 view의 큰 행렬도
동시에 보유하지 않는다.

### 4.4 Synthetic query batching

Exp407 perturbation은 기존 seed·magnitude·순서 계약으로 query마다 생성한다.
같은 fold와 perturbation type의 synthetic query를 batch로 결합하고,
Numba kernel이 각 query에서 fit identity까지의 최소 거리만 반환한다.
3차원 delta tensor나 전체 synthetic pairwise 행렬은 저장하지 않는다.

### 4.5 Adaptive backend

작은 행렬에서는 Numba compile·parallel scheduling 비용이 NumPy보다 클 수
있다. TRAIN metadata로만 계산하는 고정 cost 기준 아래에서는 NumPy direct
RMS를 사용하고, 그 이상에서는 Numba parallel kernel을 사용한다.

cost 기준은 구현 후 고정 benchmark에서 사전등록하며 full 결과를 본 뒤
변경하지 않는다. 중·대형 실제 TRAIN benchmark에서 최소 2배 speedup을
보이지 못하면 Numba를 canonical 가속 backend로 채택하지 않는다.

## 5. 수치 안전성과 parity

### 5.1 시작 전 backend gate

full SQLite scan 전에 다음을 검증한다.

- Numba thread 수가 정확히 12
- Python, NumPy, Numba, llvmlite, platform과 thread 설정 기록
- NumPy direct RMS 대비 최대 절대오차 `<= 1e-6`
- representative smoke의 query identity/index별 tag SHA256 동일
- 선택 quantile 동일
- Numba cold/warm 전체 stage 시간 기록

실패하면 NumPy로 fallback하거나, 연구 계약 검증 자체가 불완전하면
publication 없이 중단한다.

### 5.2 Dataset별 parity audit

각 dataset의 identity SHA256으로 결정하는 고정 pair 표본을 NumPy와 Numba로
이중 계산한다. 표본 수와 선택 함수는 구현 전에 config에 고정한다.

- 최대 절대오차가 `1e-6`을 넘으면 해당 dataset은 NumPy로 전체 재계산한다.
- audit pair, 최대 오차, backend, fallback 이유를 checkpoint에 기록한다.
- 표본은 TRAIN identity만 사용하고 결과를 본 뒤 바꾸지 않는다.

### 5.3 Threshold 경계 보호

calibration quantile 또는 tag threshold에서 `1e-6` 이내인 거리값은 NumPy
direct RMS로 재확인한다. 재확인된 값을 최종 threshold와 tag에 사용한다.
이 과정은 수치 허용오차가 `near`와 `unfamiliar` 판정을 뒤집는 것을
방지한다.

가속과 NumPy의 최종 tag 또는 선택 quantile이 검증 구간에서 다르면 가속
결과를 채택하지 않는다. 거리 정의나 gate를 바꾸어 같은 Exp411에서 다시
선택하지 않는다.

## 6. Checkpoint와 재개

checkpoint root는 다음으로 고정한다.

```text
/Users/mac/Documents/Dataset/exp411_near_repeat_qualification.checkpoint
```

dataset 하나가 성공할 때마다 qualification row, backend audit, elapsed
time과 SHA256을 임시 sibling에 쓴 뒤 원자적으로 rename한다. 진행 상태는
`progress.json`에 완료 dataset 수와 현재 dataset만 기록한다.

checkpoint 재사용 전에 다음 provenance가 모두 같아야 한다.

- Git commit
- DB SHA256과 허용 snapshot key
- population manifest SHA256
- Exp411 frozen config SHA256
- Exp407 source/config contract SHA256
- acceleration code/backend contract SHA256
- dataset identity와 TRAIN blob identity hash

하나라도 다르면 기존 checkpoint를 자동 사용하지 않고 fail closed한다.
손상되거나 manifest가 없는 dataset checkpoint도 재사용하지 않는다.

오류가 난 dataset은 checkpoint를 쓰지 않으며 전체 실행은 즉시 중단한다.
재실행은 검증된 완료 dataset만 건너뛴다. checkpoint는 최종 연구 artifact가
아니며 GitHub에 게시하지 않는다.

## 7. 최종 publication

검증된 checkpoint가 동결 universe 1,117개와 family 128개를 정확히
포함할 때만 aggregate를 계산한다. 최종 정본 출력은 기존과 같은 8개다.

```text
exp411_frozen_config.json
exp411_dataset_qualification.csv
exp411_quantile_summary.csv
exp411_backend_assessment.json
exp411_selected_rule.json
exp411_summary.json
EXP411_RESULT_REPORT_KO.md
MANIFEST.json
```

최종 output은 기존 경로를 사용한다.

```text
/Users/mac/Documents/Dataset/exp411_near_repeat_qualification
```

Git clean commit, DB snapshot, config, checkpoint coverage와 artifact SHA256을
rename 직전에 다시 검증한다. 일부 결과 또는 불완전 coverage는 게시하지
않는다.

## 8. 실행 방식

Codex tool process lifetime에 종속되지 않는 macOS one-shot launch job을
사용한다. stdout/stderr는 `/private/tmp`의 명시적 로그로 보낸다. 실행 중
polling은 하지 않으며, 사용자가 결과 확인을 요청할 때만 progress,
completion, manifest를 읽는다.

launch job은 다음 환경을 명시한다.

- `NUMBA_NUM_THREADS=12`
- nested BLAS/OpenMP thread `1`
- 저장소 전용 `.venv` Python
- 고정 worktree와 full runner 절대경로
- 고정 DB, population manifest, checkpoint, output 절대경로

job 종료와 checkpoint는 별개다. OS 종료나 session 중단이 있어도 다음
one-shot job이 checkpoint를 검증한 뒤 재개한다.

## 9. 테스트와 benchmark

### 9.1 단위 테스트

- large offset와 small delta에서 direct RMS parity
- NumPy/Numba 최대 오차 `<= 1e-6`
- exact/near/unfamiliar/abstain tag identity
- quantile identity
- 12-thread 설정과 nested-thread 제한
- view당 pairwise matrix 한 번 계산
- 5-fold 거리 재사용
- synthetic batch와 기존 per-query 결과 parity
- threshold 경계 NumPy 재확인

### 9.2 Checkpoint 테스트

- dataset checkpoint 원자적 write
- 정상 resume와 완료 dataset skip
- commit/config/DB/population/backend hash drift 거부
- 손상·누락·중복 checkpoint 거부
- 1,117 coverage 이전 publication 거부
- 실행 오류 후 기존 checkpoint 보존

### 9.3 실제 TRAIN benchmark

TEST를 읽지 않고 다음을 고정 benchmark한다.

- 소형: `ACSF1_normal_0`
- 중형: `HandOutlines_normal_0`
- 대형: `StarLightCurves_normal_3`

각 benchmark는 NumPy baseline, Numba cold, Numba warm, 최대 오차, tag
identity, selected quantile identity, peak RSS와 thread 설정을 기록한다.
중·대형 모두 warm end-to-end speedup이 2배 이상이어야 Numba를 canonical
가속 backend로 채택한다.

### 9.4 회귀 검증

- Exp411 전체 단위 테스트
- Exp407 synthetic anomaly 테스트
- Exp410 회귀 테스트
- representative 32-dataset smoke
- compileall과 `git diff --check`
- 독립 code review

## 10. 예상 효과와 한계

5-fold 거리 재사용만으로 대형 dataset의 중복 pairwise 계산을 구조적으로
줄인다. Numba 12-thread는 나머지 pair 연산을 모든 CPU core에 분배한다.
정확한 전체 speedup은 구현 후 실제 TRAIN benchmark로만 확정한다.

Apple M2 Max에서 12코어를 모두 사용하므로 실행 중 다른 작업의 반응성이
크게 떨어지고 발열·전력 사용이 증가할 수 있다. MPS는 현재 unavailable이며
MLX는 제외되므로 GPU 가속은 이 설계의 일부가 아니다.

## 11. 완료 조건

- 사용자 승인 설계와 구현 계획이 문서화됨
- 모든 테스트와 independent review 통과
- 중·대형 TRAIN benchmark speedup `>=2.0`
- NumPy 대비 audit distance 오차 `<=1e-6`
- representative tag와 selected quantile identity
- checkpoint resume와 fail-closed publication 검증
- PR #16에 코드·문서·benchmark 요약 push
- 사용자의 실행 승인 후 one-shot full job 시작
