# Exp419 결정론적 CPU 병렬 런타임 사전등록

## 목적과 상속 범위

Exp419는 동결된 Exp417 인과적 다중-수평 Shadow 워크플로를 독립 데이터셋 단위의
CPU 프로세스로 실행하는 실행-동등성 실험이다. 새 점수 원천이나 새 과학적 설정이
아니다. Exp419는 Exp417을 재조정하지 않고 그대로 상속한다. 즉, 동일한 1,117개
모집단과 원본 데이터베이스, 정상 TRAIN 전용 전처리, identity-grouped fold, 인과 CNN,
수평선, 손실, 시드, 조기 종료, OOF 보정, 합성 자격, exact-normal guard, frozen-lane
제외 및 workload gate를 사용한다. Hard, Standard, Priority, automatic-normal lane은
변경하지 않는다.

Exp419는 TEST label, 이상 위치, 과거 TEST/family 성능 또는 사람 검토 결과를
읽거나 사용하지 않는다. 중단된 Exp417 순차 실행과 검증된 94개 체크포인트는 지역의
변경 불가 증거일 뿐이며 Exp419가 가져오거나, 고치거나, Exp419 결과라고 주장하지
않는다.

Exp419는 Exp417 동결 구성과 정확히 결합한다. 데이터베이스 SHA-256은
`8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`이며, 동결된
lane 개수는 Hard 2,218개, Standard 806개, Priority 31개, automatic-normal 35개다.
lane identity는 Exp291 prediction
`73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848`, Exp291
manifest `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e`, Exp292
prediction `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b`, Exp292
manifest `0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0`으로 고정한다.

## 2026-07-30 공식 실행 전 사용자 판정

첫 비공개 Exp419 시도는 유효한 timing benchmark 또는 공식 산출물을 만들기 전에
두 개발 결함을 드러냈다. checkpoint universe 경계는 검토된 commit `f54ac4d`와
`b706393`에서 수정되었다. 사용자는 commit `4ea4334`의 정상 TRAIN 전용 균형
benchmark 설계를 승인했다. 이 판정은 과학 계산이나 실제 통과 gate를 바꾸지 않고
원래 benchmark anchor의 병렬 실행 가능성만 바로잡는다.

중단된 시도의 wrapper status는 `143`이고 benchmark JSON, 자격을 얻은 worker trial,
공식 checkpoint/output root는 존재하지 않는다. 다음 root와 그 하위 경로는 진단 전용
불변 증거로 남기며 승인, 재개, 병합 또는 재사용하지 않는다.

`/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark`

정정 benchmark는 다음의 새로운 정확한 no-clobber root만 사용한다.

`/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced`

## 고정된 런타임 계약

- 실험 식별자: `Exp419`; 기대 모집단: 1,117개 데이터셋.
- 후보 worker 수는 정확히 `1`, `2`, `4`, `6`이며 시작 방식은 macOS `spawn`이다.
- 모든 프로세스는 `torch.set_num_threads(1)`,
  `torch.set_num_interop_threads(1)`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`을 사용한다.
- MLX는 사용자의 명시적 결정으로 제외한다. MPS도 Exp417 벤치마크에서 CPU 대비
  `0.6709759390840965x`로 1.1x 채택 기준에 미달했으므로 제외한다. Numba는 이미
  감사된 결정론적 점수 집계 경로에만 제한한다.
- 배치 추론과 데이터셋 내부 시드/fold 병렬화는 포함하지 않는다.

부모 프로세스만 실행 상태와 정식 병합을 소유한다. worker는 모델 상태, 난수 생성기,
SQLite 연결 또는 체크포인트 root를 공유하지 않고, 자신에게 배정된 데이터셋을 기록된
순서로만 처리한다. 부모는 worker 체크포인트를 신뢰하지 않는 입력으로 취급하고,
정확한 shard, provenance, 완전한 증거, canonical payload digest 및 중복/누락/추가
레코드를 검증한 뒤에만 모집단의 canonical 순서로 병합한다.

## TRAIN 전용 비용과 shard 계획

정확한 길이 그룹에서 정상 series 수가 `n`, 길이가 `L`일 때 정수 비용은 `n * L * w`다.
가중치 `w`는 `n < 50`이면 1, `50 <= n < 100`이면 30, `n >= 100`이면 81이다.
데이터셋 비용은 모든 그룹 비용의 합이다. 이 값은 각각 전처리 전용, OOF 선택/재학습,
OOF/최종/합성 작업의 대리비용일 뿐 모델 계산을 바꾸지 않는다.

데이터셋은 추정 비용 내림차순, 데이터셋 이름 오름차순으로 정렬한다. 현재 가장 가벼운
shard에 배정하며 동률은 낮은 shard index가 우선이다. 전체 shard 계획은 worker 시작
전에 hash로 고정한다. 이 추정 및 계획에는 TEST label, 이상 위치, TEST 점수, 과거
family 성능, 사람 검토를 사용하지 않는다.

## 벤치마크와 선택 gate

TRAIN 전용 모집단은 다음 세 strata로 나눈다.

1. `preprocessing_only`: 50개 이상 정상 series인 정확한 길이 그룹이 없다.
2. `diagnostic_oof`: 50--99개 정상 series인 그룹이 하나 이상이고 100개 이상인 그룹은 없다.
3. `candidate_eligible`: 100개 이상 정상 series인 그룹이 하나 이상이다.

비어 있지 않은 각 stratum에서 데이터셋을 `(cost, dataset)`으로 정렬하고, quantile
`0`, `0.25`, `0.50`, `0.75`, `1.0`마다 `floor(q * (n - 1))`의 레코드를 선택한다.
중복 선택은 replacement 없이 제거한다. 이 결과가 변경하지 않는 원래 anchor다.

원래 15개 anchor의 총 대리비용은 `388,627,551`이다.
`StarLightCurves_normal_3`의 비용 `353,424,384`가 전체의 90.94%를 차지하여,
2/4/6 worker largest-first 대리 speedup 상한은 `1.0996059372x`에 불과하다.
정정 규칙은 anchor를 제거하거나 timing을 보고 조정하지 않는다. 남은 정상 TRAIN 전용
비용 레코드를 `(-cost, dataset)`으로 정렬하고 한 번에 정확히 하나씩 추가한다.
4-worker와 6-worker 계획 모두에서
`sum(selected costs) / max(shard total cost) >= 2.5`가 처음 성립하는 prefix에서
멈춘다.

동결 입력에서 정확히 19개가 선택된다. 원래 15개에 다음 네 데이터셋을 추가한다.

1. `StarLightCurves_normal_2`: `171,196,416`
2. `HandOutlines_normal_1`: `153,600,300`
3. `StarLightCurves_normal_1`: `88,169,472`
4. `HandOutlines_normal_0`: `86,893,884`

총 비용은 `888,487,623`이고 정확한 대리 증거는 다음과 같다.

- 2-worker load `(444,237,111, 444,250,512)`, headroom `1.9999698346x`
- 4-worker load `(353,424,384, 178,469,604, 178,343,436, 178,250,199)`,
  headroom `2.5139397937x`
- 6-worker load `(353,424,384, 171,196,416, 153,600,300, 88,169,472,
  86,893,884, 35,203,167)`, headroom `2.5139397937x`

anchor/addition/전체 dataset 이름, proxy gate `2.5`, 총 비용, 정확한 2/4/6-worker
load·headroom과 canonical proof SHA-256을 timing 전에 고정한다. 모든 worker 수에서
동일한 19개 subset, 입력 및 seed를 사용한다. timing 결과로 dataset을 추가, 제거,
재정렬 또는 선택하지 않는다. `2.5x`는 실행 가능성 여유일 뿐이며 실제 wall-clock
통과 gate `2.0x`는 바꾸지 않는다.

선택 worker 수는 다음을 모두 만족해야 한다.

1. 출력이 finite이고 설명되지 않은 worker fallback이 없다.
2. 데이터셋 상태와 candidate-coordinate identity가 정확히 같다.
3. canonical model hash가 정확히 같다.
4. score-calibration 및 TRAIN-contract hash가 정확히 같다.
5. qualification 및 lane-exclusion 결정이 정확히 같다.
6. 누락, 중복, 추가 데이터셋이 없다.
7. 총 peak resident memory가 32-GiB host의 80% 이하, 정확히 `27,487,790,694` bytes다.
8. 동일 벤치마크 subset에서 1-worker Exp419 기준보다 wall-clock speedup이 최소 `2.0x`다.

통과한 후보 중 가장 빠른 worker 수를 선택한다. 1개 초과 worker 후보가 통과하지
않으면 `no_qualified_parallel_runtime`을 기록하며 전체 모집단 실행은 승인되지 않는다.
2-worker는 proxy headroom `2.5x` 대상이 아니지만 필수 benchmark 증거로 남는다.
공식 자격은 변경하지 않은 4-worker 또는 6-worker 후보 하나 이상이 모든 실제 gate를
통과해야 한다.
선택 worker 수와 thread 환경은 전체 실행 전에 고정하며, 런타임 또는 unlabeled candidate
수를 본 뒤 parity 조건, worker 수 또는 backend를 바꿀 수 없다.

## 재개, 격리 및 공개 금지

재개에는 Git commit과 모든 source digest, 데이터베이스/모집단/lane identity, Exp417
과학 구성, Exp419 런타임 구성 및 shard-plan digest, Python/NumPy/Numba/PyTorch/platform
버전, worker 수/start method/thread 환경, 모델/전처리/seed/합성/점수/병합 계약의 정확한
동일성이 필요하다. 완료 레코드는 원래 shard에서만 재사용하며 미완료 데이터를 같은 정식
실행 중 다른 shard로 옮기지 않는다.

worker 실패, 데이터셋 terminal failure, corrupt checkpoint, timeout은 모두 fail-closed다.
부분 worker/canonical 체크포인트와 private dataset에서 나온 prediction, metric, benchmark,
데이터셋 수준 report는 지역에만 남기며 GitHub에 내보내지 않는다. GitHub에는 코드,
사전등록, runbook, synthetic test와 synthetic benchmark 증거만 포함할 수 있다.
자동 polling, monitor, cron 또는 keepalive는 만들지 않는다. 장시간 실행을 시작한 뒤에는
사용자가 요청한 시점에만 상태를 한 번 확인한다.

Exp419 패키지는 동결된 TEST-label artifact를 import, open, hash, stat, query 또는 name할
수 없다. Exp420은 성공적으로 동결된 Exp419 prediction에 대한 retrospective TEST-label
감사로 예약한다. Exp418은 원 Exp417 계약에서 prediction이 완성될 경우를 위해 계속
예약한다.

## 전체 실행 승인

모든 테스트, parity 검사와 정정된 worker-count 벤치마크가 통과한 뒤에만 1,117개 전체
실행을 요청할 수 있다. 이전의 포괄 승인, 환경 변수, 파일 또는 benchmark 성공을
승인으로 추론하지 않는다. 그 시점에 사용자가 새로 말해야 하는 정확한 문구는 다음과
같다.

`Exp419 1,117개 전체 label-free 병렬 실행 승인`

## 정식 산출물

모든 데이터셋 레코드와 global workload gate가 통과한 뒤, 새 no-clobber 디렉터리에 다음
10개 파일만 atomic하게 공개한다.

1. `exp419_shadow_predictions.jsonl`
2. `exp419_dataset_qualification.csv`
3. `exp419_oof_normal_metrics.csv`
4. `exp419_synthetic_safety.csv`
5. `exp419_backend_benchmark.json`
6. `exp419_parallel_runtime_benchmark.json`
7. `exp419_frozen_config.json`
8. `exp419_shard_plan.json`
9. `EXP419_RESULT_REPORT_KO.md`
10. `MANIFEST.json`

Prediction row는 Exp417의 label-free schema를 유지하며 TP, FP, FN, TN, precision, recall,
F1, anomaly label/location 또는 human-reviewed result를 포함하지 않는다. 런타임 metadata는
benchmark, shard-plan, report, manifest 산출물에만 제한한다.
