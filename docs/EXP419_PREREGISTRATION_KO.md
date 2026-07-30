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
중복 선택은 replacement 없이 제거한다. 선택된 정렬 데이터셋 이름과 identity를 timing
전에 벤치마크 manifest에 기록하며, 모든 worker 수에서 동일 subset, 입력 및 seed를
사용한다.

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

Exp419 패키지는 동결된 TEST-label artifact를 import, open, hash, stat, query 또는 name할
수 없다. Exp420은 성공적으로 동결된 Exp419 prediction에 대한 retrospective TEST-label
감사로 예약한다. Exp418은 원 Exp417 계약에서 prediction이 완성될 경우를 위해 계속
예약한다.

## 전체 실행 승인

모든 테스트, parity 검사와 worker-count 벤치마크가 통과한 뒤에만 1,117개 전체 실행을
요청할 수 있다. 필요한 정확한 사용자 승인 문구는 다음과 같다.

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
