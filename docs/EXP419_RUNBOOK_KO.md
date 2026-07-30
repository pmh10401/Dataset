# Exp419 결정론적 CPU 병렬 실행 런북

## 범위와 안전 경계

Exp419는 Exp417의 과학적 계산을 바꾸지 않고 데이터셋 단위로 나누어 CPU
프로세스에서 병렬 실행한다. 정상 TRAIN으로 만든 계약, causal CNN, horizon,
seed, threshold, qualification, lane 제외와 workload gate는 Exp417 그대로다.
TEST의 값은 동결된 TRAIN 계약 뒤에만 기존 값 전용 경로로 읽으며, TEST 라벨,
실제 이상 위치, 과거 TEST 성능은 열거나 조회하지 않는다.

MLX와 MPS는 사용하지 않는다. worker 후보는 정확히 `1, 2, 4, 6`이고 각
프로세스의 PyTorch/BLAS thread는 1개로 고정된다. benchmark가 고른 worker
수만 공식 실행에 쓸 수 있으며 명령행에서 바꿀 수 없다.

자동 감시, polling loop, cron, keepalive를 만들지 않는다. 장시간 실행을
시작한 뒤 상태는 사용자가 요청한 시점에 한 번만 확인한다. checkpoint,
benchmark, prediction, metrics와 보고서는 모두 반도체 설비 데이터에서
파생된 로컬 전용 파일이며 GitHub, 이슈, PR, 클라우드에 올리지 않는다.

## 1. 환경과 버전 확인

```bash
cd "/Users/mac/Documents/Time Series/.worktrees/exp409-prospective-readiness"
git status --short --branch
git rev-parse HEAD
.venv/bin/python --version
.venv/bin/python -c 'import platform, numpy, numba, torch; print(platform.platform()); print("numpy", numpy.__version__); print("numba", numba.__version__); print("torch", torch.__version__)'
.venv/bin/python -c 'from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG; FROZEN_CONFIG.validate(); print(FROZEN_CONFIG)'
```

benchmark와 공식 실행은 `git status --porcelain`이 빈 clean commit에서만
허용된다. 실행 도중 HEAD 또는 입력 파일이 바뀌면 publication 전에
fail-stop한다. 두 CLI의 argument parser는 문자열을 `Path`로 바꾸는
문법 처리만 한다. 파일 존재·종류·symlink·root 관계는 parser에서 열거나
검사하지 않고 반드시 clean Git → label guard 뒤의 동결/검증 단계에서
확인한다.

## 2. 중단된 Exp417 근거는 그대로 보존

이전 Exp417 순차 실행은 사용자의 지시에 따라 `SIGTERM`으로 종료되어
exit code `143`을 남겼고, 다음 checkpoint에 정확히 94개의 검증된 dataset
record가 있다.

```text
/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow.checkpoint
```

이 94개는 중단 복원 근거일 뿐 Exp419 결과가 아니다. Exp419 checkpoint로
복사, 이동, 병합, 이름 변경하거나 Exp419 완료 수에 합산하지 않는다.
Exp419는 별도의 새 root와 provenance를 사용한다.

## 3. 입력 해시와 전체 테스트

아래 다섯 입력만 확인한다.

```bash
shasum -a 256 \
  "/Users/mac/Documents/Dataset/univariate_ts.db" \
  "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json"

.venv/bin/python -m unittest discover \
  -s experiments/exp419_deterministic_cpu_parallel/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests -p "test_*.py"
.venv/bin/python -m py_compile \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
git diff --check
```

2026-07-30 사용자 승인에 따라 정적 격리는 동결 소스 매니페스트 방식이다.
`SOURCE_MANIFEST.json`은 Exp419 package 최상위 runtime 모듈 8개와 두 CLI,
총 10개의 정확한 repo 상대 경로와 소문자 SHA256을 고정한다. clean Git
검사 직후 manifest와 각 소스를 no-follow file descriptor로 읽어
regular-file 여부, canonical sorted path, 중복·경로 탈출, 누락·추가 파일,
해시 불일치를 모두 차단한다. 그 뒤 defense-in-depth AST 검사가 명시적인
라벨 loader import/call/attribute, 동결 TEST-label artifact 이름,
`label`/`labels_blob` SQL projection을 차단한다. 일반 Python alias/dataflow를
추론하지 않으며, 동적 코드는 검토된 소스 바이트가 정확히 일치할 때만
허용된다. guard denylist literal만 검사 데이터로 제외하고 실행 guard
코드는 예외 없이 검사한다. manifest SHA256은 각 runtime trial과 benchmark
evidence에 포함되며 공식 verifier는 현재 manifest digest와 정확히
일치해야 benchmark를 승인한다. 제안 helper는 명시적으로 호출할 때만
개발용 bytes/file을 만들며 검증 중 자동 재생성하지 않는다.

## 4. 비공식 label-free 병렬 benchmark

Task 7 구현 중에는 이 benchmark를 실행하지 않는다. 모든 회귀 검증이 끝난
뒤 Task 8에서 다음 명령을 딱 한 번 실행한다. 출력 root는 존재하지 않는 새
로컬 경로여야 한다.

```bash
.venv/bin/python scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  --database "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --exp291-prediction "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  --exp291-manifest "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  --exp292-prediction "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  --exp292-manifest "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json" \
  --benchmark-root "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark"
```

생성 파일은 다음 하나다.

```text
/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json
```

검사는 다음과 같이 수행한다.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json")
value = json.loads(path.read_bytes())
benchmark = value["benchmark"]
measurements = value["runtime_measurements"]
print("git_commit", value["git_commit"])
print("worker_candidates", value["worker_candidates"])
print("status", benchmark["status"])
print("selected_worker_count", benchmark["selected_worker_count"])
for trial in benchmark["trials"]:
    print(trial["worker_count"], trial["elapsed_seconds"], trial["peak_rss_bytes"], trial["parity"], trial["finite"], trial["fallback"])
for measurement in measurements:
    print("runtime", measurement["worker_count"], measurement["process_cpu_seconds"], measurement["process_cpu_utilization"])
    for worker in measurement["worker_utilization"]:
        print("worker", worker["shard_index"], worker["pid"], worker["elapsed_seconds"], worker["wall_occupancy_fraction"])
PY
```

통과 조건은 후보가 정확히 `1, 2, 4, 6`, 모든 trial의 dataset·과학적 digest
parity가 exact, finite, fallback 없음, peak aggregate RSS가
`27,487,790,694` bytes 이하이고, 선택된 2/4/6 worker가 1-worker 대비
최소 `2.0x` 빠른 것이다. 조건을 만족하는 병렬 후보 중 가장 빠른 수만
선택된다. `no_qualified_parallel_runtime`이면 공식 실행 승인을 요청하지
않고 중단한다.

각 trial은 parent와 종료된 child들의 실제 user+system CPU time 증가분을
`process_cpu_seconds`로 기록한다. `process_cpu_utilization`은
`CPU seconds / (wall seconds × (parent 1 + worker count))`다. 각 worker의
`wall_occupancy_fraction`은 `worker elapsed / trial wall elapsed`다.
모든 값은 finite·nonnegative이고 이 수식과 정확히 일치해야 한다. telemetry
SHA256은 scientific evidence SHA256과 결합되어 trial evidence 및 전체
benchmark SHA256에 포함되므로 숫자를 따로 고칠 수 없다.

공식 검증기는 JSON 내부 일관성만 보지 않는다. 현재 frozen TRAIN metadata로
`select_benchmark_datasets`를 다시 실행해 dataset 이름과 순서가 benchmark와
정확히 같은지 확인한다. 임의의 더 쉬운 subset으로 모든 trial과 digest를
다시 만든 파일도 거부한다.

## 5. resume와 실패 처리

Exp419 공식 checkpoint root는 다음 경로만 사용한다.

```text
/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel.checkpoint
```

처음에는 새 경로여야 한다. 중단 뒤에는 Git commit, 다섯 입력, runtime
version, scientific/runtime config, worker 수, shard plan과 source digest가
전부 정확히 같을 때만 같은 명령으로 resume한다. 완료 record는 원래 shard
에서만 재사용하고 미완료 dataset을 다른 shard로 옮기지 않는다.

worker 하나라도 실패하거나 record가 누락·중복·손상되면 merge, render,
publication은 실행되지 않는다. 부분 worker root, runtime evidence와
canonical staging은 삭제하거나 수동 병합하지 말고 그대로 보존한다. 먼저
오류 원인과 provenance를 감사한 뒤 별도의 사용자 지시가 있을 때만 재개한다.
threshold, seed, horizon, budget, source 또는 worker 수를 바꿔 실패를
우회하지 않는다. output 경로가 이미 생겼다면 덮어쓰지 않고 중단한다.

## 6. 별도 명시적 승인 뒤의 공식 명령

아래 문장은 문서와 CLI help에만 있다. 환경 변수, 파일, 이전 대화 또는
benchmark 성공을 승인으로 추론하지 않는다. 전체 실행 직전에 사용자가
정확히 다음 문장을 새로 말해야 한다.

```text
Exp419 1,117개 전체 label-free 병렬 실행 승인
```

그 승인 뒤에만 다음 명령을 한 번 실행한다.

```bash
.venv/bin/python scratch/run_exp419_deterministic_cpu_parallel.py \
  --database "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --exp291-prediction "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  --exp291-manifest "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  --exp292-prediction "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  --exp292-manifest "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json" \
  --benchmark-evidence "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json" \
  --checkpoint-dir "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel.checkpoint" \
  --output-dir "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_official"
```

공식 순서는 정확히 clean Git → label guard → 입력 동결 → benchmark 검증
→ root 열기 → run/resume → 입력·Git·benchmark 재검증 → render → verify
→ atomic publish다. 어느 경계든 실패하면 이후 단계는 호출하지 않는다.
성공 시에도 결과는 Shadow-only 검토 근거이며 기존 lane을 자동 승격하지
않는다.
