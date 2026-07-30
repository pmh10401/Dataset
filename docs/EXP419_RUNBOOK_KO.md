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

첫 비공개 Exp419 시도의 checkpoint-universe 결함은 검토된 commit `f54ac4d`와
`b706393`에서 수정되었다. 균형 benchmark 정정 설계는 사용자 승인 commit
`4ea4334`에 동결되었다. 중단된 시도의 status는 `143`이며 유효 benchmark JSON,
자격을 얻은 trial, 공식 checkpoint/output은 없다. 다음 기존 root와 모든 하위
경로는 진단 전용 불변 증거로 보존하고 승인, 재개, 병합 또는 재사용하지 않는다.

```text
/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark
```

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
  experiments/exp419_deterministic_cpu_parallel/*.py \
  scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  scratch/run_exp419_deterministic_cpu_parallel.py
.venv/bin/python - <<'PY'
from scratch.benchmark_exp419_deterministic_cpu_parallel import (
    assert_exp419_label_isolation,
)
print(assert_exp419_label_isolation())
PY
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

Guard literal 예외는 알려진 source file의 module-level denylist assignment로
한정한다. 값은 승인된 set/tuple/list/dict literal과 정적 문자열 leaf만
가능하며 Call, Attribute, comprehension, 문자열 결합, f-string은 거부한다.
Writer와 공식 verifier는 같은 binding validator를 사용하여 각 runtime
measurement의 manifest SHA256, trial evidence, benchmark digest/selection과
top-level/current manifest SHA256을 함께 재검증한다.

## 4. 정정된 비공식 label-free 병렬 benchmark

원래 15개 anchor는 총 대리비용 `388,627,551` 중
`StarLightCurves_normal_3` 하나가 `353,424,384`를 차지하여 2/4/6-worker
largest-first 대리 speedup 상한이 `1.0996059372x`였다. 승인된 정정은 원래
15개를 모두 유지하고, 남은 정상 TRAIN 비용을 `(-cost, dataset)` 순으로
하나씩 추가하여 4-worker와 6-worker의
`총 비용 / 최대 shard 비용`이 모두 `2.5x` 이상인 첫 prefix에서 멈춘다.
timing은 선택에 사용하지 않는다.

동결 결과는 원래 15개와 `StarLightCurves_normal_2` (`171,196,416`),
`HandOutlines_normal_1` (`153,600,300`),
`StarLightCurves_normal_1` (`88,169,472`),
`HandOutlines_normal_0` (`86,893,884`)를 합친 정확히 19개다. 총 비용은
`888,487,623`이고 proxy headroom은 4-worker와 6-worker 모두
`2.5139397937x`다. proxy gate `2.5x`는 실행 가능성 여유이며 실제 측정
speedup gate는 변경 없이 `2.0x`다.

모든 회귀 검증이 끝난 뒤 다음 명령을 딱 한 번 실행한다. 출력 root는
존재하지 않는 다음의 새 정확한 로컬 경로여야 한다. 자동 polling,
monitor, cron 또는 keepalive를 만들지 않는다.

```bash
.venv/bin/python scratch/benchmark_exp419_deterministic_cpu_parallel.py \
  --database "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --exp291-prediction "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  --exp291-manifest "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  --exp292-prediction "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  --exp292-manifest "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json" \
  --benchmark-root "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced"
```

생성 파일은 다음 하나다.

```text
/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json
```

검사는 다음과 같이 수행한다.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path("/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json")
value = json.loads(path.read_bytes())
benchmark = value["benchmark"]
measurements = value["runtime_measurements"]
proof = benchmark["selection_proof"]
print("git_commit", value["git_commit"])
print("source_manifest_sha256", value["source_manifest_sha256"])
print("worker_candidates", value["worker_candidates"])
print("status", benchmark["status"])
print("selected_worker_count", benchmark["selected_worker_count"])
print("benchmark_datasets", benchmark["benchmark_datasets"])
print("anchor_datasets", proof["anchor_datasets"])
print("added_datasets", proof["added_datasets"])
print("proxy_headroom_gate", proof["proxy_headroom_gate"])
print("total_cost", proof["total_cost"])
print("shard_loads", proof["shard_loads"])
print("proxy_headrooms", proof["proxy_headrooms"])
print("selection_proof_sha256", proof["sha256"])
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
`27,487,790,694` bytes 이하인 것이다. 2-worker trial은 필수 비교 증거지만
공식 선택 후보가 아니다. 4-worker 또는 6-worker가 1-worker 대비 최소
`2.0x` 빠른 것을 포함한 모든 gate를 통과해야 하며, 두 eligible 후보 중
조건을 만족하는 가장 빠른 수만 선택된다. 2-worker만 `2.0x`를 통과하거나
4/6-worker가 모두 실패하면 `no_qualified_parallel_runtime`이며 공식 실행
승인을 요청하지 않고 중단한다.

추가로 `benchmark_datasets`는 proof의 lexical `selected_datasets`와 같은
정확한 19개여야 한다. proof는 원래 15개 anchor, 위 네 addition, gate
`2.5`, total `888,487,623`과 다음 exact load/headroom을 기록해야 한다.

- 2-worker: `(444237111, 444250512)`, `1.9999698346x`
- 4-worker: `(353424384, 178469604, 178343436, 178250199)`,
  `2.5139397937x`
- 6-worker: `(353424384, 171196416, 153600300, 88169472, 86893884,
  35203167)`, `2.5139397937x`

각 trial은 parent와 종료된 child들의 실제 user+system CPU time 증가분을
`process_cpu_seconds`로 기록한다. `process_cpu_utilization`은
`CPU seconds / (wall seconds × (parent 1 + worker count))`다. 각 worker의
`wall_occupancy_fraction`은 `worker elapsed / trial wall elapsed`다.
모든 값은 finite·nonnegative이고 이 수식과 정확히 일치해야 한다. telemetry
SHA256은 scientific evidence SHA256과 결합되어 trial evidence 및 전체
benchmark SHA256에 포함되므로 숫자를 따로 고칠 수 없다.

공식 검증기는 JSON 내부 일관성만 보지 않는다. 현재 frozen TRAIN metadata로
`select_balanced_benchmark_datasets`를 다시 실행해 selection proof, dataset
이름과 순서가 benchmark와 정확히 같은지 확인한다. 임의의 더 쉬운 subset으로
모든 trial과 digest를 다시 만든 파일도 거부한다. 기존 진단 root 또는 그
하위 경로를 가리키는 `trial_roots`도 공식 root를 열기 전에 거부한다.

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

아래 문장은 문서와 CLI help에만 있다. 환경 변수, 파일, 이전 대화, 이전의
포괄 승인 또는 benchmark 성공을 승인으로 추론하지 않는다. 정정 benchmark가
모든 gate를 통과한 뒤 전체 실행 직전에 사용자가 정확히 다음 문장을 새로
말해야 한다.

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
  --benchmark-evidence "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced/EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json" \
  --checkpoint-dir "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel.checkpoint" \
  --output-dir "/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_official"
```

공식 순서는 정확히 clean Git → label guard → 입력 동결 → benchmark 검증
→ root 열기 → run/resume → 입력·Git·benchmark 재검증 → render → verify
→ atomic publish다. 어느 경계든 실패하면 이후 단계는 호출하지 않는다.
성공 시에도 결과는 Shadow-only 검토 근거이며 기존 lane을 자동 승격하지
않는다.
