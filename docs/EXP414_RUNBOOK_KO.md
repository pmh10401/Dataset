# Exp414 실행 런북

## 범위와 승인 경계

Exp414는 고정된 normal TRAIN 데이터만 사용해 결측치를 결정적으로
복구한 뒤 Exp411 near-repeat qualification을 그대로 실행한다. TEST 값,
TEST 라벨, 실제 이상 위치, 과거 TEST 성능은 어떤 단계에서도 사용하지
않는다. 전체 1,117개 데이터셋 실행은 아래 사전 검증과 별도의 사용자
승인을 모두 통과한 뒤에만 시작한다.

자동 모니터링, 반복 폴링, keepalive 재시작은 사용하지 않는다. 제출한
launchd 작업은 한 번만 실행되며 종료 시 자기 label만 제거한다.

## 1. 환경 및 소스 확인

```bash
cd "/Users/mac/Documents/Time Series/.worktrees/exp409-prospective-readiness"
git status --short --branch
git rev-parse HEAD
.venv/bin/python --version
.venv/bin/python -c 'import numpy, numba, llvmlite; print(numpy.__version__, numba.__version__, llvmlite.__version__)'
```

실제 benchmark와 전체 실행 전에는 worktree가 반드시 clean이어야 한다.
Numba 외부 수치 라이브러리 thread는 1, Numba thread는 12로 고정한다.
Exp414는 frozen 알고리즘을 바꾸지 않는 Exp411 CPU/Numba 경로를
사용하며 MLX, PyTorch, MPS는 사용하지 않는다.

## 2. 전체 단위 테스트와 컴파일 검사

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests -p "test_*.py" -v
.venv/bin/python -m unittest discover \
  -s experiments/exp411_near_repeat_qualification/tests -p "test_*.py" -v
.venv/bin/python -m compileall -q \
  experiments/exp414_missing_aware_near_repeat \
  scratch/run_exp414_missing_aware_near_repeat.py
bash -n scratch/launch_exp414_full.sh
bash -n scratch/run_exp414_launchd_once.sh
git diff --check
```

하나라도 실패하면 benchmark나 전체 실행으로 진행하지 않는다.

## 3. 읽기 전용 입력 audit

```bash
.venv/bin/python scratch/run_exp414_missing_aware_near_repeat.py \
  --input-audit \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --population-manifest "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json"
```

출력은 canonical JSON 한 줄이어야 한다. 정확한 값은 전체
`1117`, 결측 데이터셋 `19`, 결측 run `164`, non-finite 값 `983`,
all-missing run `0`, affected digest
`ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9`
이다. audit 모드는 파일을 쓰지 않는다.

## 4. 대표 smoke와 manifest 검증

매번 존재하지 않는 출력 경로를 새 임시 parent 아래에 만든다.

```bash
exp414_smoke_parent=$(mktemp -d /private/tmp/exp414-smoke-parent.XXXXXX)
.venv/bin/python scratch/run_exp414_missing_aware_near_repeat.py \
  --representative-smoke \
  --output-dir "$exp414_smoke_parent/publication"
```

publication에는 manifest를 포함해 정확히 9개 파일이 있어야 한다.
다음 검증은 manifest의 모든 payload hash, TEST-use 금지, NumPy/Numba
qualification 및 tag parity, 다섯 결측 복구 smoke regime을 확인한다.

```bash
.venv/bin/python - "$exp414_smoke_parent/publication" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = tuple(root.iterdir())
assert len(files) == 9
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
for name, expected in manifest["artifact_sha256"].items():
    assert sha256((root / name).read_bytes()).hexdigest() == expected
provenance = manifest["provenance"]
assert provenance["test_values_used"] is False
assert provenance["test_labels_used"] is False
backend = json.loads(
    (root / "exp414_backend_assessment.json").read_text(encoding="utf-8")
)
assert backend["qualification_identity"] is True
assert backend["tag_identity"] is True
assert provenance["smoke_coverage"] == [
    "clean",
    "internal_gap",
    "edge_gap",
    "repaired_identity_collision",
    "all_missing",
]
PY
```

## 5. clean-commit acceleration benchmark

실제 benchmark는 Task 7에서 clean commit에 대해 정확히 한 번 실행한다.
기존 report가 있으면 CLI가 덮어쓰지 않고 거부한다.

```bash
NUMBA_NUM_THREADS=12 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scratch/run_exp414_missing_aware_near_repeat.py \
  --acceleration-benchmark \
  --db-path "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --benchmark-report "/Users/mac/Documents/Dataset/exp414_acceleration_benchmark.json"
```

ACSF1, HandOutlines, StarLightCurves와 결측 복구 parity용
DodgerLoopDay를 모두 검사한다. 모든 row의 최대 오차는 `1e-6` 이하,
qualification/tag/selected-quantile identity는 true여야 한다.
HandOutlines와 StarLightCurves warm speedup은 각각 `2.0x` 이상이어야
하며 DodgerLoopDay는 복구 run `1`, 복구 값 `71`이어야 한다. report는
Git commit, DB hash, runtime 12/1과 모든 frozen contract hash에 묶인다.

## 6. manifest와 출력 전 조건 재확인

```bash
shasum -a 256 "/Users/mac/Documents/Dataset/exp414_acceleration_benchmark.json"
test -d "/Users/mac/Documents/Dataset/exp411_near_repeat_qualification.checkpoint"
test ! -e "/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat"
git status --short
```

Exp411 source checkpoint는 읽기 전용 migration source다. Exp414
checkpoint와 최종 출력은 각각 별도 경로를 쓴다.

## 7. 제출 명령 출력과 별도 승인

다음 명령은 preflight나 제출을 실행하지 않고 고정된 one-shot 명령만
출력한다.

```bash
bash scratch/launch_exp414_full.sh --print-command
```

출력에서 label `com.openai.exp414.full`, thread 12/1, 두 로그 경로,
여섯 고정 입력/출력 경로와 `run_exp414_launchd_once.sh` helper를
확인한다.

전체 1,117개 실행에 대한 사용자의 별도 명시적 승인을 받은 뒤에만
다음을 한 번 실행한다.

```bash
bash scratch/launch_exp414_full.sh
```

launcher는 clean Git, commit-bound benchmark, DB SHA256, SQLite
`quick_check`, population manifest 및 정확한 결측 inventory, 기존 최종
출력 부재, 같은 launchd label 미등록을 다시 확인한다. 암묵적 승인,
자동 retry, keepalive respawn은 없다.

## 8. 일회성 상태 확인과 checkpoint resume

승인된 제출 후 필요할 때 한 번만 상태를 확인한다.

```bash
launchctl print "gui/$(id -u)/com.openai.exp414.full"
tail -n 80 /private/tmp/exp414-full.stdout.log
tail -n 80 /private/tmp/exp414-full.stderr.log
```

이 확인은 일회성 수동 inspection이다. watch, while loop, cron,
주기적 상태 조회 같은 자동 monitoring/polling은 사용하지 않는다.

중단되면 최종 출력이 여전히 없고 checkpoint provenance가 동일한지
확인한 후, 다시 별도 승인을 받아 같은 launcher를 한 번 실행한다.
runner가
`/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat.checkpoint`
의 검증된 dataset record만 재사용한다. checkpoint를 삭제하거나 다른
commit, DB, manifest, repair/qualification contract와 섞지 않는다.
