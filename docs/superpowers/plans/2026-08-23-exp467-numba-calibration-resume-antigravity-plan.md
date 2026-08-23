# Exp467 Numba Calibration Resume Plan

**Spec:** `docs/superpowers/specs/2026-08-23-exp467-numba-calibration-resume-antigravity-design.md`

## 구현 체크리스트

- [x] Exp451 프로파일로 `calibration_paper.py` Python 반복문 병목 확인
- [x] float64 전용 Numba count loop와 mixed-dtype legacy fallback 구현
- [x] Hard source, MiniRocket, Hard route, science payload의 Exp467 로컬 어댑터 연결
- [x] Python·NumPy·Numba·llvmlite·CPU/MPS/MLX provenance 기록
- [x] CLI code closure에 모든 로컬 어댑터, Numba helper, 활성 의존성 SHA-256 포함
- [x] Exp467 isolated roots와 Exp450 create-only migration 계약 생성
- [x] Exp467 코드맵 triplet 생성
- [x] parity·candidate·payload byte 및 단계 속도 gate 통과
- [x] 고정 `MelbournePedestrian_normal_1` C native smoke 명령과 isolated root 검증 추가
- [ ] fresh 독립 검토 통과
- [ ] Exp450 원본 before/after snapshot 동일성 확인
- [ ] Exp467 migration 및 verify-migration 완료
- [ ] Exp467 official run 시작

## 필수 검증

```bash
EXP467_PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python
$EXP467_PY -B -m unittest paper.equipment_anomaly_detection.tests_paper.test_numba_calibration_antigravity
$EXP467_PY -B -m unittest discover -s experiments/exp467_numba_calibration_resume_antigravity/tests -p 'test_*_antigravity.py'
$EXP467_PY -B scratch/generate_exp467_codemap_antigravity.py --check
$EXP467_PY -B -m py_compile paper/equipment_anomaly_detection/numba_calibration_antigravity.py experiments/exp467_numba_calibration_resume_antigravity/*_antigravity.py scratch/run_exp467_numba_calibration_resume_antigravity.py
git diff --check
```

## 공식 전환 명령

공식 interpreter는 `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`이다.

```bash
EXP467_PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python
$EXP467_PY -B scratch/run_exp467_numba_calibration_resume_antigravity.py migrate --mode full
$EXP467_PY -B scratch/run_exp467_numba_calibration_resume_antigravity.py verify-migration --mode full
```

공식 Exp451은 코드 검토·Exp467 migration·`verify-migration`·Exp450 before/after snapshot 동일성 확인까지 계속 실행한다. 그 뒤 정확한 Exp451 Python PID에만 `SIGINT`를 보내고 PID, launcher shell, `tee`, run lock, Metal lock/FD 소멸을 확인한다. `SIGSTOP`·`SIGTERM`은 사용하지 않는다. MPS 프로세스가 없을 때만 위 smoke 명령을 실행하며, 2분 안에 종료 코드 0, `smoke-progress=success`, MPS backend, Numba backend, checkpoint reread 검증을 모두 확인한다.

macOS에 `timeout`이 없으므로 smoke는 다음 watchdog으로만 실행한다. stdout/stderr를 즉시 화면과 create-only 로그에 동시 기록한다. 120초가 지나면 자식 Python에 `SIGINT`를 보내고, 10초 후에도 종료되지 않으면 강제 종료한 뒤 코드 124를 반환한다. 종료 코드가 0이 아니면 아래 Exp451 rollback을 즉시 실행한다.

```bash
cd "/Users/mac/Documents/Time Series/.worktrees/exp467-numba-calibration-antigravity"
EXP467_PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python
EXP467_EVIDENCE_ROOT=/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_evidence_antigravity_20260823
if [ -L "$EXP467_EVIDENCE_ROOT" ] || { [ -e "$EXP467_EVIDENCE_ROOT" ] && [ ! -d "$EXP467_EVIDENCE_ROOT" ]; }; then
    echo "unsafe Exp467 evidence root" >&2
    exit 1
fi
/bin/mkdir -p "$EXP467_EVIDENCE_ROOT"
EXP467_SMOKE_LOG="$EXP467_EVIDENCE_ROOT/smoke_run_$(date -u +%Y%m%dT%H%M%SZ)_antigravity.log"
if ! "$EXP467_PY" - "$EXP467_SMOKE_LOG" <<'PY'
from pathlib import Path
import signal
import subprocess
import sys
from threading import Thread

log_path = Path(sys.argv[1])
command = [
    sys.executable,
    "-B",
    "scratch/run_exp467_numba_calibration_resume_antigravity.py",
    "smoke",
    "--mode",
    "smoke",
]
with log_path.open("x", encoding="utf-8") as log:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    def copy_output() -> None:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

    copier = Thread(target=copy_output)
    copier.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        returncode = 124
    copier.join()

if timed_out:
    print("Exp467 smoke timed out after 120 seconds", file=sys.stderr)
if returncode != 0:
    raise SystemExit(returncode if 0 < returncode < 256 else 1)
PY
then
    cd "/Users/mac/.codex/worktrees/e1be/Time Series"
    /usr/bin/nohup "$EXP467_PY" -B scratch/run_exp451_verified_accelerated_resume_antigravity.py run --mode full >> /Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_evidence_antigravity_20260823/official_run_console_antigravity.log 2>&1 &
    EXP451_ROLLBACK_PID=$!
    echo "Exp467 smoke failed; Exp451 restarted as PID $EXP451_ROLLBACK_PID" >&2
    exit 1
fi

cd "/Users/mac/Documents/Time Series/.worktrees/exp467-numba-calibration-antigravity"
"$EXP467_PY" -B scratch/run_exp467_numba_calibration_resume_antigravity.py run --mode full
```

smoke watchdog가 0을 반환한 경우에만 위 마지막 Exp467 `run --mode full`에 도달한다. 실패·timeout은 즉시 `nohup` Exp451 rollback을 시작하고 shell을 1로 종료하므로 Exp467 full run을 시작할 수 없다. 기존 durable checkpoint는 재사용되고 중단 중이던 Exp451 좌표만 다시 계산된다. 진행 상태는 별도 셀에서 공식 interpreter로 `scratch/run_exp467_numba_calibration_resume_antigravity.py status --mode full`을 실행해 확인한다.
