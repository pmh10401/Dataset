# Exp422 실행 런북

이 런북은 한 번의 작은 TRAIN-only Exp422 진단을 위한 것이다. 실제 실행은 별도
승인을 받은 뒤에만 시작하며, 기존 공식 작업의 프로세스·checkpoint·출력을
변경하지 않는다.

## 실행 전 guard

Exp419가 성공적으로 끝난 사실을 먼저 확인한다. Exp422 CLI는 제공된 Exp419 PID
파일이 살아 있으면, exit 파일이 없으면, 또는 exit code가 0이 아니면 시작을
거부해야 한다. 한 번에 하나의 MPS-heavy 프로세스만 허용한다.

데이터베이스는 읽기 전용으로 열고 동결된 SHA-256 및 `PRAGMA quick_check`를
검증한다. output root는 매 시도마다 새롭고 비어 있어야 하며, 기존 결과를
덮어쓰지 않는다.

## 승인된 새 Exp422 시도 형태

```bash
PYTHON="/Users/mac/Documents/Time Series/.venv/bin/python"
DB="/Users/mac/Documents/Dataset/univariate_ts.db"
OUT="/tmp/exp422-attempt-$(date +%Y%m%d-%H%M%S)"

"$PYTHON" scratch/run_exp422_mps_numeric_delta.py \
  --database "$DB" \
  --output-root "$OUT" \
  --exp419-pidfile "/tmp/exp419.pid" \
  --exp419-exitfile "/tmp/exp419.exit" \
  --repeats 1
```

이 명령은 새 Exp422 output root만 대상으로 한다. `--repeats`는 1~7 범위이며,
batch·seed·threshold·TRAIN-only 경계를 바꾸지 않는다.

## 결과 확인과 중단

매니페스트에서 두 단계의 `passed`, `failed`, `skipped` 상태와 각 fallback reason,
CPU/MPS timing, digest, MLX/MPS metadata를 확인한다. MPS unavailable 또는 parity
실패는 정상적인 CPU fallback 결과이며, 운영 적용 신호가 아니다.

중단하려면 Exp422 진단 프로세스에만 `Ctrl-C`를 보낸다. 부분 output은 증거 보존을
위해 즉시 삭제하지 말고, 새 root에서만 재시도한다.
