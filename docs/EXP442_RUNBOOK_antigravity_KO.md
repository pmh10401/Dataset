# Exp442 종료 ETA 저장 수리 실행 안내서

## 고정 경로

```text
RUN_ROOT=/Users/mac/Documents/Dataset/exp442_terminal_eta_publication_repair_antigravity_20260808
READINESS=$RUN_ROOT/readiness_antigravity.json
ATTEMPT=$RUN_ROOT/attempts_antigravity/attempt_0000_antigravity
PROGRESS=$ATTEMPT/progress_antigravity.json
EVENTS=$ATTEMPT/events_antigravity.jsonl
TEXT_LOG=$ATTEMPT/text_antigravity.log
PID_FILE=$ATTEMPT/pid_antigravity
PUBLICATION=$RUN_ROOT/publication_antigravity
EVALUATION=$RUN_ROOT/evaluation_antigravity
```

기존 Exp441 root는 읽기 전용 실패 증거다. Exp442 시작 전에 새 `RUN_ROOT`는 없어야
한다. 수동으로 attempt나 publication을 복사하지 않는다.

## 정확히 네 명령

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp442_terminal_eta_publication_repair_antigravity.py readiness
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp442_terminal_eta_publication_repair_antigravity.py new
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp442_terminal_eta_publication_repair_antigravity.py inspect
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp442_terminal_eta_publication_repair_antigravity.py evaluate
```

명령줄 flag, path/backend/worker/seed/budget override, resume 명령은 없다. readiness는
검토된 clean HEAD와 고정 source를 묶어 새 root를 원자적으로 만든다. `new`는
CPU 11 worker로 1,117개 source를 새로 검증하고 다음 로그를 process가 직접 갱신한다.

- PID와 active 상태
- 현재 dataset/record와 active dataset 목록
- 완료/전체, 성공/오류, 후보 단계별 수
- backend, 처리속도, 경과시간, ETA
- source checkpoint와 최신 오류

활성 실행 중 ETA는 예상 남은 초다. 종료 상태의 ETA는 정확히 JSON null이어야 한다.
오래 걸리는 실행을 Codex가 반복 감시하지 않는다. 사용자는 위 progress/text/PID
파일로 상태를 확인할 수 있다.

`new` 성공은 publication까지 만든다. `inspect`는 DB나 TEST label 없이 publication과
source 해시를 다시 확인한다. `evaluate`만 publication 동결 뒤 TEST label을 열어
사후 지표를 만든다. 실패한 root, attempt, staging, 로그는 덮어쓰지 않는다.
