# Exp422-IR 사전등록 — Exp419 중단 후 CPU/MPS 수치 차이 진단

## 1. 연구 질문

공식 Exp419가 완료되지 않고 중단된 상황에서, Exp417의 고정된 TRAIN-only
계산 경로를 CPU와 MPS에서 실행했을 때 수치 결과와 후보 좌표가 같은지
진단한다. 이 문서는 기존 Exp422의 성공한 Exp419 선행조건을 바꾸지 않고,
`Exp422-IR(Interrupted predecessor)`라는 별도 실행 변형을 등록한다.

Exp419 중단 자체를 성공 근거로 간주하지 않는다. Exp422-IR 결과는 숫자·재현성
진단 증거일 뿐이며 운영 경보, 모델·threshold·source·budget 변경의 근거가
아니다.

## 2. 선행 실행 예외

이번 변형에서만 다음 조건을 허용한다.

- 공식 Exp419 PID 파일은 존재하는 일반 파일이어야 한다.
- PID가 살아 있지 않아야 한다.
- Exp419 exit 파일은 존재하고 값이 정확히 `143(SIGTERM)`이어야 한다.
- 다른 non-zero 코드, PID 생존, exit 파일 누락·위조는 즉시 거부한다.

이는 “Exp419가 성공했다”는 의미가 아니라 “Exp419가 중단되었음을 확인한
뒤 별도 진단을 시작한다”는 의미다.

## 3. 고정 알고리즘과 데이터 경계

- 후보 batch는 정확히 `16, 32, 64, 128, 256`이다. 자동 축소·대체 batch는
  사용하지 않는다.
- Exp417의 seed, horizon, alpha, persistence, loss, fold 규칙을 그대로 쓴다.
- 데이터베이스는 고정 SHA-256과 SQLite `quick_check=ok`를 읽기 전용으로
  확인한다.
- 정상 TRAIN 값만 사용한다. TEST 값·라벨·실제 이상 위치·과거 TEST 성능·사람
  검토 결과는 읽지 않는다.
- core는 고정된 60개 `(batch, seed, length)` 조합을 모두 기록한다.
- core가 통과할 때만 제한된 정상 TRAIN score parity를 실행한다.
- MPS가 없거나 parity/speed gate가 실패하면 CPU fallback 사유를 기록하고
  운영 적용으로 승격하지 않는다.
- MLX는 이 진단에서 metadata-only로 기록한다. 알고리즘을 재작성하지 않는다.

## 4. 반복과 자원

`--repeats 3`을 사용한다. 한 번에 하나의 MPS-heavy 프로세스만 실행하며,
Exp419 checkpoint와 기존 공식 출력은 변경하지 않는다. 매 시도는 새롭고 빈
output root에서만 실행한다.

## 5. 진행 증거

각 시도 output root에 다음을 남긴다.

- `progress.jsonl`: 시작, preflight, core, TRAIN sample, score, heartbeat,
  manifest 완료/실패 이벤트. 각 줄은 즉시 flush/fsync된다.
- `execution_context.json`: Exp419 중단 예외, Git/source manifest, PID와 로그
  위치를 기록한다.
- 실행기의 stdout/stderr는 별도 `run.log`로 리다이렉트한다.
- PID 파일은 실행 중인 Exp422-IR Python 프로세스를 가리킨다.

## 6. 해석 제한

결과가 CPU와 MPS의 수치 parity를 보여도 운영 backend를 바꾸지 않는다.
후속 운영 선택은 새 실험 번호와 새 사전등록으로만 제안한다. `mean_combined_f1`
및 TEST 기반 혼동행렬은 이 진단의 의사결정 지표가 아니다.
