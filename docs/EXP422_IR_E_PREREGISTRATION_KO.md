# Exp422-IR-E 사전등록 — parity 실패 후 review-only score 수집

## 목적

Exp422-IR의 MPS core parity gate가 실패했을 때 실행을 즉시 끝내지 않고,
그 실패를 그대로 기록한 채 정상 TRAIN score 경로의 추가 증거를 수집한다.
이는 수치 오차를 없애거나 성공으로 간주하는 실험이 아니다.

## 고정 경계

- Exp419 PID가 죽어 있고 exit code가 정확히 `143`인 경우만 허용한다.
- Exp417의 seed, horizon, alpha, persistence, loss, fold 규칙과 Exp422의
  batch `16, 32, 64, 128, 256`을 바꾸지 않는다.
- DB는 고정 SHA-256과 `PRAGMA quick_check=ok`를 통과해야 한다.
- 정상 TRAIN 값만 사용한다. TEST 값·라벨·이상 위치·과거 TEST 지표는 읽지 않는다.
- MPS parity 실패는 `core_manifest.json`에 failed로 보존한다.
- 실패한 core 뒤의 score 결과는 `exploratory_manifest.json`에 별도 기록한다.
- 이 산출물은 review-only다. 운영 backend, threshold, source, lane, budget,
  자동 경보를 변경할 수 없다.

## 의도적인 예외

`core_gate_overridden=true`는 “score를 추가로 관찰한다”는 뜻이며, core
오차를 무시해 통과시킨다는 뜻이 아니다. score 자체가 실패하거나 MPS가
사용 불가능하면 해당 사유를 남긴다.

## 운영 증거

새 output root에는 `progress.jsonl`, `execution_context.json`,
`core_manifest.json`, `exploratory_manifest.json`이 생긴다. 실행기의
stdout/stderr는 launcher의 `run.log`에 기록하며, 한 번에 하나의 MPS-heavy
프로세스만 허용한다.
