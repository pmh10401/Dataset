# Exp440 구현 보고서

## 결론

Exp440의 label-free 예측, 11-worker 실행, checkpoint/resume, 자체 진행 로그, review-only 출판, 동결 후 평가 경계를 구현했다. 구현 기준 커밋은 `178d49bc9f6a418902d28031ee311e8e0270f1e1`이다. 이 시점에는 공식 prediction을 시작하지 않았고 TEST label 접근도 0이다.

Exp439의 저주파 점수가 놓친 이상을 찾는 힘은 있었지만 FP가 너무 많았던 문제를 다음처럼 좁혔다.

- Exp439 선호 깊이 주변의 최대 3개 저주파 보기 사용
- 각 TRAIN 행을 뺀 fold-local OOF 0.99 `higher` 문턱
- delete-one 문턱 비율 2.0 이하인 안정 깊이만 사용
- 최소 두 깊이가 엄격히 문턱을 넘을 때만 임시 후보
- Hard 좌표를 먼저 제외하고 전역 최대 100개
- 유한 정상 TRAIN이 256행을 넘는 exact-length 그룹은 일부 추출 없이 전체 abstain

마지막 규칙은 TEST를 보지 않고 실행 가능성과 정밀도를 보호하기 위해 prediction 전에 추가했다. 큰 그룹을 임의로 줄여 다른 알고리즘으로 바꾸지 않으며, 기존 Hard 탐지는 그대로 유지한다.

## 검증 결과

| 범위 | 테스트 | 시간 | 결과 |
|---|---:|---:|---|
| Exp440 | 35 | 약 0.17초 | 성공 |
| Exp439 재사용 경계 | 108 | 7.80초 | 성공 |
| Exp436 공유 입력·출판 경계 | 301 | 111.08초 | 성공 |
| 합계 | 444 | - | 실패 0, 오류 0 |

실제 TRAIN-only smoke:

| 데이터셋 | TRAIN | TEST | 길이 | CPU 시간 | 임시 후보 | abstain |
|---|---:|---:|---:|---:|---:|---:|
| FiftyWords_normal_49 | 4 | 50 | 270 | 0.154초 | 1 | 0 |
| CBF_normal_3 | 248 | 300 | 128 | 3.627초 | 7 | 0 |

두 smoke 모두 제외 TRAIN 0, `label_access_count=0`이었다.

## 가속 판단

CBF_normal_3 전체 단계에서 CPU와 Numba 후보는 7개로 완전히 같았고 최대 수치 차이는 `1.24155e-8`이었다. 하지만 CPU 3.628초, Numba 4.148초로 Numba 속도는 CPU의 0.875배에 불과해 1.1배 기준을 통과하지 못했다. MLX의 대표 거리 단계도 CPU의 약 0.215배였다. 따라서 공식 backend는 CPU, worker는 11, worker 내부 수치 스레드는 1개다. readiness 파일에는 두 탈락 근거를 기록한다.

## 경계 감사

- 예측 경로의 TEST label·실제 이상 위치·과거 TEST 성능 참조: 없음
- `evaluation_antigravity` 참조: 고정 평가 경로, 평가 모듈 자체, `evaluate` 명령의 lazy import, 금지 경로 테스트만 존재
- 새 파일 이름: 모두 `antigravity` 포함
- Exp439 checkpoint/publication/evaluation 변경: 없음
- publication/evaluation: no-clobber, SHA256 결합, prediction 변경 감지
- 진행 파일: PID, 현재 데이터셋, TRAIN/TEST 수와 길이, 단계, backend, 완료/전체, 성공/abstain/error, 후보 수, 처리속도, 경과시간, ETA, checkpoint, heartbeat, 최신 오류 포함

## 다음 명령

깨끗한 구현 커밋에서 readiness를 한 번 실행한다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py readiness
```

readiness가 CPU/11-worker와 label 접근 0을 확인하면 승인된 1,117개 label-free `new` 실행을 시작한다. 장시간 실행은 프로세스 자체 로그로 확인하며 Codex가 반복 감시하지 않는다.
