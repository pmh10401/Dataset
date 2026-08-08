# Exp442 구현·검증 보고서

## 결론

Exp442는 새 탐지 방법이 아니라 Exp441의 종료 ETA와 publication 계약만 고친
공학적 후속 실험이다. 정상 실행이 끝난 뒤 `eta_seconds`를 `0.0`이 아니라 JSON
`null`로 기록하도록 producer를 고쳤고, strict reporter 계약은 완화하지 않았다.
selector, source, threshold, budget, seed, routing과 TEST-label 차단선은 바꾸지 않았다.

- 기준 commit: `f5df76e377a355eaf1613b7864186c7a618b7756`
- 구현 commit: `5747a47f`
- Exp442 run identity:
  `exp442-terminal-eta-publication-repair-antigravity-20260808`
- 새 공식 root:
  `/Users/mac/Documents/Dataset/exp442_terminal_eta_publication_repair_antigravity_20260808`
- 공유 과학 설정 SHA256:
  `d6086725b68aaf3de13c006aa817faa9daa80257a36965729935f1590b389ffb`
- 동결 selected-coordinate identity SHA256:
  `db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`

## 고친 문제

Exp441은 1,117개 label-free 처리를 오류 없이 끝냈지만, writer가 종료 ETA를
`0.0`으로 기록하고 reporter가 정확한 `null`만 허용해 publication 직전에
`ValueError: attempt terminal progress is invalid`로 닫혔다.

수정 규칙은 단순하다.

- 실행 중: 기존 방식으로 남은 시간을 계산한다.
- 실행 종료: 남은 시간이 아니라 “더 이상 ETA가 없음”을 뜻하는 JSON `null`을
  기록한다.
- 이미 실패한 Exp441 root와 attempt는 수정·복사·재사용하지 않는다.

## 구조

Exp441과 Exp442는 한 개의 닫힌 lifecycle profile 구조를 공유한다. 두 profile은
실험 이름, schema namespace, run identity, root, 사전등록·설계 SHA와 CLI 이름만
다르다. 과학 설정은 한 SHA에 묶이며 임의 경로나 설정 override는 허용하지 않는다.

Exp442 package는 profile, lifecycle, CLI만 가진 얇은 wrapper다. 실제 source,
selector, runtime, reporting, evaluation 구현은 명시적 profile을 받는 공유 함수로
실행하며 기존 Exp441 공개 함수는 그대로 보존한다.

## TDD 증거

수정 전에 다음 두 실패를 재현했다.

1. 실제 writer 종료값이 `None`이어야 하지만 `0.0`이었다.
2. 실제 writer가 만든 terminal progress를 reporter에 넘기면
   `attempt terminal progress is invalid`가 발생했다.

producer 수정 후 두 테스트와 전체 회귀가 통과했다.

## 검증 결과

- Exp442 lifecycle: 5/5 통과
- Exp441 전체 회귀: 196/196 통과
- Exp440 상속 회귀: 35/35 통과
- Exp417 label·입력·모델 경계 전체 회귀: 270/270 통과
- Python compile 및 `git diff --check`: 통과
- AST/call-spy prediction firewall: 통과
- prediction 경로 SQLite 연결 차단 시험: 통과

실제 동결 source 재현 결과:

- dataset: 1,117
- provisional: 840
- outside Hard: 394
- eligible: 73
- winner dataset: 49
- selected: 30
- production workers: CPU 11
- production source events: 1,117
- label access: 0
- finite evidence: true
- 11-worker와 순차 기준 source payload: 완전 동일
- 11-worker와 순차 기준 selection payload: 완전 동일

검증 시 Exp441 불변 root의 7개 파일 tree SHA256은
`8d72eee1f04529771963da340667b049af2cab7e9aec1a5abdcc4ba5229d0880`이며,
Exp442 공식 root는 아직 존재하지 않았다.

## MLX 판단

이번 변경은 tensor 계산이 아니라 작은 JSON 상태 계약과 source-selector 재현이다.
MLX로 표현할 대상이 없으므로 `inapplicable_no_tensor_stage`로 고정하고 CPU 11
workers를 유지한다. 과학 계산을 GPU 사용 명분으로 다시 쓰지 않는다.

## 다음 gate

1. 코드맵 세 파일을 현재 source commit에 맞춰 함께 갱신한다.
2. fresh Sol High 검토에서 중요한 문제가 없어야 한다.
3. 깨끗한 reviewed commit에서 Exp442 readiness를 한 번 생성한다.
4. 새 root에서 1,117개 label-free `new`를 실행하고 `inspect`로 prediction을 동결한다.
5. prediction SHA와 label access 0을 확인한 뒤에만 `evaluate`를 실행한다.

기술 실행과 사후 재현이 성공해도 Exp442는 adaptive, descriptive, review-only다.
독립 검증, 운영 경보 정책 또는 새 champion으로 승격하지 않는다.
