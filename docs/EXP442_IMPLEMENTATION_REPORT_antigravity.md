# Exp442 구현·검증 보고서

## 결론

Exp442는 새 탐지 방법이 아니라 Exp441의 종료 ETA와 publication 계약만 고친
공학적 후속 실험이다. 정상 실행이 끝난 뒤 `eta_seconds`를 `0.0`이 아니라 JSON
`null`로 기록하도록 producer를 고쳤고, strict reporter 계약은 완화하지 않았다.
selector, source, threshold, budget, seed, routing과 TEST-label 차단선은 바꾸지 않았다.

- 기준 commit: `f5df76e377a355eaf1613b7864186c7a618b7756`
- 구현 commit: `5747a47f`
- review-fix commit: `5c946ba73d012760ba4309d46f3b4bcb499cf0cf`
- reviewed commit: `c4783d695ab3d8b9e03d605a46143c3dcaeb7a6b`
- Sol 재검토: `VERDICT: ship` (Critical 0, Important 0, Minor 0)
- reviewed design SHA256: `cf0b6e916a85cdf21c8cc28c735894f927b63e4c33aa4d62e95753540bd06a7b`
- reviewed preregistration SHA256: `ab909c35d0f4d0252d1a29fa80ea27d51d0ba56175c8cc0d8d3428092ec2746c`
- Exp442 run identity:
  `exp442-terminal-eta-publication-repair-antigravity-20260808`
- 새 공식 root:
  `/Users/mac/Documents/Dataset/exp442_terminal_eta_publication_repair_antigravity_20260808`
- 공유 과학 설정 SHA256:
  `d6086725b68aaf3de13c006aa817faa9daa80257a36965729935f1590b389ffb`
- 동결 selected-coordinate identity SHA256:
  `db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048`
- readiness SHA256: `0ea3da605eeb3ae08c719c20c895846799a79a52951881fbfccd4217ad25aa14`
- readiness source snapshot SHA256: `63a54512fbfbf7aef0db2c8ae12fe52af5f9d98a276095c2231e001c2c014f03`
- frozen prediction SHA256: `b79bad6012ad6bc07050ade9968e815796a2cc871b5c85d63f0bd7dbdf6f81f4`
- publication artifact-map SHA256: `14e2662a8880bd20658bb849d476391654ee6283f5abc8149656cf73885c69c1`
- candidate ranking SHA256: `d8d858cd74553aa64f6f0a5424064872b59f682fed1c8b2ace63fffba6ad20cc`
- source audit SHA256: `36aa46b5c8bc07b7851d84739af7ba123c840a01c85893e56cf0eb4a6b5311ee`
- selection SHA256: `06004f18dbd9b0df3fa857bb8b1a9fd7eafd4d927f3736fb8cfb25db56b72408`
- evaluation artifact-map SHA256: `e1155acf05e00935710243e875eaeb2d13047d6236d026ee93fe654fc33c670b`
- primary evaluation SHA256: `f432ffdf530b2e32bc7a62fec6c7d07cff2eff9f3049489b5e03136b445a33cf`

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

경로 안전성도 분리했다. Production generic API에는 `run_root` 같은 경로 override를
더 이상 전달할 수 없고, 고정 profile의 root만 사용한다. 별도 경로가 필요한 테스트는
private `_run_readiness_at_root_antigravity`와
`_validate_readiness_at_root_antigravity` seam 아래에서만 실행한다. 따라서 테스트
편의용 경로 주입이 공식 readiness/new 경로를 풀어버리지 않는다.

첫 Sol High fix-first에서 지적된 두 Important와 두 Minor는 다음처럼 처리했다.

- Important: profile의 design/preregistration hash binding을 최신 동결 SHA로 맞췄다.
- Important: production root override를 제거하고 private test seam으로 격리했다.
- Minor: codemap status 근거를 장부와 검증 기록에 반영하고, 최종 codemap 세 파일을
  commit `04560d0d`에서 함께 재생성했다.
- Minor: active 상태의 원래 ETA 계산을 보존하는 회귀 테스트를 추가했다.

## TDD 증거

수정 전에 다음 두 실패를 재현했다.

1. 실제 writer 종료값이 `None`이어야 하지만 `0.0`이었다.
2. 실제 writer가 만든 terminal progress를 reporter에 넘기면
   `attempt terminal progress is invalid`가 발생했다.

producer 수정 후 두 테스트와 전체 회귀가 통과했다.

## 검증 결과

- Exp442 lifecycle: 7/7 통과
- Exp441 전체 회귀: 197/197 통과
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
- pre-freeze label access: 0
- finite evidence: true
- 11-worker와 순차 기준 source payload: 완전 동일
- 11-worker와 순차 기준 selection payload: 완전 동일
- 11-worker와 순차 기준 snapshot payload: 완전 동일

검증한 뒤 Exp442 공식 root가 생성되었고 readiness, `new`, `inspect`, `evaluate`가
모두 끝났다. 공식 실행은 1,117개를 모두 처리했고 오류는 0개였다. 종료 상태에서
`eta_seconds`는 `null`이며, 마지막 실행 기록은 PID 32029(현재 비활성), CPU 11개
worker, 경과 49.68초, 처리량 약 22.48개/초다. 최종 후보 수는 provisional 840,
outside Hard 394, eligible 73, winner 49, selected 30이다.

## 결과를 쉽게 읽기

Exp442가 추가한 30개 후보 중 실제 TEST 이상과 맞은 것은 15개였고, 틀린 경보도
15개였다. 후보 정밀도는 50%다. 따라서 놓쳤던 이상 15개를 새로 찾았으며, 기존
강한 데이터셋에서 성능이 떨어진 경우는 0개였다.

| 구분 | 기존 Exp404 Hard | Exp442 union |
|---|---:|---:|
| 맞게 찾음(TP) | 1,915 | 1,930 |
| 잘못 경보(FP) | 303 | 318 |
| 놓침(FN) | 2,003 | 1,988 |
| 정상(TN) | 191,679 | 191,664 |

전체 F1은 `0.6242`에서 `0.6260`으로 `+0.00183` 올라갔고, 데이터셋별 평균
F1 변화는 `+0.01244`였다. 사전등록한 paired bootstrap 95% 구간도 micro
`[0.000054, 0.003814]`, macro `[0.006535, 0.019158]`로 모두 0보다 컸다.
다만 이는 TEST label을 prediction에 쓰지 않은 뒤 동결 후 평가한 회고적 결과다.
prediction 단계에서는 label 접근이 0회였고, TEST label은 freeze 이후 `evaluate`에서만
열었다.

## MLX 판단

이번 변경은 tensor 계산이 아니라 작은 JSON 상태 계약과 source-selector 재현이다.
MLX로 표현할 대상이 없으므로 `inapplicable_no_tensor_stage`로 고정하고 CPU 11
workers를 유지한다. 과학 계산을 GPU 사용 명분으로 다시 쓰지 않는다.

## 실행 완료 및 일시정지

공식 readiness, 1,117개 label-free `new`, `inspect`, prediction 동결, 그리고
freeze 이후 TEST label을 사용하는 `evaluate`까지 모두 완료했다. 모든 사전등록
회고 평가 gate는 통과했지만, 이번 결과는 기존 연구 흐름에서 정한 adaptive
follow-up의 기술 재현이므로 `independent_validation=false`, `review_only=true`다.
따라서 운영 자동 경보나 새 scientific champion으로 승격하지 않는다.

현재 Exp442를 완료·일시정지하고 다음 방향을 함께 결정할 때까지 새 실험은 자동으로
시작하지 않는다. 다음 실험을 시작할 때는 반드시 새 번호와 새 사전등록으로
분리한다.
