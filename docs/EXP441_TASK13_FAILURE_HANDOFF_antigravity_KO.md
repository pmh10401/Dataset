# Exp441 Task 13 공식 실행 실패 및 Exp442 인계

## 쉬운 결론

Exp441은 1,117개 source를 모두 정상적으로 읽고 후보 30개를 골랐다. 문제는 탐지
계산이 아니라 마지막 저장 단계였다. 실행기는 완료 후 남은 시간을 숫자 `0.0`으로
썼고, 저장 검증기는 `null`만 허용해 publication 직전에 멈췄다.

## 동결된 실패 증거

```text
reviewed_commit = f5df76e377a355eaf1613b7864186c7a618b7756
source_and_report_commit = c38b933e5ff1d76c70d90b2b5cac9ee72a1898f5
run_root = /Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808
attempt = attempts_antigravity/attempt_0000_antigravity
pid = 28062
pid_active = false
readiness_sha256 = 9d433c59451f6121e11445d724a4f48497f744ecc0917b5b6fc6567449c8948a
terminal_authority_sha256 = 726d50e831acdad6728503156137f72aba71c541af82b8b37e6f959d62b0e95b
progress_sha256 = 04eddec6bb05e7b27f90c5ed6b47c3234a0ad169795d65c638692bfe89c03c1e
source_snapshot_sha256 = 36aa46b5c8bc07b7851d84739af7ba123c840a01c85893e56cf0eb4a6b5311ee
selection_file_sha256 = 5823ea0a72222d67517eaa68ee48d5b1c89783d4ea61a0db4a1fc358f4278b88
events_sha256 = e521a7807784faad3fee4893630bfb15a878d96fde202de8e19cda25040d56a8
text_log_sha256 = 1fbf02a6a7e60edb08c3ed2bf6894a13dbefa705107afd02ff496b088929818e
pid_file_sha256 = 4e16b5cd608e9578bb77043ded30c34345108387f184d839cc7cd1fa582891a7
```

Terminal progress facts:

```text
completed = 1117 / 1117
success = 1117
error = 0
provisional = 840
outside_hard = 394
eligible = 73
dataset_winners = 49
selected = 30
elapsed_seconds = 47.46669329202268
throughput = 23.532290170879136
eta_seconds = 0.0
```

정확한 예외는 `ValueError: attempt terminal progress is invalid`이며
`reporting_antigravity.py`의 terminal progress 검증에서 발생했다.

publication과 evaluation 디렉터리는 생성되지 않았다. prediction 단계의 TEST
label 접근 수는 0이며 SQLite label query도 실행되지 않았다.

## 처리 결정

기존 Exp441 root, readiness, attempt, 로그는 모두 불변 실패 증거로 남긴다. 코드를
바꾼 뒤 같은 root를 구제하거나 오래된 attempt를 publication에 넣지 않는다.

Exp442를 과학 규칙이 같은 기술 수리 후속으로 사전등록한다. 종료 ETA만 `null`로
정규화하고, 새 identity와 새 빈 root에서 readiness 및 1,117개 label-free replay를
처음부터 수행한다. 선택 좌표 identity가 Exp441과 다르면 publication 전에 실패한다.
