# Exp443 정상 TRAIN 결측 처리 정책 비교 실행 안내서

## 실행 전 동결

이 runbook은 승인 설계의 동일 bytes와 구현 전 Exp443 control state에 묶인다.
SHA가 다르면 아래 명령을 실행하지 말고 새 실험 번호와 새 사전등록을 만든다.

```text
design_source = docs/superpowers/specs/2026-08-09-exp443-missing-normal-train-policy-comparison-antigravity-design.md
design_source_sha256 = b52142a0a48f9c557a56180ee96dc1a7a8e46760a15a547aa92f8b5999808f9b
experiment_id = Exp443
run_identity = exp443-missing-normal-train-policy-comparison-antigravity-20260809
run_root = /Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809
```

이 문서의 명령은 구현·회귀·실데이터 label-free 제어 검증을 통과했다. 직전
clean HEAD는 Sol High에서 `ship`을 받았지만, 첫 readiness에서 찾은 BLIS thread
설정 누락을 TDD로 수정했다. 이 수정이 포함된 새 clean HEAD의 scoped re-review가
다시 `ship`을 주기 전에는 실행하지 않는다. 현재 official
root·attempt·prediction·evaluation은 없다. `REVIEWED_COMMIT`은
readiness/new/resume 시점에
호출자가 넣는 runtime argument일 뿐 preregistration에 저장하는 placeholder가
아니다.

## 고정 경로

```text
RUN_ROOT=/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809
READINESS=$RUN_ROOT/readiness_antigravity.json
ATTEMPTS=$RUN_ROOT/attempts_antigravity
ATTEMPT=$ATTEMPTS/attempt_0000_antigravity
PROGRESS=$ATTEMPT/progress_antigravity.json
EVENTS=$ATTEMPT/events_antigravity.jsonl
TEXT_LOG=$ATTEMPT/text_antigravity.log
PID_FILE=$ATTEMPT/pid_antigravity.json
CHECKPOINT=$ATTEMPT/checkpoint_antigravity
PUBLICATION=$ATTEMPT/publication_antigravity
EVALUATION=$RUN_ROOT/evaluation_antigravity
```

실행 프로세스가 `progress_antigravity.json`, append-only
`events_antigravity.jsonl`, `text_antigravity.log`, `pid_antigravity.json`을
원자적·순차적으로 갱신한다. progress에는 PID/active 상태, 현재 policy와
dataset, 정상 TRAIN 전체·결측·사용·제외 행 수, TEST 행 수와 target length,
단계/backend, 완료/전체 `57`, import/recompute/success/abstention/error,
처리속도·경과시간·ETA, checkpoint 경로와 마지막 오류를 기록한다. checkpoint는
ordered `(policy, dataset)` prefix와 payload SHA를 보존한다. 프로세스가 살아
있는 동안에는 사용자가 이 파일을 직접 수정하지 않는다.

`readiness`는 source·설계 SHA·환경·clean reviewed commit·label firewall·빈
출력 root를 확인한다. `new`는 가장 작은 absent attempt sequence를 새로 만들고
기존 실패 attempt를 resume하거나 덮지 않는다. `resume`은 inactive PID, 고정
input/config/environment hash와 유효 ordered prefix를 검증한 뒤 미완료 unit만
계산한다. 성공한 `new`/`resume`은 terminal-complete progress를 먼저 기록하고
세 prediction publication을 create-only로 만든다. publication 실패 시 DB pin을
남긴다. 같은 attempt의 `resume`은 complete checkpoint와
progress/PID/events/text bytes를 읽기 전용으로 확인하고, 완료 로그나 저장된
process identity를 바꾸지 않은 채 publication만 재시도한다. 성공 후에만 pin을
정리한다. `inspect`는 SQLite label query나 파일 생성 없이 기존
publication/checkpoint/source bytes만 재검증한다. `evaluate`만 세 prediction
freeze 후 TEST label을 열어 별도 evaluation을 만든다. 실패 root와 staging은
감사 증거로 보존한다.

## 고정 명령 (scoped re-review 통과 후에만)

```bash
bash scripts/create_exp443_environment_antigravity.sh
EXP443_REVIEWED_COMMIT_ANTIGRAVITY="$(git rev-parse HEAD)"
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py readiness --reviewed-commit "$EXP443_REVIEWED_COMMIT_ANTIGRAVITY"
nohup /Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py new --reviewed-commit "$EXP443_REVIEWED_COMMIT_ANTIGRAVITY" > /Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809/launch_antigravity.log 2>&1 &
echo $!
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py resume --reviewed-commit "$EXP443_REVIEWED_COMMIT_ANTIGRAVITY" --attempt attempt_0000_antigravity
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py inspect
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py evaluate
```

`new`는 위 한 프로세스만 시작한다. 성공 반환 시 publication까지 이미 만들어져
있다. Codex가 반복 감시하지 않고 프로세스가 고정
progress/events/text/PID/checkpoint 파일을 직접 갱신한다. 완료 또는 오류는 사용자가
알린 뒤 한 번 확인한다. `evaluate`는 complete publication의 세 prediction SHA를
검사한 이후에만 실행한다.

환경 script는 Python 3.14.6, `aeon==1.5.0`, `numpy==2.3.5`,
`scikit-learn==1.8.0` 고정 환경을 저장소 밖에 만들도록 설계한다. 공식
MiniROCKET은 third-party CPU transformer이므로 MLX rewrite를 하지 않고 outer
11 worker/inner 1 thread CPU를 사용한다. wrapper가 runtime을 import할 때
OMP/OpenBLAS/MKL/vecLib/NumExpr/Numba/BLIS의 7개 thread 환경값을 모두 `1`로
설정하므로, 환경 생성 shell이 끝난 뒤 별도 Python 명령으로 실행해도 같은
계약을 유지한다. MLX adoption이나 실행 속도 성공을 이 문서에서 미리 주장하지
않는다.

## 단계 경계와 fail-closed 규칙

`readiness`와 `new` prediction 흐름은 TEST `label`, `labels_blob`, anomaly
position, evaluator, Exp416/Exp442 evaluation을 읽지 않는다. 먼저 Policy A
`linear_edge_nearest_control`의 finite score·Exp249 후보·scalar `1e-6`·최종
2,218 Hard parity를 확인한다. control이 실패하면 상태를
`control_reproduction_failed`로 기록하고 Policy B/C prediction과 모든 TEST
평가를 생성하지 않는다. Policy B와 C는 각각 독립적으로 19개 affected dataset을
계산하고, 1,098개 unaffected dataset은 frozen Exp249 record를 byte identity로
검증해 import한다. 세 policy의 57 recompute unit이 모두 성공하고 prediction
publication SHA가 동결된 뒤에만 `evaluate`를 허용한다.

publication에는 세 개의 label-free prediction, 정책 audit, manifest와 artifact
SHA map만 둔다. evaluation에는 confusion, macro/micro, rescued/regressed,
strong-dataset loss, paired bootstrap와 성공표를 별도 기록한다. 같은 TEST 결과를
보고 정책 하나를 Exp443 승자로 선택하지 않으며, Exp404/Exp416은 계속 verified
reference다.

모든 governing artifact는 다음 flat governance block을 유지한다.

```text
official_run_purpose = controlled_missing_normal_train_policy_comparison
adaptive_followup = true
independent_validation = false
external_generalization_supported = false
confirmatory_statistical_claim = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
test_informed_reselection_allowed = false
```

결과가 성공하더라도 독립 검증·외부 일반화·운영 경보·과학적 champion을 뜻하지
않는다. 결과 기록 후에는 Exp444를 만들거나 시작하지 않고 방향 논의를 위해
멈춘다.
