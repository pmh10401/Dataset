# Exp443 구현 검증 보고서

## 핵심 판단

Exp443의 제품 코드는 구현·회귀·실데이터 label-free 대조 검증을 마쳤다.
다섯 명령 `readiness`, `new`, `resume`, `inspect`, `evaluate` 외에는 열지
않았고, worker·정책·alpha·kernel·seed·입력·출력 경로를 실행 시점에 바꾸는
옵션은 없다. 공식 root는 아직 없고 official prediction 및 TEST-label 평가는
시작하지 않았다.

이 보고서 시점의 상태는 `implementation_verified_awaiting_scoped_rereview`다.
두 차례 Sol High 검토에서 나온 문제를 수정했으며, 수정된 clean HEAD가 다시
`ship`을 받기 전에는 official readiness를 실행하지 않는다.

## 구현 경계

- RED CLI 계약 commit: `8d4b345d`
- review-fix RED commits: `943e8ce0`, `0639db79`, `d9907603`
- reviewed-fix source commit: `e463f26e975376f3507306408ca04490eb588df8`
- source-bound code-map commit: `58e8267e2b2a245341f77bf0f7b7b0f943455855`
- 고정 wrapper:
  `scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py`
- 고정 interpreter:
  `/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python`
- explicit resume 이름: `attempt_[0-9]{4}_antigravity`
- official CPU topology: logical CPU 12, outer worker 11, inner numerical thread 1
- MLX/MPS 판단: `inapplicable_official_cpu_transformer`; aeon MiniROCKET을
  GPU용으로 다시 쓰지 않는다.

`new` 또는 `resume`이 57개 계산과 세 prediction을 끝내고 progress를
`complete`로 기록한 다음 label-free publication을 create-only로 만든다.
publication 중 실패하면 완료 progress를 오류로 되돌리지 않고 DB pin을 남긴다.
같은 attempt의 `resume`은 complete checkpoint와 progress/PID/events/text를
읽기 전용으로 검증하고, 그 bytes나 소유 PID를 바꾸지 않은 채 prediction/control
authority를 replay해 publication만 재시도한다. publication이 성공한 뒤에만 DB
pin을 정리한다. `inspect`는 기존 publication bytes만 읽어 재검증하며 아무
파일도 만들지 않는다. `evaluate`는 그 검사를 먼저 통과한 뒤에만
`evaluation_antigravity`를 lazy import한다. 다른 네 명령은 evaluator를 import할
수 없다.

## 첫 검토 지적과 수정

clean HEAD `40ddc4e5`의 첫 Sol High 검토는 `fix-first`였다.

- `inspect`가 publication을 만드는 문제: 읽기 전용 검사로 분리했다.
- 잘못된 Python으로 wrapper를 실행할 때 traceback이 나오는 문제: stderr에
  단일 `exp443.wrapper-status.v1` JSON만 기록하도록 고쳤다.
- 상태 문서의 검토 대상이 오래된 commit을 가리키는 문제: 고정된 과거 HEAD
  문자열 대신 현재 clean HEAD를 재검토 대상으로 삼도록 바꿨다.

추가 자체 검토에서 publication 실패 전에 DB pin을 지울 수 있는 순서 문제도
발견했다. 이제 `complete progress → publication 성공 → pin 정리` 순서이며,
publication이 실패하면 같은 attempt를 명시적으로 resume할 수 있다.

clean HEAD `5eb7b39a`의 두 번째 Sol High 검토도 `fix-first`였다. DB pin은
남았지만 기존 progress reopen이 `complete`를 거부해 실제 publication 재시도가
불가능하다는 지적이었다. `d9907603`에서 실패 재현 테스트를 먼저 고정했고,
`e463f26e`에서 complete checkpoint용 읽기 전용 finalization-resume을 추가했다.
실제 제어 흐름 테스트는 첫 publication 실패에서 pin 정리값 `false`, 같은
attempt 재시도 성공에서 `true`, 네 progress/log bytes 불변을 확인한다.

## 자동 검증

- CLI + runtime + progress focused: 33/33 passed.
- 전체 Exp443: 174 passed, 실데이터 opt-in 2개 expected skip.
- inherited Exp249/250 MiniROCKET regression: 13/13 passed.
- compileall: passed.
- `git diff --check`: passed.
- wrapper external-CWD smoke: passed.
- code map: 20 nodes / 61 edges / 5 flows / unknown 0, HTML embedded JSON
  byte-identical.

## 실제 Policy A 대조

19개 결측 정상 TRAIN 데이터셋 전체를 pinned source로 다시 계산했다.

- 8/8 control tests passed in 53.843 s.
- score comparisons: 76.
- 최대 절대 수치 차이: `4.5397329895280336e-08` (`1e-6` 이내).
- candidate mismatch: 0.
- control status: `control_reproduction_passed`.
- routing identity: 1,117 datasets / promotion 137 / pre-exact 2,253 /
  exact-normal removal 35 / final Hard 2,218.
- TEST-label access: 0.

## CPU 1개와 worker 11개 동일성

같은 19개 Policy A 작업을 순차와 11-worker로 각각 실행했다.

- sequential: 46.705213875 s.
- parallel: 7.030636542 s.
- informative speed ratio: 6.643098900x.
- sequential/parallel semantic SHA256:
  `317368dfef5e8a6653a70343ba7e428abf50ba0c187bb4b2bab8107ab954c41e`.
- prediction bytes SHA256:
  `9766c1939fd7c45473ccd35230ca489e2bedc35dabc58080eb26582e4e51e10a`.
- semantic equality: true.
- TEST-label access: 0.

속도 값 자체는 장비 상태에 따라 바뀔 수 있는 참고값이다. 결과 bytes와 후보
동일성이 통과 조건이며, 이번 재실행에서도 두 경로가 동일했다.

## 아직 하지 않은 일

- `/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809`
  root 생성
- official readiness
- 57-unit A/B/C prediction
- prediction publication freeze
- TEST-label query
- retrospective evaluation
- Exp444 설계 또는 실행

## scoped re-review 뒤의 공식 첫 명령

scoped re-review가 현재 clean HEAD를 `ship`한 뒤 그 정확한 40자 commit을
사용한다.

```bash
EXP443_REVIEWED_COMMIT_ANTIGRAVITY="$(git rev-parse HEAD)"
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python \
  scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py \
  readiness --reviewed-commit "$EXP443_REVIEWED_COMMIT_ANTIGRAVITY"
```

공식 `new` 또는 `resume`이 성공하면 prediction과 publication은 이미 동결되어
있다. 그다음 `inspect`로 bytes를 읽기 전용 재검증하고, 사용자가 완료를 알린 뒤
한 번만 `evaluate`한다. 결과 기록 후 Exp444를 자동 시작하지 않고 방향 논의를
위해 멈춘다.
