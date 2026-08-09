# Exp443 구현 검증 보고서

## 핵심 판단

Exp443의 제품 코드는 구현·회귀·실데이터 label-free 대조 검증을 마쳤다.
다섯 명령 `readiness`, `new`, `resume`, `inspect`, `evaluate` 외에는 열지
않았고, worker·정책·alpha·kernel·seed·입력·출력 경로를 실행 시점에 바꾸는
옵션은 없다. 공식 root는 아직 없고 official prediction 및 TEST-label 평가는
시작하지 않았다.

이 보고서 시점의 상태는 `implementation_verified_awaiting_fresh_review`다.
fresh Sol High 검토가 `ship`을 주기 전에는 official readiness를 실행하지 않는다.

## 구현 경계

- RED CLI 계약 commit: `8d4b345d`
- CLI/runtime/wrapper source commit: `41019f1c`
- source-bound code-map commit: `639400a6`
- 고정 wrapper:
  `scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py`
- 고정 interpreter:
  `/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python`
- explicit resume 이름: `attempt_[0-9]{4}_antigravity`
- official CPU topology: logical CPU 12, outer worker 11, inner numerical thread 1
- MLX/MPS 판단: `inapplicable_official_cpu_transformer`; aeon MiniROCKET을
  GPU용으로 다시 쓰지 않는다.

`inspect`는 complete checkpoint에서 세 label-free prediction publication을
create-only로 만들거나 기존 bytes를 재검증한다. `evaluate`는 그 publication
검사를 먼저 통과한 뒤에만 `evaluation_antigravity`를 lazy import한다. 다른 네
명령은 evaluator를 import할 수 없다.

## 자동 검증

- CLI + runtime focused: 19/19 passed.
- 전체 Exp443: 167 passed, 실데이터 opt-in 2개 expected skip.
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

## fresh review 뒤의 공식 첫 명령

fresh review가 clean HEAD를 `ship`한 뒤 그 정확한 40자 commit을 사용한다.

```bash
EXP443_REVIEWED_COMMIT_ANTIGRAVITY="$(git rev-parse HEAD)"
/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python \
  scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py \
  readiness --reviewed-commit "$EXP443_REVIEWED_COMMIT_ANTIGRAVITY"
```

공식 실행이 완료되면 prediction을 `inspect`로 동결·재검증하고, 사용자가 완료를
알린 뒤 한 번만 `evaluate`한다. 결과 기록 후 Exp444를 자동 시작하지 않고 방향
논의를 위해 멈춘다.
