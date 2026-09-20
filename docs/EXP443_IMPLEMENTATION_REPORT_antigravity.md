# Exp443 구현 검증 보고서

## 핵심 판단

Exp443의 제품 코드는 구현·회귀·실데이터 label-free 대조 검증을 마쳤다.
다섯 명령 `readiness`, `new`, `resume`, `inspect`, `evaluate` 외에는 열지
않았고, worker·정책·alpha·kernel·seed·입력·출력 경로를 실행 시점에 바꾸는
옵션은 없다. clean HEAD `081b94b0`이 Sol High `ship`을 받은 뒤 official
readiness, 57-unit label-free prediction, publication freeze, read-only inspect,
post-freeze TEST-label 평가까지 완료했다.

이 보고서 시점의 상태는 `official_evaluated_paused_for_direction`이다. 세 정책 중
leave-one-out peer median만 기존 미탐 이상 1개를 오경보 없이 복구했지만 paired
bootstrap 95% 구간의 하한이 micro와 macro 모두 `0`이어서 사전등록 개선 신호를
통과하지 못했다. winner를 선택하지 않고 Exp444 전에 방향 논의를 위해 멈춘다.

## 구현 경계

- RED CLI 계약 commit: `8d4b345d`
- review-fix RED commits: `943e8ce0`, `0639db79`, `d9907603`
- reviewed-fix source commit: `e463f26e975376f3507306408ca04490eb588df8`
- source-bound code-map commit: `58e8267e2b2a245341f77bf0f7b7b0f943455855`
- first-readiness RED commit: `f4be2c6c`
- first-readiness source fix commit: `16d012d3`
- pre-NumPy-import RED commit: `88a14ea0`
- pre-NumPy-import source fix commit: `93471c0462fa85e33f280b0496d756c62a25d71c`
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

## 첫 official readiness에서 찾은 환경 문제

clean HEAD `e3dfb5aa`의 최종 Sol High 검토는 blocking finding 없이 `ship`이었다.
그 정확한 commit으로 official readiness를 한 번 실행했지만 다음 오류로 즉시
중단됐다.

```text
Exp443 requires one-thread environment controls: BLIS_NUM_THREADS=None
```

이 실패는 official root·attempt·prediction을 만들기 전이며 TEST label 접근도
없었다. 원인은 환경 검사기가 BLIS를 포함한 6개 BLAS/OpenMP 변수를 요구하지만,
CLI가 import하는 runtime의 사전 설정 목록은 BLIS를 빼고 NUMBA를 포함한 6개만
설정한 것이었다. 환경 생성 shell의 BLIS export는 별도 child shell 안에서만
유효하므로 다음 Python 명령으로 이어지지 않는다.

`f4be2c6c`에서 worker/runtime이 BLIS까지 7개를 설정해야 한다는 RED 테스트를
먼저 고정했고, 실제로 `BLIS_NUM_THREADS=None` 한 항목만 실패함을 확인했다.
`16d012d3`에서 runtime 사전 설정과 worker initializer에 BLIS를 추가했다. 수정 후
환경·readiness·runtime·CLI focused 37개 테스트와 실제 공식 Python 환경 검사가
통과했고, `BLIS_NUM_THREADS=1`, outer worker `11`을 확인했다. 점수·후보·정책·
threshold·데이터 처리는 바뀌지 않았다.

첫 수정을 포함한 clean HEAD `1c4a7880`의 Sol High 검토는 `fix-first`였다. CLI가
`reporting → checkpoint → contracts`를 runtime보다 먼저 import해 NumPy가 7개
값 설정 전에 로드된다는 지적이었다. `88a14ea0`에서 별도 fresh Python process의
첫 NumPy import 순간 환경값을 기록하는 RED 테스트를 추가했고, 실제로 7개가 모두
`None`임을 재현했다. `93471c04`에서 공식 wrapper가 CLI를 import하기 전에 7개
값을 모두 `1`로 설정하도록 고쳤다. 이제 fresh process의 첫 NumPy import와 CLI
import 완료 후 값이 모두 `1`이며, child worker도 같은 7개 계약을 사용한다.

## 자동 검증

- environment + readiness + runtime + CLI focused: 38/38 passed.
- 전체 Exp443: 175 passed, 실데이터 opt-in 2개 expected skip.
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

## 공식 실행과 동결 결과

- reviewed commit: `081b94b0485da7e59b1523d481031daf11d3fc65`
- readiness SHA256:
  `aedf2908d71a809d663408dde3fb2167b00a45a1902c7e34a70739a9439944ad`
- PID: `34590`
- 57/57 recompute success, scientific error 0
- elapsed: 49.55091841612011 s
- publication manifest SHA256:
  `0fdfc6c9af65d4b9434c58cc6a34fd03ae1e5917cd97b4e7ae5f68bd801b49aa`
- evaluation SHA256:
  `85a64e626aff1049a588c8013d1b6ff836502b5c0a619246347529236bc5d7a8`
- prediction 이전 TEST-label access: 0
- prediction freeze 이후 label query: 1,117 calls / 195,900 rows

`inspect`가 동결한 policy별 prediction SHA256은 다음과 같다.

| 정책 | SHA256 |
|---|---|
| linear-edge-nearest control | `2974d7da3967653c1530e4c2b830d8e689e8befb99e1339bf78b0dd88c60f965` |
| 결측 정상행 전체 제외 | `a304ba2deba886964e5487e15776e3fd36e0ecb0ea697cba1712107c79c37e3e` |
| leave-one-out peer median | `1e5801fdd9f8eb60e4f1f0ab75f2cec697d7fa953aa1b78ec2efb7b003e1a07d` |

## 공식 평가 결과

혼동행렬 순서는 `[TP, FP, FN, TN]`이다.

| 정책 | 혼동행렬 | 추가 TP/FP | rescued | micro F1 | 판정 |
|---|---:|---:|---:|---:|---|
| 기준/linear control | `[1915,303,2003,191679]` | `0/0` | 0 | 0.624185 | 동일 |
| 결측 정상행 전체 제외 | `[1915,303,2003,191679]` | `0/0` | 0 | 0.624185 | 동일 |
| peer median | `[1916,303,2002,191679]` | `1/0` | 1 | 0.624409 | 좁은 양성 신호 |

peer median은 `MelbournePedestrian_normal_8`의 TEST index `343`을 새로 맞혔다.
이 데이터셋은 기존에 실제 이상 7개를 모두 놓쳐 F1이 0이었으나, 새 정책은 1개를
맞혀 F1 `0.25`, precision `1.0`, recall `1/7`이 됐다. 결측 정상 TRAIN 행 37개 중
36개는 해당 행을 제외한 정상 동료 파형의 median으로 보완했고, 동료 근거가 없는
1개만 제외했다. 다른 1,116개 데이터셋은 악화되지 않았고 추가 FP도 없다.

전체 micro F1 delta는 `+0.0002241836`, macro F1 delta는 `+0.0002238138`이다.
그러나 20,000회 paired-bootstrap 95% 구간은 각각
`[0, 0.0007423410]`, `[0, 0.0006714414]`로 하한이 0이다. 따라서 공식 상태는
`no_retrospective_improvement_signal`이며 scientific winner나 운영 경보로
승격하지 않는다.

## 종료 오류와 보존 상태

publication과 progress가 완성된 뒤 `new` 명령은 `pinned database close failed`로
비정상 종료했다. 그러나 DB pin 파일은 실제로 제거됐고 원본 DB nlink는 `1`이며,
read-only `inspect`와 post-freeze evaluation은 모든 artifact SHA를 검증했다.
과학 결과는 동결해 보존하고, 이 종료 검증 문제는 별도 engineering repair로
다룬다. Exp444는 시작하지 않았다.
