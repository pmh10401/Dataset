# Exp444 TRAIN-only 안정성 정밀화 실행 안내서

## 검토 후 고정 상태

아래 명령은 승인 설계와 이 사전등록의 bytes, clean reviewed commit, 그리고 별도
Exp444 환경에 묶인다. 공식 실행은 fresh Sol/High final review가 `ship`이고 미해결
Critical/Important finding이 없을 때까지 금지된다. `readiness`와 `new`는 모두
official root를 만들기 전에 source values를 materialize하고 session을 닫는다.

```text
DESIGN_SOURCE=docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md
DESIGN_SHA256=1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8
PREREGISTRATION=docs/EXP444_PREREGISTRATION_antigravity_KO.md
RUN_ROOT=/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810
READINESS=$RUN_ROOT/readiness_antigravity.json
ATTEMPTS=$RUN_ROOT/attempts_antigravity
ATTEMPT=$ATTEMPTS/attempt_0000_antigravity
CHECKPOINT=$ATTEMPT/checkpoint_antigravity
PROGRESS=$ATTEMPT/progress_antigravity.json
EVENTS=$ATTEMPT/events_antigravity.jsonl
TEXT_LOG=$ATTEMPT/text_antigravity.log
PID_FILE=$ATTEMPT/pid_antigravity.json
PUBLICATION=$ATTEMPT/publication_antigravity
EVALUATION=$RUN_ROOT/evaluation_antigravity
```

실행 process는 `progress_antigravity.json`, append-only
`events_antigravity.jsonl`, `text_antigravity.log`, `pid_antigravity.json`을
직접 원자적으로 갱신한다. progress에는 PID/active 상태, 현재 dataset/candidate/view,
TRAIN·TEST 크기와 raw length, 단계/backend, 완료/전체와 import/recompute/success/error
counter, 처리속도·경과시간·ETA, checkpoint, 마지막 오류를 쓴다. checkpoint는
30개 후보의 ordered prefix, context/state hash와 오류·중단 상태를 보존한다. 사용자는
실행 중 이 파일을 수정하지 않으며 Codex는 장기 process를 반복 polling하지 않는다.

## 고정된 다섯 명령 (final review 전 실행 금지)

`REVIEWED_COMMIT`은 명령 호출자가 넣는 runtime argument이며 문서나 publication에
placeholder로 저장하지 않는다. 아래 고정 명령은 clean reviewed implementation과
fresh review가 `ship`으로 끝난 뒤에만 실행한다.

```bash
bash scripts/create_exp444_environment_antigravity.sh
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py new --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py resume --reviewed-commit REVIEWED_COMMIT --attempt attempt_0000_antigravity
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py inspect
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp444_train_only_stability_precision_gate_antigravity.py evaluate
```

`readiness`는 설계·사전등록 SHA, clean reviewed HEAD, 고정 input bytes와 30개
좌표, 1,117개 population, environment lock, backend decision, 1-worker/11-worker
parity, zero label access, 그리고 빈 `RUN_ROOT`를 확인한다. `new`는 같은 process에서
source provenance를 다시 확인하고 values-only source session을 닫은 뒤에만 absent
root를 만들며 정상 TRAIN control replay가 Exp442와 정확히 맞은 뒤 omission view를
계산한다. `resume`은 하나의 `attempt_####_antigravity` basename만 받고, interrupted
checkpoint의 read-only authority 확인과 source materialization/close를 모두 끝낸 뒤에만
progress/PID/checkpoint owner를 rebind한다. complete checkpoint의 publication 실패에는
checkpoint/progress/PID/events/text bytes를 수정하지 않는 source-free finalization-only
resume만 허용한다.

성공한 `new`/`resume`는 먼저 label-free publication을 create-only로 동결한다.
publication에는 1,117개 summary, 30개 원본 좌표, mask/view/score evidence,
stable·unstable lane, source/control/backend audit, checkpoint/progress/PID identity,
manifest와 artifact SHA map이 포함된다. `inspect`는 worker·SQLite·TEST label을
열지 않고 publication/checkpoint/source bytes와 lane conservation (`stable + unstable = 30`)
만 다시 확인한다. `evaluate`만 publication/hash를 두 번 확인한 뒤 TEST label을
읽어 `descriptive_complete_no_promotion` evaluation artifact를 만들며 prediction
publication은 바꾸지 않는다.

## 경계와 실패 보존

prediction/readiness/checkpoint/publication은 TEST `label`, `labels_blob`, 실제 이상
위치, 과거 TEST metric, family 성능, TEST missingness gate, candidate index 위치와
tail 규칙을 읽지 않는다. 원본 Exp442 30개 control 또는 source/hash가 다르면
stability를 실행하지 않고 오류 root를 보존한다. TEST label 접근이 감지되면 즉시
실패하고 publication을 만들지 않는다. worker/checkpoint/log/publication 오류는
dataset·candidate·view와 최신 오류를 기록하며 자동 수리·덮어쓰기를 하지 않는다.
Exp440/442/443 root와 artifact는 절대 수정하지 않는다.

## 운영자 확인 순서

1. 환경 생성과 `readiness`의 clean reviewed commit·빈 root·zero-label 증거를 확인한다.
2. 한 개의 `new` process만 시작하고, 오래 걸리면 위 고정 경로의 progress/text/PID를
   사용자가 확인한다. Codex는 반복 polling하지 않는다.
3. process가 complete checkpoint와 create-only publication을 남긴 뒤 `inspect`를
   실행한다.
4. prediction SHA와 label-access count가 동결된 뒤 `evaluate`를 한 번 실행한다.
5. 결과는 설명용으로만 기록하고 Exp444 내부에서 규칙을 다시 선택하지 않는다.

`inspect`는 publication, checkpoint, progress, PID, events, text, prediction bytes를
변경하지 않는다. `evaluate`는 publication을 먼저 완전히 inspect한 뒤에만 evaluator를
lazy import하며, evaluator도 label access 전 독립된 두 번째 inspection을 수행한다.
Wrapper 실패는 stdout에 하나의 canonical JSON error object만 남긴다.

이 runbook은 구현된 고정 인터페이스를 설명할 뿐 준비 완료나 공식 실행 증거가 아니다.
Codex는 장기 process를 polling하지 않는다.
