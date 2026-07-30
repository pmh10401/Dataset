# Exp417 실행 런북

## 범위와 절대 경계

Exp417은 정상 TRAIN만으로 causal multi-horizon CNN을 학습하고, 동결된
TRAIN qualification을 통과한 뒤에만 TEST의 값과 instance identity를
읽어 비공식 Shadow 후보를 만든다. TEST 라벨, 실제 이상 위치, 과거
TEST/family 성능은 열기·조회·해시·stat·이름 참조를 포함해 전부 금지한다.
기존 Hard/Standard/Priority/automatic-normal lane은 읽기 전용이며 승격,
tail replacement, top-N 정리는 없다.

이 런북의 smoke는 정식 prediction도, 정식 checkpoint도 아니다.
`official_prediction=false`, `publication_eligible=false`,
`contains_test_labels=false`, `official_checkpoint_reusable=false`가
고정된다. 전체 1,117개 실행은 이 문서의 검증과 사용자의 정확한 별도
승인을 모두 받은 뒤에만 실행한다.

자동 감시, watch, while-loop polling, cron, keepalive를 만들지 않는다.
상태 확인이 필요하면 사용자가 요청한 시점에 한 번만 확인한다.

## 1. 환경과 clean commit

```bash
cd "/Users/mac/Documents/Time Series/.worktrees/exp409-prospective-readiness"
git status --short --branch
git rev-parse HEAD
.venv/bin/python --version
.venv/bin/python -c 'import numpy, numba, torch; print(numpy.__version__, numba.__version__, torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
```

실제 smoke와 전체 실행은 clean commit에서만 허용된다. Apple M2 Max
호스트에서는 MPS가 built/available이며 동기화된 Conv1d
forward/backward probe도 통과했다. 다만 일부 Codex 격리 프로세스는
Metal을 숨길 수 있으므로 실제 MPS probe/benchmark는 범위를 좁힌 호스트
프로세스 하나로만 실행한다. Task 7의 동일 입력·동일 seed 감사에서 후보
identity, 최대 정규화 오차 `4.0948700075205124e-07`, Spearman
`0.9999999999845686`은 통과했지만, MPS speedup은 `0.6709759390840965x`로
동결 gate `1.1x`에 못 미쳤다. 따라서 Exp417의 canonical 실행 backend는
PyTorch CPU다. Numba score 집계는 후보 identity가 일치하며 synthetic
speedup `32.49217872476399x`인 감사 근거만 사용한다. MLX는 Exp417에서
제외한다.

Task 7 로컬 감사 파일은 다음 위치에 있어야 하며 GitHub에 올리지 않는다.

```bash
shasum -a 256 \
  "/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow/task7_backend_audit.json"
```

기대 SHA256은
`f77efa941f6b33fe6687b8ac0c2b01c152dac9c15e8b0966e3d9bb41b119b592`다.
파일이 없거나 다르면 smoke와 전체 실행을 시작하지 않는다.

## 2. 전체 합성·회귀 검증

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp417_causal_multihorizon_shadow/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests -p "test_*.py"
.venv/bin/python -m compileall -q \
  experiments/exp417_causal_multihorizon_shadow \
  scratch/run_exp417_causal_multihorizon_shadow.py \
  scratch/smoke_exp417_causal_multihorizon_shadow.py
git diff --check
```

하나라도 실패하면 실제 값 smoke나 전체 실행으로 진행하지 않는다.

## 3. 동결된 비라벨 입력 SHA256

아래 다섯 파일만 해시한다. TEST 라벨 artifact는 절대로 해시, stat,
open하지 않는다.

```bash
shasum -a 256 \
  "/Users/mac/Documents/Dataset/univariate_ts.db" \
  "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json"
```

기대 순서는 다음과 같다.

```text
8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848
761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e
f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b
0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0
```

## 4. 비공식 label-free readiness smoke

다음 명령은 정상 TRAIN의 normal-count와 raw length만으로 low/middle/high
count×length strata를 만들고 각 strata에서 사전식 tie-break로 정확히
3개를 선택한다. dataset 이름을 직접 지정하는 옵션은 없다. 출력 경로는
매번 존재하지 않는 새 경로여야 한다.

```bash
.venv/bin/python scratch/smoke_exp417_causal_multihorizon_shadow.py \
  --database "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --exp291-prediction "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  --exp291-manifest "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  --exp292-prediction "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  --exp292-manifest "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json" \
  --smoke-datasets 3 \
  --output-dir "/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow_smoke_20260729"
```

출력에는 `EXP417_NON_OFFICIAL_SMOKE.json` 하나만 존재한다. 정식 여덟
파일 이름, 정식 manifest, 정식 checkpoint record는 생성하지 않는다.
smoke의 status와 workload는 TEST 라벨 지표가 아니며, precision, recall,
F1, confusion matrix를 계산하거나 기록하지 않는다. smoke 전후에 3절의
다섯 입력 SHA256이 동일해야 한다.

smoke JSON은 정확히 15개 top-level key를 사용한다. disk 근거는 단일
`estimated_full_disk_bytes` 값이 아니라 `disk_projection`에 기록한다.
`measured_sample_checkpoint_bytes`는 Task 8 canonical JSON schema로
메모리에서 직렬화한 smoke 결과의 실제 byte 수다. 이를 1,117개에 선형
환산한 `failed_terminal_projection_bytes`와, 정확한 전체 TEST identity
수 및 정상 TRAIN 수에 all-candidate/every-series-unique-group schema
비용을 더한 `candidate_bearing_conservative_upper_bytes`를 별도로
해석한다. 전자는 실패 terminal 중심 projection이고 후자는 의도적으로
느슨한 보수 상한이다. 둘을 하나의 정확한 전체 실행 disk 예측처럼
보고하지 않는다.

기존 smoke가 terminal stage/reason을 보존하지 않았다면 `oof / "x"`라는
가장 짧은 유효 terminal schema로 checkpoint record를 메모리에서
재구성하고 `sample_basis=reconstructed_minimum_terminal_schema`로
명시한다. 이 재구성은 공식 checkpoint를 만들거나 재사용하지 않는다.

## 5. checkpoint 검사와 resume

정식 checkpoint는
`/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow.checkpoint`
만 사용한다. root provenance에는 Git commit, 다섯 입력 SHA256, frozen
config, preprocessing/fold/model/backend contract, runtime version, 정확한
1,117 dataset universe가 함께 묶인다. 기존 경로는 이 provenance와
dataset universe가 정확히 일치할 때만 열리고, 완료된 canonical record만
재사용된다.

writer는 checkpoint root 디렉터리 inode의 exclusive lock을, 검사·resume
reader는 shared lock을 잡는다. 따라서 active writer가 끝나기 전에는 다른
writer나 reader가 중간 record/progress 상태를 관찰하지 않는다.

새 root는 예측 불가능한 `.exp417-init-<uuid>` staging 이름으로 만든다.
생성 직전·직후 parent inventory를 descriptor 기준으로 비교해 새 inode가
정확히 하나이고 그 inode가 staging 이름에 결합된 경우에만 내용을 쓴다.
교체나 다른 동시 변경이 보이면 초기화를 중단하며 staging 또는 이동된
inode를 자동 삭제하지 않는다. 최종 no-replace rename 뒤에도 root와
datasets inode/inventory, provenance/progress bytes와 inode를 모두 다시
검증한다. publish 뒤 검증이 실패하면 canonical root를 recovery 이름으로
원자적으로 옮겨 공식 이름을 비운다. 이는 우발적 동시 실행과 같은 parent
안의 pathname 교체를 fail-closed로 막는다. 다만 portable POSIX에는
`mkdir`와 생성 inode의 descriptor 획득을 하나로 묶는 API가 없으므로,
임의의 다른 디렉터리까지 감시·이동할 수 있는 악의적 동일 UID 프로세스는
운영 위협 모델 밖이다. checkpoint parent는 해당 실행 계정만 쓸 수 있게
권한을 제한해야 한다.

중단되면 정식 output이 아직 없는지, Git과 입력 SHA가 그대로인지 확인한다.
다음 실행에서 동일 provenance의 root에 hard-crash transaction marker,
임시 파일, record/progress 불일치가 발견되면 구현이 그 root inode 전체를
같은 parent의 고유한 `.exp417_causal_multihorizon_shadow.checkpoint.quarantine.*`
경로로 원자적으로 격리하고, 보존 경로를 포함한 오류로 중단한다. 다른
provenance의 foreign root는 이동하거나 수정하지 않는다. 격리본의 파일을
부분 삭제·병합·원래 경로로 복사하지 말고 그대로 보존한 뒤 원인을
검사한다. 다시 별도 승인을 받은 다음 호출에서만 새 checkpoint root를
생성한다. smoke 출력은 정식 checkpoint로 재사용할 수 없다. 이미 존재하는
정식 output을 덮어쓰거나 같은 경로에 재실행하지 않는다.

transaction rollback이나 원자적 progress 교체 과정에서 밀려난 기존
record/progress는 checkpoint parent의 고유한
`.exp417_causal_multihorizon_shadow.checkpoint.rollback-*.json` recovery
capsule로 이동한다. commit 전 임시 payload나 경쟁 target을 안전하게
원위치로 돌릴 수 없으면
`.exp417-checkpoint.rollback-temporary.*.bin` 또는
`.exp417-checkpoint.rollback-publish-target.*.bin`으로 보존한다. 이
파일들 외에 실패한 최종 root target은
`.exp417-checkpoint.rollback-root-publish-target.*.entry`로 보존한다.
recovery entry와 실패한 `.exp417-init-*` staging은 자동 삭제하지 않는다.
정상 실행 중에도 이전 progress 보존본이 생길 수 있다. 실행 중 이름을
바꾸거나 지우지 말고, 실행·검증 완료 뒤에도 provenance와 최종 산출물
보존 정책에 따라 별도 보관한다.

## 사용자의 별도 명시적 승인 후에만 실행

아래 명령은 사용자가 정확히
`Exp417 1,117개 전체 label-free 실행 승인`이라고 지시한 뒤에만 한 번
실행한다.

```bash
.venv/bin/python scratch/run_exp417_causal_multihorizon_shadow.py \
  --database "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --exp291-prediction "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl" \
  --exp291-manifest "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json" \
  --exp292-prediction "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl" \
  --exp292-manifest "/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json" \
  --checkpoint-dir "/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow.checkpoint" \
  --output-dir "/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow_official"
```

공식 호출 순서는 clean Git → package label guard → frozen hashes/lanes →
checkpoint → runner → input snapshot 재검증 → render/verify → atomic
publication이다. 실패하면 이후 단계와 정식 출력 publication은 실행되지
않는다. 전역 신규 Shadow 비율이 `0.01`을 넘거나 dataset 실패가 하나라도
있으면 공식 여덟 파일을 발행하지 않는다.

모든 checkpoint, prediction, metrics, report와 Task 7 audit는 반도체
설비 데이터에서 파생된 로컬 전용 artifact다. GitHub, 이슈 첨부, PR,
클라우드 저장소로 내보내지 않는다. 이후 개선은 Exp417 결과를 보고
threshold/source/budget을 바꾸는 방식이 아니라 별도 사전등록 Exp418로
분리한다.
