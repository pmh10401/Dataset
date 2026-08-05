# Exp426 검증 기준 결과 완성 사전등록

## 1. 실험의 뜻

Exp426은 새 탐지 모델을 시험하는 실험이 아니다. Exp424에서 이미 정상적으로
완료되고 검증된 983개 결과를 그대로 가져오고, 학습 근거가 부족해 끝난 134개는
현재 DB의 정상 TRAIN 데이터만으로 다시 확인한다. 이렇게 1,117개 전체를 빠짐없이
채운 결과를 이후 연구가 넘어야 할 비교 기준으로 고정한다.

이 결과는 연구 검토용이다. 자동 경보로 사용할 수 없고, 성능 개선을 주장하지
않는다. TEST 정답과 실제 이상 위치는 prediction과 manifest의 SHA256이 고정된
뒤 별도 평가에서만 연다.

## 2. 고정 입력

- DB: `/Users/mac/Documents/Dataset/univariate_ts.db`
- DB SHA256:
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Exp424 출처:
  `/Users/mac/Documents/Dataset/exp424_mps_score_cache_20260802-153831.checkpoint`
- Exp424 root provenance SHA256:
  `f7526bf638f4c1051a5970271bdbbccd753986f7d067ec1260817da884c1b757`
- Exp424 progress SHA256:
  `5e9aa4a4b7240ac89c078bdc1dce4b41bda6b6dbc505dd9ad49782556da7bd84`
- Exp424 record-set SHA256:
  `d74a0def7788db89a1d3e32f46662abf07c44c7cbbce5e02612b510e2fdf4258`
- Exp291 prediction SHA256:
  `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848`
- Exp291 manifest SHA256:
  `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e`
- Exp292 prediction SHA256:
  `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b`
- Exp292 manifest SHA256:
  `0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0`
- fold contract SHA256:
  `ed67b46a7a777ba3e6ef319344338d5b2f72aa4bbdec678af1db7ef6f29e0998`
- 실행 코드는 readiness와 실제 실행 시점의 동일한 clean Git commit으로 고정한다.
- 실행 환경의 Python, NumPy, Numba, Torch 버전은 checkpoint context와 최종
  manifest에 함께 고정한다.

동결 lane 수는 Hard 2,218, Standard 806, Priority 31,
automatic-normal 35다. Exp424 출처 구성은 정확히 전체 1,117개, 가져오기 983개,
TRAIN 재확인 134개여야 한다.

## 3. 사용할 정보와 금지할 정보

prediction 단계에서 사용할 수 있는 정보는 다음뿐이다.

- 검증된 Exp424 record bytes와 그 SHA256
- 정상 TRAIN 값
- 데이터셋 이름
- TRAIN/TEST 개수와 길이 같은 정답 없는 고정 메타데이터
- 동결 lane 좌표

다음 정보는 prediction이 고정될 때까지 사용할 수 없다.

- TEST 정답과 실제 이상 위치
- 과거 family별 TEST 성능
- 혼동행렬, precision, recall, F1, human-assisted 지표
- TEST 결과를 보고 다시 고른 alpha, threshold, source 또는 budget
- tail replacement

983개는 새 CPU 결과로 가장하지 않고 `exp424_import`로 기록한다. 134개는 후보
점수를 만들지 않고 각 TRAIN 길이 그룹의 서로 다른 정상 파형 수가 부족한지만
확인한다. 모든 유효 그룹이 `oof_identity_insufficient` 또는 전처리 단계 판단
보류이며, 적어도 한 그룹이 `oof_identity_insufficient`일 때만 후보 없는
`verified_train_abstention`을 기록한다. 하나라도 학습 가능해지면 Exp426을
중단하고 새 실험 번호로 분리한다.

## 4. 계산 및 병렬 처리 고정

- backend는 CPU다.
- 외부 worker는 정확히 12개다. CLI에 worker 선택 옵션을 두지 않는다.
- 미리 준비하는 작업은 최대 24개다.
- 각 worker의 OMP, OpenBLAS, MKL, VecLib, NumExpr, Numba thread는 1개다.
- 부모 프로세스만 DB snapshot, 출처 descriptor, checkpoint와 로그를 소유한다.
- worker에는 경로나 DB 연결이 아니라 검증된 bytes 또는 `RawTrainDataset`만 준다.
- worker 완료 순서와 관계없이 데이터셋 이름 순서로 저장한다.

Exp426에는 신경망 학습, 점수 행렬 또는 tensor fusion 단계가 없다. 같은 계산을
MLX로 옮길 대상이 없으므로 MLX를 채택하지 않는다. MPS로 새 점수를 계산하지도
않는다. readiness에서 1-worker와 12-worker 결과 SHA256이 같아야 하며 속도는
기록만 하고 worker 수 선택에 사용하지 않는다.

## 5. 고정 경로와 명령

- output root:
  `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805`
- checkpoint root:
  `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint`
- readiness artifact:
  `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_readiness_20260805.json`
- run identity: `exp426-verified-baseline-antigravity-20260805`

허용 명령은 `readiness`, `new`, `resume`, `inspect` 네 개뿐이다. `new`는 두
root가 모두 없고, 현재 clean Git commit과 같은 commit으로 만든 readiness가
검증될 때만 실행한다. 검증한 commit은 실행 함수에 전달되고, 실행 함수가 clean
HEAD와 readiness 원본을 다시 대조한다. `resume`은 같은 지문의 재개 가능한
미완료 root만 잇는다.
완료된 `resume`과 `inspect`는 DB와 Exp424 출처를 열지 않는다. 임의 경로와
덮어쓰기는 허용하지 않는다.

## 6. 진행 기록과 재개

프로그램은 Codex의 반복 감시 없이 다음 파일을 직접 기록한다.

- `progress.json`: 원자적으로 교체되는 현재 상태
- `events.jsonl`: fsync되는 상태·데이터셋 완료 기록
- `parent.log`: 사람이 읽는 상태 기록
- `run.pid`: PID, run identity, active 상태

상태 순서는 새 실행에서 다음과 같다.

```text
initializing -> validating_source -> importing -> verifying_train ->
finalizing -> complete_review_only
```

사용자가 정상적으로 중단하면 상태를 `interrupted`로 기록하고 오류 수를 늘리지
않는다. 이 상태와 강제 종료로 마지막 안전 상태에 머문 root만 재개할 수 있다.
계산·무결성 오류로 `failed`가 된 root는 재개하지 않고 그대로 보존한다.

로그에는 현재 데이터셋, TRAIN/TEST 개수와 길이, CPU/12 workers, 완료/전체,
가져오기/재확인/판단 보류/오류 수, 처리속도, 경과시간, ETA, checkpoint와 최신
오류를 기록한다. 30초 heartbeat는 프로그램이 직접 갱신한다. record commit,
완료 event, progress 교체 사이에서 중단되어도 완료 record를 다시 계산하거나
완료 event를 중복시키지 않는다.

## 7. 성공과 실패 기준

성공은 다음을 모두 만족하는 경우뿐이다.

- completed 1,117
- imported 983
- verified TRAIN abstention 134
- failed 0
- 누락·중복·알 수 없는 origin/status 0
- 후보 좌표가 정렬·고유하며 TEST 개수 범위 안이고 동결 lane과 겹치지 않음
- source, checkpoint, runtime event 수와 SHA256이 서로 일치함
- current dataset이 `null`, PID가 inactive임
- TEST 값·정답 접근 0
- 결과를 독립 재파싱해 같은 prediction 및 artifact SHA256을 얻음

입력 지문 변경, 983/134 불일치, TRAIN에서 학습 가능한 그룹 발견, label 접근,
worker 결과 불일치, checkpoint 손상 또는 저장 실패가 하나라도 있으면 실패다.
실패 root를 삭제하거나 덮어쓰지 않는다.

## 8. 결과와 관리 표시

완성 결과는 output의 `publication_antigravity` 디렉터리에 한 번에 공개한다.

- `prediction_antigravity.jsonl`
- `dataset_summary_antigravity.jsonl`
- `import_ledger_antigravity.jsonl`
- `train_abstention_ledger_antigravity.jsonl`
- `backend_decision_antigravity.json`
- `manifest_antigravity.json`
- `status_antigravity.json`
- `artifact_sha256_antigravity.json`

manifest와 status에는 다음을 고정한다.

- `review_only = true`
- `operational_alert_eligible = false`
- `contains_test_labels = false`
- `test_labels_opened = false`
- `workers = 12`

artifact SHA map은 자기 자신의 해시는 순환 때문에 제외한다고 명시하고 나머지
결과 파일 전부의 SHA256을 기록한다. manifest는 그보다 먼저 만들어지는 과학적
결과 다섯 파일의 SHA256을 기록한다.

## 9. prediction 이후 경계

prediction과 manifest SHA256이 고정되면 Exp426 prediction 단계는 닫힌다.
그 다음에만 별도 프로그램이 TEST 정답을 열 수 있다. 평가는 새로 찾은
데이터셋, 놓친 데이터셋, false alarm, 기존 강한 데이터셋의 악화, 전체 paired
결과와 bootstrap 신뢰구간을 함께 비교한다. 그 결과를 보고 Exp426의 설정을
바꾸지 않는다. 개선 연구는 Exp427 이상의 새 번호와 새 사전등록을 사용한다.
