# Exp422 MPS 수치 차이 연구 설계

Date: 2026-08-02  
Status: approved design; implementation pending

## 1. 연구 목적

Exp422는 동결된 Exp417 계산을 CPU와 MPS에서 같은 입력으로 실행했을 때
발생하는 수치 차이를 정량화한다. 목적은 평균 F1을 높이거나 운영 경보를
바꾸는 것이 아니라, MPS를 향후 계산 백엔드로 사용할 수 있는지 수치적으로
판단할 근거를 만드는 것이다.

현재 공식 Exp419는 CPU 4워커로 독립 실행 중이다. Exp422는 새 worktree,
새 브랜치, 새 attempt root와 새 매니페스트를 사용하며 Exp419의 process,
checkpoint, prediction, lane 파일을 읽거나 수정하지 않는다.

## 2. 연구 질문과 단계

연구는 두 단계를 순서대로 수행한다.

### 2.1 1단계: 핵심 연산 parity

동일한 float32 입력·가중치·seed를 사용하여 Exp417의 Causal CNN과 같은
causal Conv1d, GELU, residual, horizon head, loss/reduction의 CPU/MPS
출력을 비교한다. 입력은 고정된 합성 tensor와 정상 TRAIN에서 얻은 길이
stratum별 읽기 전용 tensor를 함께 사용한다.

고정 batch 후보는 `(1, 4, 8, 16, 32)`이고, 모델 seed는 `41701, 41702,
41703`이다. batch 1은 CPU 기준으로 남기며 MPS로 강제하지 않는다. 각
조합은 같은 순서로 CPU를 먼저 실행한 뒤 MPS를 실행하고, MPS 동기화를
호출한 후 결과를 CPU로 복사한다.

각 연산·batch·seed에 대해 다음을 기록한다.

- 모든 출력의 finite 여부
- 최대 절대 오차, 최대 상대 오차, 평균 절대 오차
- 최대 ULP 차이와 오차가 발생한 tensor 위치
- CPU/MPS raw output digest와 candidate-coordinate digest
- 입력 전송, device 계산, 결과 전송, synchronization 포함 총 시간

1단계가 MPS unavailable, 비유한 출력, 허용오차 초과, 후보 좌표 불일치,
또는 실행 오류로 끝나면 2단계는 실행하지 않고 실패 원인만 매니페스트에
남긴다.

### 2.2 2단계: 실제 Exp417 score parity

1단계가 모든 필수 조합에서 통과한 경우에만, 정상 TRAIN 표본에서 동결된
Exp417 batched benchmark를 실행한다. 표본 선택 규칙은 기존 Exp421의
읽기 전용 exact-length TRAIN selector를 재사용하며, 길이·TRAIN 행 수
stratum을 사전에 고정하고 digest를 기록한다. 표본은 최대 64 series와
256 MiB로 제한한다.

2단계는 다음을 CPU/MPS로 비교한다.

- frozen model forward 출력
- OOF residual과 calibration 입력
- horizon score와 정규화 오차
- 동결 alpha에서 candidate index와 tie-aware rank
- synthetic qualification의 결과와 evidence digest

Exp421의 모델 seed·perturbation seed·horizon·alpha·persistence·loss·fold
계약을 그대로 사용한다. threshold, source, budget, lane, alert 승격은
계산하거나 재선택하지 않는다.

## 3. 입력 경계와 금지 사항

- 허용 입력은 SHA-256으로 고정된 SQLite의 정상 TRAIN values와 단계별로
  생성한 finite synthetic tensor뿐이다.
- TEST values, TEST label, 실제 이상 위치, ground truth, human review,
  과거 TEST metric, Exp419 checkpoint/prediction/lane 파일은 읽지 않는다.
- 데이터베이스 연결은 read-only이며 `PRAGMA quick_check`가 `ok`가 아니면
  즉시 중단한다.
- Exp422는 새 output root에만 쓰고 기존 파일이 있으면 시작하지 않는다.
- MPS-heavy 프로세스는 한 번에 하나만 실행한다. 공식 Exp419 CPU 작업이
  끝나기 전에는 실제 MPS 측정을 시작하지 않는다.
- MLX는 이 연구의 알고리즘을 다시 쓰지 않는다. 설치 여부, 버전, device,
  `package_absent` 또는 `algorithm_rewrite_required` 사유만 metadata로
  기록한다.

## 4. 판정 기준

각 단계와 전체 결과를 다음 기준으로 판정한다.

1. CPU와 MPS 출력은 모두 finite여야 한다.
2. 최대 정규화 오차는 `1e-4` 이하여야 한다.
3. 동결 alpha의 후보 좌표는 exact 일치해야 한다.
4. tie-aware Spearman 순위 상관은 `0.999` 이상이어야 한다.
5. 2단계에서는 model state, stop epoch, loss history, OOF, calibration,
   perturbation, qualification digest가 일치해야 한다.
6. batch 1을 제외한 MPS 총시간 speedup은 transfer와 synchronization을
   포함하여 `1.1x` 이상이어야 한다.

하나라도 실패하면 해당 조합은 CPU fallback으로 기록한다. `passed`는
사전등록된 표본·조합에서만 통과했다는 의미이며 운영 백엔드 채택이나
Exp419 재실행 승인을 의미하지 않는다.

## 5. 산출물과 provenance

새 매니페스트는 다음을 포함한다.

- `experiment_id=Exp422`와 schema version
- Git commit, source manifest와 source blob digest
- database SHA-256, SQLite quick-check 결과, frozen config digest
- Python·PyTorch·MPS·MLX·mlx-metal 버전과 device identity
- 1단계 연산별 tensor shape·seed·batch·delta·ULP·digest·timing
- 2단계 표본 count/digest와 Exp421 benchmark evidence
- 단계별 pass/fail, fallback reason, partial-run 상태
- 공식 Exp419와 분리되었음을 나타내는 fresh attempt root

매니페스트는 canonical JSON으로 한 번만 기록하고, 검증 실패 시 운영
artifact로 복사하지 않는다. 부분 실행도 삭제하지 않고 원인 분석용으로
보존한다.

## 6. 구현과 검증 순서

1. Exp422 worktree에서 이 설계와 사전등록·runbook을 고정한다.
2. Exp421의 동결 모델 adapter와 TRAIN-only boundary를 재사용하는 얇은
   numeric-delta 모듈을 작성한다.
3. 1단계 tensor parity의 실패·오차·ULP·후보 좌표 테스트를 먼저 통과시킨다.
4. 2단계 benchmark wrapper, 매니페스트 검증, no-TEST/no-clobber 테스트를
   추가한다.
5. Exp421 회귀 테스트와 Exp422 전체 단위 테스트를 실행한다.
6. Exp419 종료를 확인한 뒤에만 새 attempt root에서 실제 측정을 시작한다.

## 7. 해석과 후속 조치

Exp422가 통과해도 MPS는 운영에 자동 승격하지 않는다. 결과는 별도 검토용
근거로만 남기며, 운영 채택에는 새 사전등록·전체 표본 benchmark·별도 승인과
재현성 검증이 필요하다. 실패한 경우에도 CPU 경로를 유지하고 실패한 연산,
환경, 수치 범위와 fallback 사유를 기록하는 것이 연구 결과다.
