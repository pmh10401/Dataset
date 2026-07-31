# Exp419 실행 복구 설계

## 결정과 범위

Exp419는 Exp417의 과학적 계산을 바꾸지 않는 실행 복구 실험으로 유지한다. 정상
TRAIN 전처리, identity-grouped fold, causal CNN 구조, horizon, seed, early-stop,
OOF 보정, qualification, threshold, lane 제외, TEST 값 접근 시점 및 publication
규칙은 변경하지 않는다. TEST label, 실제 이상 위치, 과거 TEST 성능 및 사람 검토
결과는 어떤 새 경로나 선택에도 사용하지 않는다.

이번 변경은 아직 유효 benchmark 또는 공식 Exp419 산출물이 없는 상태에서만 적용한다.
이전에 중단된 benchmark root는 진단 증거로 보존하고 재사용하지 않는다. 복구 root는
새 no-clobber 경로를 사용하며, 모든 파생 산출물은 로컬에만 남긴다.

## 관찰된 원인

`StarLightCurves_normal_3`는 정상 TRAIN 4,261개, 길이 1,024의 큰 group이다.
현재 구조는 이 group의 5 OOF fold x 3 seed와 3 final seed를 한 worker가 순차로
끝낸 뒤에야 dataset checkpoint를 쓴다. 따라서 수시간 동안 완료 수가 0인 것은
정지의 증거가 아니다. 이전 시도는 worker 예외가 아니라 외부 종료 status 143으로
끝났다.

또한 모델의 PyTorch CPU thread 수를 바꾸면 동일한 input/seed라도 모델 상태 hash가
달라진다. 따라서 thread 수를 늘려 속도를 얻는 방식은 execution-equivalence를
보장하지 못하며 채택하지 않는다.

## 선택한 구조

### 1. 한 모델당 결정론적 한 thread

각 CNN fit은 계속 `torch.set_num_threads(1)`과
`torch.set_num_interop_threads(1)`을 사용한다. MLX와 MPS는 사용하지 않는다.
외부 dataset worker도 최대 4개만 사용한다. 이는 12 logical CPU host에서 최대
12개의 독립 seed-fit 작업을 만들 수 있고, project rule의 7 dataset-worker 상한을
지킨다.

### 2. seed 단위 병렬 학습

한 length group 안에서 서로 독립인 세 frozen model seed를 세 Python 작업으로
동시에 실행한다. 각 작업은 자신의 seed에 속한 다섯 OOF fit과 필요하면 final fit을
순서대로 수행한다. 동시에 실행되는 작업은 파라미터, optimizer, residual buffer를
공유하지 않는다.

초기 모델 상태는 기존 순차 seed 순서로 먼저 생성하고 고정한다. 그 뒤의 각 학습은
그 복사본에서 시작한다. 따라서 전역 난수 상태를 여러 작업이 동시에 바꾸지 않는다.
작업 결과는 원래 `(fold, seed)` 정렬 순서로 다시 조립한다.

채택 전 반드시 같은 synthetic group에서 순차 경로와 새 경로의 model SHA-256,
stop epoch, loss history, OOF residual bytes 및 candidate index가 정확히 같은지
검사한다. 하나라도 다르면 seed 병렬 기능은 fail-closed이며 기존 순차 one-thread
경로만 남긴다.

### 3. 사람이 읽는 진행 상태

각 outer worker root에 원자적으로 교체되는
`exp419_worker_status.json`을 둔다. 이것은 다음 상태만 기록한다.

- worker 시작/실패/완료 시각과 PID
- 현재 dataset, shard 안의 순번, 완료 dataset 수
- 현재 단계: context 준비, TRAIN load, preprocessing, seed별 OOF, seed별 final,
  qualification, TEST values-only scoring, checkpoint
- 현재 seed 및 OOF fold(해당할 때)
- 마지막 성공 단계와 오류 문자열

학습 결과, raw series, score, TEST label, anomaly 위치 또는 metric은 status 파일에
쓰지 않는다. 별도 자동 monitor, polling loop, cron, keepalive는 만들지 않는다.
사용자가 요청한 한 번의 상태 확인은 이 파일만 읽는다.

### 4. 실행 자격 검증

과거의 19-dataset 전체 pipeline을 1/2/4/6 worker로 네 번 완주하는 benchmark는
가장 큰 dataset 하나 때문에 실행 가능하지 않았다. 새 검증은 두 증거를 분리한다.

1. **동등성 smoke:** synthetic input과 작은 고정 normal-TRAIN smoke에서 순차와
   seed-parallel 결과를 byte-for-byte 비교한다. 이 단계는 prediction을 만들지 않는다.
2. **런타임 smoke:** 기존 19개 label-free TRAIN selection을 유지하되, 각 group에서
   고정된 두 training batch만 사용해 outer-worker x seed-worker layout의 실제
   scheduling throughput과 peak RSS를 측정한다. dataset 선택, 순서 및 batch 수는
   timing 전에 source manifest와 함께 고정한다. TEST 값과 label은 읽지 않는다.

이 smoke는 scheduler의 통과 여부만 정한다. full population 실행에서 선택된 layout의
출력 동등성은 각 dataset checkpoint digest와 seed-parity 검사로 계속 fail-closed
검증한다. runtime smoke가 2.0x wall-clock speedup, memory 80% 이하, deterministic
parity를 통과하지 못하면 전체 1,117개 실행은 시작하지 않는다.

## 실패와 재개

상태 기록 실패, worker 예외, corrupted checkpoint, parity 실패 및 memory 초과는
모두 fail-closed다. dataset checkpoint는 기존처럼 dataset 결과가 완성된 뒤에만
commit한다. 진행 status는 재개 근거나 prediction evidence가 아니며, 재개 시에는
checkpoint provenance·source manifest·input digest·layout이 정확히 일치해야 한다.

공식 전체 실행에는 검증이 끝난 뒤에도 별도의 명시 승인이 필요하다.

## 테스트

1. status writer가 시작·단계 전환·완료·실패를 원자적으로 기록하고 민감한 field를
   거부하는지 검사한다.
2. 직렬과 seed-parallel synthetic fits의 모든 증거 bytes가 일치하는지 검사한다.
3. seed task가 실패하면 partial output을 publication에 넘기지 않는지 검사한다.
4. worker 수와 seed concurrency가 host CPU 예산 및 memory 계약을 넘지 않는지
   검사한다.
5. source-manifest, label-isolation, Exp417/Exp419 회귀 test 전체를 실행한다.

## 비목표

- CNN/feature/score/threshold 변경
- PyTorch thread 수 증가 또는 MPS/MLX 채택
- TEST label 기반 속도·결과 선택
- 자동 감시 또는 원격/TencentDB에 실험 데이터·checkpoint·결과 저장
- 중단된 Exp417 94개 checkpoint를 Exp419 결과로 복사하거나 병합
