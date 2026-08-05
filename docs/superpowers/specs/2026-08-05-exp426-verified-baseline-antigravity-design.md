# Exp426 검증 기준 결과 완성 설계

## 1. 쉬운 설명

Exp426의 목적은 새 탐지 방법을 시험하는 것이 아니다. 이전 Exp424 실행에서
정상적으로 저장되고 검증된 983개 결과를 그대로 가져오고, 학습 근거 부족으로
끝난 134개 데이터셋을 현재 TRAIN 데이터로 다시 확인하여 1,117개 전체 기준
결과를 빠짐없이 완성한다.

새 MPS 또는 MLX 계산은 하지 않는다. TEST 정답은 예측 결과와 그 SHA256이
완전히 고정되기 전까지 열지 않는다. 이 결과는 다음 탐지 방법이 실제로
좋아졌는지를 비교하기 위한 연구 기준이며 자동 경보 정책이 아니다.

## 2. 범위와 성공 조건

Exp426은 다음 상태에서만 성공한다.

- 데이터셋 수가 정확히 1,117개다.
- Exp424에서 가져온 결과가 정확히 983개다.
- 다시 확인할 데이터셋이 정확히 134개다.
- 134개 모두 TRAIN 데이터만으로 `oof_identity_insufficient` 상태가 다시
  확인되고 명시적인 판단 보류 결과로 저장된다.
- 134개 처리 중 TEST 값, TEST 정답, 평가 지표는 한 번도 열리지 않는다.
- 오류, 손상된 결과, 알 수 없는 상태, 누락 데이터셋이 0개다.
- 최종 후보 좌표가 정렬되고 중복되지 않으며 동결 lane과 겹치지 않는다.
- 예측 JSONL, 데이터셋 요약, 출처 기록, 실행 기록, manifest, 상태 파일의
  SHA256이 고정된다.
- 결과 고정 전에는 혼동행렬, F1, precision, recall 또는 TEST 기반 선택을
  계산하지 않는다.

이 실험은 성능 개선을 주장하지 않는다. 전체 비교 기준을 완성하는 기반
실험이다. 이후의 별도 평가가 현재 champion보다 좋거나 나쁘다는 사실을
보여줄 수 있지만, 그 결과로 Exp426 자체의 설정을 바꾸지 않는다.

## 3. 사용하고 사용하지 않는 정보

### 사용한다

- `/Users/mac/Documents/Dataset/univariate_ts.db`
- DB SHA256
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Exp424 checkpoint
  `/Users/mac/Documents/Dataset/exp424_mps_score_cache_20260802-153831.checkpoint`
- Exp424의 고정된 root, progress, record-set SHA256
- Exp291/292의 동결된 Hard, Standard, Priority, automatic-normal lane
- 정상 TRAIN 값, 데이터셋 이름, TRAIN/TEST 개수와 길이 같은 고정 메타데이터
- Exp425에서 이미 검증된 checkpoint, 출처 검증, DB snapshot, 자원 정리 코드

### 사용하지 않는다

- TEST 정답과 실제 이상 위치
- TEST 결과로 계산한 family별 과거 성능
- human-assisted 지표
- Exp425 retry3에서 저장되지 않은 MPS 차이 추정값
- TEST 결과를 본 뒤 선택한 alpha, threshold, source 또는 budget
- tail replacement

## 4. 과학적 출처와 정직한 이름

983개 결과는 CPU로 새로 계산하지 않는다. 이 결과들은 Exp424에서 생성된
검증된 MPS 출처 결과다. Exp426은 각 결과를 `exp424_import`로 표시하고 원본
record SHA256, result SHA256, 출처 root SHA256과 호환성 확인 SHA256을 함께
보존한다. CPU 결과인 것처럼 표시하지 않는다.

134개 결과는 후보 점수를 새로 계산하지 않는다. 현재 DB의 TRAIN 데이터를
CPU로 읽어 동일한 길이별 분할 가능 여부를 다시 확인한다. 모든 길이 그룹이
학습 검증에 필요한 서로 다른 TRAIN 개수가 부족한 경우에만 후보가 없는
`verified_train_abstention` 기록을 만든다.

134개 중 하나라도 현재 TRAIN 데이터에서 충분한 학습 근거를 갖는 것으로
바뀌면 Exp426 전체를 중단한다. 같은 실험에서 새 모델 계산으로 전환하지
않는다. 그런 경우는 새 실험 번호와 새 사전등록이 필요하다.

## 5. 실행 구조

Exp426은 Exp425 파일을 수정해 CPU 모드로 바꾸지 않는다. 새 Exp426 package와
실행 파일을 만들고, Exp425의 검증된 읽기 전용 부품만 재사용한다.

구성은 다섯 부분이다.

1. **출처 확인**: Exp424 파일 1,117개와 `983/134` 구성을 읽기 전용으로
   확인한다.
2. **DB 확인**: DB를 한 번 해시하고 실행 동안 같은 파일인지 계속 확인한다.
3. **983개 가져오기**: 검증된 Exp424 결과를 새 checkpoint에 복사하되 모든
   출처 지문을 보존한다.
4. **134개 TRAIN 확인**: TEST loader를 사용할 수 없게 막은 상태에서 TRAIN
   근거 부족을 다시 확인하고 판단 보류 기록을 만든다.
5. **결과 만들기**: 1,117개를 다시 읽어 후보 좌표, 요약, 출처 기록, manifest,
   상태와 SHA256을 만든 뒤 독립 검증한다.

## 6. CPU, MLX, MPS와 병렬 처리

Exp426은 새 점수 계산 실험이 아니며, 행렬 계산이나 신경망 학습을 하지 않는다.
따라서 MLX로 옮길 계산 단계가 없고 MLX를 사용하지 않는다.

MPS는 Exp425 한 데이터셋 비교에서 기존 기준과 같은 최종 판단을 보장하지
못했고 기존 측정에서 1.1배 속도 조건도 통과하지 못했다. Exp426은 MPS를
사용하지 않는다. 이 결정과 근거를 manifest에 기록한다.

현재 Mac은 Apple M2 Max, CPU 12코어(성능 코어 8개, 효율 코어 4개), 메모리
32GiB다. 프로젝트 규칙이 데이터셋 worker를 최대 7개로 제한하므로 Exp426은
허용 범위 안에서 CPU를 최대한 사용한다.

기본 실행은 7개 worker다. 실행 전 작은 고정 입력으로 worker 1, 4, 7을 각각
측정하여 모든 결과 SHA256과 후보 좌표가 완전히 같은지 확인한다. 7개가 가장
빠르면 7개를 사용한다. 파일 읽기 경쟁 때문에 4개 또는 1개가 실제로 더
빠른 경우에만 측정상 가장 빠른 수를 사용하고 그 이유와 시간을 manifest에
기록한다. CPU 사용률만 높이고 전체 시간은 느려지는 빈 작업은 만들지 않는다.

각 worker 내부의 NumPy/BLAS/Numba thread 수는 1로 제한하여 `7 workers × 내부
여러 thread` 형태의 과도한 경쟁을 막는다. 병렬 worker는 독립적인 출처 검증과
TRAIN 근거 확인만 담당한다. 부모 프로세스가 잠긴 파일 descriptor와 SQLite
snapshot으로 원본 bytes를 읽고, worker에는 경로나 DB 연결이 아니라 읽기 전용
bytes와 고정 메타데이터만 전달한다. SQLite snapshot, 결과 정렬, checkpoint 및
로그 저장은 부모 프로세스 하나가 소유한다.

worker 완료 순서와 관계없이 데이터셋 이름 순서대로 저장한다. 1/4/7 worker의
과학적 데이터셋 record SHA256과 prediction 좌표 SHA256은 같아야 한다. worker
수, 처리 시간과 실행 환경을 기록하는 로그/manifest SHA256은 서로 다를 수
있으며 과학적 동일성 비교 대상에 넣지 않는다.

## 7. 저장 위치와 재개

사전등록에서 실제 날짜가 포함된 다음 세 절대 경로를 한 번만 고정한다.

- output root:
  `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805`
- checkpoint root:
  `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint`
- readiness artifact:
  `/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_readiness_20260805.json`

`new`, `resume`, `inspect`, `readiness` 네 명령을 제공한다.

- `new`: 두 root가 모두 없을 때만 새 실행을 만든다.
- `resume`: 같은 실행 지문을 가진 미완료 root만 이어서 처리한다.
- `inspect`: 파일을 바꾸지 않고 현재 상태만 읽는다.
- `readiness`: 실제 output/checkpoint를 만들지 않고 입력과 테스트 준비 상태를
  별도 파일에 기록한다.

checkpoint는 데이터셋 하나가 끝날 때마다 안전하게 교체 저장한다. 중단되면
이미 끝난 데이터셋은 다시 계산하지 않는다. 손상된 checkpoint를 자동으로
덮어쓰거나 무시하지 않고 실패 상태로 남긴다.

## 8. 진행 로그

장시간 실행은 프로그램이 직접 다음 파일을 계속 갱신한다.

- `progress.json`: 현재 상태 한 개
- `events.jsonl`: 완료와 오류 기록을 줄 단위로 추가
- `parent.log`: 사람이 읽기 쉬운 로그
- `run.pid`: 실행 PID

로그에는 PID, 현재 데이터셋, TRAIN/TEST 개수와 길이, 현재 단계, CPU worker 수,
완료/전체, 가져오기/재확인/성공/판단 보류/오류 수, 속도, 경과 시간, 예상 남은
시간, checkpoint 경로와 최신 오류를 기록한다. TEST 값과 정답은 기록하지 않는다.

## 9. 실패 처리

다음 중 하나라도 발생하면 전체 실행을 실패로 끝내고 원인을 보존한다.

- Exp424 파일, DB, lane 또는 Git 지문이 바뀜
- Exp424 구성이 `1117/983/134`와 다름
- 134개 중 하나가 TRAIN 근거 부족이 아님
- TEST 값 또는 TEST 정답 loader 호출
- 중복, 누락, 잘못된 데이터셋 또는 후보 좌표
- checkpoint와 로그 수가 다름
- worker 1과 병렬 worker의 결과가 다름
- 쓰기 공간 부족, 저장 실패 또는 자원 정리 실패

실패한 output/checkpoint root는 삭제하거나 재사용하지 않는다. 수정이 필요하면
새 실험 번호 또는 사전등록된 새 root가 필요하다.

## 10. 결과 파일과 안전 표시

최종 결과에는 다음을 포함한다.

- 후보 좌표 prediction JSONL
- 1,117개 데이터셋 요약
- 983개 출처 가져오기 기록
- 134개 TRAIN 근거 부족 기록
- DB 및 lane 확인 기록
- CPU/MLX/MPS 결정 기록
- manifest와 status
- 모든 파일 SHA256

manifest는 다음을 명시한다.

- `review_only = true`
- `operational_alert_eligible = false`
- `contains_test_labels = false`
- `test_labels_opened = false`
- `imported_datasets = 983`
- `verified_train_abstentions = 134`
- `completed_datasets = 1117`
- `failed_datasets = 0`

Priority와 Shadow 결과를 자동 경보로 승격하지 않는다.

## 11. 테스트와 실행 전 조건

구현은 테스트를 먼저 작성하고 실패를 확인한 뒤 최소 코드로 통과시킨다.
최소한 다음을 자동 검증한다.

- Exp424 `983/134` 출처 확인과 모든 지문 변조 거부
- 983개 결과를 다시 계산하지 않는지 확인
- 134개에서 TRAIN만 열고 TEST loader를 호출하지 않는지 확인
- TRAIN 근거가 충분한 데이터셋이 나타나면 전체 중단
- 1,117개 보존과 후보 좌표 정렬/중복/lane 겹침 거부
- `new`, `resume`, `inspect`, `readiness`와 중단 후 재개
- progress/events/checkpoint 수 일치
- worker 1/4/7 결과 동일성 및 작은 속도 시험
- 결과 파일을 다시 읽어 같은 SHA256이 나오는지 확인
- 기존 Exp425 및 Exp417 테스트가 그대로 통과하는지 확인

실제 1,117개 실행 전에 clean Git, 정확한 입력 지문, DB 한 번 해시, TEST 접근
차단, 전체 테스트 통과, readiness 성공과 새 Sol/High 검토가 필요하다.

## 12. 실행 후 단계

예측 파일과 manifest SHA256이 고정되면 prediction 단계는 닫힌다. 그 다음에만
별도 평가 프로그램이 TEST 정답을 열 수 있다. 평가는 기존 기준과 다음을 함께
비교한다.

- 새로 찾아낸 데이터셋
- 놓친 데이터셋
- false alarm
- 기존에 잘하던 데이터셋의 악화
- 전체 paired 결과
- bootstrap 신뢰구간

평가 결과로 Exp426을 수정하지 않는다. 개선 실험은 Exp427 이상 새 번호와 새
사전등록으로 시작한다.

## 13. 파일 이름 규칙

새로 만드는 모든 repository 파일 이름에는 `antigravity`를 포함한다. 기존
Exp425/Exp417 파일은 필요한 경우에도 직접 변경하지 않는 것을 기본으로 한다.
새 Exp426 directory는 namespace package로 사용하고, 규칙을 어기는 새
`__init__.py`는 만들지 않는다.
