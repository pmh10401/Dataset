# Exp411 실패 보고서

## 동결된 실패 증거

Exp411은 1,117개 전체 모집단의 TRAIN-only near-repeat qualification으로
사전등록되었다. 정상 TRAIN 입력의 비유한값을 발견하면 fail-closed 하도록 한
계약에 따라 실행을 중단했으며, 최종 publication은 존재하지 않는다.

- 검증 가능한 dataset record는 187개다.
- `progress.json`의 186은 마지막 record를 기록한 뒤 progress를 갱신하기 전
  프로세스가 중단되어 남은 stale 값이다. 따라서 진행 또는 migration coverage의
  권위 있는 근거가 아니며, dataset record 자체만 증거로 사용한다.
- 첫 실패 입력은 `DodgerLoopDay_normal_1`의 정상 TRAIN run 6이다. 시작 위치의
  NaN이 71개였으며 Exp411은 이를 보정하거나 우회하지 않았다.
- 동결 DB와 모집단을 읽기 전용 TRAIN-only로 전수 검사한 inventory는 영향
  dataset 19개, 영향 TRAIN run 164개, 비유한값 983개, 전체 값이 비유한 run
  0개 (`19 / 164 / 983 / 0`)다.

## 주장 및 운영 경계

이 실패는 Exp411 입력 계약 실패로 보존한다. 동일 실험 번호에서 입력을
보정하거나 checkpoint를 재개하지 않는다. TEST split, `labels_blob`, 실제
이상 위치와 과거 TEST 성능에는 접근하지 않았다. 기존 Hard, Standard,
Priority, exact-normal 및 prediction과 운영 lane은 변경하지 않으며, 자동
정상 lane 변경도 수행하지 않는다.
