# Exp421 사전등록: MPS 배치 추론 가속 진단

상태: **진단용 사전등록**. 이 문서는 공식 Exp419 결과를 다시 만들거나
운영 경보를 바꾸기 위한 승인이 아니다.

## 1. 질문과 목적

Exp417에서 이미 고정한 모델과 점수 계산을 그대로 사용하면서, 같은 정상
TRAIN 입력을 여러 개 묶어 계산할 때 CPU보다 MPS가 실제로 빠르고 안전한지
확인한다. 목적은 평균 F1을 높이는 것이 아니라, 후보 좌표와 판단을 바꾸지
않으면서 계산 시간을 줄일 수 있는지 알아보는 것이다. 결과는 한 번의 작은
로컬 진단 시도와 그 매니페스트로 끝난다.

## 2. 바꾸지 않는 기준

- 기준은 동결된 Exp417 설정이다. 모델 seed는 `41701, 41702, 41703`, 합성
  섭동 seed는 `41711, 41712, 41713`이다.
- 길이별 horizon은 `1`, `ceil(0.02*L)`, `ceil(0.10*L)`이며, 후보 alpha는
  `0.01`, persistence는 `0.05`로 고정한다.
- 최종 모델 적합, OOF 잔차, calibration, qualification, 합성 섭동의 정의와
  입력 순서를 그대로 재사용한다. 학습에 쓰는 전체 후보 적격 정상 TRAIN
  모집단의 개수와 digest도 매니페스트에 남긴다.
- 동결한 정확한 batch 후보는 `(1, 4, 8, 16, 32)`이다. batch 1은 항상 CPU가
  기준이며 MPS로 바꾸지 않는다. 장치에서 측정하는 표본은 길이가 같은 최대
  64개 series, 최대 256 MiB로 제한한다. 전체 1,117개를 이 진단에서 다시
  실행하지 않는다.

## 3. 입력과 경계

입력은 읽기 전용 SQLite의 정상 TRAIN 값뿐이다. 사전등록된 database SHA-256이
맞고 `PRAGMA quick_check`가 통과해야 하며, 출력은 매번 새롭고 비어 있는
attempt root에만 쓴다. `--dataset`을 지정하면 그 정상 TRAIN 이름만
전달한다. TEST 값, TEST label, 실제 이상 위치, ground truth, 위치 정보,
과거 TEST 성능, Exp419 checkpoint/prediction/lane 파일은 읽지 않는다.

실험은 한 프로세스에서 순차적으로 수행한다. CPU thread 수는 1로 고정하며,
측정 repeat는 1~7의 작은 범위만 허용한다. warm-up은 측정에서 제외하고,
전송·계산·동기화 시간을 모두 포함한다. 중첩 병렬 실행이나 공식 Exp419
작업 디렉터리 재사용은 금지한다.

## 4. 통과 게이트와 선택 규칙

다음 조건을 **모든 실제 batch와 합성 TRAIN qualification**에서 확인한다.

1. CPU와 MPS 출력이 유한하다.
2. 같은 입력·seed에서 후보의 정확한 좌표가 일치한다.
3. 최대 상대 정규화 오차가 `1e-4` 이하이다.
4. tie-aware rank 상관(Spearman)이 `0.999` 이상이다.
5. fit/stop/loss/OOF/calibration/qualification/섭동 digest와 합성
   qualification 결과가 일치한다.
6. batch 1을 제외하고 전송과 동기화를 포함한 MPS speedup이 `1.1x` 이상이다.

하나라도 실패하면 해당 batch는 CPU로 남기고 구체적인 fallback reason을
기록한다. MPS가 없거나, 유한성·메모리·동기화에서 오류가 나도 CPU fallback으로
종료한다. MLX는 이 실험에서 알고리즘을 다시 쓰지 않고 버전·장치·적용 불가
사유만 metadata로 남긴다(`package_absent` 또는
`algorithm_rewrite_required`).

## 5. 해석과 후속 조치

`passed`는 이 작은 TRAIN 표본에서 모든 게이트를 통과했다는 뜻이고, `failed`는
하나 이상의 게이트가 실패했거나 CPU fallback이 있었다는 뜻이다. 어느 쪽도
TEST 성능, mean combined F1, lane 등급, 자동 경보, 공개 결과를 바꾸지 않는다.
Exp421 매니페스트의 후보 좌표는 진단 증거이지 운영 prediction이 아니다.

더 큰 표본이나 1,117개 전체 실행이 필요하다고 판단되면 새 실험 번호와 새
사전등록, 별도 승인으로 계획한다. 이 문서의 결과를 보고 alpha, threshold,
source, budget을 다시 고르거나 Exp419에 덮어쓰지 않는다.
