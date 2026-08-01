# Exp422 사전등록: MPS 수치 차이 및 점수 동등성 진단

상태: **진단용 사전등록**. Exp422는 동결된 Exp417 계산의 CPU/MPS 수치 차이를
검사하는 작은 TRAIN-only 연구이며, Exp419 또는 Exp421의 설정·산출물·운영 경로를
변경하지 않는다.

## 고정된 범위

- batch 크기는 정확히 `(16, 32, 64, 128, 256)`만 사용한다.
- model seed는 `(41701, 41702, 41703)`, 섭동 seed는
  `(41711, 41712, 41713)`으로 Exp417에 묶인다.
- 최대 64개 series와 `256 * 1024 * 1024` bytes만 사용한다.
- 수치 게이트는 최대 정규화 오차 `<= 1e-4`, tie-aware Spearman `>= 0.999`,
  그리고 speedup `>= 1.1x`이다.

## 입력·단계 순서

입력은 유한한 정상 TRAIN 값 또는 결정론적 합성 tensor뿐이다. TEST 값, label,
실제 이상 위치, ground truth, 과거 TEST 지표와 공식 Exp419 checkpoint 및 출력은
읽거나 사용하지 않는다.

1. 1단계는 모든 고정 batch와 seed에 대해 CPU 기준 출력과 MPS 출력을 비교한다.
   유한성, tensor digest, ULP, 정규화 오차, 순위 및 후보 좌표를 확인한다.
2. 2단계의 제한된 TRAIN 점수 비교는 1단계가 모든 필수 게이트를 통과했을 때만
   실행한다. model state, stop epoch, loss history, OOF, calibration, perturbation,
   qualification digest와 후보 좌표의 동등성을 확인한다.

시간은 입력·출력 전송, 장치 계산, 명시적 동기화 시간을 각각 측정하고 그 합을
총시간으로 사용한다. 총시간의 CPU/MPS 비율로 speedup을 계산한다.

MPS를 사용할 수 없거나 수치·좌표·속도·메모리 게이트가 하나라도 실패하면 해당
행은 명시적인 사유와 함께 CPU fallback으로 기록한다. batch를 임의로 줄이거나
threshold를 다시 고르지 않는다. MLX는 알고리즘 재작성 없이 availability, 버전,
device와 `package_absent` 또는 `algorithm_rewrite_required` 사유만 기록한다.

## 해석

`passed`는 이 제한된 사전등록 표본에서의 동등성 증거이고, `failed` 또는
`skipped`는 보수적인 진단 결과다. 어떤 결과도 운영 backend, threshold, source,
lane, 예산 또는 Exp419 실행을 자동으로 바꾸지 않는다. 운영 채택에는 별도
사전등록과 승인, 더 넓은 재현성 검증이 필요하다.
