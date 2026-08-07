# Exp423 MPS 전체 TEST 값 평가

상태: **검토 전용 실행**.

Exp423은 사용자 요청에 따라 frozen Exp417 모델을 실제로 Apple MPS에서 학습하고,
1,117개 전체 데이터셋의 TEST **값**을 같은 실행에서 추론하는 별도 장치 연구다.
Exp417 공식 산출물, 운영 lane, threshold, 예산 또는 자동 경보를 변경하지 않는다.

## 목적

- CPU가 아닌 MPS에서 OOF 및 최종 seed 모델이 실제로 학습되는지 확인한다.
- 전체 데이터셋에서 MPS 추론 결과를 재현 가능한 checkpoint와 prediction으로 남긴다.
- CPU Task 7 결과와 수치·후보 차이를 사후에 비교할 수 있는 증거를 만든다.

## 고정 경계

- TRAIN 값만으로 모델·보정·계약을 만든다.
- TEST에서는 값과 instance identity만 읽는다. TEST label, 실제 이상 위치,
  과거 TEST 성능은 실행 중 열지 않는다.
- Exp291/Exp292 입력과 DB는 SHA-256으로 고정한다.
- `mps`와 `mps:0`은 같은 장치로 취급하지만, CPU fallback은 허용하지 않는다.
- 실행은 단일 MPS-heavy 프로세스와 순차 dataset loop를 사용한다.
- 결과는 `review_only=true`, `official_prediction=false`로 기록한다.

## 게이트와 해석

Task 7의 frozen backend gate는 MPS speedup이 1.1배 이상일 때만 공식 채택을
허용한다. 현재 감사 결과는 CPU 선택이며 MPS는 수치 후보 identity는 유지하지만
속도 gate를 통과하지 못했다. 따라서 Exp423은 명시적 요청에 의한 예외적인 검토용
실행이며, 완료되어도 Exp417 운영 backend를 MPS로 바꾸지 않는다.

전체 실행이 끝난 뒤에만 별도 post-hoc evaluator로 TEST label을 열어 혼동행렬과
성능을 계산한다. 그 지표는 연구 보고용이며 threshold/source/lane 재선택에 쓰지
않는다.

## 산출물

- `progress.json`: dataset별 checkpoint 완료 수와 현재 상태
- `exp423_mps_predictions.jsonl`: label-free MPS 추론 후보
- `manifest.json`: 장치, gate override, 입력 hash, 완료 수, prediction hash
- `status.json`: 종료 상태와 label guard
