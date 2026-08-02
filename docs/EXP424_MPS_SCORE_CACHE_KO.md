# Exp424 MPS 점수 보정 캐시 최적화

상태: **검토 전용 실행**.

Exp424는 Exp423에서 확인한 반복 NumPy 점수 계산과 모델 재생성 비용을 줄인 뒤,
MPS에서 학습하고 1,117개 전체 TEST 값을 추론하는 별도 실험이다. Exp417 공식 CPU
경로와 운영 lane은 변경하지 않는다.

## 최적화 내용

- OOF residual의 median, MAD, 99% threshold, OOF score를 길이·seed·horizon별로
  한 번만 계산한다.
- 최종 모델을 query마다 다시 만들지 않고 reusable predictor에 한 번 materialize한다.
- TEST와 합성 query는 최대 256개 단위로 묶고, 내부 MPS batch는 32개로 제한한다.
- 기존 점수 공식, float64 계산, threshold, seed, perturbation, routing rule은 고정한다.

원시 TEST label, 실제 이상 위치, 과거 TEST 성능은 실행 중 사용하지 않는다. TEST 값과
instance identity만 읽고, label 기반 혼동행렬은 prediction hash를 고정한 뒤 별도
post-hoc 평가에서 계산한다.

## 안전 게이트

대표 CPU/MPS parity 테스트에서 residual·후보 좌표·유한성을 확인한다. parity 또는
속도 게이트가 실패하면 최적화 경로를 채택하지 않고 명시적인 fallback 사유를 남긴다.
Exp424 결과는 `review_only=true`, `official_prediction=false`이며 완료되어도
Exp417 backend, threshold, source, lane, budget을 재선택하지 않는다.

## 진행 기록

checkpoint의 기존 `progress.json`은 동결된 dataset 완료 회계를 유지한다. 상세한
현재 dataset, stage, query 완료 수는 output의 `runtime_progress.json`에 기록한다.
