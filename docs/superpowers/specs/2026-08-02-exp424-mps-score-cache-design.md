# Exp424 MPS 점수 보정 캐시·batch 추론 설계

상태: **승인된 설계 / review-only 실험**

## 목표와 경계

Exp424는 Exp423에서 발견한 반복 NumPy 점수 계산과 MPS 모델 재생성 비용을 줄이는
별도 장치·속도 연구다. Exp417 공식 CPU 산출물과 운영 lane은 변경하지 않는다.
현재 Exp423의 102개 완료 checkpoint와 로그는 보존하며, Exp424 checkpoint는 새
provenance와 새 실행번호로 만든다.

실행 중에는 TRAIN 값으로만 모델·OOF 보정·계약을 만들고, TEST에서는 값과 identity만
읽는다. TEST label과 실제 이상 위치는 post-hoc 평가 전까지 열지 않는다. Exp424
결과는 `review_only=true`, `official_prediction=false`다.

## 설계

### 1. OOF 보정 캐시

`score_horizon()`의 고정 OOF 부분을 `ScoreCalibration`으로 분리한다. `(dataset,
length, seed, horizon)`마다 다음을 정확히 한 번 계산한다.

- OOF residual 입력 검증
- pooled median, MAD scale
- normalized OOF와 99% exceedance threshold
- 각 OOF series의 q95+persistence score

query마다 수행하는 작업은 기존 수식과 같은 방식의 정규화, q95, persistence, p-value
비교뿐이다. `np.median`, `np.percentile(method="linear")`, `np.count_nonzero`의
수식과 dtype을 바꾸지 않는다. 기존 `score_horizon()`와 동일 입력을 비교해 점수와
candidate decision의 차이가 0 또는 사전등록 허용오차 이내인지 검증한다.

### 2. 모델 재사용과 batch 추론

OOF 모델과 최종 seed 모델을 query마다 materialize하지 않고 길이·seed별로 한 번
materialize한다. TEST 및 고정 synthetic query는 동일한 순서로 batch를 만들어 한 번에
추론한다. batch 크기는 메모리와 MPS parity를 확인한 뒤 고정하며, 학습 epoch·seed·
optimizer·threshold·routing rule은 변경하지 않는다.

batch와 기존 단일 query 경로를 동일 TRAIN 입력에서 비교한다. 모든 출력은 finite,
후보 좌표 동일, score 차이 허용오차 이내여야 한다. 하나라도 실패하면 batch 경로를
채택하지 않고 캐시만 사용한 경로 또는 기존 경로로 명시적으로 fallback한다.

### 3. 검증과 진행 기록

finite·shape·identity 검사는 immutable group/calibration 경계에서 유지한다. hot loop의
중복 복사와 동일 배열 재검증만 제거한다. dataset별로 `stage`, `completed_queries`,
`expected_queries`, `current_dataset`을 checkpoint progress에 기록한다. 이 기록은
TEST label을 읽지 않는다.

## 실행 게이트

1. unit test: cache가 기존 `score_horizon()`과 동일한 점수·p-value·persistence를
   반환한다.
2. integration smoke: 대표 TRAIN-only 데이터셋에서 단일 query와 batch query의
   residual·candidate parity를 확인한다.
3. benchmark: 입력 전송·MPS 계산·동기화를 포함한 wall time을 비교한다. 캐시/배치가
   의미 있는 속도 향상을 보이지 않으면 기존 경로를 유지한다.
4. full run: 한 개의 MPS-heavy 프로세스만 사용하고 새 Exp424 output/checkpoint에
   기록한다.

## 실패와 해석

최적화 parity 또는 finite gate가 실패하면 해당 최적화를 채택하지 않는다. Exp424가
완료되어도 Exp417 backend, threshold, source, lane, budget을 재선택하지 않는다.
TEST label을 사용하는 혼동행렬과 성능은 prediction hash를 고정한 뒤 별도 post-hoc
작업으로만 계산한다.
