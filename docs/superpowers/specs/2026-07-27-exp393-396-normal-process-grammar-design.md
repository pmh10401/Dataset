# Exp393-396 정상 운전 문법 이탈 점수 설계

## 1. 목적

Exp391과 Exp392는 기존 점수원의 임계값 확장과 조합만으로는 놓친 이상을
안정적으로 복구하기 어렵다는 신호를 보였다. 특히 후보 범위를 넓힐수록
FP가 빠르게 증가했고, 가장 유망한 조합도 자동 경보 확대보다 기존 검토
후보의 우선순위 개선에 적합했다.

Exp393-396은 기존 점수의 다른 조합을 추가하는 실험이 아니다. 정상 운전
과정의 순서와 지속시간을 측정하는 신규 점수원인 **정상 운전 문법 이탈
점수(Normal Process Grammar Score, NPGS)**를 개발하고 검증한다.

이번 단계의 목적은 놓친 FN을 검토 후보에서 복구하는 것이다. 기존 자동
경보는 변경하지 않는다.

이 점수원은 이 프로젝트에서 새로운 정보축이라는 뜻이다. 선행연구 대비
학술적 신규성은 별도 문헌 감사 전까지 주장하지 않는다.

## 2. 변경하지 않는 원칙

- feature fitting, 상태 모델, score calibration, threshold는 TRAIN 정상
  데이터만 사용한다.
- TEST 라벨과 실제 이상 위치는 동결된 prediction을 사후 평가할 때만
  사용한다.
- dataset 이름, family 이름, 과거 family별 TEST 성능으로 정책을 바꾸지
  않는다.
- TEST 길이 기반 top-N budget을 사용하지 않는다.
- tail replacement를 사용하지 않는다.
- Exp137/Exp291 계열의 기존 자동 경보 후보와 lane은 변경하지 않는다.
- 신규 점수는 Exp396까지 일반 검토의 근거 및 우선순위에만 사용한다.
- 평가 결과를 보고 상태 수, alpha, 가중치 또는 threshold를 다시 선택하지
  않는다.
- 모든 결과는 retrospective counterfactual로 표시한다.

## 3. 데이터 단위

- 하나의 dataset은 동일한 sensor-step 조건의 가상 탐지 환경이다.
- 각 TRAIN instance는 정상 run trace 한 개로 취급한다.
- 각 TEST instance는 평가할 run trace 한 개로 취급한다.
- 최종 후보 index는 TEST instance의 행 번호다. 파형 내부의 시점 index가
  아니다.
- TRAIN instance가 실제 현장의 독립 wafer run을 의미한다는 가정은 현재
  데이터 구성에 대한 연구 가정이며 실제 설비 메타데이터로 검증된 사실은
  아니다.

## 4. 핵심 아이디어

기존 거리 기반 점수는 주로 "TEST 파형이 정상 TRAIN 파형과 얼마나 다른가"를
측정한다. NPGS는 다음 질문을 측정한다.

1. 정상 TRAIN에서 거의 관찰되지 않은 상태 전환이 발생했는가?
2. 특정 상태에 정상보다 지나치게 오래 머물렀는가?
3. 비정상적인 전환 후 정상적으로 자주 나타나는 상태로 복귀하지 못했는가?

예를 들어 정상 운전이 다음 순서를 반복한다고 가정한다.

`대기 -> 상승 -> 안정 -> 하강 -> 대기`

TEST가 `대기 -> 상승 -> 진동 -> 상승`으로 움직이거나, `상승` 이후 안정
상태로 복귀하지 않으면 파형 전체의 값 범위가 정상이어도 높은 NPGS를 낼 수
있다.

## 5. 점수 생성 구조

### 5.1 전처리

전처리 순서는 다음으로 고정한다.

1. NaN/Inf 및 결측 보정
2. TRAIN 기준 길이 정렬
3. TRAIN 정상 통계 기반 robust normalization

TEST 자체의 평균, 표준편차, 최솟값 또는 최댓값으로 보정 기준을 만들지
않는다. 기존 전처리 helper를 재사용하되 위 순서를 테스트로 고정한다.

### 5.2 상태 표현

TRAIN으로 결정한 정렬 길이를 기준으로 구간 수를 다음처럼 고정한다.

`n_segments = min(32, max(8, target_length // 8))`

각 trace를 `n_segments`개의 같은 폭 구간으로 나눈다. 경계 손실을 줄이기
위해 동일한 폭의 50% overlap window를 추가하고, 구간마다 다음 descriptor를
계산한다. TEST 길이나 TEST 결과에 따라 구간 수를 바꾸지 않는다.

- robust level
- 기울기
- 곡률 에너지
- 변동성(MAD)
- lag-1 자기상관
- spectral entropy

descriptor는 상태를 정의하기 위한 입력이다. descriptor의 TEST 이상 거리를
그대로 신규 점수로 사용하지 않는다.

TRAIN 정상 구간만 이용해 결정적 상태 prototype을 학습한다.

- 알고리즘: MiniBatchKMeans
- 상태 수: 8
- random seed: 20260727
- `n_init=10`, `max_iter=200`
- batch size는 `min(1024, TRAIN window 수)`로 정한다.
- 서로 다른 TRAIN descriptor가 8개보다 적으면 유효 상태 수를 서로 다른
  descriptor 수로 낮춘다. 유효 상태가 3개보다 적으면 abstention 처리한다.
- 상태 prototype 학습이 불가능한 상수 trace 또는 유효 구간 부족
  데이터셋은 abstention으로 기록한다.

### 5.3 세 개의 독립 구성 점수

#### 전환 놀라움

TRAIN 정상 상태 전이 행렬에 `alpha=1.0`인 Dirichlet smoothing을 적용한다.
TEST trace의
상태 경로에서 `-log P(next_state | current_state)`를 계산하고, 상위 5%의
평균을 trace 점수로 사용한다.

#### 체류시간 이탈

TRAIN 정상에서 각 상태의 연속 체류 길이 분포를 만든다. TEST의 체류 길이가
해당 TRAIN 분포의 꼬리에서 얼마나 드문지 `-log(tail_probability)`로
측정한다. trace 점수는 체류 이탈값 상위 5%의 평균으로 고정한다.

#### 정상 복귀 실패

TRAIN 전체 window에서 점유율이 5% 이상인 상태를 high-support normal
state로 정의한다.
TRAIN 전환 놀라움의 95번째 percentile을 넘는 전환이 발생한 뒤 high-support
상태로 돌아오기까지 걸린 구간 수를 측정한다. trace 안에서 복귀하지 않으면
남은 구간 수에 1을 더한 값으로 기록한다. trace 점수는 가장 긴 복귀시간으로
고정한다.

### 5.4 합성 점수

세 구성 점수를 TRAIN 교차 적합 percentile로 각각 변환한다. 합성 점수는
동일 가중 평균으로 고정한다.

`NPGS = mean(transition_percentile, dwell_percentile, recovery_percentile)`

개별 점수와 합성 점수를 모두 저장한다. TEST 평가 결과를 보고 가중치를
변경하지 않는다.

## 6. TRAIN 정상 calibration

각 TRAIN 정상 instance를 한 번씩 pseudo-test로 두고, 해당 instance를
제외한 TRAIN 데이터만으로 상태 prototype과 calibration reference를 만든다.

- `n_train >= 5`: instance 단위 cross-fitting을 수행한다.
- `n_train < 5`: 이번 실험에서는 calibration abstention으로 처리한다.
- pseudo-test instance 자신의 구간, 상태 또는 점수는 자신의 model과
  threshold 생성에 포함하지 않는다.
- alpha는 `(0.005, 0.01, 0.02, 0.05)`를 모두 보고한다.
- TEST trace의 conformal p-value는
  `(1 + count(TRAIN_OOF_score >= TEST_score)) / (n_train_oof + 1)`로 계산한다.
- 각 alpha에서 `p_value <= alpha`인 TEST instance만 해당 점수원의 후보로
  기록한다.
- TEST 결과로 특정 alpha를 primary 또는 optimal로 선택하지 않는다.
- finite-sample resolution이 alpha보다 거친 데이터셋은 별도로 표시한다.

Prediction 함수는 `y_test`, 실제 이상 위치 또는 평가 metric을 입력받지
않는다. Prediction JSONL에는 TP, FP, F1과 같은 평가값을 저장하지 않는다.

## 7. 실험 구성

### Exp393: NPGS 점수 생성

- 전체 1,117개 데이터셋에서 네 점수 벡터를 생성한다.
- 전환 놀라움, 체류시간 이탈, 정상 복귀 실패, 동일 가중 합성 점수를
  독립적으로 저장한다.
- 데이터셋별 coverage, abstention, 오류와 실행시간을 기록한다.
- labels를 읽지 않는 prediction artifact를 먼저 동결한다.

### Exp394: TRAIN 정상 calibration 감사

- TRAIN cross-fit pseudo-normal false-alert rate를 alpha별로 계산한다.
- calibration resolution, 정상 점수 분포, seed 재현성을 감사한다.
- 기존 점수원과 TRAIN OOF 상관을 계산해 신규 정보축 여부를 진단한다.
- TEST 평가 metric은 계산하지 않는다.

### Exp395: FN 상보성 잠금 평가

- Exp393 prediction hash와 Exp394 calibration contract를 검증한 뒤 TEST
  라벨을 읽는다.
- 기존 Hard, Standard, Priority 후보와 정확한 index 단위로 비교한다.
- 기존 FN 중 NPGS가 찾은 수, 새 FP, 기존 review와의 중복, review 밖의
  신규 후보를 계산한다.
- exact index와 dataset-level 결과를 모두 보고한다.
- 결과를 보고 score 구성이나 alpha를 수정하지 않는다.

### Exp396: Review-only 통합

- 기존 자동 경보는 그대로 둔다.
- NPGS 후보는 일반 검토에 추가하거나 기존 일반 검토의 우선순위를 높이는
  두 정책으로 분리해 보고한다.
- 자동 경보 지표와 human-assisted review 지표를 분리한다.
- 어떤 variant도 prospective 검증 전 자동 경보로 승격하지 않는다.

## 8. 산출물 계약

공통 출력 디렉터리는 다음과 같다.

`/Users/minho/Documents/Dataset/exp393_396_normal_process_grammar`

필수 산출물:

- prediction JSONL 및 SHA256
- dataset별 score와 candidate CSV
- TRAIN cross-fit calibration audit
- abstention/error manifest
- 기존 점수원과의 상관 및 후보 중복표
- lane transition 및 review workload 표
- 독립 재집계 결과
- 실행 명령, 코드 hash, 입력 hash가 포함된 MANIFEST
- retrospective/prospective 및 autonomous/human-assisted 범위 문서

기존 Exp391/392 및 운영 결과 파일은 덮어쓰지 않는다.

## 9. 성공 기준

성공은 Mean F1 하나로 판정하지 않는다.

### Calibration gate

- 같은 입력과 seed에서 prediction hash가 일치한다.
- pseudo-normal false-alert가 사전 등록 alpha와 명백히 모순되지 않는다.
- 오류 데이터셋과 abstention을 누락 없이 보고한다.

### New-information gate

- 적어도 하나의 NPGS 구성 점수가 모든 기존 점수원과 완전히 동일한 후보
  signature를 만들지 않는다.
- TRAIN OOF score 상관과 TEST value-only score 상관을 별도로 보고한다.
- 높은 상관이 나오더라도 숨기지 않으며, TEST 성능을 보고 descriptor를
  변경하지 않는다.

### Review-value gate

- 기존 FN 복구 수, 추가 FP, review 안팎의 precision을 함께 보고한다.
- TP가 한 dataset lineage에 집중되는지 leave-one-lineage-out 방식으로
  검사한다.
- 신규 후보가 기존 review보다 나쁘면 추가하지 않고 연구 결과로 종료한다.

### Future Hard-promotion gate

이번 Exp393-396에서는 평가하지 않는다. 향후 별도 사전등록과 신규 데이터
검증에서 autonomous precision 하한, workload, prospective stability를
통과해야만 자동 경보 승격을 검토한다.

## 10. 중단 조건

다음 조건에서는 후속 실험을 중단하고 blocker를 기록한다.

- prediction 단계가 TEST label 또는 실제 이상 위치를 읽음
- pseudo-test TRAIN instance가 자신의 model/calibration에 포함됨
- 유효한 source coverage가 누락되었는데 빈 후보로 성공 처리함
- TEST 결과를 보고 상태 수, descriptor, alpha 또는 가중치를 변경하려 함
- 기존 자동 경보 index가 Exp396 이전 단계에서 변경됨
- candidate/lane index가 TEST instance 범위를 벗어남
- Hard, Standard, Priority lane이 중복됨

## 11. 검증 테스트

- prediction 함수에 `y_test` 인자가 없는지 구조 검사
- label permutation 및 anomaly-position 변경 후 prediction hash 불변
- dataset/family 이름 변경 후 점수와 후보 불변
- 입력 순서 변경 후 dataset별 결과 불변
- 동일 seed 재실행 hash 일치
- TRAIN leave-one-out exclusion 검사
- 상수 trace, 짧은 trace, NaN/Inf, variable length 처리
- `n_train < 5` abstention
- candidate 범위 및 lane 상호 배타성
- evaluation 전 prediction hash tamper detection
- 기존 자동 경보 index 불변

## 12. 해석 범위

Exp393-396이 성공해도 다음만 말할 수 있다.

> 정상 운전의 상태 순서와 지속시간을 측정하는 신규 점수원이 기존 점수원과
> 다른 검토 근거를 제공하며, 과거 잠금 평가 데이터에서 일부 FN 복구 가능성을
> 보였다.

다음 주장은 사용할 수 없다.

- 학술적으로 완전히 새로운 SOTA 알고리즘이다.
- 전체 pipeline이 end-to-end strict TRAIN-only로 검증됐다.
- 실제 반도체 설비에서 성능이 검증됐다.
- 검토 후보가 자동 경보로 승격될 준비가 끝났다.
