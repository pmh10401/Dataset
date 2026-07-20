# Exp254 MatrixProfile non-finite correction

## 발견된 경고

초기 full run의 105/1,117 시점에 MatrixProfile 계산에서 `divide by zero`와
`invalid value` RuntimeWarning이 반복됐다. 실패 데이터셋은 없었지만 로그에
278개의 RuntimeWarning이 기록됐다.

## 원인

일정한 값이 반복되는 subsequence의 표준편차는 0이다. sktime
`MatrixProfileFeatures`가 z-normalized 거리를 계산할 때 이 subsequence에서
분모가 0이 되어 profile에 `inf` 또는 `NaN`이 생성됐다. 기존 구현은 profile
전체에 mean/quantile을 바로 적용하여 일부 유한 거리가 있어도 여러 요약 특징이
함께 비유한 값으로 오염됐다.

## 수정

- 각 series의 matrix profile에서 유한한 거리만 선택해 8개 통계를 계산한다.
- 유한한 거리가 전혀 없는 행은 8개 NaN을 유지한다.
- 해당 NaN은 기존 TRAIN-normal median sanitizer가 보정한다.
- sktime 내부의 알려진 0-분산 RuntimeWarning은 MatrixProfile 호출 범위에서만 억제한다.
- policy 이름을 source config에 추가해 기존 cache와 config hash를 분리한다.

## 검증

- 혼합 finite/inf/NaN profile 요약 테스트 통과
- 완전 상수 profile이 RuntimeWarning을 내지 않는 테스트 통과
- 전체 Exp254 단위 테스트 12개 통과
- 실제 경고 데이터셋 `AllGestureWiimoteZ_normal_10` 재계산 결과:
  - score arrays: 10
  - all finite: true
  - RuntimeWarning: 0

초기 105개 partial cache는 결과로 사용하지 않고 `--reset-cache`로 전체 재생성한다.

## SAX panel-input correction

재실행 중 SAX의 scipy z-score에서 catastrophic-cancellation RuntimeWarning이
발견됐다. 조사 결과 `SAX`는 한 Series 입력용인데 panel 전체를 전달하여 z-score가
instance 축을 함께 사용하고 있었다. 이는 한 TEST instance의 표현이 다른 TEST
instance에 의존하게 만들 수 있다.

- SAX를 시계열별 1차원 입력으로 독립 변환한다.
- 값 범위가 scale의 `1e-7` 이하인 near-constant series는 중립 심볼 4로 변환한다.
- scipy가 정밀도 RuntimeWarning을 보고한 series도 중립 심볼로 안전 처리한다.
- 단일 변환과 여러 instance 동시 변환에서 첫 series 결과가 동일한지 테스트한다.
- 수정 전 partial cache는 결과에 사용하지 않고 다시 전체 생성한다.
