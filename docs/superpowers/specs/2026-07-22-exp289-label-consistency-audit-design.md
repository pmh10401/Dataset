# Exp289 Recovery Priority Label Consistency Audit Design

## 목적

Exp288에서 확인된 Recovery Priority 후보 43건(11 TP, 32 FP)이 TRAIN 정상 또는 반대 TEST 라벨 파형과 동일하거나 거의 동일한지 감사한다. 이 실험은 데이터 정합성의 한계를 설명하기 위한 retrospective diagnostic이며 자동 경보 정책을 변경하지 않는다.

## 고정 모집단

- Exp286 `synthetic_shape_alpha_005` Recovery Priority 후보 43건
- 전체 prediction coverage 1,117개 데이터셋
- 실제 후보가 존재하는 27개 데이터셋
- Exp286 prediction SHA256 검증 후에만 TEST 라벨을 읽는다.

## 비교 대상

각 후보를 다음 두 집합과 비교한다.

1. 같은 데이터셋의 모든 TRAIN 정상 인스턴스
2. 같은 데이터셋에서 후보와 반대 TEST 라벨을 가진 모든 인스턴스

추가로 후보가 존재하는 27개 데이터셋 전체에서 TEST 정상과 TEST 이상 사이의 완전 동일 파형 충돌을 해시 기반으로 집계한다.

## 고정 유사도 정의

결과를 본 뒤 기준을 바꾸지 않는다.

- `raw_exact`: 원래 길이가 같고 sanitize 후 float32 값이 완전히 동일
- `aligned_exact`: TRAIN 기준 길이 정렬 후 float32 값이 완전히 동일
- `normalized_equivalent`: 인스턴스별 표준화 후 RMSE가 `1e-6` 이하
- `near_duplicate`: 표준화 RMSE가 `0.01` 이하이고 correlation distance가 `1e-4` 이하
- 그 외는 `distinct`

상수 파형은 두 파형이 모두 상수이고 정렬된 값이 같을 때만 normalized equivalent로 인정한다. 각 연속 거리값도 함께 저장하여 고정 범주가 정보를 숨기지 않게 한다.

## 출력

- 후보 43건의 TRAIN 정상 및 반대 TEST 라벨 최근접 비교
- 데이터셋별 후보·중복·라벨 충돌 집계
- 후보가 있는 27개 데이터셋의 TEST 정상/이상 exact conflict 목록
- TP와 FP별 중복 유형 분포
- 대표 충돌 사례의 양쪽 파형 CSV
- `RESULT_REPORT.md`

## 해석 규칙

- 반대 라벨 exact/near duplicate는 파형 단독 분류의 식별 가능성 문제로 해석한다.
- 이것만으로 라벨이 잘못됐다고 단정하지 않는다. 공정 문맥이 라벨에 반영되었을 가능성이 있다.
- TEST 라벨을 사용한 결과이므로 prediction, threshold 또는 routing 개선 근거로 직접 사용하지 않는다.
- 동일 파형 제외 규칙을 자동 경보에 추가하지 않는다.
- Exp253 Hard alert와 Exp287 Recovery lane은 변경하지 않는다.

## 성공 조건

- 43건이 누락 없이 감사됨
- 1,117개 prediction coverage와 Exp286 hash가 검증됨
- 후보 인덱스 범위 오류와 비교 오류가 0건
- exact/near 기준이 코드 상수로 고정됨
- TP/FP별 결과와 연속 거리값이 독립 재집계 가능함

## 제한

Exp289는 retrospective counterexample audit이다. 전체 데이터 생성 과정의 정답성을 증명하지 않으며, 실제 설비의 prospective 성능이나 end-to-end strict TRAIN-only 정책을 검증하지 않는다.
