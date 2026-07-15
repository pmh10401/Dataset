# Exp165 사전등록: 소표본 TRAIN-only 유사도 풀링 교정 감사

## 목적

Exp164에서 `5 <= n_train < 20`인 데이터셋은 네 alpha가 같은 임계값으로
퇴화하고 정상 오검토 위험도 높았다. Exp165는 TEST를 전혀 사용하지 않고
다른 TRAIN-normal 데이터셋의 유사한 점수 분포를 참조하여 교정 표본을
늘렸을 때 이 문제가 완화되는지 감사한다.

## 동결 입력

- DB: `/Users/minho/Documents/Dataset/univariate_ts.db`
- DB SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Exp164 contract SHA256: `cd5af2984efb88d85d8de380dacb5c11016d2e3f7c9a2dbaf61d9b98c78448e0`
- Exp164 manifest SHA256: `a3226c4c30f5217440126ebab1b502fd11e3dceec64451b7be88437e845e439e`
- 전체 population: 1,117개
- 대상: `5 <= n_train < 20`인 316개
- donor: `n_train >= 20`인 639개
- `n_train < 5`: 162개 완전 기권 유지

## 고정 설정

- sources: 미분·스펙트럼 통계, 부분 형태 일치, 구간 분포
- alphas: `0.005, 0.01, 0.02, 0.05`
- variants: `2-of-3`, `3-of-3`
- minimum pooled reference: source별 200
- threshold: `np.quantile(reference, 1-alpha, method="higher")`
- candidate: normalized query score가 threshold를 strict 초과

## 유사도 풀링

각 target pseudo-run은 target의 정규화와 descriptor에서 제외한다. 나머지
target OOF 점수로 source별 robust normalization과 7개 분포 통계를 만들고,
세 source를 연결한 21차원 descriptor를 사용한다. Donor descriptor는 donor
TRAIN-normal OOF 점수만으로 만든다.

거리순으로 donor를 추가해 source별 reference가 200개 이상이 되게 한다.
경계 거리 동률 donor는 모두 포함한다. Target 자신은 reference에 들어가지
않으며 dataset/family 이름은 조회 key 외 정책에 사용하지 않는다.

## 금지 사항

- TEST 값, 라벨, 이상 위치, 길이 또는 후보 수 사용
- 과거 family별 TEST 성능 사용
- 결과를 본 뒤 alpha, pool size, descriptor 또는 variant 변경
- Hard/Priority 생성 또는 승격
- tail replacement 또는 TEST-length budget 도입
- `n_train < 5` 데이터에 임의 후보 생성

## 결과 해석

Exp165는 정상 false-review calibration audit이다. anomaly F1이나 실제 설비
성능을 측정하지 않는다. 모든 variant는 sensitivity로 함께 보고하며 특정
설정을 운영 기본값으로 선택하지 않는다.
