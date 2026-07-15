# Exp163 Alternative-Source Tail Agreement Review Design

## 목적

Exp163은 Exp160에서도 후보가 하나도 남지 않은 데이터셋에 Exp162의 세 대안 점수원을 적용해 일반 검토 후보를 보충하는 후향적 반사실 실험이다. 자동 경보와 우선 검토는 변경하지 않는다.

Exp162의 TRAIN 점수는 교차적합 OOF 점수가 아니다. 따라서 Exp163은 이 점수를 conformal p-value라고 부르지 않는다. 각 점수원의 TRAIN 경험적 상위 분위수를 고정 alpha별 threshold로 사용하고, 서로 다른 점수원이 같은 TEST index를 지목하는지 확인한다.

## 동결 입력

- 기준 lane: `outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`
- 기준 prediction SHA256: `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`
- 대안 점수 manifest: `outputs/exp162_full_coverage_alternative_score_cache/02_cache_manifest.csv`
- 대안 점수 manifest SHA256: `ec8e1a1c617aa2216a81da1959bd1aec529ad7aadb908ababe3f7e05b7f288c1`
- 대안 점수 contract: `outputs/exp162_full_coverage_alternative_score_cache/01_source_contract.json`
- 대안 점수 contract SHA256: `0a80d8a26387d81cd7b2e18ae36f3116c829d20742dd1fa79bb8a3c5bc3d6461`
- 데이터셋 population: 위 Exp160 prediction에 존재하는 1,117개 이름
- alpha sensitivity: `(0.005, 0.01, 0.02, 0.05)`

입력 해시나 1,117개 coverage가 다르면 prediction을 시작하지 않는다.

## 사용자용 점수 명칭

- `spectral`: 미분·스펙트럼 통계 점수
- `shapelet`: 부분 형태 일치 점수
- `interval`: 구간 분포 점수

원시 점수는 규모가 크게 다르므로 서로 더하거나 평균하지 않는다.

## TRAIN 경험적 꼬리 기준

각 dataset, source, alpha에 대해 다음을 수행한다.

1. 해당 source의 finite TRAIN 점수만 사용한다.
2. `q = 1 - alpha`를 계산한다.
3. `numpy.quantile(train_scores, q, method="higher")`를 threshold로 고정한다.
4. `test_score > threshold`인 TEST index만 source 후보로 만든다.

동점은 후보로 만들지 않는다. `n_train`이 작으면 여러 alpha가 같은 threshold를 만들 수 있으므로 dataset별 threshold 고유 개수와 source별 후보 수를 기록한다. 이는 오류가 아니라 경험적 분위수 해상도 한계다.

## 사전등록 variant

두 variant를 모두 실행하고 TEST 성능으로 하나를 선택하지 않는다.

### `tail_rank_2of3`

세 점수원 중 두 개 이상이 동일 TEST index를 후보로 선택하면 동의 후보로 인정한다.

### `tail_rank_3of3`

세 점수원 모두가 동일 TEST index를 후보로 선택해야 동의 후보로 인정한다. 더 보수적인 workload sensitivity다.

## 대상 gate와 lane 통합

각 Exp160 dataset/alpha row에서 다음 조건을 모두 만족할 때만 Exp163 후보를 계산한다.

- `n_train >= 5`
- Exp160 `candidate_count == 0`

이 gate는 TEST label이나 실제 이상 위치를 사용하지 않지만 frozen TEST score에서 파생된 후보 유무를 사용한다. 따라서 Exp163은 end-to-end strict TRAIN-only가 아니며 `uses_test_candidate_empty_gate=1`로 기록한다.

새 동의 후보는 일반 검토에만 추가한다.

- 자동 경보: Exp160과 index까지 완전 동일
- 우선 검토: Exp160과 index까지 완전 동일
- 일반 검토: 기존 일반 검토와 새 동의 후보의 합집합에서 자동 경보와 우선 검토를 제외
- 알림 없음: 어느 lane에도 포함되지 않은 index

`n_train < 5`는 완전 calibration abstention을 유지한다. top-N, TEST 길이 budget, family/dataset 이름 기반 분기, tail replacement는 사용하지 않는다.

## Prediction과 평가 분리

Prediction은 Exp160 prediction과 Exp162 score cache만 읽는다. TEST label, anomaly position, TP, FP, FN, precision, recall, F1은 읽거나 기록하지 않는다.

Prediction JSONL을 원자적으로 쓴 뒤 SHA256 manifest를 생성한다. 평가 함수는 manifest의 prediction SHA256을 검증한 다음에만 DB에서 TEST label을 읽는다. 동결 이후 prediction 파일은 수정하지 않는다.

예상 prediction 행 수는 다음과 같다.

`1,117 datasets x 4 alphas x 2 variants = 8,936 rows`

## 평가 계약

모든 variant/alpha를 sensitivity로 보고한다.

- 자동 경보 수, TP, FP, micro precision, mean dataset F1
- 일반 검토 수, TP, FP, micro precision
- 우선 검토 수, TP, FP, micro precision
- 새 일반 검토 수, TP, FP, precision
- 후보 0 dataset 수 전후
- dataset별 표시 후보 수 mean, median, p90, p95, max
- lane overlap, union mismatch, count mismatch, index range 오류
- Exp160 대비 자동 경보와 우선 검토 index 불변성
- source별 후보 수와 2-of-3/3-of-3 합의 수
- alpha별 threshold 고유 개수와 해상도 제한 dataset 수
- `mean_combined_f1`은 human-assisted 진단으로만 표시

TEST 결과를 보고 alpha, variant, threshold 규칙을 수정하거나 운영값으로 선택하지 않는다.

## 불변성 및 중단 조건

다음을 테스트한다.

- prediction 함수가 `y_test`를 받지 않는다.
- prediction JSONL에 평가 field가 없다.
- TEST label 변경은 prediction에 영향을 주지 않는다.
- 입력 행 순서 변경 후 dataset별 정렬 출력 hash가 같다.
- 동일 입력과 seed 없는 결정적 규칙에서 결과 hash가 같다.
- cache 또는 Exp160 prediction hash가 바뀌면 실행이 실패한다.
- dataset 이름은 lookup key로만 사용되며 정책 분기에 사용되지 않는다.
- Hard/Priority lane은 Exp160과 완전히 같다.
- 모든 lane은 상호 배타적이고 합집합이 candidate와 같다.

1,117개 cache coverage, 필수 배열, 길이, finite 값, config hash 중 하나라도 맞지 않으면 full prediction을 중단한다. 일부 실패를 빈 후보로 바꾸지 않는다.

## 해석 경계

Exp163이 성공해도 말할 수 있는 것은 고정된 세 대안 점수원의 TRAIN 경험적 꼬리 합의가 Exp160 후보 0 상태의 일반 검토 coverage를 후향적으로 보충했는지 여부다. 전체 pipeline의 end-to-end strict TRAIN-only, 특정 alpha/variant의 운영 최적성, 실제 설비 prospective 성능을 증명하지 않는다.
