# Exp163 불변성 검증 보고서

## 결론

Exp163의 label-free prediction과 분리 평가 계약은 통과했다. Full prediction은 1,117개 데이터셋, 4개 alpha, 2개 variant의 8,936행이며 누락과 중복이 없다.

## 검증 결과

- Python compile: 통과
- 단위/구조 테스트: 18/18 통과
- prediction rows: 8,936/8,936
- datasets: 1,117/1,117
- duplicate/missing dataset-alpha-variant key: 0/0
- prediction 평가 field: 0
- lane overlap: 0
- candidate/lane union mismatch: 0
- lane count mismatch: 0
- TEST index range 오류: 0
- Exp160 Hard index mismatch: 0
- Exp160 Priority index mismatch: 0
- candidate-zero target gate mismatch: 0
- source agreement 재계산 mismatch: 0
- 3-of-3이 2-of-3의 부분집합이 아닌 경우: 0
- 독립 TEST metric 재집계 mismatch: 0

## Prediction 동결

- prediction SHA256: `54fb3f1a94a268428b00835c7433d2ef0b033c8880a254c96c59d49e1477c655`
- 평가 전 SHA256: 동일
- 평가 후 SHA256: 동일
- full deterministic replay SHA256: 동일
- 10-dataset 정방향/역방향 입력 prediction SHA256: 동일

## 입력 동결

- Exp160 prediction SHA256: `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`
- Exp162 manifest SHA256: `ec8e1a1c617aa2216a81da1959bd1aec529ad7aadb908ababe3f7e05b7f288c1`
- Exp162 contract SHA256: `0a80d8a26387d81cd7b2e18ae36f3116c829d20742dd1fa79bb8a3c5bc3d6461`
- Exp163 prediction config SHA256: `be398861ff848a2674fc58893ad79c2bbd682d09f2d08268974e358856b062ed`

## 계약별 판정

| 항목 | 판정 | 근거 |
|---|---|---|
| Prediction 함수 TEST label 입력 없음 | PASS | 구조 테스트 및 prediction field 전수 검사 |
| TEST label은 prediction SHA 동결 후에만 읽음 | PASS | `run_evaluate()` 선행 manifest 검증, 평가 전후 SHA 동일 |
| 세 source 원시 점수 합산 없음 | PASS | source별 TRAIN quantile과 index 합의만 사용 |
| n_train < 5 완전 abstention | PASS | target gate 테스트와 전수 gate 재계산 |
| Hard/Priority 보존 | PASS | Exp160 대비 전수 index mismatch 0 |
| 새 후보 Standard-only | PASS | lane 및 added Standard 전수 재계산 |
| family/dataset 이름 정책 분기 없음 | PASS | dataset 이름은 cache/baseline lookup key로만 사용 |
| TEST 길이 budget/top-N 없음 | PASS | prediction config와 코드 감사 |
| 입력 순서 불변 | PASS | 10개 역순 smoke 및 full 정렬 출력 |
| 재현성 | PASS | full prediction SHA 재실행 동일 |

## 제한

`candidate_count == 0` gate는 TEST label이나 실제 이상 위치를 사용하지 않지만 frozen TEST score 결과에서 파생된다. 따라서 이 검증은 Exp163 policy의 label-free 재현성과 lane 안전성을 증명하며, 전체 개발 계보의 end-to-end strict TRAIN-only 또는 prospective 설비 성능을 증명하지 않는다.
