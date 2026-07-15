# Exp167/168 결과 분석

## 한 줄 결론

동일 fold 모델로 TRAIN 정상과 TEST를 점수화해 생성 조건은 맞췄지만, 316개
소표본 데이터셋에서는 empirical p-value의 해상도가 너무 낮아 후보가 사실상
사라졌다. Exp167/168은 parity 구현 검증에는 성공했지만 운영 탐지 정책 후보로는
기각해야 한다.

## 실험 범위

- 대상: `5 <= n_train < 20`인 316개 데이터셋
- TEST instance: 데이터셋당 50개, 총 15,800개
- 고정 alpha: 0.005, 0.01, 0.02, 0.05
- 점수원: 주파수 변화, 정상 패턴 거리, 구간 통계
- 조합: 세 쌍, 2-of-3, 3-of-3
- 출력: 20개 고정 조합
- lane: retrospective research review candidate
- autonomous Hard alert: 평가하지 않음
- 선택된 alpha 또는 조합: 없음

## 핵심 결과

alpha 0.005, 0.01, 0.02에서는 모든 조합의 후보가 0개였다. alpha 0.05에서만
`n_train=19`인 일부 데이터셋에 후보가 생성됐다.

| alpha | 조합 | 후보 | TP | FP | Precision | Mean F1 | Zero-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.005/0.01/0.02 | 모든 조합 | 0 | 0 | 0 | 0.00% | 0.0000 | 316 |
| 0.05 | 주파수 변화 + 정상 패턴 거리 | 2 | 2 | 0 | 100.00% | 0.0063 | 314 |
| 0.05 | 주파수 변화 + 구간 통계 | 4 | 3 | 1 | 75.00% | 0.0084 | 313 |
| 0.05 | 정상 패턴 거리 + 구간 통계 | 2 | 2 | 0 | 100.00% | 0.0063 | 314 |
| 0.05 | 2-of-3 | 4 | 3 | 1 | 75.00% | 0.0084 | 313 |
| 0.05 | 3-of-3 | 2 | 2 | 0 | 100.00% | 0.0063 | 314 |

표본이 극히 적으므로 alpha 0.05의 높은 precision은 성능 우위로 해석하면 안 된다.

## 후보가 사라진 이유

finite-sample 보정 p-value의 최소값은 `1 / (n_train + 1)`이다. 예를 들어 정상
TRAIN이 10개면 가능한 최소 p-value는 약 0.0909다. alpha 0.05보다 작아질 수
없으므로 아무리 큰 TEST 점수도 후보가 되지 않는다.

316개 중 `n_train=19`인 데이터셋은 7개뿐이다. 따라서 alpha 0.05에서 309개가
해상도 제한 상태였고, 더 작은 세 alpha에서는 316개 전부가 제한됐다. 실제 후보는
3개 데이터셋의 4개 위치에만 나타났다.

## Exp166 Local과 비교

| 방법 | 조합 | 후보 | TP | FP | Mean F1 |
|---|---|---:|---:|---:|---:|
| Exp166 Local | 2-of-3 | 265 | 131 | 134 | 0.3797 |
| Exp168 alpha 0.05 | 2-of-3 | 4 | 3 | 1 | 0.0084 |
| Exp166 Local | 3-of-3 | 50 | 41 | 9 | 0.1271 |
| Exp168 alpha 0.05 | 3-of-3 | 2 | 2 | 0 | 0.0063 |

Exp168 후보는 Exp166 Local 후보의 부분집합이었다. 2-of-3에서는 261개 후보가
제거되며 TP 128개와 FP 133개가 함께 사라졌다. FP를 줄였지만 탐지 가능성도 거의
모두 잃었으므로 false-alarm 개선으로 채택할 수 없다.

## parity 감사에서 확인된 점

fold 모델과 full-TRAIN 모델의 TEST 순위 상관 중앙값은 주파수 변화 0.990,
정상 패턴 거리 0.868, 구간 통계 0.991이었다. 정상 패턴 거리 점수는 316개 중
206개 데이터셋에서 상관이 0.9 미만이었고, 구간 통계도 109개에서 0.9 미만이었다.
즉 Exp166에서 의심한 fit 조건 민감성은 실제로 존재한다. 다만 조건을 정확히
맞추자 소표본 p-value 해상도라는 더 근본적인 제한이 드러났다.

## 판단

- Exp168을 운영 기본 calibration으로 채택: **기각**
- Exp168 후보를 자동 경보로 사용: **금지**
- 특정 alpha 또는 점수원 조합 선택: **하지 않음**
- fold-matched scoring 구현: 향후 calibration 연구의 검증 도구로 유지
- 다음 과제: 데이터셋 내부 local empirical p-value만으로 해결하려 하지 말고,
  TEST를 보지 않고 사전 등록한 pooled/hierarchical reference 또는 소표본에 맞는
  연속 calibration 방법을 별도 실험 번호로 검증

## 근거 파일

- `02_parity_audit.csv`: 316개 데이터셋의 p-value 해상도 감사
- `03_fold_score_summary.csv`: 점수원별 fold/full-fit 비교
- `04_frozen_predictions.jsonl`: label-free prediction 6,320행
- `05_prediction_manifest.json`: 동결 SHA와 실행 계약
- `06_exp168_dataset_metrics.csv`: 후향 데이터셋별 평가
- `07_exp168_global_summary.csv`: 20개 고정 조합 요약
- `08_candidate_transition.csv`: Exp166 Local 대비 후보 이동

모든 성능 수치는 동일 TEST benchmark에 대한 retrospective human-assisted review
지표다. 실제 설비 prospective 또는 autonomous Hard alert 성능이 아니다.
