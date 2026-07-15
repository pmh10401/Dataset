# Exp169/170 실행 안내

## 경로

- 저장소: `/Users/minho/Documents/timesries project`
- DB: `/Users/minho/Documents/Dataset/univariate_ts.db`
- 결과: `outputs/exp169_170_oob_pooled_calibration`

## 검증

```bash
python3 -m py_compile experiments/exp169_170/*.py
python3 -W error::ResourceWarning -m unittest discover \
  -s experiments/exp169_170 -p 'test_*.py'
```

## 순차 실행

```bash
OUT=outputs/exp169_170_oob_pooled_calibration

python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase cache --workers 7 --reset-checkpoint --output-dir "$OUT"

python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase audit --output-dir "$OUT"

python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase predict --workers 7 --output-dir "$OUT"

python3 experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py \
  --phase evaluate --output-dir "$OUT"

python3 experiments/exp169_170/independent_reaggregate_exp169_170.py \
  --output-dir "$OUT"
```

## 실행 장벽

- cache 단계는 DB SHA256과 Exp164 입력 hash를 검증한다.
- audit와 prediction은 cache 파일 316개의 hash, config, shape, finite value를 검증한다.
- prediction은 pool manifest를 독립 재계산하여 donor와 reference hash를 대조한다.
- evaluation은 contract, prediction, cache manifest, pool manifest hash를 먼저 검증한다.
- prediction은 316개 데이터셋마다 60개 variant가 정확히 한 번씩 있어야 한다.
- 독립 재집계는 prediction과 metrics key, 후보 수, 인덱스 범위, `TP+FP`를 다시 확인한다.

## 대시보드

cache와 audit/prediction 단계는 `/Users/minho/Documents/Dataset/rank_dashboard_external_live_run.json`을 갱신한다. Exp169/170은 완료 후에도 연구 결과로 표시되며 운영 모델 비교에서는 제외된다.
