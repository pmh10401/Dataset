# Exp397-399 실행 안내

전체 실행:

```bash
python3 experiments/exp397_399/run_experiment_397_399_pipeline.py \
  --stage all \
  --output-dir /Users/minho/Documents/Dataset/exp397_399_review_ranking
```

20개 데이터셋 smoke:

```bash
python3 experiments/exp397_399/run_experiment_397_399_pipeline.py \
  --stage all \
  --dataset-limit 20 \
  --output-dir /Users/minho/Documents/Dataset/exp397_399_review_ranking_smoke
```

결과는 `exp398_precision_at_k.csv`, `exp398_report.md`, `exp399_claims.md`에서 확인한다. TEST 결과가 가장 좋은 구성을 운영 기본값으로 선택하지 않는다.
