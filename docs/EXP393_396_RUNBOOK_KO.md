# Exp393-396 실행 안내

## 테스트

```bash
python3 -m unittest \
  experiments.exp393_396.test_preprocessing \
  experiments.exp393_396.test_grammar_core \
  experiments.exp393_396.test_structure \
  experiments.exp393_396.test_pipeline
```

## 10개 데이터셋 smoke

```bash
python3 experiments/exp393_396/run_experiment_393_396_pipeline.py \
  --stage all \
  --dataset-limit 10 \
  --workers 2 \
  --output-dir /Users/minho/Documents/Dataset/exp393_396_normal_process_grammar_smoke
```

## 전체 실행

```bash
python3 experiments/exp393_396/run_experiment_393_396_pipeline.py \
  --stage all \
  --workers 7 \
  --output-dir /Users/minho/Documents/Dataset/exp393_396_normal_process_grammar
```

## 독립 재집계

```bash
python3 experiments/exp393_396/independent_reaggregate_exp393_396.py \
  --output-dir /Users/minho/Documents/Dataset/exp393_396_normal_process_grammar
```

## 완료 확인

- `status.json`의 `status`가 `completed`
- `exp393_prediction_manifest.json`의 prediction SHA256 일치
- 전체 실행에서 dataset 또는 명시적 abstention 합계 1,117
- prediction JSONL에 TEST 라벨과 TP/FP/F1 없음
- `exp396_integration_manifest.json`의 `hard_lane_unchanged=true`
- `independent_consistency_report.json`의 `consistent=true`

중간 실패 시 후속 단계를 완료로 간주하지 않는다. 동일 output에 다시
실행하기 전 기존 prediction과 manifest의 hash를 먼저 보존한다.
