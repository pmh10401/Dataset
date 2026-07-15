# Exp165 실행 가이드

## 테스트

```bash
python3 -m py_compile experiments/exp165/*.py experiment_progress.py
python3 -m unittest \
  experiments/exp165/test_exp165_small_sample_calibration.py \
  experiments/exp165/test_exp165_structure.py \
  scratch/test_rank_dashboard_operational.py \
  scratch/test_rank_queue_safety.py
```

## Smoke

```bash
python3 experiments/exp165/run_experiment_165_small_sample_calibration.py \
  --dataset-limit 10 \
  --workers 2 \
  --reset-checkpoint \
  --output-dir outputs/exp165_small_sample_calibration_smoke
```

`--dataset-limit`은 target만 제한하며 donor universe 639개는 유지한다.

## Queue 실행

```bash
python3 run_rank_experiments_sequential.py queue experiment_165_small_sample_calibration
RANK_EXPERIMENT_WORKERS=7 python3 run_rank_experiments_sequential.py run-queue
```

## 대시보드

```bash
python3 dashboardctl.py restart
python3 dashboardctl.py status
```

로컬 주소: `http://127.0.0.1:8765/`
