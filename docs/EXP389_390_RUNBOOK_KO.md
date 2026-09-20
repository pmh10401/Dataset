# Exp389-390 실행 방법

```bash
python3 -m unittest experiments.exp389_390.test_exp389_390
python3 experiments/exp389_390/run_experiment_389_dual_confidence_prediction.py
python3 experiments/exp389_390/run_experiment_390_dual_confidence_evaluation.py
```

순차 실행 큐에서는 반드시 Exp389를 먼저 실행하고 Exp390을 뒤에 실행한다.

```bash
python3 run_rank_experiments_sequential.py queue experiment_389_dual_confidence_prediction
python3 run_rank_experiments_sequential.py queue experiment_390_dual_confidence_evaluation
python3 run_rank_experiments_sequential.py run-queue
```

결과 경로:

`/Users/minho/Documents/Dataset/exp389_390_dual_confidence_tiers`
