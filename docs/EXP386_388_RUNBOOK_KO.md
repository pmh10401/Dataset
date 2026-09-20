# Exp386-388 실행 안내

```bash
python3 -m unittest experiments.exp386_388.test_exp386_388
python3 experiments/exp386_388/run_experiment_386_exact_replay.py
python3 experiments/exp386_388/run_experiment_387_alpha_sweep.py
python3 experiments/exp386_388/run_experiment_388_alpha_comparison.py
```

결과 위치:

`/Users/minho/Documents/Dataset/exp386_388_existing_evidence_alpha_sweep`

순차 큐에서는 Exp386 실패 시 자동으로 멈추며 Exp387을 실행하지 않는다.
대시보드 진행률은 역사 코드의 status JSON을 바깥 실행기가 읽어 표시한다.
