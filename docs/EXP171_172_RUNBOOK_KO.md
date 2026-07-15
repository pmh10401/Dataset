# Exp171/172 실행 안내

## 검증

```bash
python3 -m py_compile experiments/exp171_172/*.py
python3 -m unittest discover -s experiments/exp171_172 -p 'test_*.py'
```

## Smoke

```bash
SMOKE=outputs/exp171_172_train_risk_policy_smoke
python3 experiments/exp171_172/run_experiment_171_172_train_risk_policy.py \
  --phase audit --output-dir "$SMOKE"
python3 experiments/exp171_172/run_experiment_171_172_train_risk_policy.py \
  --phase predict --dataset-limit 10 --output-dir "$SMOKE"
python3 experiments/exp171_172/run_experiment_171_172_train_risk_policy.py \
  --phase evaluate --output-dir "$SMOKE"
```

## Full

```bash
OUT=outputs/exp171_172_train_risk_policy
python3 experiments/exp171_172/run_experiment_171_172_train_risk_policy.py \
  --phase audit --output-dir "$OUT"
python3 experiments/exp171_172/run_experiment_171_172_train_risk_policy.py \
  --phase predict --output-dir "$OUT"
python3 experiments/exp171_172/run_experiment_171_172_train_risk_policy.py \
  --phase evaluate --output-dir "$OUT"
python3 experiments/exp171_172/independent_reaggregate_exp171_172.py \
  --output-dir "$OUT"
```

`evaluate`는 frozen policy와 prediction SHA256 검증 후에만 TEST label을 읽는다.
결과가 좋거나 나쁘더라도 이번 실험에서 policy를 다시 고르지 않는다.
