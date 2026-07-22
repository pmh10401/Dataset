# Exp173/174 실행 런북

## 작업 위치

```bash
cd "/tmp/timeseries-exp171-172-train-risk-policy"
```

기준 DB는 `/Users/minho/Documents/Dataset/univariate_ts.db`이며 경로와 SHA256을
변경하지 않는다.

## 1. 테스트

```bash
python3 -m py_compile experiments/exp173_174/*.py
python3 -m unittest discover -s experiments/exp173_174 -p 'test_*.py'
python3 scratch/test_rank_queue_safety.py
python3 scratch/test_rank_dashboard_operational.py
```

테스트가 실패하면 smoke 또는 full 실행을 시작하지 않는다.

## 2. Smoke

```bash
SMOKE=outputs/exp173_174_balanced_train_selector_smoke
python3 experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py \
  --phase audit --output-dir "$SMOKE"
python3 experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py \
  --phase predict --dataset-limit 10 --output-dir "$SMOKE"
python3 experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py \
  --phase evaluate --output-dir "$SMOKE"
```

10개 prediction에 label, TP, FP, F1이 없어야 한다. 후보 인덱스는 정렬되고
중복이 없으며 각 TEST 범위 안에 있어야 한다.

## 3. Full 순차 실행

다른 experiment Python process가 없음을 확인한 뒤 실행한다.

```bash
python3 run_rank_experiments_sequential.py status
pgrep -af 'run_experiment_|run_rank_experiments_sequential' || true

python3 run_rank_experiments_sequential.py \
  run experiment_173_balanced_train_selector_freeze
```

Exp173 policy가 abstain이면 Exp174를 실행하지 않는다. 선택된 경우
`03_exp173_frozen_policy.json`의 SHA256을 기록한 뒤 실행한다.

```bash
python3 run_rank_experiments_sequential.py \
  run experiment_174_balanced_frozen_review
```

## 4. 독립 검증

```bash
python3 experiments/exp173_174/independent_reaggregate_exp173_174.py
git diff --check
```

`11_independent_reaggregation.json`의 `mismatch_count`가 0이어야 한다.

## 5. 보고 범위

다음을 서로 분리해 보고한다.

- TRAIN 정상 위험: 정상 OOB review rate와 동시 Wilson 상한
- 사후 탐지 범위: review TP, recall, mean F1, zero-F1 데이터셋
- 사용자 부담: review FP, precision, 데이터셋별 후보 수 분위수

Exp174는 review-only retrospective 결과다. Hard alert 성능, 실제 설비 성능,
end-to-end strict TRAIN-only 개발 계보 또는 prospective 검증 완료라고 표현하지
않는다.
