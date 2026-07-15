# Exp153–157 실행 가이드

## 1. 파일 복사

ZIP 내부 `experiments/exp153_157`을 저장소의 동일 경로에 복사합니다.

## 2. 단위·구조 테스트

```bash
python3 -m py_compile experiments/exp153_157/*.py
python3 -m unittest \
  experiments/exp153_157/test_exp153_157_train_only_improvements.py \
  experiments/exp153_157/test_exp153_157_structure.py
```

## 3. 10개 dataset smoke test

```bash
SMOKE=outputs/exp153_157_train_only_improvements_smoke

python3 experiments/exp153_157/run_experiment_153_157_pipeline.py \
  --stage cache \
  --dataset-limit 10 \
  --workers 2 \
  --reset-cache \
  --skip-db-hash \
  --output-dir "$SMOKE"

python3 experiments/exp153_157/run_experiment_153_157_pipeline.py \
  --stage 153 \
  --stage 154-predict --stage 154-evaluate \
  --stage 155-predict --stage 155-evaluate \
  --output-dir "$SMOKE"
```

Exp156/157은 기존 Exp152 full prediction과 Exp153 audit를 입력으로 사용하므로 full 저장소 artifact가 존재하는 환경에서 smoke 여부를 별도 확인합니다.

## 4. Full cache와 Exp153

```bash
OUT=outputs/exp153_157_train_only_improvements

python3 experiments/exp153_157/run_experiment_153_157_pipeline.py \
  --stage cache \
  --workers 7 \
  --reset-cache \
  --output-dir "$OUT"

python3 experiments/exp153_157/run_experiment_153_157_pipeline.py \
  --stage 153 \
  --output-dir "$OUT"
```

Exp153 결과를 먼저 검토합니다. 정상 pseudo-run false-alert control이 심각하게 실패하면 이후 policy 비교는 구조 진단으로만 실행합니다.

## 5. Exp154–157

```bash
python3 experiments/exp153_157/run_experiment_153_157_pipeline.py \
  --stage 154-predict --stage 154-evaluate \
  --stage 155-predict --stage 155-evaluate \
  --stage 156-predict --stage 156-evaluate \
  --stage 157-predict --stage 157-evaluate \
  --stage independent --stage manifest \
  --output-dir "$OUT"
```

## 6. 검증

Prediction manifest의 SHA256을 확인하고 prediction JSONL에 다음 필드가 없는지 검사합니다.

```text
y_test
hard_tp
hard_fp
hard_f1
combined_f1
```

각 result CSV에서 다음을 독립 재집계합니다.

- lane overlap 0
- candidate = Hard ∪ Standard ∪ Priority
- index 범위 오류 0
- dataset coverage 일치
- Hard와 review 지표 분리

## 7. 결과 선택 금지

현재 TEST 성능으로 아래를 선택하지 않습니다.

- Exp154 pooling variant
- Exp155 fusion variant
- Exp156 confirmation variant
- Exp157 risk target
- alpha

선택은 신규 잠금 데이터 또는 외부 운영 목표로만 수행합니다.
