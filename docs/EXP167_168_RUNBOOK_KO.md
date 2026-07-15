# Exp167/168 실행 안내

## 준비

저장소 루트에서 실행한다.

```bash
python3 -m py_compile experiments/exp167_168/*.py
python3 -m unittest \
  experiments/exp167_168/test_exp167_168_parity_core.py \
  experiments/exp167_168/test_exp167_168_structure.py
```

## Smoke

```bash
SMOKE=outputs/exp167_168_score_calibration_parity_smoke
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase audit --dataset-limit 10 --workers 2 --reset-checkpoint \
  --output-dir "$SMOKE"
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase predict --dataset-limit 10 --workers 2 --reset-checkpoint \
  --output-dir "$SMOKE"
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase evaluate --dataset-limit 10 --output-dir "$SMOKE"
```

Smoke prediction은 200행이어야 하며 TEST 라벨과 TP/FP/F1 필드를 포함하면 안 된다.

## 전체 실행

```bash
OUT=outputs/exp167_168_score_calibration_parity
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase audit --workers 7 --reset-checkpoint --output-dir "$OUT"
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase predict --workers 7 --reset-checkpoint --output-dir "$OUT"
python3 experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py \
  --phase evaluate --output-dir "$OUT"
```

## 완료 조건

- audit: 316개 데이터셋, 오류 0개
- source audit: 948행
- prediction: 6,320행, 데이터셋당 20행
- prediction SHA256이 manifest와 일치
- prediction에 TEST 라벨과 평가 지표 없음
- candidate index 범위 오류 없음
- evaluation summary: 20행
- `selected_policy=0`

`cache/`, checkpoint, smoke 출력은 재실행용 로컬 산출물이며 Git에 포함하지 않는다.
