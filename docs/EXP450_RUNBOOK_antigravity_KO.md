# Exp450 실행 Runbook

## 고정 인터프리터와 명령

작업 디렉터리는 저장소 루트이며, 모든 명령은 아래 고정 인터프리터를 사용한다.

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py readiness --mode smoke
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py smoke
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py verify --mode smoke
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py run
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py verify --mode full
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py evaluate
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python -B scratch/run_exp450_resampling_image_ablation_antigravity.py report
```

`status`는 언제든 실행할 수 있다. `smoke`와 `run`은 완료 또는 실패한 데이터셋×조건마다 JSON 한 줄을 내보내며 `elapsed_seconds`와 `eta_seconds`를 포함한다.

## 순서와 회복

1. `readiness`에서 고정 root와 상태를 확인한다.
2. 별도 권한이 확인된 뒤에만 23개 데이터셋 `smoke`를 실행한다.
3. 실패하면 root를 삭제하거나 덮어쓰지 말고 같은 `smoke` 명령을 다시 실행한다. 두 wave의 create-only checkpoint/failure ledger가 재개 위치를 결정한다.
4. smoke 검증과 별도 전체 실행 권한 뒤에만 `run`을 실행한다. `verify`가 A–D 네 full manifest의 존재를 확인한다.
5. `evaluate`는 A–D full manifest와 세 public-baseline root 검증 전에는 거부된다. 그 후에만 TEST label capability가 열린다.
6. `report`는 canonical evaluation artifact가 존재할 때만 실행한다.

어떤 단계에서도 root를 수동으로 수정하거나 결과를 이용해 A–D를 재선택하지 않는다. MLX는 평가되지만 동결된 공식 CPU transformer를 대체하지 않으며, MPS heavy 작업은 한 번에 하나만 실행한다.
