# Exp206-210 실행 가이드 (Runbook)

## 1. 개요
본 배치는 총 5개의 스테이지(`206` ~ `210`)로 구성되며 순차적으로 실행되어야 합니다.

## 2. 스테이지 상세 및 실행 명령어

### 2.1. Smoke 테스트 (2개의 데이터셋만 사용하여 신속 검증)
각 스테이지가 정상 작동하는지 확인하기 위해 데이터셋 제한을 걸어 실행합니다:
```bash
python3 experiments/exp206_210/run_experiment_206_210.py --stage 206 --dataset-limit 2 --workers 1 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 207 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 208 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 209 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
python3 experiments/exp206_210/run_experiment_206_210.py --stage 210 --dataset-limit 2 --output-root /tmp/exp206_210_smoke
```

### 2.2. 전체 배치 실행 (Full Run)
전체 251개 데이터셋에 대해 7개의 워커로 캐싱을 실행하는 대규모 배치 명령어입니다:
```bash
# 1. 시퀀셜 배치 큐 등록
python3 run_rank_experiments_sequential.py queue-exp206-210

# 2. 백그라운드로 배치 실행 시작
nohup python3 run_rank_experiments_sequential.py run \
  > /Users/minho/Documents/Dataset/exp206_210_residual_coverage/queue_runner.log 2>&1 &
```

## 3. 리스크 통제 및 승인 조건
- **retrospective_80pct_point_gate**: 사후 정밀도(Point Precision)가 최소 80% 이상이고 자율 경보의 Zero-F1 상태가 순 감소해야만 향후 승인 논의를 시작할 수 있습니다.
- **Exp204 Prospective Gate**: 실제 운영 시스템 반영은 대규모의 prospective shadow 결과 및 엄격한 Wilson 하한 검증을 통과해야 논의될 수 있으며 본 배치 결과만으로 프로모션될 수 없습니다.
