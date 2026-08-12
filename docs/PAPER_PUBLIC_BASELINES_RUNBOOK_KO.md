# 공개 이상 탐지 비교 실행서

## 1. 실행 원칙

저장소 root에서 실행한다. 대용량 산출물은 저장소가 아니라
`/Users/mac/Documents/Dataset/paper_public_baselines_v1` 아래에 둔다.
완료 shard는 검증 후 재사용되므로 중단 뒤 같은 명령을 다시 실행할 수 있다.
기존 산출물이 현재 설정과 다르면 덮어쓰지 않고 실패한다.

전체 실행 전에 아래 테스트를 통과시킨다.

```bash
python3 -m unittest discover -s experiments/paper_public_baselines/tests -v
```

## 2. 소규모 smoke

공식 root와 분리된 root에서 모집단 앞 3개만 실행한다.

```bash
python3 scratch/run_paper_public_baselines.py prepare \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_smoke \
  --limit 3

python3 scratch/run_paper_public_baselines.py assess-mlx \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_smoke

python3 scratch/run_paper_public_baselines.py score \
  --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest,deep_svdd_1d,drocc_1d \
  --workers 1 --epochs 2 \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_smoke

python3 scratch/run_paper_public_baselines.py freeze \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_smoke
```

smoke의 2 epochs 결과는 논문 성능표에 사용하지 않는다.

## 3. 공식 1,117개 실행

### 3.1 Prepare와 MLX 판정

```bash
python3 scratch/run_paper_public_baselines.py prepare \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1

python3 scratch/run_paper_public_baselines.py assess-mlx \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
```

`prepared.json`에서 DB hash, `full_dataset_count=1117`, 선택 데이터셋 1,117개를
확인한다. `acceleration.json`에서는 6개 방법 모두 CPU를 선택하고, MLX 및
mlx-metal 버전과 조건 1 미충족 사유가 기록되어야 한다.

### 3.2 고전 방법

```bash
python3 scratch/run_paper_public_baselines.py score \
  --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest \
  --workers 7 --epochs 50 \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
```

scikit-learn 내부 `n_jobs`는 1이며 dataset worker만 최대 7개 사용한다.

### 3.3 심층 방법

두 방법은 한 worker로 순차 실행한다. 명령을 나누더라도 같은 50 epochs와
같은 run root를 유지한다.

```bash
python3 scratch/run_paper_public_baselines.py score \
  --methods deep_svdd_1d \
  --workers 1 --epochs 50 \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1

python3 scratch/run_paper_public_baselines.py score \
  --methods drocc_1d \
  --workers 1 --epochs 50 \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
```

심층 pilot로 산출한 예상 시간이 현실적으로 불가능하면 완료된 고전 비교는
보존하고 심층 coverage 미완료를 명시한다. TEST 결과를 보고 epochs를 줄여
공식 결과로 바꾸지 않는다.

### 3.4 Freeze

```bash
python3 scratch/run_paper_public_baselines.py freeze \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1
```

6 × 1,117 = 6,702개 dataset-method pair가 모두 complete 또는 사전등록
abstain이어야 freeze가 성공한다. shard가 하나라도 없거나 추가·변조되었으면
실패한다.

## 4. 장애 복구

- 같은 명령 재실행: digest가 일치하는 완료 shard는 재사용한다.
- shard 또는 meta 한쪽만 존재: 불완전 artifact로 간주해 실패한다.
- DB, 코드, epochs, 모집단 또는 method 설정 변경: 기존 root와 충돌하므로
  새 run root를 사용한다.
- detector 예외: 다른 detector 점수로 대체하지 않는다.
- 파일 변조: 원인을 확인하고 새 root에서 전체 identity를 다시 만든다.

TEST label을 여는 `evaluate` 단계는 frozen manifest 검증 기능과 함께 별도
구현·검증한다. 그 전까지 score root를 성능 결과로 해석하지 않는다.
