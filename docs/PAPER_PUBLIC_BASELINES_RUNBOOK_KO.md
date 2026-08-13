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
구현되어 있다. 그 전까지 score root를 성능 결과로 해석하지 않는다.

## 5. 평가와 보고서

freeze가 성공한 뒤에만 다음 명령을 실행한다.

```bash
python3 scratch/run_paper_public_baselines.py evaluate \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1 \
  --exp448-root /Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812

python3 scratch/run_paper_public_baselines.py report \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1

python3 scratch/run_paper_public_baselines.py report \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_v1 \
  --verify-only
```

`evaluate`는 frozen score manifest와 모든 shard digest를 먼저 검증한 뒤
TEST label capability를 지연 import한다. `report --verify-only`는 CSV, JSON,
Markdown을 다시 계산해 저장된 byte와 정확히 같은지 검증한다.

최종 문서는 다음 경로에 생성된다.

- `report_public_baselines/PAPER_PUBLIC_BASELINE_RESULT_KO.md`
- `report_public_baselines/public_baseline_summary.csv`
- `report_public_baselines/public_baseline_dataset_metrics.csv`
- `report_public_baselines/public_baseline_bootstrap.json`

## 6. 2026-08-12 공식 실행 범위

6개 방법의 50-epoch pilot 결과, Deep SVDD는 큰 데이터셋에서 단일 pair가
132초까지 소요되었고 DROCC는 첫 3개만으로 평균 29.28초였다. 전체 TRAIN
크기·길이 분포를 적용한 DROCC 예상치는 약 21시간이었다. 사전등록 파라미터를
줄이지 않고 다음과 같이 산출물을 분리했다.

- 논문 사용 가능:
  `/Users/mac/Documents/Dataset/paper_public_baselines_classical_v1`
  — 고전 공개 방법 4종, 1,117개 전체, freeze/evaluate/report 완료
- partial evidence:
  `/Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1`
  — Deep SVDD 50 epochs 107개 완료, freeze/evaluate하지 않음
- superseded evidence:
  `/Users/mac/Documents/Dataset/paper_public_baselines_v1_superseded_unbound_dispatch_20260812`
  — 초기 source-hash inventory 누락으로 논문에서 제외

고전 전용 공식 root는 다음 prepare 명령으로 method population 자체를 4종으로
고정했다.

```bash
python3 scratch/run_paper_public_baselines.py prepare \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_classical_v1 \
  --methods knn_distance,lof_novelty,ocsvm_rbf,isolation_forest
```

## 7. 보류 심층 baseline 독립 실행 키트

이 절의 명령은 아직 하나도 실행하지 않았다. Deep SVDD와 DROCC는 각각
독립된 새 공식 root를 사용하며 동시에 실행하지 않고 반드시 한 번에 하나만
실행한다. 설정은 deterministic PyTorch CPU, worker 1, PyTorch thread 1,
50 epochs, seed 20260812, 1,117개 데이터셋으로 고정된다.

Deep SVDD의 향후 실행 명령은 다음과 같다.

```bash
python3 scratch/manage_deferred_deep_baselines_antigravity.py launch \
  --method deep_svdd_1d \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_1d_full_v1_antigravity
```

Deep SVDD가 freeze와 verify까지 끝난 뒤에만 DROCC를 시작한다.

```bash
python3 scratch/manage_deferred_deep_baselines_antigravity.py launch \
  --method drocc_1d \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_drocc_1d_full_v1_antigravity
```

아래 `ROOT`에는 확인할 공식 root의 절대경로를 넣는다.

```bash
python3 scratch/manage_deferred_deep_baselines_antigravity.py status --run-root ROOT
tail -f ROOT_ops_antigravity/progress_antigravity.log
python3 scratch/manage_deferred_deep_baselines_antigravity.py verify --run-root ROOT
```

같은 launch 명령을 다시 실행하면 authority가 정확히 같은 경우에만 완전히
검증된 shard/meta pair를 재사용해 resume한다. method, DB, interpreter, Git
commit, wrapper 또는 score-source identity가 다르거나 shard/meta 한쪽만 있거나
byte가 변조됐으면 덮어쓰지 않고 실패한다. 기존 107-pair mixed root
`/Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1`은 증거로만
보존하며 두 새 공식 root 어느 쪽에도 복사하지 않는다.

`run-root`는 canonical 경로로 고정하며 symlink leaf는 거부한다. sibling
`worker_lock_antigravity.lock`의 비차단 `flock`만 실행 중 여부를 판단한다.
`pid_antigravity.txt`와 heartbeat는 관찰용 metadata이며 liveness 판정에는
사용하지 않는다.
