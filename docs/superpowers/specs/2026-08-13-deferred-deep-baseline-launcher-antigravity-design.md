# Deferred Deep Public-Baseline Launcher Design

## 1. 목적

논문용 공개 이상탐지 비교에서 계산비용 때문에 보류한 `Deep SVDD 1D`와
`DROCC 1D`를 나중에 안전하게 실행할 수 있는 운영 키트를 준비한다. 이번 작업은
학습을 시작하거나 공식 실행 root를 생성하지 않는다.

준비 완료의 의미는 다음과 같다.

- 두 기법을 서로 독립된 공식 root에서 한 명령으로 시작할 수 있다.
- PID, 실시간 로그, 완료 데이터셋 수, 경과시간과 ETA를 확인할 수 있다.
- 중단 후 같은 실행 identity로 재개할 수 있다.
- 1,117개가 모두 완료된 경우에만 score root를 동결하고 검증한다.
- TEST label은 평가 단계 전까지 import하거나 읽지 않는다.

## 2. 현재 상태

두 공개 기법의 과학 구현은 이미 존재한다.

- `experiments/paper_public_baselines/methods/deep_svdd.py`
- `experiments/paper_public_baselines/methods/drocc.py`
- 공통 계약: 정상 TRAIN만 사용한 5-fold OOF score와 별도 full-TRAIN fit TEST
  score, 결정론적 PyTorch CPU, 공식 50 epochs
- 기존 CLI 계약: deep method는 정확히 `--workers 1`
- 기존 artifact 계약: dataset-method별 JSON shard와 meta를 create-only로 게시하고,
  재실행 시 해시가 검증된 shard만 재사용한다.

기존 부분 root
`/Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1`에는 Deep SVDD
107개와 기존 고전 기법 결과가 섞여 있다. 이 root는 보존하지만, 새 단일-method
공식 root로 shard를 복사하지 않는다. shard meta가 기존 mixed-method run-config
hash에 결합돼 있기 때문이다.

현재 실행 중인 제안기술 r3는
`paper/equipment_anomaly_detection/**` 코드 폐쇄를 고정했다. 이번 준비 작업은 그
namespace, DB, r3 root와 paper codemap을 수정하지 않는다.

## 3. 선택한 방식

기존 baseline CLI와 학습 코드를 변경하지 않는 독립 운영 래퍼를 추가한다.

대안으로 기존 `runner.py`와 `cli.py`에 진행 출력을 직접 추가할 수 있지만, 해당
파일은 score-run source hash에 포함된다. 이를 변경하면 기존 부분 root의 재개
identity가 달라지고 변경 범위와 회귀 위험이 커진다. 문서에 명령만 적는 방식은
PID 중복 방지, 상태 확인과 자동 동결 검증이 부족하다.

따라서 새 래퍼는 기존 CLI를 subprocess로 호출하고 이미 게시된 shard/meta만
관찰한다. 점수, 모델, seed, fold, 전처리, conformal alpha와 artifact schema를
변경하지 않는다.

## 4. 공개 인터페이스

새 파일은 AGENTS.md의 파일명 정책에 따라 다음 이름을 사용한다.

```text
scratch/manage_deferred_deep_baselines_antigravity.py
```

지원 명령은 세 개뿐이다.

### 4.1 launch

```bash
python3 scratch/manage_deferred_deep_baselines_antigravity.py launch \
  --method deep_svdd_1d \
  --db-path /Users/mac/Documents/Dataset/univariate_ts.db \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_full_v1_antigravity
```

`drocc_1d`도 별도 root로 같은 명령을 사용한다. `launch`는 background worker를
시작하고 즉시 PID, run root와 log path를 출력한다. method는 두 값만 허용하며
공식 설정은 사용자 옵션으로 바꾸지 못한다.

```text
epochs = 50
workers = 1
seed = 20260812
population = frozen 1,117 datasets
backend = canonical CPU
```

내부 worker는 다음 순서를 실행한다.

```text
prepare(method 한 개만 등록)
→ assess-mlx
→ score(method 한 개, 50 epochs, worker 1)
→ freeze
→ verify_score_root
```

### 4.2 status

```bash
python3 scratch/manage_deferred_deep_baselines_antigravity.py status \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_full_v1_antigravity
```

`status`는 학습을 시작하지 않는 read-only 명령이다. 다음을 JSON 한 줄로 출력한다.

- method와 run identity
- `not_started`, `running`, `failed`, `score_complete`, `frozen` 상태
- PID와 heartbeat age
- 검증 가능한 shard/meta pair 수 / 1,117
- abstain 수
- 경과시간, 최근 완료 dataset, 완료율
- 완료 데이터가 충분할 때만 계산한 단순 ETA
- log path와 마지막 오류

PID가 존재하더라도 heartbeat가 오래됐으면 `running`으로 오판하지 않는다.

### 4.3 verify

```bash
python3 scratch/manage_deferred_deep_baselines_antigravity.py verify \
  --run-root /Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_full_v1_antigravity
```

기존 `verify_score_root`를 호출해 manifest, run identity, DB hash, 모든 shard와 exact
coverage를 검증한다. label loader를 import하지 않는다. 미완료 또는 변조 root는
0이 아닌 종료 코드로 실패한다.

## 5. 운영 artifact와 진행 로그

과학 score root와 운영 상태를 섞지 않기 위해 sibling directory를 사용한다.

```text
<run-root>_ops_antigravity/
  authority_antigravity.json
  pid_antigravity.txt
  state_antigravity.json
  progress_antigravity.log
```

`authority_antigravity.json`은 create-only이며 다음 identity를 고정한다.

- method, DB resolved path와 SHA-256, run root
- epochs, seed, worker 수
- 실행 Python의 resolved executable과 버전
- Git commit과 clean-worktree 여부
- 래퍼 SHA-256
- detected logical CPU count와 실제 worker count
- PyTorch/MLX/mlx-metal 버전
- MPS built/available 여부
- tensor eligibility, canonical CPU fallback 사유
- 가속 경로가 canonical이 되기 위한 동일 seed/input, finite score, 모든 alpha의
  candidate identity, score tolerance `1e-5`, end-to-end `1.1x` 속도 gate

Deep SVDD/DROCC는 frozen PyTorch 초기화, optimizer와 autograd 경로를 바꾸지 않는
exact MLX 구현이 없고 CPU-MPS 결정론·candidate parity 및 end-to-end 속도 gate가
검증되지 않았으므로 CPU가 canonical이다. 실행 단계 자체는 병렬 stage가 아니며
PyTorch thread도 1로 고정된다. 감지 CPU 수와 worker 1을 provenance에 함께 남긴다.

`state_antigravity.json`은 원자적으로 교체 가능한 운영 cache다. 과학 결과의
authority가 아니며 동결 검증에 사용하지 않는다. log는 UTC timestamp를 포함한
append-only text다. worker는 일정 간격과 shard 게시 감지 시 다음을 기록한다.

```text
timestamp method completed/1117 elapsed eta recent_dataset state
```

실시간 확인은 다음 하나로 통일한다.

```bash
tail -f <run-root>_ops_antigravity/progress_antigravity.log
```

## 6. 재개와 오류 처리

- 같은 root의 살아 있는 worker가 있으면 중복 launch를 거부한다.
- authority의 method, DB, interpreter, commit, wrapper와 공식 상수가 모두 같을
  때만 재개한다.
- 기존 baseline runner가 shard/meta를 검증한다. 둘 중 하나만 있거나 byte가
  다르면 조용히 삭제하거나 덮어쓰지 않고 실패한다.
- 완료된 pair는 즉시 게시되므로 process 종료 후 같은 launch 명령이 이를 재사용한다.
- worker 실패 시 `failed` 상태, 종료 코드와 마지막 오류를 남기며 score root와 log를
  보존한다.
- 1,117개 exact coverage 전에는 freeze하지 않는다.
- 공식 root가 frozen이면 launch를 재실행하지 않고 검증 결과를 반환한다.
- Deep SVDD와 DROCC는 서로 다른 root와 ops directory를 사용한다. 실행서는 한 번에
  하나씩 실행하도록 안내한다.
- 자동 삭제, 강제 종료 또는 손상 복구 명령은 제공하지 않는다.

## 7. 파일 범위

추가 또는 수정 허용 범위는 다음으로 제한한다.

- 신규 `scratch/manage_deferred_deep_baselines_antigravity.py`
- 신규
  `experiments/paper_public_baselines/tests/test_deferred_deep_launcher_antigravity.py`
- 기존 `docs/PAPER_PUBLIC_BASELINES_RUNBOOK_KO.md`
- 기존 `docs/PAPER_PUBLIC_BASELINES_IMPLEMENTATION_REPORT_KO.md`
- 활성 Exp448 codemap HTML/JSON/lock triplet

다음은 수정하지 않는다.

- `experiments/paper_public_baselines`의 scoring, model, runner, CLI, evaluation 코드
- `paper/equipment_anomaly_detection/**`
- `paper/equipment_anomaly_detection/codemap_paper.*`
- DB와 현재 r3 run root
- 기존 classical 또는 deep partial artifact

현재 Exp448 codemap은 byte는 최신이지만 public-baseline package의 caller, effect와
test를 포함하지 않아 AGENTS.md 기준으로 coverage가 부족하다. 구현 전에 현재
public-baseline closure가 포함되도록 Exp448 triplet 세 파일을 함께 재생성하고,
구현 후 새 wrapper/test edge까지 포함해 다시 함께 갱신한다.

새 source, test와 운영 파일은 모두 파일명에 `antigravity`를 포함한다. 기존
public-baseline core가 생성하는 `prepared.json`, `acceleration.json`,
`run_config.json`, hash-token shard와 frozen manifest 이름은 이미 동결된 v1 artifact
schema이므로 이름을 바꾸거나 복제하지 않는다. 이 기존 schema의 runtime 파일은
사용자가 승인한 논문 전용 실행 경계 안에서 그대로 사용하며, 이번 작업에서 새
artifact schema를 만들지 않는다.

## 8. 테스트 전략

테스트는 실제 50-epoch 학습이나 새 공식 root 생성을 수행하지 않는다. TDD로 다음
경계를 먼저 실패시키고 최소 구현으로 통과시킨다.

1. 허용되지 않은 method와 공식 설정 변경을 거부한다.
2. launch authority가 method, DB, interpreter, Git, CPU/backend 평가와 wrapper
   hash를 고정한다.
3. 같은 root의 중복 실행을 거부하고 동일 identity의 실패 root는 재개할 수 있다.
4. 서로 다른 Deep SVDD/DROCC root가 상태를 섞지 않는다.
5. published shard/meta만 완료로 세며 partial pair는 오류로 표시한다.
6. heartbeat가 신선할 때만 running이며 stale PID를 running으로 표시하지 않는다.
7. worker 단계 순서가 prepare, assess-mlx, score, freeze, verify로 고정된다.
8. verify는 label module을 import하지 않고 frozen manifest 변조를 거부한다.
9. wrapper import만으로 process나 학습이 시작되지 않는다.

외부 process 경계는 짧게 종료되는 test double command로 검증하고, 실제 scoring
동작은 기존 public-baseline 61개 테스트와 deep method 결정론 테스트를 그대로
재실행한다. 현재 제안기술 r3와 자원을 경쟁하는 deep smoke는 수행하지 않는다.

## 9. 완료 조건

- 새 래퍼의 launch/status/verify 집중 테스트가 통과한다.
- 기존 public-baseline 전체 unit suite가 통과한다.
- 어떤 Deep SVDD/DROCC 공식 학습 process도 시작되지 않는다.
- 두 예정 공식 root와 ops directory가 존재하지 않는다.
- 기존 107-pair partial root의 byte가 변하지 않는다.
- TEST label module이 준비 및 검증 과정에 import되지 않는다.
- Exp448 codemap triplet이 함께 갱신되고 caller/effect/test를 답할 수 있다.
- 본 세션 검증 후 fresh Sol reviewer가 `ship`을 판정한다.
