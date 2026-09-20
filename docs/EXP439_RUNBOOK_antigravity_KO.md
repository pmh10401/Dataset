# Exp439 크기 대응 FFT 복원 검증 실행 안내서

과학 규칙은 `docs/EXP439_PREREGISTRATION_antigravity_KO.md`가 정본이다.

## 고정 경로

```text
WORKTREE=/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence
DB=/Users/mac/Documents/Dataset/univariate_ts.db
EXP438=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807
RUN_PARENT=/Users/mac/Documents/Dataset/exp439_scale_aware_reconstruction_repair_scheduler_antigravity_20260807
READINESS=$RUN_PARENT/readiness_antigravity.json
OUTPUT=$RUN_PARENT/output_antigravity
CHECKPOINT=$RUN_PARENT/checkpoint_antigravity
PUBLICATION=$RUN_PARENT/publication_antigravity
EVALUATION=$RUN_PARENT/evaluation_antigravity
```

## 실행 전 gate

1. Git worktree가 clean reviewed HEAD다.
2. Exp438 failed root가 read-only로 exact 89-record prefix를 증명한다.
3. DB SHA256은
   `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`다.
4. DB link count는 snapshot 전 1이다.
5. Exp439 run parent는 존재하지 않는다.
6. focused test와 실제 Chinatown TRAIN smoke가 통과한다.
7. prediction label access는 0이다.

## 정확한 다섯 명령

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py readiness

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py new

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py inspect

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py resume

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py evaluate
```

workers, backend, seed, threshold, depth, alpha, budget 또는 경로 override는 없다.

## 직접 확인할 로그

```text
$OUTPUT/progress_antigravity.json
$OUTPUT/events_antigravity.jsonl
$OUTPUT/parent_antigravity.log
$OUTPUT/run_antigravity.pid
$CHECKPOINT/import_manifest_antigravity.json
```

`progress_antigravity.json`에서 PID, current dataset, dataset metadata,
completed/total, import/recompute count, error, 처리속도와 ETA를 확인한다. heartbeat와
checkpoint는 프로세스가 직접 쓴다. Codex 반복 감시는 필요하지 않다.

## 시작·재개·실패

- `new`: exact readiness 후 absent output/checkpoint에만 사용한다.
- `resume`: PID가 죽은 interrupted root에만 사용한다.
- `inspect`: DB와 worker를 열지 않는 read-only 검사다.
- `evaluate`: publication 동결 뒤에만 TEST label을 연다.
- failed Exp438 또는 Exp439 root는 삭제·수정·덮어쓰기·resume하지 않는다.

Exp438 89개 prefix가 exact 검증되면 Exp439 progress는 `import_count=89`에서
시작하고 첫 recompute dataset은 `BeetleFly_normal_2`다. 검증 실패 시 run root를
만들지 않으며 부분 가져오기를 하지 않는다.

## 공식 실행 시작

지속되는 terminal session에서 다음처럼 실행한다.

```bash
echo $$ > "$RUN_PARENT/launcher_antigravity.pid"
exec env PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py new
```

시작 직후 한 번만 PID 생존, CPU worker 11, import 89, error 0, DB nlink 2와 로그
경로를 확인한다. 이후 사용자가 완료를 알릴 때까지 반복 polling하지 않는다.

## 2026-08-07 실행 병목 복구

첫 공식 root
`exp439_scale_aware_reconstruction_repair_antigravity_20260807`은 101/1,117에서
사용자 승인에 따라 정상 인터럽트했다. progress와 checkpoint는 모두
`interrupted`, 마지막 checkpoint는 `Chinatown_normal_2`이며 이 root는 수정하거나
덮어쓰지 않는다.

새 reviewed implementation은 과학 규칙을 바꾸지 않고 두 실행 병목만 제거한다.

- 완료 순서가 늦은 결과 때문에 작업자가 쉬지 않도록 최대 worker 수만큼 계속
  dispatch하며, pending+buffered 범위는 worker 수의 제곱으로 제한한다.
- TRAIN 행별 preferred depth와 depth별 decomposition feature를 TRAIN-only로 한 번
  계산해 leave-one-out fold에서 재사용한다. 제외 행, lower-median depth,
  robust transform, OOF score, threshold와 후보 규칙은 그대로다.

기존 `Chinatown_normal_2` record와 새 구현 record는 SHA256
`01dabf95e22175ba09b01d80df4002981bbe149e8112a7d9771aec3de7478446`로
byte-identical이었다. `ChlorineConcentration_normal_1` 실제 TRAIN 800 / TEST 1,000,
길이 166 label-free CPU smoke는 42.576초에 끝났고 label access는 0이었다.
