# Exp438 구현 완료 보고서

## 범위와 상태

- Worktree: `/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence`
- Branch: `codex/exp438-frequency-complexity-evidence`
- 검토 구현 commit: `c89d5b47fa6d18ae6fba0c6fba7f06400ee84088`
- 상태: 병행 실행 보완 구현 완료, Exp437 복구·Exp438 readiness 대기
- readiness/prediction/publication/evaluation: 시작하지 않음
- TEST label access: 0
- Exp437 제품 코드 변경: 없음. 별도 운영 런처는 exact clean commit의 기존 공개
  runtime API만 호출한다.

이 보고서는 구현과 회귀 검증만 동결한다. backend 선택이나 공식 1,117개 결과를
주장하지 않는다. backend는 나중의 exact readiness에서만 정해진다.

## 구현된 경계

Exp438은 별도 패키지
`experiments/exp438_frequency_complexity_evidence_antigravity`에 구현했다.

- `config_antigravity.py`, `contracts_antigravity.py`: 1,117 population, 정확한
  상수, review-only·label-free 계약과 max-minus-one CPU 정책
- `inputs_antigravity.py`: descriptor-bound DB snapshot, frozen lane 선행 검증,
  exact-length value-only TRAIN/TEST 접근과 행 단위 nonfinite 처리
- `aspe_antigravity.py`, `features_antigravity.py`: 출처가 고정된 ASPE와 프로젝트
  자체 FFT high/low split, TRAIN-only depth 선택
- `scoring_antigravity.py`: 세 채널 분리, leave-one-out TRAIN calibration,
  q=0.99 higher/strict threshold, nearest-normal k=3와 fail-closed budget
- `numba_antigravity.py`, `mlx_antigravity.py`, `mps_antigravity.py`,
  `backend_antigravity.py`: exact-stage parity·속도 gate와 truthful fallback
- `checkpoint_antigravity.py`, `progress_antigravity.py`, `runtime_antigravity.py`:
  ordered-prefix checkpoint, immutable failed root, interrupt/resume byte parity,
  self-written heartbeat/JSON/log와 결정적 11-worker CPU scheduler
- `readiness_antigravity.py`, `cli_antigravity.py`, wrapper: Exp437 terminal/inactive,
  clean HEAD와 fixed root를 강제하는 다섯 명령 lifecycle
- `reporting_antigravity.py`: 1,117 conserved population과 zero-label-access를
  요구하는 separate-channel review-only publication/no-clobber hash replay
- `evaluation_antigravity.py`: publication과 comparator를 먼저 재검증한 뒤에만
  TEST labels를 읽고 세 채널별 혼동행렬, dataset 결과, 20,000회 bootstrap,
  direct/integration 판정을 8개 hash-bound 파일로 동결

Prediction 경로는 evaluation 모듈을 import하지 않는다. CLI의 `evaluate` 명령만
평가기를 지연 import한다. Exp438 outside-lane 후보가 frozen Hard와 겹치거나 TEST
identity 밖이면 평가도 fail closed한다.

## 출처와 금지 코드

- ASPE: `Brioal/ASPE` commit
  `f17cf8d796c4d25bc1b5ace0e9a7ef7be2944a89`, Apache-2.0, 출처 표기와 독립
  golden vector 사용
- AFD toolbox: CC BY-NC-ND 4.0 연구 참고만; copy/import/vendor/runtime 사용 없음
- `VUT-HFUT/PhysDiff`: 다른 physiology 프로젝트이며 license 미확인; 완전 제외
- PhysDiff NeurIPS 2025: high/low와 amplitude-sensitive complexity 연구 아이디어만
  참고; diffusion 코드·CUDA 경로·결과 호환성 주장 없음

자세한 provenance는 `docs/EXP438_THIRD_PARTY_PROVENANCE_antigravity.md`에 있다.

## 테스트 증거

### Exp438 집중 테스트

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp438_frequency_complexity_evidence_antigravity/tests \
  -p 'test_*_antigravity.py'
```

결과: **82 tests**, failures 0, errors 0, `OK`.

### Exp436 공용 엔진 회귀

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p 'test_*_antigravity.py'
```

결과: **299 tests in 115.489s**, failures 0, errors 0, `OK`.

### Exp425 선행 경계 회귀

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp425_selective_recovery/tests -p 'test_*.py'
```

결과: **180 tests in 27.479s**, failures 0, errors 0, `OK`.
256 MiB DB benchmark 출력은 `snapshot=0.154s`, `verified_opens=3.387s`였다.

### 평가 계산 성능 확인

1,117개 합성 dataset row에 20,000회 cluster bootstrap 한 채널을 실행한 결과
약 **2.75초**였고 finite interval을 반환했다. 공식 평가는 세 채널을 별도로
계산하지만 전체 prediction 병목과 비교하면 작은 사후 비용이다.

## 코드맵과 파일 정책

완료된 module boundary를 `docs/codemap/codemap.html`, `codemap.json`,
`codemap.lock`에 함께 반영한다. map은 20개 primary nodes 이내를 유지하고,
wrapper/CLI에서 runtime/input/scoring/checkpoint로 가는 label-free 흐름과 lazy
evaluation에서 SQLite label로 가는 post-freeze 흐름을 source evidence로 구분한다.

모든 새 파일명은 `antigravity`를 포함한다. `docs/codemap` 세 파일은 기존 필수
고정 이름이므로 함께 재생성한다. Exp437 파일이나 runtime path는 변경하지 않았다.

## 정확한 다음 단계

사용자는 stale-PID Exp437 복구 후 살아 있는 CPU Exp437와 Exp438을 병행하도록
승인했다. Exp437는 운영 런처로 복구하고 실제 PID와 fresh heartbeat를 한 번
확인한다. 그다음 clean HEAD에서 다음 readiness 명령을 실행한다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py readiness
```

복구 실패, 죽은 PID, 오래된 heartbeat, 오류, MLX/MPS predecessor 또는 카운터
후퇴가 있으면 Exp438 run parent를 만들지 않는다. 두 프로세스가 살아난 뒤에는
각자 atomic 로그를 쓰며 Codex가 반복 감시하지 않는다.
