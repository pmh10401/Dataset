# Exp450 구현 검증 기록

## 구현 범위

- 구현 기준 commit: `a0bd6839` (Task 1–7 완료 기준), Task 8 구현 commit은 이 문서와 함께 기록된다.
- 코드: `reporting_antigravity.py`, `run_exp450_resampling_image_ablation_antigravity.py`.
- 산출물: canonical AP/AUROC·paired·curve CSV/JSON, PNG/SVG, self-contained HTML, 한국어 사전등록과 runbook.

## 검증

- Exp450 bounded test discovery: 86 tests passed (139.056s; rerun after the PNG-label regression).
- Public baseline test discovery: 91 tests passed (6.413s).
- Task 8 회귀 테스트는 report 축/열 방법, scorer·label-free `status`/`verify`, root 부재, progress schema, evaluate gate를 검사한다.
- `generate_exp450_codemap_antigravity.py --check`, `py_compile`, `git diff --check`가 통과했다.

## TDD 증거

- Task 8 최초 RED는 report/CLI API 부재를 확인했다.
- 수정 RED는 canonical evaluation round-trip, v2 verifier mode, safe embedded JSON/table rendering, short-write completion을 추가해 기존 구현에서 실패하는 것을 확인했다.
- GREEN은 fixture-only regression으로 위 계약과 durable checkpoint 뒤 live-progress callback 순서를 검증한다.
- PNG 가시성 RED는 제목, 축 라벨, 10개 method legend 영역이 비어 있음을 확인했고, GREEN은 dependency-free bitmap glyph renderer로 그 영역의 실제 raster ink를 검증한다.

## Backend 사실

- Tensor-eligible 부분: adaptive pooling과 recurrence distance다.
- PyTorch/MPS availability는 실행 시 provenance로 기록한다. MPS가 있으면 해당 tensor 단계에 사용하며 MPS-heavy 작업은 직렬화한다.
- MLX availability도 기록한다. MLX는 CPU와 동일 입력/seed smoke, 모든 alpha의 동일 후보, 수치 tolerance, transfer/synchronization 포함 1.1x benchmark를 통과하지 않았으므로 canonical path가 아니다.
- CPU fallback은 공식 sktime 변환과 `max(1, logical_cpu_count - 1)` worker 정책이며 inner numerical threads는 1이다.

## 결과 상태

| field | value |
|---|---|
| smoke_result | `v1_failed_preserved; v2_preserved; v3_verified_92_of_92` |
| full_run_result | `not_run` |
| evaluation_result | `not_run` |
| report_result | `not_run` |
| test_labels_opened | `not_run` |
| automatic_promotion | `not_run` |

Exp450은 ablation일 뿐이며 자동 promotion 주장을 하지 않는다.

## Smoke v1/v2 보존과 Smoke v3 권한

- v1 root: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_antigravity_20260820` 및 log: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_progress_antigravity_20260820.log`는 변경하지 않고 보존한다.
- v1은 92개 좌표 중 70개 성공, C/D 22개 좌표는 두 wave에서 총 44회 실패로 종료했다. 실패는 2차원 MPS adaptive pooling에서 한 축은 output보다 작고 다른 축은 큰 mixed-axis 입력의 native shape 제한이었다. TEST label은 열지 않았다.
- 수정은 해당 native 오류 문구를 기존 MPS-window fallback으로 인식하는 한 줄뿐이다. fallback의 mean/window 수식, 요인 A–D, threshold, score 정의와 CPU 경로는 바꾸지 않는다.
- Smoke v2 root는 `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_v2_antigravity_20260820`이며 v1 checkpoint와 code closure를 혼합하지 않는 검증 완료 증거로 보존한다.
- full input의 DB SHA-256을 데이터셋마다 다시 읽던 운영 병목을 제거하면서 `data_paper.py`가 v3 code closure에 추가되었다. 따라서 새 create-only v3 root `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_v3_antigravity_20260820`를 사용하며 v2 checkpoint를 혼합하지 않는다. 이 변경은 행·값·요인·점수·임계값을 바꾸지 않는다.

## Smoke v2 검증 완료

- 실행: 23 datasets × A--D = 92/92 dataset-condition checkpoint가 첫 wave에서 성공했고 실패와 2차 재시도는 없었다. 로컬 단계 elapsed time은 1,115.429초였다.
- 고정 backend: PyTorch 2.13.0, MPS available/selected, MLX unavailable/unused, logical CPU 12, CPU workers 11, inner numerical threads 1. MPS-heavy 작업은 한 번에 하나만 실행했다.
- scorer-free 신선 프로세스 검증: A--D 각각 23 shards와 6,500 TEST-row identities를 검증했다. manifest SHA-256은 각각 `2d13d54c…85dd0`, `624e6d00…9e8616`, `205d9938…4d6d2`, `c1dfe1c7…80afb`이다.
- 승격 게이트: 모든 score가 finite이고 고정 image shape/TEST identity가 정확하며, backend 변경 및 MPS OOM이 없었다. TEST label, AP/AUROC, alert, 우열 정보는 열거나 검사하지 않았다.
- durable progress log: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_v2_progress_antigravity_20260820.log`.

## Full-run DB 검증 병목 보정

- 원인: full 입력 준비는 1,117개 dataset마다 `load_dataset_values`를 호출했고, 기존 reader는 각 호출의 시작과 종료에 1.6 GB DB 전체 SHA-256을 계산했다. 따라서 적어도 2,234회의 전체 파일 읽기가 발생했다.
- 보정: Exp450의 한 번의 science invocation에서 catalog가 계산한 DB 전체 SHA-256을 한 번만 사용해 frozen digest·resolved path·device/inode·size·mtime과 묶는다. 이후 각 dataset read는 새 read-only descriptor와 pathname의 identity를 시작/종료에 계속 대조하고, 교체·stat 변경·변조가 감지되면 fail-closed 한다. 일반 reader의 기존 호출은 계속 매 호출 양쪽 SHA-256 검사를 사용한다.
- 과학적 불변성: SQL, TRAIN/TEST row 선택, raw blob/float32 값, A--D 요인, 점수와 threshold는 변경하지 않았다. 임시 SQLite fixture에서 bound reader와 기존 reader의 TRAIN/TEST raw bytes가 동일함을 검사했다. TEST label은 열지 않았다.
- 새 code closure에는 `paper/equipment_anomaly_detection/data_paper.py`가 포함된다. 그래서 v2 checkpoint를 섞지 않는 새 root `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_v3_antigravity_20260820`를 만든 권한으로 고정했다. full run은 아직 시작하지 않았고 official score root·TEST label에는 접근하지 않았다.
- TDD: binding API 부재 RED를 확인한 뒤, 다중 load 한 번 해시, 교체·content/stat 변조 거부, forged digest 거부, 새 invocation 재해시, 기존 loader와 byte 동등성 회귀를 추가했다.
- 다음 단계: 아래 v3 smoke의 독립 검증이 완료됐으므로, 1,117-dataset × A--D full score run만 라벨 없이 실행할 수 있다.

## Smoke v3 검증 완료

- 실행: DB digest-once closure에서 23 datasets × A--D = 92/92 dataset-condition checkpoint가 첫 wave에서 성공했고 실패와 2차 재시도는 없었다. 첫 checkpoint까지 10.164초, 로컬 단계 elapsed time은 1,080.141초였다.
- 고정 backend: PyTorch 2.13.0, MPS available/selected, MLX unavailable/unused, logical CPU 12, CPU workers 11, inner numerical threads 1. MPS-heavy 작업은 한 번에 하나만 실행했다.
- scorer-free 신선 프로세스 검증: A--D 각각 23 shards와 6,500 TEST-row identities를 검증했다. manifest SHA-256은 각각 `fa38d4e2…9d712`, `4d6a5448…d9bde`, `e4603aee…7201e`, `28f14d44…751cd`이다.
- 승격 게이트: 모든 score가 finite이고 고정 image shape/TEST identity가 정확하며, backend 변경 및 MPS OOM이 없었다. TEST label, AP/AUROC, alert, 우열 정보는 열거나 검사하지 않았다.
- durable progress log: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_v3_progress_antigravity_20260820.log` (92 events; SHA-256 `c32ed65b…6deaa`). v1과 v2 root는 변경하지 않고 보존한다.
