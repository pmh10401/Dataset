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
| smoke_result | `v1_failed_preserved; v2_not_run` |
| full_run_result | `not_run` |
| evaluation_result | `not_run` |
| report_result | `not_run` |
| test_labels_opened | `not_run` |
| automatic_promotion | `not_run` |

Exp450은 ablation일 뿐이며 자동 promotion 주장을 하지 않는다.

## Smoke v1 보존과 Smoke v2 권한

- v1 root: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_antigravity_20260820` 및 log: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_progress_antigravity_20260820.log`는 변경하지 않고 보존한다.
- v1은 92개 좌표 중 70개 성공, C/D 22개 좌표는 두 wave에서 총 44회 실패로 종료했다. 실패는 2차원 MPS adaptive pooling에서 한 축은 output보다 작고 다른 축은 큰 mixed-axis 입력의 native shape 제한이었다. TEST label은 열지 않았다.
- 수정은 해당 native 오류 문구를 기존 MPS-window fallback으로 인식하는 한 줄뿐이다. fallback의 mean/window 수식, 요인 A–D, threshold, score 정의와 CPU 경로는 바꾸지 않는다.
- Smoke v2 root는 `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_smoke_v2_antigravity_20260820`이며 아직 실행하지 않았다. v2는 v1 checkpoint와 code closure를 혼합하지 않는다.
