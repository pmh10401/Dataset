# Exp467 Numba Calibration Resume Design

## 목적

Exp451에서 확인된 empirical upper-tail p-value의 Python `O(n_train × n_test)` 반복 병목을 제거한다. Exp450의 검증된 4,234개 결과는 다시 계산하지 않고 읽기 전용 검증 후 Exp467 루트로 이관하며, 남은 C/D 234개만 새 실행 컨텍스트로 계산한다.

## 변경하지 않는 과학 계약

- 데이터베이스와 1,117개 데이터셋, TEST 195,900행
- A–D 조건, 특징, 점수, finite-sample 보정식 `(1 + count) / (n + 1)`
- `reference >= query`의 동률 포함 규칙
- alpha `0.005, 0.01, 0.015, 0.02, 0.03`의 후보 판정
- 임계값, 라우팅, AP/AUROC, ROC/PR 정의
- MPS 우선 영상 처리와 관찰된 비지원 pooling 모양의 PyTorch CPU native fallback

## Numba 변경

`paper/equipment_anomaly_detection/numba_calibration_antigravity.py`가 다음 계약을 구현한다.

- reference와 query가 모두 정확히 `float64`일 때만 Numba nopython 반복문을 사용한다.
- `fastmath=False`, `parallel=False`, `nogil=True`이다.
- Numba는 정수 upper-tail count만 계산한다. 최종 finite-sample 나눗셈은 기존 Python 순서를 유지한다.
- 혼합 자료형과 다른 dtype은 기존 `calibration_paper.upper_tail_pvalues`로 내려간다. 특히 `2**53` 경계의 Python scalar 비교 의미를 보존한다.
- Hard source, MiniRocket, Hard route, native payload 직렬화·재검증과 고정 Exp450 payload migration 검증은 Exp467 로컬 seam을 통해 같은 Numba 함수를 사용한다. 과거 serializer의 schema, count-to-p-value Python 나눗셈, canonical payload bytes는 바꾸지 않는다.
- label-gated 평가의 public baseline micro-score loader도 Exp467 로컬 adapter가 담당한다. Exp450 evaluator의 dataclass, metric/curve 함수와 label gate는 유지하고, 기존 baseline `upper_tail_p_values`만 호출하지 않는다.
- Exp450·Exp451의 공통 보정 파일과 공식 결과는 수정하지 않는다.

## 채택 근거

공식 Python 3.14.6 환경에서 NumPy 2.3.5, Numba 0.63.1, llvmlite 0.46.0을 사용했다. 결정적 합성 float64 입력에서 68개 parity 사례, 등록 alpha의 후보 인덱스, 반복 실행 및 canonical science payload 바이트가 일치했다.

| reference × query | 기존 중앙값 | Numba warm 중앙값 | 배수 |
|---:|---:|---:|---:|
| 256 × 128 | 0.014071초 | 0.00001725초 | 815.7× |
| 1,024 × 512 | 0.229162초 | 0.00008537초 | 2,684.2× |
| 2,048 × 1,024 | 0.919076초 | 0.00025579초 | 3,593.1× |

빈 JIT 캐시에서 module import는 0.286초, 첫 compile+call은 0.486초, 합계는 0.772초였다. 이는 정확한 보정 단계의 합성 벤치마크이며 전체 공식 좌표 속도를 직접 주장하지 않는다. 채택 기준인 1.1배를 크게 넘고 한 프로세스에서 반복 호출되므로 Exp467에서 채택한다.

## 가속기 판단

- Numba: 현재 병목이 Python/GIL 반복 비교이므로 canonical 가속 경로로 채택한다.
- MPS: 기존 영상 특징과 pooling 경로에서 계속 사용한다. Numba calibration과 동시에 별도 GPU 프로세스를 늘리지 않는다.
- MLX: 공식 환경에 설치되어 있지 않고 frozen 전체 점수 경로의 정확한 표현도 검증되지 않아 사용하지 않는다.
- PyTorch CPU: MPS가 관찰된 pooling 모양을 지원하지 않을 때의 기존 native fallback으로 유지한다.
- CPU worker: 새 CPU 병렬 단계가 생기면 논리 CPU 수 12에서 1을 뺀 11개를 사용한다. 현재 좌표 실행은 MPS-heavy 직렬 정책을 유지한다.

## 실행 정체성과 출처

- 실험 ID: `Exp467`
- 코드: `experiments/exp467_numba_calibration_resume_antigravity/`
- 실행기: `scratch/run_exp467_numba_calibration_resume_antigravity.py`
- 원본: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_official_antigravity_20260820`
- 공식 결과: `/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_official_antigravity_20260823`
- 평가: `/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_evaluation_antigravity_20260823`
- 보고서: `/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_report_antigravity_20260823`
- 코드맵: `docs/codemap/codemap_exp467_antigravity.{html,json,lock}`

실행 컨텍스트는 CLI, Exp467 production 모듈 16개, 고정 Exp450 serializer를 시작점으로 하는 deterministic local-import transitive closure와 import-time package `__init__.py`, `lineage_lock_paper.json`의 SHA-256을 포함한다. 현재 69개 실행 모듈과 resource를 고정한다. Imported envelope은 고정 Exp450 backend를, native envelope은 Exp467 Numba backend를 사용한다.

## 전환 및 실패 처리

1. Exp450 원본의 파일 목록·SHA-256 snapshot을 만든다.
2. 새 Exp467 root로 4,234개를 create-only 이관하고 manifest를 재검증한다.
3. 이관 전후 Exp450 snapshot이 완전히 같은지 확인한다.
4. Exp451의 최신 durable progress와 native checkpoint 수를 기록한 뒤 정확한 Python PID에 `SIGINT`를 보낸다. PID·shell·`tee`가 종료되고 Exp451 run lock과 해당 PID의 Metal lock/FD가 사라졌는지 확인한다. `SIGSTOP`과 `SIGTERM`은 사용하지 않는다.
5. MPS 프로세스가 없는 상태에서 공식 interpreter `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`로 `scratch/run_exp467_numba_calibration_resume_antigravity.py smoke --mode smoke`를 실행한다. 직접 무제한 실행하지 않고 plan에 기록된 120초 watchdog을 사용한다. watchdog은 만료 시 smoke Python PID에 `SIGINT`를 보내고 10초 유예 후에도 종료되지 않으면 강제 종료하며 코드 124로 실패를 기록한다. smoke는 `MelbournePedestrian_normal_1`의 C(`linear`, `area`) 한 좌표만 `SMOKE_ROOT`에서 실행하고, frozen smoke inventory의 실제 값만 사용하며 label을 읽지 않는다. create-only native checkpoint를 즉시 reread 검증하고 JSON progress를 남긴다.
6. smoke가 실패하면 동일 Exp451 명령과 기존 공식 root로 즉시 재개한다. smoke가 성공한 경우에만 Exp451 중단을 영구 전환으로 확정한다.
7. Exp467 `run --mode full`로 남은 234개를 실행한다. 다른 context의 checkpoint를 섞지 않는다.

Numba import/JIT가 실패하거나 parity·candidate·payload byte gate가 깨지면 Exp467을 시작하지 않고 기존 Exp451을 유지한다. 다른 bytes가 있는 대상 파일은 덮어쓰지 않는다.

## 완료 기준

- Numba 및 Exp467 테스트 전체 통과
- p-value, 모든 등록 alpha 후보, canonical payload byte 동일
- warm 단계 속도 1.1배 이상
- Exp467 코드맵 triplet 일치
- Exp450 원본 이관 전후 byte 동일
- 최종 4,468개 성공과 네 condition manifest 검증 후에만 평가·보고
