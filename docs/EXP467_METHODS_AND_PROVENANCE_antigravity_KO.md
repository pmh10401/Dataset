# Exp467 방법 및 재현성 기록

## 논문용 요약

Exp467은 제안 기술의 점수·임계값·라우팅을 바꾸지 않고 empirical upper-tail p-value 계산만 가속한다. 기존 구현은 query마다 reference 전체를 Python에서 순회했다. Exp467은 양쪽 점수 배열이 `float64`일 때 동일한 `reference >= query` count loop를 Numba nopython으로 실행하며, 혼합 자료형은 기존 Python 경로를 사용한다.

Exp450에서 검증된 4,234개 결과는 원본 context와 payload SHA-256을 확인한 후 새 Exp467 context로 create-only 이관한다. 남은 C/D 234개만 Exp467 native 경로로 계산한다. Imported/native 결과는 각각의 backend와 execution origin을 따로 기록한다.

과거 Exp450 payload를 읽어 검증하는 직렬화기에서도 동일한 float64 Numba count helper를 사용한다. 이때 고정된 serializer의 입력·출력 schema와 최종 Python 나눗셈은 바꾸지 않으므로, 원본 payload와 재직렬화 canonical JSON 바이트는 변하지 않는다. migration과 verification에서 기존 Python 반복 보정이 다시 호출되지 않는 회귀 검사를 둔다.

평가 단계의 public baseline micro-score 보정도 Exp467 로컬 evaluator adapter에서 같은 helper를 사용한다. Exp450의 dataclass, AP/AUROC 표, ROC/PR 곡선과 label gate는 그대로 재사용하며, representative fixture의 canonical metric payload가 동결 evaluator와 일치함을 확인한다.

## 재현 환경

- macOS arm64
- CPython 3.14.6
- NumPy 2.3.5
- Numba 0.63.1
- llvmlite 0.46.0
- PyTorch 2.13.0, MPS 사용 가능
- MLX 사용 불가
- 논리 CPU 12개, 정책상 CPU worker 11개
- Numba 옵션: `fastmath=False`, `parallel=False`, `nogil=True`

## 동일성 검증

- 결정적·경계 입력 68개에서 p-value `np.array_equal`
- alpha `0.005/0.01/0.015/0.02/0.03` 후보 인덱스 완전 일치
- 동률, 중복, signed zero, `nextafter`, 최대 finite 값 검증
- float64/int64 혼합 `2**53` 경계는 legacy fallback으로 기존 결과 유지
- source 6개와 route의 canonical payload byte 일치
- 확인 payload SHA-256: `254b49db99336789edb28f8d6d46df156ca53b3876dc9d018fa627af6e280f3a`

## 속도 검증

입력 검증과 contiguous 변환을 포함한 보정 함수 전체를 측정했다. 1,024×512에서 기존 중앙값 0.229162초, Numba warm 중앙값 0.00008537초로 2,684.2배였다. 빈 캐시의 import+첫 compile/call은 0.772초였다. 이는 정확한 보정 단계의 합성 benchmark이며 공식 234개 좌표 전체의 최종 속도는 진행 로그로 별도 보고한다.

## 실행 및 로그

```bash
PY=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python
$PY -B scratch/run_exp467_numba_calibration_resume_antigravity.py migrate --mode full
$PY -B scratch/run_exp467_numba_calibration_resume_antigravity.py verify-migration --mode full
$PY -B scratch/run_exp467_numba_calibration_resume_antigravity.py run --mode full
$PY -B scratch/run_exp467_numba_calibration_resume_antigravity.py status --mode full
```

smoke는 위 일반 명령과 분리하여 [Exp467 공식 계획](superpowers/plans/2026-08-23-exp467-numba-calibration-resume-antigravity-plan.md)의 공식 interpreter·120초 `SIGINT` watchdog 명령으로만 실행한다. 무제한 직접 smoke 실행은 허용하지 않는다.

실행 context의 code closure는 CLI, Exp467 production module, 고정 Exp450 serializer를 시작점으로 하여 로컬 Python import를 전이적으로 추적한 SHA-256 목록이다. import 시 실행되는 package `__init__.py`와 lineage lock JSON도 포함한다. 현재 69개 실행 모듈과 resource를 고정하며, 코드맵 triplet은 테스트와 생성기를 포함한 77개 노드를 기록한다.

공식 cutover 전 smoke는 고정된 `MelbournePedestrian_normal_1`의 C 조건만 실행한다. 이 좌표는 frozen smoke inventory에 속하며, 292 TRAIN·350 TEST·길이 24의 실제 값만 사용한다. C의 image-area와 Numba p-value 경로를 그대로 지나고 TEST label은 읽지 않으며, `SMOKE_ROOT` 밖의 checkpoint에는 쓰지 않는다. 단, 하나의 MPS-heavy 프로세스만 허용하므로 Exp451을 먼저 `SIGINT`로 정상 중단하고 PID·run lock·Metal lock 소멸을 확인한 뒤 2분 이내에 smoke를 실행한다. smoke가 실패하면 기존 Exp451 공식 root를 바꾸지 않고 동일 명령으로 즉시 재개하며, 성공한 경우에만 Exp467로 영구 전환한다.

진행 이벤트는 공식 결과 루트의 `progress_antigravity.jsonl`에 기록한다. 각 native 성공 checkpoint는 진행 이벤트보다 먼저 durable create-only 파일로 발행된다. 실패는 별도 ledger에 남기고 첫 wave가 끝난 뒤 실패 좌표만 한 번 재시도한다.

## 해석 제한

Numba는 CPU JIT 가속이며 MPS를 대체하지 않는다. 영상 특징과 adaptive pooling의 기존 MPS/CPU 정책은 그대로다. 합성 보정 단계의 큰 속도 향상을 전체 실험 wall time의 동일한 배수로 해석하지 않는다. 전체 결과는 4,468개 checkpoint와 네 condition manifest가 검증된 뒤에만 논문 평가에 사용한다.
