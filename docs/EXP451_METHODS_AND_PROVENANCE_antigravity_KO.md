# Exp451 방법 및 실행 출처 기록

## 목적

Exp451은 Exp450의 검증된 결과 4,234개를 다시 계산하지 않고 별도 Exp451 결과 루트로 이관한 뒤, 남은 234개만 가속 경로로 계산하는 실험이다. Exp450 원본 결과 루트는 읽기 전용이며 수정하지 않는다.

## 조건과 실행 출처

아래 표는 **실행 전 고정 검증 계약**이다. 아직 공식 Exp451 실행이 끝나지
않았다면 이 수치는 관측 결과가 아니라, 평가를 열기 위해 반드시 충족해야 하는
예정된 분포다. 실제 관측된 checkpoint SHA-256, context SHA-256, 실행 시간과
조건 manifest SHA-256은 `verify` 이후 생성되는 manifest와 최종 report provenance
artifact에서만 확정한다.

| 조건 | 길이 정렬 | 영상 특징 | 이관 결과 | 신규 native 결과 | 최종 |
| --- | --- | --- | ---: | ---: | ---: |
| A | linear | linear | 1,117 | 0 | 1,117 |
| B | polyphase_pchip | linear | 1,117 | 0 | 1,117 |
| C | linear | area | 1,000 | 117 | 1,117 |
| D | polyphase_pchip | area | 1,000 | 117 | 1,117 |
| 합계 |  |  | 4,234 | 234 | 4,468 |

이관 결과의 `execution_origin`은 `imported_exp450_mps`이고, 새 결과의 `execution_origin`은 `exp451_native_cpu_pool_fallback`이다. 두 수는 평가 및 보고서에서도 합치지 않고 별도로 기록한다. 실행 전 이 문서는 결과 우위를 주장하지 않으며, 실제 report 생성 시 provenance artifact가 관측된 출처 분포를 덧붙인다.

## 고정 권한과 데이터베이스

- Exp450 원본 context SHA-256: `5d5b44aec0e738ae61d9dafd85d9eaea1448baba49705c4d042c5ed1d6886f83`
- Exp450 원본 transforms SHA-256: `28b5fcf731fc7ed31546939bbbaa932e38770f3f6e77864fc353328ddaf059c7`
- Exp450 원본 catalog SHA-256: `99aeceb59e98cac160ddc87b252a9be7aca087ba78627847bfbe4005f4b39c4b`
- Exp450 원본 Git commit: `eac0d3b4`
- 데이터베이스는 `FrozenPaperConfig`의 고정 경로와 SHA-256을 사용한다. 권한을 발급하기 전에 해당 파일 자체를 SHA-256으로 다시 계산하고, Exp451 catalog의 SHA-256 및 고정 값과 모두 일치해야 한다. catalog에 적힌 값만으로는 평가 권한이 열리지 않는다.
- Exp451 context SHA-256은 실행 시 현재 Exp451 code closure, catalog, backend, config로 생성되며, migration manifest의 `target_context_sha256` 및 모든 Exp451 checkpoint와 일치해야 한다.

## 가속 및 동등성 기준

MPS의 native adaptive-average pooling을 먼저 사용한다. 관찰된 non-divisible 또는 mixed-axis MPS 오류일 때만 같은 입력을 PyTorch CPU native adaptive-average pooling으로 전환한다. 다른 오류는 숨기지 않고 실패 ledger에 남긴다.

이 전환은 pooling window, 점수, 임계값, 라우팅 규칙, 데이터베이스, A–D 조건 및 평가 정의를 바꾸지 않는다. 장치에 따른 부동소수점 누산 순서 차이로 미세한 연속 점수 차이는 허용한다. 이는 알고리즘 정의의 변경이 아니라 수치 실행 경로의 차이이며, 각 좌표의 실행 출처와 payload SHA-256으로 추적한다.

## MLX 실행 전 평가

Exp451의 전체 점수 생성 경로는 부분적으로만 tensor 연산에 적합하다. Adaptive
pooling과 dense recurrence-distance 하위 단계는 tensor 대상이지만, 고정 알고리즘은
SciPy의 `resample_poly`·PCHIP, scikit-learn의 KNN, aeon의
MiniRocket·MultiRocket/Hydra 계열, 그리고 PyTorch 영상 연산의 현재 의미를 함께
요구한다. 이 전체 경로를 동일한 알고리즘과 수치 계약으로 표현하고 검증한 MLX
구현은 없다. 따라서 MLX 정책의 첫 번째 조건인 “고정 알고리즘의 정확한 표현”이
실패했으며 MLX를 실행 backend로 선택하지 않는다.

검토 시점의 로컬 환경에서는 `mlx==0.32.0`, `mlx-metal==0.32.0`이 설치되어
있고 metal 사용 가능 여부는 `true`, 기본 장치는 `Device(gpu, 0)`으로 확인됐다.
이는 설치·장치 관측값일 뿐 실행 적합성이나 성능 결과가 아니다. 정확한 MLX 경로가
없으므로 다음 항목은 실행하지 않았으며 결과를 추정해 기록하지 않는다.

- 전송·컴파일 overhead 측정: `not_run_gate_1_failed`
- CPU 대비 MLX parity smoke: `not_run_gate_1_failed`
- 동일 입력과 seed에서 finite output, 모든 사전 등록 alpha의 동일 candidate index,
  허용 오차 이내 점수라는 parity 판정: `not_run_gate_1_failed`
- 장치 전송·컴파일·동기화를 포함해 MLX가 최소 1.1배 빨라야 한다는 benchmark:
  `not_run_gate_1_failed`

정식 fallback은 SciPy·scikit-learn·aeon 단계를 기존 CPU 구현으로 유지하고,
PyTorch tensor 단계는 MPS를 우선 사용하되 승인된 shape 오류에서만 CPU native
adaptive pooling으로 전환하는 현재 경로다. 명시적인 fallback 사유는
`mlx_gate_1_failed_exact_frozen_algorithm_not_expressible_and_validated`이다. 이
판정과 버전·장치·미실행 상태는 `Exp451BackendDecision`의
`mlx_assessment`에 모두 저장되어 run context SHA-256에 포함된다.

## 평가와 보고서의 잠금 조건

TEST label은 다음이 모두 확인되기 전에는 열지 않는다.

1. 정확히 4,468개 성공 checkpoint가 존재한다.
2. 네 조건 A–D가 각각 1,117개 checkpoint를 포함한다.
3. 모든 imported/native envelope의 context, catalog, backend, raw identity, execution origin, payload SHA-256 및 typed payload 역직렬화가 검증된다. Imported envelope은 migration이 사용한 고정 Exp450 backend record와 정확히 같아야 하고, native envelope은 현재 Exp451 backend record와 정확히 같아야 한다.
4. 네 condition manifest가 동결되고 검증된다.

검증, 연속 점수, 평가 결과는 각각 이전 단계가 실제로 발급한 불투명 권한 객체에 묶인다. JSON으로 다시 읽은 값이나 변경 후 SHA-256을 다시 계산한 값은 새 권한을 만들 수 없다. 따라서 네 persisted condition manifest를 create-only 방식으로 발행한 뒤, 같은 검증 권한과 바이트 단위로 다시 확인하고서만 TEST label을 읽는다.

AP/AUROC, ROC, PR 및 비교 방법 순서는 Exp450의 고정 정의를 그대로 사용한다. ROC 축은 `false_positive_rate`(x)와 `true_positive_rate`(y)이며, PR 축은 `recall`(x)과 `precision`(y)이다. 비교 그래프에는 A–D와 모든 고정 baseline 방법을 함께 표시한다.

## 재현 명령

프로젝트 루트에서 다음 순서로 실행한다. 실제 전체 실행 전에는 의존성 버전과 read-only 데이터베이스 바인딩을 확인한다.

```bash
python3 scratch/run_exp451_verified_accelerated_resume_antigravity.py migrate --mode full
python3 scratch/run_exp451_verified_accelerated_resume_antigravity.py verify-migration --mode full
python3 scratch/run_exp451_verified_accelerated_resume_antigravity.py run --mode full
python3 scratch/run_exp451_verified_accelerated_resume_antigravity.py status --mode full
python3 scratch/run_exp451_verified_accelerated_resume_antigravity.py verify --mode full
python3 scratch/run_exp451_verified_accelerated_resume_antigravity.py evaluate --mode full
python3 scratch/run_exp451_verified_accelerated_resume_antigravity.py report --mode full
```

기본 Python 환경에 `pycatch22` 또는 `aeon`이 없으면 공식 통합 실행을 시작하지 않는다. 이 Task의 fixture 테스트는 해당 선택 의존성 없이 수행할 수 있으며, 의존성 부재는 실험 결과가 아닌 실행 환경의 보류 사유로 기록한다.

보고서 명령은 먼저 최종 디렉터리의 sibling stage에 모든 Exp450 원본 표·그래프, Exp451 provenance 및 완료 marker를 기록하고 디렉터리를 `fsync`한다. 그런 다음 배타적 파일 잠금 아래에서 Darwin `RENAME_EXCL`을 사용해 완성된 디렉터리 전체를 한 번에 no-clobber 발행한다. Rename 직전 장애는 최종 디렉터리를 만들지 않고, rename 직후 장애는 완료 marker가 이미 들어 있는 유효한 최종 디렉터리를 남기므로 동일 바이트 재시도가 가능하다. Marker가 없거나 다른 marker를 가진 기존 최종 경로는 외부 소유일 수 있으므로 삭제하거나 재사용하지 않는다.

보고서 발행 잠금은 고정된 일반 파일에 대한 운영체제의 배타적 파일 잠금을 사용한다. 다른 프로세스가 잠금을 보유하는 동안에는 발행을 거부하고 그 프로세스의 잠금 파일이나 결과 디렉터리를 삭제하지 않는다. 잠금 보유 프로세스가 종료되면 운영체제가 소유권을 자동으로 해제하므로, 남은 잠금 파일을 안전하게 다시 사용해 중단된 발행을 재시도한다.
