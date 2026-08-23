# Exp451 방법 및 실행 출처 기록

## 목적

Exp451은 Exp450의 검증된 결과 4,234개를 다시 계산하지 않고 별도 Exp451 결과 루트로 이관한 뒤, 남은 234개만 가속 경로로 계산하는 실험이다. Exp450 원본 결과 루트는 읽기 전용이며 수정하지 않는다.

## 조건과 실행 출처

| 조건 | 길이 정렬 | 영상 특징 | 이관 결과 | 신규 native 결과 | 최종 |
| --- | --- | --- | ---: | ---: | ---: |
| A | linear | linear | 1,117 | 0 | 1,117 |
| B | polyphase_pchip | linear | 1,117 | 0 | 1,117 |
| C | linear | area | 1,000 | 117 | 1,117 |
| D | polyphase_pchip | area | 1,000 | 117 | 1,117 |
| 합계 |  |  | 4,234 | 234 | 4,468 |

이관 결과의 `execution_origin`은 `imported_exp450_mps`이고, 새 결과의 `execution_origin`은 `exp451_native_cpu_pool_fallback`이다. 두 수는 평가 및 보고서에서도 합치지 않고 별도로 기록한다.

## 고정 권한과 데이터베이스

- Exp450 원본 context SHA-256: `5d5b44aec0e738ae61d9dafd85d9eaea1448baba49705c4d042c5ed1d6886f83`
- Exp450 원본 transforms SHA-256: `28b5fcf731fc7ed31546939bbbaa932e38770f3f6e77864fc353328ddaf059c7`
- Exp450 원본 catalog SHA-256: `99aeceb59e98cac160ddc87b252a9be7aca087ba78627847bfbe4005f4b39c4b`
- Exp450 원본 Git commit: `eac0d3b4`
- 데이터베이스는 `FrozenPaperConfig`의 고정 경로와 SHA-256을 사용한다. Exp451 catalog의 데이터베이스 SHA-256은 이 값과 일치해야 한다.
- Exp451 context SHA-256은 실행 시 현재 Exp451 code closure, catalog, backend, config로 생성되며, migration manifest의 `target_context_sha256` 및 모든 Exp451 checkpoint와 일치해야 한다.

## 가속 및 동등성 기준

MPS의 native adaptive-average pooling을 먼저 사용한다. 관찰된 non-divisible 또는 mixed-axis MPS 오류일 때만 같은 입력을 PyTorch CPU native adaptive-average pooling으로 전환한다. 다른 오류는 숨기지 않고 실패 ledger에 남긴다.

이 전환은 pooling window, 점수, 임계값, 라우팅 규칙, 데이터베이스, A–D 조건 및 평가 정의를 바꾸지 않는다. 장치에 따른 부동소수점 누산 순서 차이로 미세한 연속 점수 차이는 허용한다. 이는 알고리즘 정의의 변경이 아니라 수치 실행 경로의 차이이며, 각 좌표의 실행 출처와 payload SHA-256으로 추적한다.

## 평가와 보고서의 잠금 조건

TEST label은 다음이 모두 확인되기 전에는 열지 않는다.

1. 정확히 4,468개 성공 checkpoint가 존재한다.
2. 네 조건 A–D가 각각 1,117개 checkpoint를 포함한다.
3. 모든 imported/native envelope의 context, catalog, backend, raw identity, execution origin, payload SHA-256 및 typed payload 역직렬화가 검증된다.
4. 네 condition manifest가 동결되고 검증된다.

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
