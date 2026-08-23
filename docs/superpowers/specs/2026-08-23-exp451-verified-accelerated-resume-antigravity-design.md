# Exp451 Verified Accelerated Resume Design

## 목적

Exp450의 완료된 4,234개 dataset-condition 결과를 다시 계산하지 않고 보존하면서, MPS가 지원하지 않는 adaptive-average pooling 모양에서 발생한 장시간 scalar-window 병목을 제거한다. Exp451은 기존 결과를 검증 가능한 방식으로 가져오고 남은 좌표만 가속된 실행 경로로 계산한다.

## 범위

- Exp450 공식 결과 루트는 읽기 전용 원본으로 취급한다.
- Exp451은 별도의 실험 ID, 코드 네임스페이스, 결과 루트, 평가 루트, 보고서 루트와 코드맵 3종을 사용한다.
- 점수 정의, 조건 A–D, 데이터베이스, 데이터셋 1,117개, TEST 195,900행, 임계값, 라우팅 규칙과 평가 방법은 Exp450과 동일하다.
- 변경되는 실행 세부사항은 MPS native adaptive pooling이 특정 모양을 거부할 때 PyTorch CPU native adaptive pooling으로 전환하는 것뿐이다.
- 보안 강화, 공격 테스트, UI 변경과 무관한 리팩터링은 범위에서 제외한다.

## 고려한 접근법

### 선택: 검증 후 새 Exp451 루트로 이관

Exp450의 옛 코드 closure를 재구성하여 각 성공 체크포인트의 envelope, context, catalog, raw identity, backend와 payload hash를 검증한다. 검증된 payload만 Exp451의 새 context로 다시 봉인하고, 원본 증거와 새 결과 사이의 일대일 대응을 별도 migration manifest에 기록한다. 기존 결과를 보존하면서도 혼합 실행 출처를 명시할 수 있어 이 방식을 채택한다.

### 기각: Exp450 체크포인트의 context hash 수정

기존 파일을 다시 쓰면 create-only 계약과 원본 증거가 훼손되므로 사용하지 않는다.

### 기각: 4,468개 전체 재실행

과학적으로 단순하지만 이미 검증 가능한 4,234개 결과를 버리고 시간이 크게 증가하므로 사용하지 않는다.

## 실험 정체성과 경로

- 실험 ID: `Exp451`
- 코드: `experiments/exp451_verified_accelerated_resume_antigravity/`
- 실행기: `scratch/run_exp451_verified_accelerated_resume_antigravity.py`
- 원본: `/Users/mac/Documents/Dataset/exp450_resampling_image_ablation_official_antigravity_20260820`
- 공식 결과: `/Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_official_antigravity_20260823`
- 평가 결과: `/Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_evaluation_antigravity_20260823`
- 보고서: `/Users/mac/Documents/Dataset/exp451_verified_accelerated_resume_report_antigravity_20260823`
- 이관 manifest: Exp451 공식 결과 루트의 `migration_manifest_antigravity.json`
- 코드맵: `docs/codemap/codemap_exp451_antigravity.html`, `.json`, `.lock`

모든 새 파일 이름에는 `antigravity`를 포함한다.

## 구성 요소

### Exp450 원본 검증기

원본 체크포인트가 생성될 당시의 Exp450 context를 고정 상수로 복원한다. 최소한 다음 값을 고정하고 실제 원본과 대조한다.

- 원본 context SHA-256: `5d5b44aec0e738ae61d9dafd85d9eaea1448baba49705c4d042c5ed1d6886f83`
- 원본 transforms SHA-256: `28b5fcf731fc7ed31546939bbbaa932e38770f3f6e77864fc353328ddaf059c7`
- 원본 catalog SHA-256: `99aeceb59e98cac160ddc87b252a9be7aca087ba78627847bfbe4005f4b39c4b`
- 원본 backend: PyTorch `2.13.0`, MPS 사용, 논리 CPU 12개, CPU worker 11개, 내부 수치 thread 1개

검증기는 현재 수정된 Exp450 context를 원본 권한으로 사용하지 않는다. 원본 context를 구성하는 모든 code-closure digest는 Git의 원본 commit `eac0d3b4`에서 읽어 재구성하고, 재구성된 context SHA-256이 고정 원본 값과 일치해야만 이관을 시작한다.

### 이관기

이관기는 다음 순서를 지킨다.

1. Exp450 원본 root와 catalog를 읽기 전용으로 연다.
2. 성공 checkpoint 전체를 원본 context로 검증한다.
3. 중복 `(dataset, condition)` 좌표, 실패 기록, 잘못된 순서, 누락된 raw identity를 거부한다.
4. 검증된 4,234개 좌표의 typed result를 Exp451 context로 create-only 저장한다.
5. 원본 상대 경로, 원본 envelope SHA-256, payload SHA-256, 원본 context SHA-256, Exp451 envelope SHA-256을 manifest에 기록한다.
6. manifest는 정확히 4,234개의 고유 좌표를 포함하고 A=1,117, B=1,117, C=1,000, D=1,000이어야 한다.
7. 중간 실패 시 완료된 Exp451 checkpoint는 재실행에서 다시 검증하여 건너뛰고, manifest는 전체 이관이 끝나기 전에는 발행하지 않는다.

이관은 원본 파일을 수정·이동·삭제하지 않는다. payload의 점수 배열도 다시 계산하지 않는다.

### 가속 실행 경로

MPS native `adaptive_avg_pool1d/2d`를 먼저 호출한다. MPS가 이미 관찰된 non-divisible 또는 mixed-axis 모양 오류를 반환할 때만 동일한 입력을 CPU로 복사하여 PyTorch native adaptive-average pooling을 실행한다. 그 외 RuntimeError는 그대로 실패로 기록한다.

이 전환은 pooling window 정의를 바꾸지 않는다. 부동소수점 누산 순서가 달라 미세한 점수 차이가 발생할 수 있으며, 사용자가 허용한 “알고리즘적으로 동일하면 일부 결과 차이 인정” 기준을 적용한다. 기존 Exp450 이관 결과와 Exp451 신규 결과의 실행 출처는 각각 `imported_exp450_mps`와 `exp451_native_cpu_pool_fallback`으로 구분한다.

MPS-heavy 프로세스는 동시에 하나만 실행한다. CPU 병렬 단계가 새로 시작되는 경우 논리 CPU 수에서 1을 뺀 11개 worker와 내부 thread 1개를 기록한다.

## 실행 흐름

1. `migrate` 명령이 원본 context와 4,234개 checkpoint를 검증한다.
2. Exp451 create-only checkpoint와 migration manifest를 생성한다.
3. `verify-migration` 명령이 좌표 수, payload hash, 원본-대상 대응과 원본 비변경을 다시 검증한다.
4. `run --mode full`이 Exp451 inventory에서 누락된 234개 좌표만 실행한다.
5. 실패 좌표는 기존 파이프라인처럼 별도 failure ledger에 기록하여 전체 진행을 막지 않는다.
6. 모든 4,468개 성공 결과가 존재하면 네 조건 manifest를 동결한다.
7. 기존 Exp450 평가 규칙을 재사용하여 AP, AUROC, ROC, PR과 비교 표를 생성한다.

## 실패 처리

- 원본 context 또는 code closure가 다르면 이관 전 즉시 중단한다.
- 원본 checkpoint 하나라도 검증되지 않으면 해당 오류 좌표를 보고하고 manifest를 발행하지 않는다.
- 대상에 다른 bytes의 파일이 존재하면 덮어쓰지 않고 중단한다.
- 이미 정확히 같은 대상 checkpoint가 있으면 검증 후 재사용한다.
- 신규 계산 실패는 `(dataset, condition, wave)` 단위로 저장하고 나머지 좌표를 계속 처리한다.
- 평가와 보고서는 4,468개 성공 및 네 manifest 검증 전에는 시작하지 않는다.

## 검증

### 단위 테스트

- 원본 context 재구성이 고정 SHA-256과 일치한다.
- 변조된 원본 context, payload, raw identity와 backend를 거부한다.
- 이관 결과가 원본 typed payload와 동일하다.
- 중복·누락 좌표와 부분 manifest를 거부한다.
- MPS 비지원 오류만 CPU native pooling으로 전환한다.
- 관련 없는 MPS 오류는 삼키지 않는다.
- mixed-axis 입력이 scalar-window 구현을 호출하지 않고 유한한 정확한 shape를 반환한다.

### 통합 테스트

- 임시 root의 작은 A–D fixture를 이관하고 재검증한다.
- 실제 원본 4,234개를 읽기 전용 검증하고 조건별 개수를 확인한다.
- 이관 전후 Exp450 원본 파일 목록과 SHA-256 집합이 동일하다.
- Exp451 readiness가 남은 좌표를 정확히 234개로 계산한다.
- 실제 MPS 비정형 pooling smoke가 완료되고 실행 시간이 로그에 기록된다.

### 완료 기준

- Exp450 원본의 모든 파일 SHA-256이 이관 전후 동일하다.
- Exp451 migration manifest가 4,234개 고유 좌표를 검증한다.
- Exp451 run이 정확히 234개 누락 좌표에서 시작한다.
- 진행 로그에 완료·실패·남은 개수와 ETA가 지속적으로 기록된다.
- 최종 4,468개 결과와 네 조건 manifest가 검증되기 전에는 완료를 선언하지 않는다.

## 논문 기록

최종 방법 및 재현성 문서에는 다음을 명시한다.

- A/B 전체와 C/D 각 1,000개는 원래 Exp450 MPS 실행에서 검증 후 이관했다.
- 남은 C/D 좌표는 MPS native 미지원 모양에서 CPU native adaptive-average pooling으로 전환했다.
- pooling window 정의와 점수·임계값·라우팅·평가 정의는 동일하다.
- 실행 장치 차이로 인한 미세한 수치 차이는 허용했으며 각 좌표의 실행 출처를 manifest에 보존했다.
