# Exp439 구현 완료 보고서

## 핵심 판단

Exp438을 멈춘 원인은 이상 탐지 방법이 아니라, 정상적인 float32 반올림 오차를
고정 `1e-5` 한계로 잘못 거부한 검증 코드였다. Exp439는 탐지 규칙을 바꾸지 않고
재구성 검증 한계만 `max(1e-5, component 최대 크기에서 float32 1 ULP)`로
수정했다.

실제 `Chinatown_normal_1` TRAIN 83행은 결측·무한대가 0개였고, 최대 재구성
오차 `6.103515625e-05`가 scale-aware 허용치 `0.000244140625` 안에 있었다.
TRAIN-only 보정에서 high-frequency, low-frequency, amplitude-complexity 세 채널이
모두 사용 가능했다. TEST label 접근은 0이었다.

## 동결된 변경 범위

- 변경: float32 재구성 검증 한계만 1 ULP scale-aware 방식으로 계산한다.
- 유지: FFT bin, cutoff, inverse FFT, high 계산, float32 cast와 반환 component bytes.
- 유지: depth 선택, ASPE, feature, score, threshold, lane, candidate budget.
- 유지: prediction의 TEST label·실제 이상 위치·과거 family TEST 성능 접근 금지.
- 유지: 비유한 TRAIN 행은 행 전체 제외, 보간 없음.
- 유지: CPU worker 11개, worker 내부 수치 스레드 1개.

MLX, MPS, Numba는 Exp438 readiness의 동일 입력 parity와 end-to-end 속도 gate를
통과하지 못했으므로 CPU가 정본이다. 이 결정은 모델이나 점수 정의를 바꾸지 않는다.

## TDD 증거

### 1. 실제 Chinatown 회귀

little-endian float32 fixture SHA256:

```text
33bd5923753b1a3b6e4903ad9e8405d0c761a08f432b7a76c2177921034702a7
```

수정 전에는 `decomposition reconstruction tolerance failed`가 재현됐다. 수정 후
같은 행이 통과했고, 기존 Exp438 통과 입력의 low/high component bytes는 Exp439와
완전히 같았다. component를 4 ULP 변조한 테스트는 계속 실패 폐쇄했다.

### 2. Exp438 89개 prefix

Exp438 root는 읽기 전후 모든 파일 SHA가 같았다. 다음 고정 증거를 모두 검증했다.

```text
reviewed commit  d1da286e9c74c64abe76e0c255c86c30141813bd
readiness SHA256 8ac3c67395002e064f295c5aa8cd5472412579bcbe65d7c268934243a4ec1c0a
context SHA256   0630e884ab51d9d175bae3c82715694fa582375cf559f316cd83963fbe9f2fb6
state SHA256     9bbbbc58569e18dddbab211ffa1e2f267e23d7b8688256d0b9e3100093100fdc
progress SHA256  13a978b8f042d6345cd9fd000e32a9532930355e1320c8d3729dafffaa5791e3
record-set SHA   a97422f66338d1a2a633c3c8441560b525d364199aa8f2681ff7b07d3826eb8d
records          89
last             BeetleFly_normal_1
next             BeetleFly_normal_2
label access     0
```

89개는 임시 checkpoint에서 전부 변환·검증된 뒤 최종 경로에 한 번에 나타난다.
하나라도 변조·누락·재배열·추가되면 최종 checkpoint가 생성되지 않는다. import
manifest는 source/destination record SHA를 모두 기록하며 자체 SHA가 checkpoint
context에 묶인다. 중단·재개 후에도 `import_count=89`, `recompute_count=0`을
보존하고 `BeetleFly_normal_2`부터 계속한다.

### 3. 실패 데이터셋 로그

worker나 accelerator 오류는 부모가 traceback 문자열을 추측하지 않고 dataset,
TRAIN/TEST 크기 metadata, stage, backend, 원래 예외 type/message를 구조적으로
기록한다. 같은 내용이 progress JSON, events JSONL, parent text log에 남는다.

## 검증 결과

```text
Exp439   106 tests passed
Exp438    94 tests passed
Exp436   299 tests passed
Exp425   180 tests passed
total    679 tests passed, failures 0, errors 0
```

실제 DB 검증:

```text
PRAGMA quick_check  ok
DB SHA256           8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
DB nlink            1 -> 2 while snapshot open -> 1 after close
Chinatown TRAIN     83 rows, raw length 24, exclusions 0
selected depth      1
available channels  high_frequency, low_frequency, amplitude_complexity
TEST label access   0
```

구현 커밋:

```text
b4a88ea9 isolate Exp439 runtime
a783cd6a one-ULP reconstruction guard
6e68d769 verified 89-record prefix import
9230efa0 dataset-attributed failure evidence
```

## 다음 명령

작업 트리가 clean인 상태에서 readiness를 한 번만 생성한다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp439_scale_aware_reconstruction_repair_antigravity.py readiness
```

readiness가 1,117개 population, 89개 import manifest, prediction label access 0,
CPU fallback과 clean reviewed commit을 동결한 뒤에만 `new`를 실행한다.
