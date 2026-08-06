# Exp435 정상 패턴 합의 기반 복구 설계

## 1. 문서 상태

- 실험 번호: Exp435
- 상태: 사용자 승인 설계, 구현 전
- 승인일: 2026-08-06
- 작업 위치: `/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery`
- 기준 데이터셋 수: 1,117
- 결과 성격: 연구 검토용(`review_only`)
- 자동 경보 승격: 금지

이 문서는 Exp435의 과학적 의도와 안전 경계를 고정한다. 정확한 입력 SHA256,
실행 경로, Git commit, 환경 버전과 실행 명령은 구현 전에 별도 사전등록과
runbook에서 고정한다. 사전등록을 완료하기 전에는 TEST 예측을 시작하지 않는다.

## 2. 쉬운 설명

현재 Hard 기준은 경보의 정확성은 높지만 실제 이상 중 약 절반만 찾는다. Exp435는
Hard 경보를 지우거나 바꾸지 않고, 정상 TRAIN 데이터와 확실히 다른 TEST 항목을
추가로 찾는 실험이다.

한 가지 특징만 이상하다고 바로 후보로 만들지 않는다. 먼저 정상 설비에서 자주
반복되는 대표 파형과 충분히 달라야 한다. 그 상태에서 파형의 기울기 또는 반복
주기까지 달라야 검토 후보가 된다.

```text
대표 정상 패턴과 다름 AND (기울기가 다름 OR 반복 주기가 다름)
```

새 후보는 사람이 검토할 연구 결과다. Exp435가 성공하더라도 같은 실험 안에서
자동 경보로 승격하지 않는다.

## 3. 배경과 기준

### 3.1 검증된 비교 기준

Exp404 Hard prediction을 Exp416에서 사후 평가한 현재 기준은 다음과 같다.

- prediction SHA256:
  `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`
- TP 1,915 / FP 303 / FN 2,003 / TN 191,679
- precision 0.8633904418
- recall 0.4887697805
- micro F1 0.6241851369
- dataset-balanced macro F1 0.6089825087

Exp426은 새 탐지기가 아니라 1,117개 전체의 검증 가능한 실행 기준이다. 공식
attempt 2는 `1117/983/134/0`의 완료/가져오기/TRAIN 판단 보류/오류 구성을
완료했다. 후보는 0개이며 TEST 라벨을 사용하지 않았다. Exp435는 Exp426의
입력·실행·checkpoint·publication 안전장치를 재사용 가능한 기반으로 본다.

### 3.2 폐기한 대안의 교훈

비공식 antigravity Exp425~434 구현은 아이디어 참고용일 뿐 실험 근거로 사용하지
않는다. 특히 다음 방식을 Exp435에 가져오지 않는다.

- 여러 TRAIN 또는 TEST instance를 하나의 긴 시계열로 연결
- 점수 계산 직후 prediction SHA256 동결 없이 TEST 라벨 열기
- 데이터셋마다 일정 수의 후보를 강제로 생성
- 전체 후보가 많을 때 상위 N개만 잘라내기
- TEST 결과를 보고 alpha, threshold, source 또는 budget 다시 선택
- 실제로 사용하지 않은 MLX·CEEMD·XAI 이름을 결과에 표시

다만 정상 패턴 거리, 기울기 변화, ACF 주기 변화와 반복 정상 파형을 강한 정상
증거로 사용하는 생각은 올바른 경계 안에서 다시 구현한다.

## 4. 연구 질문과 가설

연구 질문은 다음과 같다.

> 정상 TRAIN에서 반복되는 대표 파형과의 거리, 기울기 거리, 주기 거리의 보수적
> 합의를 사용하면 Hard가 놓친 데이터셋을 새로 구조하면서 추가 오탐을 30개
> 이하로 제한할 수 있는가?

가설은 세 특징 중 아무 두 개가 아니라, 대표 패턴 거리를 반드시 요구하면 단순한
노이즈나 한 특징의 흔들림으로 생기는 오탐이 줄어든다는 것이다. 기울기와 주기는
서로 다른 이상 형태를 보완한다.

## 5. 과학적 방화벽

### 5.1 prediction 단계에서 허용

- 정상 라벨(`label = 0`)의 TRAIN 원시 값
- 정답과 이상 위치가 제거된 TEST 원시 값
- 데이터셋 이름
- TRAIN/TEST instance 개수, instance index와 시계열 길이
- 고정된 lane 좌표와 그 SHA256
- 사전등록된 점수식, threshold, alpha, 후보 상한과 실행 backend
- 동일 데이터셋 안에서 반복되는 정상 TRAIN 파형의 빈도

### 5.2 prediction 동결 전 금지

- TEST 라벨과 실제 이상 위치
- 혼동행렬, precision, recall, F1
- 과거 family별 TEST 성능
- human-assisted 지표
- TEST 결과를 본 뒤 다시 고른 alpha, threshold, source 또는 budget
- Priority·Shadow 근거의 자동 경보 승격
- tail replacement

prediction 과정과 사후 평가 과정은 별도 entrypoint와 별도 import 경계로 만든다.
prediction package에는 TEST label loader를 넣지 않는다.

## 6. 데이터 단위와 전처리

### 6.1 instance 경계 보존

한 후보 좌표는 반드시 `(dataset, TEST instance_index)` 한 개를 뜻한다. TRAIN과
TEST의 여러 instance를 연결하지 않는다. 길이가 다른 파형끼리도 연결하거나
강제로 늘리지 않는다.

각 데이터셋의 TRAIN은 같은 길이의 instance끼리 묶는다. TEST instance는 같은
길이의 정상 TRAIN 그룹이 있을 때만 그 그룹과 비교한다. 비교 가능한 그룹이
없으면 판단을 보류한다.

### 6.2 반복 정상 파형

완전히 같은 TRAIN 파형은 데이터 오류로 제거하지 않는다. 반도체 설비가 같은
상태를 안정적으로 반복했다는 정상 증거로 취급한다.

거리 계산에서는 모든 TRAIN instance를 유지한다. TRAIN 점수를 만들 때는 현재
행 하나만 제외하고, 같은 값을 가진 다른 TRAIN 행은 남긴다. 따라서 반복 빈도가
높은 정상 파형은 가까운 이웃을 여러 개 제공하고 정상 판단을 강화한다.

### 6.3 TRAIN-only 정규화

각 길이 그룹에서 TRAIN 값만 사용해 중앙값과 MAD 기반 크기를 계산한다. TEST 값은
이 고정된 값으로 변환만 한다. 크기가 0이거나 유한하지 않은 그룹은 억지로 점수를
만들지 않고 명시적으로 판단을 보류한다.

## 7. 세 가지 점수

모든 점수는 값이 클수록 정상 TRAIN과 다르다는 뜻이다. 정확한 수식과 상수는
사전등록에서 한 번 고정하고 TEST 결과로 바꾸지 않는다.

### 7.1 대표 정상 패턴 거리 — 필수

정규화한 원시 파형을 정상 TRAIN 파형들과 비교한다. 가장 가까운 소수의 정상
파형 거리들을 합쳐 대표 패턴 거리를 만든다. 반복 정상 파형을 그대로 유지하므로
설비가 자주 반복한 상태와 같은 TEST 파형은 낮은 점수를 얻는다.

### 7.2 기울기 거리 — 보조 1

각 파형의 한 시점 차이값을 계산한다. 정상 TRAIN의 차이값 패턴과 비교하여 갑작스런
상승·하락, 모서리 모양 또는 변화 속도의 차이를 찾는다.

### 7.3 ACF 주기 거리 — 보조 2

TRAIN으로 고정한 lag 범위에서 자기상관(ACF) 특징을 만든다. 정상 설비의 반복
주기가 흔들리거나 사라지는 변화를 찾는다. 길이가 너무 짧거나 분산이 없어 ACF를
안전하게 계산할 수 없으면 이 채널만 사용할 수 없는 것으로 기록한다.

기울기와 ACF가 모두 사용할 수 없으면 해당 TEST instance는 판단을 보류한다.

## 8. TRAIN-only 기준 결정

각 채널의 정상 점수 분포는 leave-one-row-out 방식으로 만든다. 한 TRAIN 행의
점수를 계산할 때 그 행만 기준 집합에서 제외하고 나머지 정상 TRAIN 행과 비교한다.
동일한 값을 가진 다른 행은 제거하지 않는다.

Exp435는 하나의 고정 alpha와 하나의 고정 점수 정의만 공식 prediction에 사용한다.
alpha, 최소 TRAIN 수, 이웃 수, ACF lag와 수치 허용오차는 다음 단계의 사전등록에
정확히 기록한다. TRAIN-only readiness로 계산 가능성과 유한성은 확인할 수 있지만,
그 결과로 여러 공식 설정 중 하나를 TEST 직전에 고르는 절차는 두지 않는다.

## 9. 후보 결정과 전체 안전장치

한 TEST instance는 다음 조건을 모두 만족할 때만 Exp435 후보가 된다.

1. 대표 정상 패턴 점수가 그 TRAIN-only 기준을 넘는다.
2. 기울기 점수 또는 ACF 점수 중 적어도 하나가 각자의 TRAIN-only 기준을 넘는다.
3. 좌표가 고정 lane 좌표와 겹치지 않는다.
4. 모든 사용 점수와 판정 근거가 유한하고 완전히 기록된다.

공식 새 후보 수의 전체 상한은 100개다.

- 0~100개: 정렬·중복 검증 후 prediction 동결 단계로 진행
- 101개 이상: Exp435 prediction을 실패 처리
- 초과 시 상위 100개 선택 금지
- 초과 결과를 보고 같은 Exp435의 기준 변경 금지

후보가 0개인 경우에도 빈 prediction을 동결할 수 있지만 개선 성공 조건은 통과할
수 없다.

## 10. MLX와 CPU 결정

새 점수원 실험이므로 전체 실행 전에 MLX 가능성을 반드시 평가한다. 대표 패턴,
기울기와 ACF의 dense distance 계산은 MLX 후보 단계다.

smoke seed `43501`로 만든 동일한 float32 입력과 동일한 seed를 CPU와 MLX에
그대로 전달해 비교하고 다음을 모두 만족할 때만 MLX를 채택한다. 이 seed는
사후 bootstrap seed와 분리된 MLX parity 계약의 일부다.

고정 batch sizes `16, 32, 64, 128, 256`은 reference 축의 묶음 크기다. 각 묶음의
RMS 거리에서 최근접 k개를 남기고 이를 다음 묶음과 전역적으로 병합해, 전체
reference를 한 번에 계산한 것과 같은 k-neighbor median을 만든다. readiness에서
선택한 reference batch size는 production에서도 바꾸지 않는다.

1. 점수식, threshold, alpha와 routing이 바뀌지 않는다.
2. 모든 출력이 유한하다.
3. 공식 alpha에서 후보 좌표가 완전히 같다.
4. 각 유한 score 쌍의 정규화오차를
   `abs(cpu_score - mlx_score) / max(abs(cpu_score), abs(mlx_score), 1e-12)`로
   계산했을 때 모든 쌍의 최댓값이 `1e-5` 이하이고, 최대 절대오차도 `1e-5` 이하이다.
5. 전송과 동기화 시간을 포함해 정확히 같은 단계가 1.1배 이상 빠르다.

하나라도 실패하면 CPU를 사용한다. CPU 경로는 저장소 정책에 따라 최대 7개의
dataset worker를 사용하고 worker 내부의 BLAS·Numba thread 경쟁을 제한한다.
MLX/MPS-heavy 프로세스는 동시에 하나만 실행한다. backend 결정과 버전, 장치,
시간, parity, fallback 이유를 manifest에 남긴다.

## 11. 실행 상태와 재개

Exp426의 검증된 구조를 참고해 `readiness`, `new`, `resume`, `inspect`를 분리한다.
프로세스가 직접 다음 파일을 원자적으로 또는 append-only로 기록한다.

- 현재 상태 JSON
- 데이터셋 완료 event JSONL
- 사람이 읽는 parent text log
- PID 파일
- 데이터셋별 checkpoint record

진행 상태에는 최소한 다음 정보를 둔다.

- PID와 heartbeat
- 현재 데이터셋 이름
- TRAIN/TEST instance 수와 시계열 길이
- 현재 단계와 backend
- 완료/전체, 성공/판단 보류/오류 수
- 처리속도, 경과시간과 ETA
- checkpoint 경로
- 최근 안정적인 오류

checkpoint는 데이터셋 하나가 끝날 때마다 고정한다. 저장 완료 순서는 데이터셋
이름 순서로 결정하여 worker 완료 순서가 결과 SHA256을 바꾸지 못하게 한다.

## 12. prediction 동결

1,117개 처리가 끝나고 오류와 전체 후보 상한 검사를 통과하면 다음을 canonical
형식으로 만든다.

- prediction JSONL
- 데이터셋 요약 JSONL
- 채널별 판정 근거 JSONL
- 판단 보류 ledger JSONL
- backend 결정 JSON
- manifest와 status JSON
- artifact SHA256 map

모든 새 파일 이름에는 `antigravity`를 포함한다. prediction과 manifest의 SHA256을
독립적으로 다시 계산하여 일치한 뒤 prediction 단계를 닫는다. 이 시점까지 TEST
라벨 접근 수는 0이어야 한다.

## 13. 별도 사후 평가

동결 뒤 별도 evaluator만 TEST 라벨을 연다. 비교 대상은 기존 Hard와
`Hard + Exp435 review-only 후보`다. Hard 좌표는 삭제하거나 이동하지 않는다.

다음 항목을 1,117개 전체에서 보고한다.

- TP, FP, FN, TN
- precision, recall, specificity, FPR와 F1
- micro와 dataset-balanced macro 결과
- 기존 TP가 0이었으나 새 TP가 생긴 구조 데이터셋
- 기존 강한 데이터셋의 악화
- 데이터셋별 paired delta
- 고정 seed의 dataset-cluster bootstrap 95% 신뢰구간

bootstrap은 데이터셋을 단위로 복원 추출하고, 같은 표본에서 Hard와 결합 결과를
함께 계산한다. micro F1 delta와 macro F1 delta의 95% 구간을 각각 만든다.

## 14. 사전등록된 성공 조건

Exp435는 다음을 모두 만족할 때만 개선 실험으로 통과한다.

1. 1,117개 전체 prediction과 평가가 보존된다.
2. 새 후보는 100개 이하이다.
3. 기존 Hard에서 TP가 0이었던 데이터셋 중 최소 10개가 TP 1개 이상으로 구조된다.
4. Exp435 후보가 추가한 FP는 전체 30개 이하이다.
5. micro F1 delta가 양수이고 95% bootstrap 신뢰구간의 최저값도 0보다 크다.
6. dataset-balanced macro F1 delta가 양수이고 95% bootstrap 신뢰구간의 최저값도
   0보다 크다.
7. prediction 동결 전 TEST 라벨 접근, 같은 실험 재선택 또는 금지 정보 사용이 0이다.

일부 데이터셋의 F1 악화는 위 조건이 모두 통과할 때만 허용한다. 악화한 데이터셋은
숨기지 않고 이름, 기존/새 혼동 수와 원인을 별도 표로 남긴다.

조건을 하나라도 통과하지 못하면 Exp435를 실패 ledger에 보존한다. Exp435 설정을
고치지 않고 새 실험 번호로 다음 가설을 사전등록한다.

## 15. 구현 경계

제품 코드는 기존 Exp417, Exp425 또는 Exp426 파일을 직접 바꾸는 방식보다 새
Exp435 package에 둔다. 검증된 읽기 전용 DB snapshot, frozen lane, canonical
artifact와 checkpoint 부품만 명시적으로 재사용한다.

구현 전에 코드맵에서 다음 세 질문을 확인한다.

1. 재사용할 모듈을 무엇이 호출하는가?
2. 그 모듈을 바꾸면 무엇에 영향을 주는가?
3. 어떤 테스트가 그 모듈을 덮는가?

새 모듈 경계와 데이터 흐름이 추가되므로 구현 commit에는
`docs/codemap/codemap.html`, `codemap.json`, `codemap.lock`을 함께 갱신한다.

## 16. 구현 전 필수 문서

이 설계가 사용자 검토를 통과한 뒤 다음을 순서대로 만든다.

1. Exp435 구현 계획
2. Exp435 한국어 사전등록
3. Exp435 한국어 runbook
4. TRAIN-only readiness와 MLX/CPU parity 계약
5. prediction 동결 후에만 실행 가능한 별도 평가 계약

사전등록에는 아직 이 문서에서 숫자로 고정하지 않은 alpha, 최소 TRAIN 수, 이웃 수,
ACF lag, bootstrap 반복 수·seed, 수치 허용오차와 절대 artifact 경로를 빠짐없이
고정한다.
