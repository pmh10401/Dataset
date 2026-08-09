# Exp443 정상 TRAIN 결측 처리 정책 비교 설계

작성일: 2026-08-09

상태: 사용자 설계 승인 완료, 구현 전 동결

실험 ID: `Exp443`

실행 ID: `exp443-missing-normal-train-policy-comparison-antigravity-20260809`

## 1. 쉬운 설명

현재 자동 Hard 기준선은 결측값을 전혀 처리하지 않는 모델이 아니다. 기존
전처리는 정상 TRAIN과 TEST 시계열의 `NaN`, `+Inf`, `-Inf`를 원래 시간축에서
선형 보간하고, 시작과 끝의 결측은 가장 가까운 유한값으로 채운다.

Exp443은 이 기존 보간을 그대로 대조군으로 두고, 정상 TRAIN 결측 행을 버리는
방법과 같은 길이의 다른 정상 TRAIN을 이용해 복원하는 방법을 비교한다. TEST
전처리와 탐지 규칙은 바꾸지 않는다. 따라서 성능 차이가 생기면 정상 TRAIN의
결측 처리 차이로 설명할 수 있다.

전체 Hard를 역사적으로 처음부터 다시 만드는 대신, 원시 정상 TRAIN을 실제로
읽고 Standard 후보를 Hard로 승격했던 공식 MiniROCKET 부분만 다시 계산한다.
나머지 기존 Hard와 exact-normal 보호 규칙은 동결한다.

## 2. 확인된 현재 상태

동결 DB의 정상 TRAIN에는 다음 결측 inventory가 있다.

- 전체 모집단: 1,117개 데이터셋
- 결측 영향: 19개 데이터셋, 4개 family
- 결측 정상 TRAIN: 164행
- 비유한 값: 983개
- 전부 결측인 행: 0개
- 결측 행 전체를 제외한 뒤 남는 정상 TRAIN의 데이터셋별 최솟값: 15행

Exp414는 동일한 19개 데이터셋을 선형 보간했지만 TRAIN-only near-repeat
자격 검증만 수행했다. prediction과 TEST 평가를 만들지 않았으므로, 결측 처리법이
현재 Hard 성능에 미치는 영향에는 답하지 않는다.

Exp416이 감사한 현재 기준선은 다음과 같다.

```text
TP 1,915 / FP 303 / FN 2,003 / TN 191,679
micro F1 = 0.6241851368970013
```

결측 영향 19개를 사후 진단하면 `TP 21 / FP 12 / FN 53 / TN 3,614`, micro
F1 `0.3925233644859813`이다. 이 subgroup 결과는 연구 필요성을 설명하는
진단일 뿐이며 dataset/family별 prediction routing에는 사용하지 않는다.

## 3. 연구 질문과 주장 범위

주 질문은 다음과 같다.

> 정상 TRAIN의 결측 처리만 바꿨을 때 기존 자동 Hard의 MiniROCKET 승격 부분이
> 더 많은 놓친 이상을 복구하면서 전체 오경보와 강한 데이터셋 성능을 보존하는가?

Exp443은 같은 1,117개 벤치마크를 반복 분석하는 적응형 후속 실험이다. 성공해도
독립 검증, 외부 일반화, 반도체 현장 성능, 운영 자동 경보 승격 또는 과학적
champion을 주장하지 않는다. 모든 결과는 `review-only`다.

## 4. 동결 입력

| 입력 | 경로 | SHA256 |
|---|---|---|
| SQLite DB | `/Users/mac/Documents/Dataset/univariate_ts.db` | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |
| Exp249 prediction | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/frozen_predictions.jsonl` | `fc72d2dfbaf6a1653ec804dbe349397214992387083b22bdab0e8c26a66876c2` |
| Exp249 prediction manifest | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/prediction_manifest.json` | `25ade7c3f8362559a29c15e43fead6070a7fac6cab648f03f62bcb20382b8c8f` |
| Exp249 artifact manifest | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/MANIFEST.json` | `42fee6f9a26b2c6fd19747f83887a431bf5edad144e2af3464033256a0108da9` |
| Exp249 cache manifest | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/cache_manifest.csv` | `44c0ecc80ff7cbae383d49d2b8b0cefed442c49b8a12a7bd0cf8569ce121c971` |
| Exp249 audit | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/audit.json` | `144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee` |
| Exp291 routing | `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl` | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` |
| Exp291 manifest | `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json` | `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e` |
| Exp414 repair audit | `/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat/exp414_missing_repair_audit.csv` | `548f493a1608521bc81e4b742cc061fab7298f0d03c69f23845e37528b26ab91` |
| Exp414 manifest | `/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat/MANIFEST.json` | `99556922dc8f5178feaf03bc61eca4bfae628ae8f6b6092c2f6159da8c5dd3cc` |
| Exp404 Hard comparator | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` |
| Exp404 Hard manifest | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json` | `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8` |

모든 파일은 descriptor-bound read-only 방식으로 열고, parse에 사용한 동일 bytes의
SHA256을 확인한다. Exp249 cache manifest가 선언한 1,117개 NPZ의 SHA256도 원본
scalar score를 읽기 전에 확인한다. symlink, FIFO, inventory 변경, 중복 데이터셋,
범위 밖 좌표는 TEST label 접근 전에 실패한다.

## 5. 기존 Hard 재구성 증명

Exp249와 Exp291 동결 산출물만으로 다음 관계가 1,117개 모두 정확히 재생된다.

```text
Exp236 기존 Hard                  2,116
+ Exp249 MiniROCKET 승격           137
= exact-normal 적용 전 Hard       2,253
- Exp290/291 exact-normal           35
= 현재 automatic Hard             2,218
```

데이터셋 `d`의 동결 집합을 다음처럼 정의한다.

```text
promotion(d)
  = exp236_standard(d)
    AND (mini1024_seed_intersection_alpha005(d)
         OR mini4096_seed_intersection_alpha005(d))

pre_exact_hard(d) = exp236_hard(d) OR promotion(d)
automatic_hard(d) = pre_exact_hard(d) MINUS frozen_exact_normal(d)
```

Exp443은 이 식의 MiniROCKET 후보만 정책별로 다시 계산한다. Exp236 lane과
frozen exact-normal은 변경하지 않는다.

## 6. 세 결측 정책

세 정책은 정상 TRAIN에만 적용한다. TEST 전처리는 기존 선형 보간을 그대로 쓴다.
target length는 정책 적용 전 원본 정상 TRAIN 길이만으로 기존 방식과 동일하게
계산하여 정책 사이의 길이 차이를 막는다. 유한 정상 TRAIN 행은 세 정책에서
같은 값과 같은 순서를 유지한다.

Exp443 입력 adapter는 기존 loader가 결측을 먼저 지우기 전 단계의 원본 `<f4`
`values_blob`을 descriptor-bound read-only connection으로 읽는다. TRAIN에서는
정상 행을 고르기 위한 `split='TRAIN' AND label='0'`만 허용하고, TEST에서는
`values_blob`과 instance identity만 읽는다. TEST `label`, `labels_blob`, 평가
테이블은 prediction SQL에 존재할 수 없다. 원본 blob SHA256과 결측 mask를 먼저
고정한 뒤 아래 정책을 적용하고, 이후 alignment와 normalization은 기존 Exp249
순서를 그대로 사용한다.

### A. `linear_edge_nearest_control`

- 내부 `NaN`, `+Inf`, `-Inf`: 원래 시간좌표에서 선형 보간
- 시작·끝 결측: 가장 가까운 유한값으로 연장
- 기존 Exp249 전처리를 그대로 재현하는 대조군

### B. `exclude_nonfinite_row`

- 비유한값이 하나라도 있는 정상 TRAIN 행 전체를 모델 fit과 OOF calibration에서 제외
- 제외 행의 dataset, instance index, length, 결측 위치와 원본 blob SHA256 기록
- 현재 inventory에서는 모든 영향 데이터셋에 정상 TRAIN이 최소 15행 남음

### C. `leave_one_out_peer_median`

- 원래 길이가 같은 다른 정상 TRAIN 행들만 참조
- 각 결측 위치에서 유한한 동료 값의 leave-one-out 중앙값으로 복원
- 자기 행은 복원 reference에 포함하지 않음
- 결측 위치 하나라도 유한 동료가 0개면 해당 행 전체를 제외하고 fallback 보간하지 않음
- 현재 inventory에서는 162행을 복원할 수 있고 2행은 무지원으로 제외됨
- mask는 복원과 audit에만 사용하며 MiniROCKET 입력 feature로 전달하지 않음

전부 결측인 행은 현재 0개다. 실행 시 발견되면 세 정책 모두 해당 데이터셋을
`unexpected_all_missing_train`으로 실패 처리하며 0 대체를 하지 않는다.

## 7. 고정 MiniROCKET 계약

다음은 세 정책에서 동일하다.

- 공식 `aeon` MiniROCKET
- requested kernels `1,024`와 `4,096`
- actual dimensions `1,008`과 `4,032`
- maximum dilations per kernel `32`
- seeds `20260721`, `20261721`
- inner `n_jobs=1`
- five-fold TRAIN OOF transformer cross-fit
- fold-fit TRAIN reference 최대 `1,000`
- query chunk `128`
- fold-fit median centering and 10th-to-90th percentile robust scaling
- KNN-3 local-gap score
- `seed_intersection`
- alpha `0.005`
- per-instance z-normalization after alignment

threshold, seed, kernel 수, alpha, KNN, fold, target length, Standard lane, exact-normal
규칙을 TEST 결과로 변경하지 않는다.

## 8. 환경과 대조군 재현 gate

과거 Exp249 환경은 다음과 같다.

```text
aeon = 1.5.0
numpy = 2.3.5
scikit-learn = 1.8.0
```

현재 공용 환경은 `aeon`이 없고 NumPy/scikit-learn 버전이 다르므로 공식 실행은
별도의 고정 로컬 환경을 사용한다. 이 환경은 저장소 밖에 만들고 package lock,
Python·OS·CPU 정보를 readiness manifest에 기록한다.

정책 B와 C를 계산하기 전에 정책 A를 결측 영향 19개에서 실행한다. 다음을 모두
만족해야 한다.

1. 모든 OOF와 TEST score가 유한하다.
2. 원본 Exp249의 두 seed-intersection 후보 index가 두 kernel 설정과 alpha
   `0.005`에서 완전히 같다.
3. 대응하는 scalar score 최대 절대오차가 `1e-6` 이하이다.
4. A를 기존 1,098개 동결 결과와 결합하고 routing 식을 적용한 최종 Hard가
   Exp291/Exp404 2,218개 좌표와 완전히 같다.

하나라도 실패하면 `control_reproduction_failed`로 종료하고 B/C prediction과
TEST 평가는 만들지 않는다. 환경 또는 코드 변경은 같은 Exp443 공식 attempt를
덮어쓰지 않고 새 attempt에서 readiness부터 다시 시작한다.

## 9. Label-free prediction 흐름

공식 단위 작업은 `3 policies x 19 datasets = 57`개다.

1. 고정 입력과 19개 결측 inventory를 검증한다.
2. A 대조군 재현 gate를 통과한다.
3. A/B/C의 57개 단위를 결정론적 dataset 순서와 11개 CPU worker로 계산한다.
4. 각 정책에서 결측 없는 1,098개 Exp249 후보를 byte-identity 검증 후 가져온다.
5. 정책별 promotion과 final Hard를 1,117개 전체에 대해 계산한다.
6. 세 prediction, 정책 audit, manifest와 artifact SHA map을 원자적으로 동결한다.
7. `inspect`가 입력·checkpoint·prediction·manifest·SHA를 read-only로 재생한다.

prediction 경로는 TEST label, `labels_blob`, 실제 이상 위치, Exp416 metrics,
Exp442 evaluation, 과거 family별 TEST 성능을 import하거나 읽지 않는다. prediction
모듈에는 label-capable DB 함수와 evaluator import가 존재하지 않는다.

## 10. 사후 평가와 성공 기준

세 prediction SHA가 모두 동결되고 `inspect`가 통과한 뒤에만 별도 evaluator가
TEST instance label을 연다. 같은 label snapshot으로 정책 A/B/C와 Exp404/Exp416
기준선을 paired 평가한다.

각 정책에 대해 다음을 모두 기록한다.

- TP, FP, FN, TN, precision, recall, micro F1
- dataset-balanced macro precision, recall, F1과 false-positive rate
- 현재 기준 대비 추가·제거 좌표와 각각의 TP/FP
- 결측 영향 19개와 전체 1,117개 결과
- rescued, regressed, 기존 강한 데이터셋 악화
- dataset-paired bootstrap 20,000회 micro/macro F1 delta 95% 구간
- family concentration과 19개 밖 좌표 불변성

dataset별 precision, recall, F1과 false-positive rate는 기존 프로젝트 방식과
같이 분모가 0이면 `0.0`으로 둔다. dataset-balanced macro 값은 1,117개 모든
dataset의 이 값을 단순 평균한다. `rescued`는 기준선 TP가 0이었지만 해당 정책의
TP가 1 이상이 된 dataset, `regressed`는 dataset F1이 기준선보다 감소한
dataset으로 고정한다.

bootstrap은 `numpy 2.3.5`, `PCG64`, seed `44320260809`, `int64`, endpoint
`false`, quantile method `linear`로 동결한다.

정책 하나가 `retrospective_improvement_signal`이 되려면 다음을 모두 만족해야 한다.

1. 1,117개 전체 micro F1 delta `> 0`
2. 1,117개 dataset-balanced macro F1 delta `> 0`
3. micro와 macro paired-bootstrap 95% 구간 하한이 모두 `> 0`
4. 전체 FP가 기준선 `303`보다 증가하지 않음
5. 새로 복구한 데이터셋이 최소 1개
6. 기준선 F1 `>= 0.8`인 강한 데이터셋의 F1 감소가 0개
7. 결측 없는 1,098개의 최종 Hard 좌표가 기준선과 완전히 같음
8. prediction 이전 label 접근 0회, 오류와 비유한 출력 0개

여러 정책이 통과해도 TEST 결과로 하나를 같은 실험의 승자로 다시 선택하지 않는다.
통과 정책은 모두 review-only 신호로 보고한다. 후속 사용은 새 실험 번호와 새
사전등록에서 하나의 규칙을 다시 동결해야 한다.

## 11. CPU·MLX 판단

공식 aeon MiniROCKET CPU transformer를 MLX로 다시 구현하면 제3자 알고리즘과
수치 경로가 바뀐다. AGENTS의 MLX 채택 조건 1을 만족하지 않으므로 Exp443의
MiniROCKET 단계는 `inapplicable_official_cpu_transformer`로 기록하고 CPU를 쓴다.

- outer dataset workers: 논리 CPU 12개 중 11개
- inner numerical threads: 1개
- 동시에 실행하는 MLX/MPS-heavy process: 0개

JSON routing과 SHA 검증은 GPU 대상이 아니다. 구현 후 실제 19개 smoke timing으로
ETA를 계산하며 worker 1과 worker 11의 prediction payload가 완전히 같아야 한다.

## 12. Checkpoint와 진행 로그

실행 루트는 다음으로 고정한다.

```text
/Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809
```

프로세스가 직접 원자적 JSON과 append-only text/event log를 갱신한다. 로그에는
다음을 포함한다.

- PID와 active 상태
- 현재 policy와 dataset 이름
- 정상 TRAIN 전체·결측·사용·제외 행 수
- TEST 행 수와 target length
- 현재 단계와 backend
- 완료/전체 `57`
- import/recompute/success/abstention/error 수
- 처리속도, 경과시간, ETA
- checkpoint 경로와 마지막 오류

checkpoint는 `(policy, dataset)`의 동결 순서 prefix와 payload SHA를 저장한다.
재개는 비활성 PID, 고정 input/config/environment hash, 유효한 ordered prefix를
모두 확인한 뒤 미완료 단위만 계산한다. 완료된 record를 덮어쓰거나 오류를
성공으로 바꾸지 않는다. Codex의 반복 감시는 요구하지 않는다.

## 13. 실패 처리

다음은 publication 전 실패다.

- 입력 또는 DB SHA 불일치
- 1,117개 모집단, 19개 결측 inventory 또는 983개 값 불일치
- 대조군 후보·score·최종 Hard parity 실패
- package 버전 또는 device provenance 불일치
- 누락·중복·범위 밖 dataset/coordinate
- 비유한 score, prediction 또는 timing scalar
- TEST label firewall 위반
- 1-worker/11-worker payload 불일치
- 1,098개 비결측 좌표 변경
- checkpoint chain, PID 또는 create-only artifact 위반

실패 attempt는 증거로 보존한다. 같은 root를 삭제·수정해 재사용하지 않는다.

## 14. 테스트와 검증

구현은 다음 테스트를 먼저 만든다.

1. 내부·시작·끝 결측의 기존 선형 보간
2. `NaN`, `+Inf`, `-Inf` 행 전체 제외와 audit
3. leave-one-out 동료 중앙값, 자기참조 금지, 무지원 행 제외
4. 정책 사이 유한 행·target length·TEST preprocessing 동일성
5. Exp249 19개 대조군 parity와 실패 폐쇄
6. 2,116 + 137 - 35 = 2,218 routing 재생
7. label/evaluation import 및 SQL firewall
8. 1-worker와 11-worker 결정성
9. checkpoint 중단·재개·tamper·PID 검증
10. 원자적 progress와 create-only prediction publication
11. prediction freeze 전 evaluator 차단과 freeze 후 한정 label read
12. 혼동행렬, paired bootstrap, strong-regression과 성공표 재계산

전체 실행 전 단위 테스트, compile, 실제 결측 대표 데이터셋 smoke, 패키지/CPU
manifest, control reproduction을 통과해야 한다.

## 15. 예정 코드와 산출물

모든 새 파일 이름에는 `antigravity`를 포함한다.

```text
experiments/exp443_missing_normal_train_policy_comparison_antigravity/
scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py
docs/EXP443_PREREGISTRATION_antigravity_KO.md
docs/EXP443_RUNBOOK_antigravity_KO.md
docs/EXP443_IMPLEMENTATION_REPORT_antigravity.md
```

공식 root는 readiness, attempts/checkpoint/logs, 세 prediction publication,
inspection, 별도 evaluation과 artifact SHA map을 각각 create-only 경계로 둔다.

## 16. 다음 단계 경계

이 설계가 문서 검토를 통과하면 별도 구현 계획을 작성한다. 구현·테스트·독립
검토·고정 환경의 대조군 재현을 마친 뒤에만 공식 Exp443 실행을 시작한다.
Exp443 결과를 보기 전까지 Exp444 설정을 만들지 않는다.
