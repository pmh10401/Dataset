# 현재 프로젝트 연구 맥락 및 의사결정 기준

기준일: 2026-07-28

> 이 문서는 현재 프로젝트를 이해하기 위한 공식 입구다. 과거 실험 문서는
> 각 시점의 계획과 결과를 보존하는 역사적 근거이며, 이 문서는 이를 소급해
> 변경하지 않는다.

## 1. 프로젝트가 해결하려는 문제

이 프로젝트의 목표는 하나의 평균 F1 수치를 최대화하는 것이 아니다. 정상
TRAIN 데이터만으로 정상 동작의 기준과 후보 생성 규칙을 만들고, 정답을
모르는 TEST 입력에 그 규칙을 적용하면서 false alarm(오경보)을 억제하는
것이 핵심이다. 사용자가 반복되는 오경보 때문에 경보를 무시하기 시작하면
실제 이상을 찾아도 시스템의 목적을 달성했다고 보기 어렵기 때문이다.

예를 들어 새 방법이 실제 이상 한 건을 더 찾으면서 정상 샘플 열 건을
추가로 경보한다면 평균 점수 일부는 좋아질 수 있어도 현장 사용자의 신뢰는
낮아질 수 있다. 따라서 후보를 넓히는 것뿐 아니라 자동 경보, 사람 검토
후보, 연구용 Shadow 근거를 분리하고 각각의 false alarm 비용을 관리해야
한다.

이 목적은 현재 handoff에 “정상 TRAIN 데이터로만 후보를 만들고 자동 경보의
오탐을 억제하여 사용자 신뢰를 높인다”라고 명시되어 있다. 자세한 운영
원칙은 [Exp403-408 handoff](HANDOFF_20260728_EXP403_408_KO.md)를 따른다.

## 2. 데이터의 실제 성격과 영구 제약

현재 SQLite 데이터는 반도체 설비에서 반출한 원본 신호가 아니다. 원래
데이터베이스는 UCR Time Series Archive를 class별 one-vs-all 문제로 변환한
1,118개 데이터셋과 Hugging Face `CornellWhaleChallenge` 1개로 구성된
연구용 벤치마크였다. 현재 실험 모집단은 여기서
`CornellWhaleChallenge`와 `Wafer_normal_1`을 제외한 1,117개 데이터셋으로
동결되어 있다. 초기 구성과 제외 이력은
[과거 작업 감사 기록](archive/task_history_20260709_before_compaction.md)에,
현재 동결 모집단은
[Exp403-408 handoff](HANDOFF_20260728_EXP403_408_KO.md)에 기록되어 있다.

이 데이터는 알고리즘을 비교하고 실패 양상을 연구하는 데 사용할 수 있지만,
데이터 출처 자체가 반도체 설비 현장을 대표하지는 않는다. 반도체 설비는
이 연구의 최종 적용 목표다.

그러나 보안과 외부 반출 제한 때문에 신규 반도체 실데이터를 추가로 확보할
수 없으며, 앞으로도 새 실데이터가 들어오지 않는다는 조건을 연구 계획의
고정 전제로 둔다. 따라서 “새 데이터가 들어오면 확인한다”는 경로를 현재
실행 가능한 연구 계획으로 사용하지 않는다.

현재 데이터에서 얻은 결과는 retrospective benchmark, 즉 이미 동결된
평가셋을 사후 분석한 근거로만 해석한다. 이를 반도체 현장 성능, 실제 배포
검증 또는 새 데이터에 대한 prospective 성능이라고 부르지 않는다. 후속
연구는 반복 정상이라는 의도적 운전 조건은 보존하면서 family 과대대표,
설정 선택과 최종 평가의 혼입을 최대한 줄이는 방향으로 설계한다.

## 3. 데이터셋과 평가의 주요 위험

현재 데이터는 사용할 수 없는 데이터가 아니다. 다만 다음 위험 때문에
전체 평균 하나를 실제 일반화 성능으로 바로 해석해서는 안 된다.

| 위험 | 확인된 현상 | 해석 시 주의점 |
|---|---|---|
| 작은 TEST와 단일 이상 | 788개 데이터셋은 이상이 1개이고 TEST 크기가 50 이하 | 한 샘플의 성공·실패가 F1을 크게 바꿀 수 있음 |
| 작은 TRAIN | 478개 데이터셋은 TRAIN이 20 미만 | calibration과 conformal 해상도가 제한됨 |
| 의도적 repeated-normal 운전 조건 | 460개 데이터셋은 정상 TEST의 90% 이상이 TRAIN 정상과 정확히 겹침 | 안정된 설비·recipe 상태의 반복을 모사하므로 낮은 오경보는 긍정적이다. 다만 미관측 상태 일반화와는 분리해 보고함 |
| 전체 정상 반복 | 정상 TEST 221,284개 중 179,652개, 약 81.2%가 TRAIN 정상과 정확히 겹침 | 반복 정상 운영 성능의 주 근거이며, 독립·저반복 조건의 성능을 함께 뜻하지는 않음 |
| family 반복 | `ShapesAll` 60개, Pig 세 family 각각 52개, `FiftyWords` 50개처럼 동일 원자료 계열이 반복됨 | family 유사성 누수와 큰 family의 과대대표를 막아야 함 |
| label 의미 | `labels_blob`은 instance label을 전체 series 길이에 반복한 값 | 실제 이상 시점 위치로 해석하면 안 됨 |
| 가변 길이 | 153개 데이터셋에서 길이 관련 메타데이터와 실제 blob의 불일치가 보고됨 | 고정 길이 가정을 적용하기 전에 원자료 확인이 필요함 |
| 난이도 구분 | model-hard 218개, evaluation-hard 789개, 상대적으로 신뢰 가능한 집합 110개 | 모델 실패와 평가 불안정을 같은 문제로 취급하지 않음 |

작은 표본, 작은 TRAIN, 90% 이상 정상 반복, 가변 길이, 난이도 분류 수치는
[이상치 검증 난이도 보고서](anomaly_validation_difficulty_report_20260707.md)에
정리되어 있다. 전체 정상 중복률, family 반복, `labels_blob`의 의미는
[과거 데이터 구조 감사 기록](archive/task_history_20260709_before_compaction.md)에
근거한다.

정상 TRAIN/TEST의 정확 중복은 이 프로젝트에서 안정된
equipment/recipe/sensor/step 상태가 반복되는 의도적 repeated-normal 운전
조건이다. 반복 정상에 대한 낮은 오경보는 반도체 적용 목표와 맞는 긍정적
특성이다. 다만 UCR waveform 중복이 실제 동일 wafer run을 뜻하거나 미관측
설비의 prospective 성능을 증명하지는 않는다.

따라서 결론은 “중복을 제거하거나 데이터를 버린다”가 아니다. 반복 정상
성능, 덜 반복되는 정상에 대한 보조 스트레스 결과, family별 차이를 나누어
보고하고 주장의 범위를 현재 벤치마크 안으로 제한한다. 자세한 결정은
[반복 정상 해석 결정 기록](REPEATED_NORMAL_INTERPRETATION_KO.md)에 고정한다.

## 4. 연구가 현재 구조에 도달한 과정

1. **초기 VAE 중심 탐색:** 초기
   [README](../README.md)는 947개 데이터셋에서 VAE, 증강, threshold,
   hybrid routing을 비교한 당시 연구 단계의 기록이다. 현재 1,117개 동결
   모집단의 공식 종합 결과가 아니라 역사적 출발점으로 읽는다.
2. **데이터와 평가 구조 감사:** 이후 작은 TEST, 단일 이상, 의도적인 정상
   반복, one-vs-all family 반복이 확인되면서 단일 평균만으로 서로 다른
   운전 조건을 설명하지 않게 되었다. 평가 불안정과 실제 모델 실패도
   분리되었다.
3. **넓은 교체와 합집합의 한계:** global replacement와 candidate union은
   놓친 이상을 일부 복구했지만 false positive가 더 빠르게 늘어나는 경우가
   반복되었다. 정답 후보 수만 늘리는 방식은 사용자 신뢰라는 목표와 맞지
   않았다.
4. **family별 신호와 누수 경계:** family별 표현과 score source에서 유용한
   신호가 발견되었지만, 과거 family별 TEST 성능을 보고 routing하는 것은
   prediction에 정답 정보를 간접 사용하므로 금지되었다.
5. **exact-normal 자동 정상 guard:** Exp290은 기존 Hard 후보 중 같은
   데이터셋의 정상 TRAIN과 byte-for-byte 같은 35개를 `automatic normal
   exact`로 분리했다. 사후 평가에서 35개 모두 FP였고 TP 손실은 0이었다.
   Exp291은 이 규칙을 최종 routing에 고정했다.
6. **보수적인 lane 분리:** 자동 경보인 Hard를 동결하고 Standard와
   Priority를 사람 검토 영역으로 분리했다. 새 근거는 먼저 Shadow 또는
   review-only로 검토하며 자동 경보에 바로 합치지 않는다.
7. **Exp403-410:** Exp403-408은 feature 안정성, confidence, demotion,
   self-supervised·synthetic 근거와 독립 consensus를 검증했다. Exp409는
   이 근거를 새 데이터에 적용할 준비가 되었는지 label isolation, lane
   불변성, budget, provenance 관점에서 확인했다. Exp410은 동결 Hard가
   family 크기 차이와 반복 정상 구간에서 어떻게 보이는지 읽기 전용으로
   감사했다.

이 흐름은 초기 결과가 잘못되었다는 뜻이 아니다. 초기 결과는 당시 모집단과
평가 계약에서 얻은 역사적 결과다. 이후 감사와 실험이 데이터의 성격을 더
분명히 밝혔기 때문에 현재는 같은 수치를 더 좁고 엄격한 범위에서 해석한다.

## 5. 현재 동결 기준선

현재 운영·연구 비교의 기준점은 다음 세 lane이다.

| Lane | 개수 | 현재 의미 | 자동 승격 여부 |
|---|---:|---|---|
| Hard | 2,218 | 현재 동결된 자동 경보 기준선 | 기준선 자체 |
| Standard | 806 | 사람이 검토하는 보조 후보 | 자동 승격 금지 |
| Priority | 31 | 더 우선적으로 사람이 검토하는 후보 | 자동 승격 금지 |

별도의 `automatic normal exact` 35개는 Exp290/291에서 정상 TRAIN과
byte-for-byte 같은 입력으로 판정되어 자동 이상 경보에서 제외된 정상
lane이다. 이는 현재 Hard 2,218개와 서로소이며 Standard·Priority review로
이동하지 않는다.

동결 Hard의 retrospective benchmark 결과는 다음과 같다.

> TP 1,915 / FP 303 / precision 86.339% / mean dataset F1 0.608983

Hard TP/FP는
[Exp403-408 독립 재집계 결과](/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md)에서,
precision과 mean dataset F1은
[Exp381-385 결과](EXP381_385_RESULTS_KO.md)에서 교차 확인할 수 있다.

이 수치는 현재 1,117개 동결 벤치마크에서 얻은 autonomous Hard의
retrospective 결과다. 반도체 현장 또는 prospective 성능이 아니며,
Standard·Priority의 사람이 확인한 결과를 합친 human-assisted 성능도
아니다.

## 6. Exp403-410이 제공한 증거

Exp403-408은 하나의 새 모델로 기존 운영 정책을 교체하는 실험이 아니다.
기존 feature의 안정성, 자동 Hard의 confidence와 하향 가능성, 새로운
Shadow score, 독립 근거의 합의를 순서대로 분리해 검증한 묶음이다.

| 실험 | 질문 | 현재 결론 |
|---|---|---|
| Exp403 | 9개 FeatureBlock이 정상 TRAIN에서 안정적인가? | feature별 결측·상수·중복·fold 안정성을 감사해 이후 근거의 품질을 확인하되, 그 자체를 새 routing 정책으로 사용하지 않음 |
| Exp404 | 기존 automatic Hard에 TRAIN-only confidence를 붙일 수 있는가? | 정상성 confidence 위원회의 등급을 보조 진단으로 사용하며 기존 Hard를 자동 변경하지 않음 |
| Exp405 | 낮은 confidence Hard를 안전하게 하향할 수 있는가? | 재확인 473개 중 실제 TP 395, FP 78이어서 TP 손실이 지나치게 큼; 자동 demotion gate 실패 |
| Exp406 | tri-view self-supervised 표현이 놓친 이상 후보를 찾는가? | 시간·주파수·잔차 표현의 새 근거지만 human-assisted review-only Shadow |
| Exp407 | 정상 TRAIN으로 만든 합성 이상이 놓친 이상 후보를 찾는가? | 합성 교란 기반의 보조 근거지만 human-assisted review-only Shadow |
| Exp408 | 서로 독립적인 근거의 합의가 후보 품질을 보장하는가? | 가장 엄격한 변형도 62개 중 TP 29, FP 33으로 자동 경보 승격 근거가 부족함 |
| Exp409 | 고정 계약을 label-blind하게 재실행할 준비가 되었는가? | representative smoke에서 label isolation, lane 불변성, budget, provenance를 확인했지만 성능은 평가하지 않음 |
| Exp410 | 큰 family가 전체 평균을 지배하는가? 반복 정상 구간에서 Hard는 어떻게 보이는가? | family별 성능 차이가 크다는 감사 결과와 함께, 높은 반복 정상 구간에서 상대적으로 강한 성능을 확인함. 운영 정책은 변경하지 않음 |

Exp405의 결과는 “재확인”을 곧 “오경보”로 간주할 수 없음을 보여 준다.
473개 중 395개가 실제 TP이므로 이를 자동으로 내리면 false alarm 78개를
줄이는 대신 이미 맞힌 이상을 대량으로 잃는다. Exp405는 실제 lane 변경이
아닌 counterfactual 진단으로 남는다.

Exp408의 가장 엄격한 변형은
`two_new_plus_existing_independent`, alpha `0.005`다. 사후 평가에서 후보
62개 중 TP 29, FP 33, retrospective precision 46.77%였다. 이 평가는
동결 prediction이 끝난 뒤 연구 분석을 위해 정답과 비교한 것이며 TEST
정답이 prediction 입력으로 들어간 것은 아니다. 동시에 이 결과를 보고
variant나 alpha를 다시 고르는 것도 허용되지 않는다. 해당 후보는 계속
review-only다.

Exp409 smoke는 8개 대표 데이터셋에서 같은 고정 variant와 alpha를
실행했고 후보는 0개였다. 보고서의 성능 상태는 `not_evaluated`,
`prospective_validated: false`, `deployment_eligible: false`,
`operational_policy_changed: false`다. 즉 Exp409가 입증한 것은 실행
준비성과 실패 폐쇄(fail-closed) 거버넌스이지 이상 탐지 성능이 아니다.

Exp410의 `severe_family_imbalance` 판정은 “정상 반복이 나쁘다”는 뜻이
아니다. 이는 dataset마다 한 표를 줄 때와 family마다 한 표를 줄 때 결과가
크게 달라지고 family별 성능 편차가 크다는 뜻이다. 별도로 높은 정상 반복
구간의 더 좋은 결과는 안정된 recipe 운전에서 false alarm을 억제하려는
방향과 맞는 긍정적 관찰이다. 다만 family 구성과 TRAIN 크기 등이 함께
다르므로 overlap 하나의 인과 효과라고 부르지 않는다.

이 해석의 직접 근거는
[Exp403-408 설계](superpowers/specs/2026-07-27-exp403-408-feature-ml-validation-design.md),
[Exp408 구현 보고](../.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-10-report.md),
[독립 재집계 보고](/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md),
[Exp409 설계](superpowers/specs/2026-07-28-exp409-prospective-readiness-design.md),
[Exp409 smoke 보고](/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/EXP409_READINESS_REPORT_KO.md)다.

## 7. 반복 실험에서 확정된 교훈

### 후보를 넓히는 것 자체는 개선이 아니다

global replacement와 여러 source의 단순 union은 놓친 이상을 추가로 찾을
수 있지만 정상 후보도 훨씬 빠르게 늘리는 경우가 반복되었다. 따라서 새
후보의 개수나 TP만 보지 않고 FP, review workload, 기존 lane과의 중복을
함께 평가해야 한다.

### model-hard에는 새로운 score 표현이 필요하다

model-hard 데이터셋은 현재 score 공간에서 이상 샘플이 정상보다 위로
올라오지 않는 문제다. 순서가 잘못된 상태에서 threshold만 이동하면 이상
대신 정상 경보가 늘어난다. 이런 집합에는 시간·주파수·형상 등 다른
representation이 필요하며, 그 표현도 TEST 성능으로 선택해서는 안 된다.

### 작은 TRAIN은 calibration의 해상도를 제한한다

TRAIN 샘플이 매우 적으면 empirical 또는 conformal p-value가 취할 수 있는
값 자체가 거칠다. 예를 들어 정상 TRAIN이 몇 개뿐이면 작은 alpha를 정밀하게
구분할 근거가 없다. 낮은 p-value가 계산됐다는 사실만으로 높은 confidence를
주지 않고 작은 TRAIN을 별도 불확실성으로 표시해야 한다.

### review도 무제한 자원이 아니다

review-only는 자동 경보보다 안전하지만 비용이 0이라는 뜻은 아니다.
사람에게 수천 개 후보를 넘기면 중요한 후보를 놓치거나 검토가 중단될 수
있다. review budget은 성능 점수를 보고 상위 후보를 고르는 장치가 아니라,
예산을 넘으면 공개를 중단하는 운영 gate로 사용한다.

### family별 과거 TEST 성능은 routing 입력이 아니다

특정 family에서 어떤 source가 과거 TEST에서 좋았다는 사실을 이용해 새
prediction source를 고르면 TEST 정답이 간접적으로 설정에 반영된다.
family는 grouped evaluation과 error analysis에는 사용할 수 있지만 과거
family별 TEST 성능 기반 자동 routing에는 사용할 수 없다.

### tail replacement는 금지한다

과거 tail replacement는 현재 합성 평가셋에서 이상이 뒤쪽에 놓이는 배열
특성을 이용한다. 이는 실제 설비에서도 유지된다고 보장할 수 없는 TEST
layout 정보이므로 배포 가능한 탐지 규칙이 아니라 leakage에 가깝다.
[프로젝트 작업 기록](task.md)도 Exp91/92 tail replacement를 운영 사용에
부적합한 것으로 명시한다.

### 파일 존재와 재현 가능한 checkpoint는 다르다

checkpoint 파일이 남아 있어도 입력 artifact, 코드, 설정의 hash 계보를
다시 확인할 수 없다면 안전하게 이어 실행할 수 없다.
[Exp403-408 checkpoint provenance 감사](EXP403_408_CHECKPOINT_PROVENANCE_AUDIT_KO.md)는
기존 data-stage checkpoint의 입력 hash를 현재 파일로 재현하지 못했으므로
직접 resume하지 말고 필요한 stage를 새 output에서 재실행하도록 결론냈다.

## 8. 성능 주장과 해석의 경계

서로 다른 종류의 성능을 한 숫자로 합치지 않는다.

| 구분 | 의미 | 현재 허용되는 표현 |
|---|---|---|
| Autonomous | 정답을 본 사람의 개입 없이 생성·실행되는 경보 | 동결 Hard의 retrospective 자동 결과 |
| Retrospective | 현재 동결 TEST의 정답으로 사후 평가 | 벤치마크 안의 비교와 실패 진단 |
| Human-assisted | review 후보를 사람이 완벽히 판별한다고 가정한 상한·보조 지표 | `mean_combined_f1`처럼 사람 개입을 명시한 결과 |

이 구분에 따라 다음 경계를 지킨다.

1. `mean_combined_f1`은 autonomous 성능이 아니다.
2. Standard, Priority와 Shadow 후보는 자동 경보로 승격하지 않는다.
3. 현재 벤치마크 결과는 반도체 설비 현장 성능이 아니다.
4. Exp409 smoke는 prospective 성능 검증이 아니다.
5. retrospective TEST-label 분석은 동결 결과를 설명할 수는 있지만 다음
   prediction 설정을 선택하는 입력이 될 수 없다.

## 9. 앞으로의 연구 방향

신규 실데이터가 영구적으로 없는 상황에서는 현재 데이터 안에서 학습·선택과
평가의 독립성을 더 엄격하게 만드는 것이 현실적인 방향이다.

| 방법 | 쉬운 예 | 줄이려는 위험 |
|---|---|---|
| family-grouped validation | `ShapesAll` 계열을 쪼개 학습·평가 양쪽에 넣지 않고 family 전체를 한쪽 fold에 둠 | family 유사성 누수와 큰 family의 과대대표 |
| nested validation | 안쪽 fold에서 alpha·threshold를 고르고 바깥 fold는 최종 평가에만 사용 | TEST를 본 뒤 설정을 재선택하는 과적합 |
| repeatability-aware reporting | exact·near·덜 익숙한 정상 조건을 나누고 원본 repeated-normal 결과와 보조 스트레스 결과를 함께 제시 | 안정 운전의 장점과 덜 반복되는 조건의 한계를 혼동하는 해석 |
| uncertainty-aware reporting | 이상 1개인 데이터셋과 충분한 이상이 있는 데이터셋을 분리하고 신뢰구간·민감도를 제시 | 작은 분모에 의한 F1 급변 |

예를 들어 ShapesAll의 여러 one-vs-all 데이터셋을 무작위로 나누면 매우
비슷한 원자료 계열이 개발 fold와 평가 fold 양쪽에 들어갈 수 있다.
family-grouped validation은 ShapesAll 전체를 같은 fold에 두어 “본 적 없는
family에 얼마나 버티는가”를 더 엄격하게 본다.

nested validation에서는 안쪽 fold만 사용해 여러 alpha 중 하나를 고른다.
선택이 끝난 뒤 바깥 fold를 딱 한 번 평가한다. 바깥 결과가 마음에 들지
않는다고 alpha를 다시 고르면 바깥 fold도 사실상 TRAIN이 되므로 새 실험과
사전등록 없이 변경하지 않는다.

repeatability-aware reporting은 정상 반복을 제거하자는 방법이 아니다.
원본 repeated-normal을 정본 운영 시나리오로 유지하고, `clean-balanced`는
반복성이 약해졌을 때의 보조 스트레스·하한 검증으로만 사용한다.

이 네 방법은 신규 데이터에 대한 진정한 prospective validation을
대체하지 않는다. 추가 데이터를 얻을 수 없는 조건에서 family 반복,
운전 반복성 차이, 설정 선택 과적합, 작은 표본의 불확실성을 숨기지 않는
엄격한 retrospective 연구 방식이다.

## 10. 후속 작업의 불변 원칙

- prediction 단계에서 TEST label과 실제 이상 위치를 사용하지 않는다.
- 과거 family별 TEST 성능으로 source나 routing을 선택하지 않는다.
- feature, score, calibration, alpha와 threshold는 정상 TRAIN 근거로
  결정한다.
- `mean_combined_f1`은 human-assisted 지표로만 취급한다.
- Standard, Priority review와 Shadow 후보를 automatic Hard로 승격하지
  않는다.
- tail replacement를 사용하지 않는다.
- TEST 결과를 본 뒤 alpha, threshold, source, review budget을 재선택하지
  않는다.
- 새 개선은 새 실험 번호와 사전등록으로 분리한다.
- 기존 Hard, Standard, Priority lane과 동결 artifact를 새 실험이 조용히
  덮어쓰지 못하게 한다.

## 11. 근거 문서 지도

처음 프로젝트를 이어받는 사람은 모든 문서를 시간순으로 읽기보다 목적에
맞는 최소 묶음부터 읽는 편이 안전하다.

### 복원과 현재 상태

- [Exp403-408 handoff](HANDOFF_20260728_EXP403_408_KO.md)
- [다른 Mac 재개 프롬프트](RESUME_PROMPT_OTHER_PC_KO.md)
- [Exp403-408 checkpoint provenance 감사](EXP403_408_CHECKPOINT_PROVENANCE_AUDIT_KO.md)

### 데이터 품질과 평가 난이도

- [반복 정상 해석 결정 기록](REPEATED_NORMAL_INTERPRETATION_KO.md)
- [이상치 검증 난이도 보고서](anomaly_validation_difficulty_report_20260707.md)
- [실험 일관성 감사](experiment_consistency_audit_20260707.md)
- [과거 데이터 구조 감사 기록](archive/task_history_20260709_before_compaction.md)

### 역사적 연구 결과와 해석 변화

- [초기 README](../README.md)
- [실험 39-50 결과 분석](experiment_39_50_results_analysis_20260707.md)
- [model-hard 대응 연구](model_hard_response_research_20260707.md)
- [Exp381-385 결과](EXP381_385_RESULTS_KO.md)

### Exp403-408 설계·실행·결과

- [설계 명세](superpowers/specs/2026-07-27-exp403-408-feature-ml-validation-design.md)
- [실행 runbook](EXP403_408_RUNBOOK_KO.md)
- [Exp407 구현 보고](../.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-9-report.md)
- [Exp408 구현 보고](../.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-10-report.md)
- [독립 재집계 구현 보고](../.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-12-report.md)
- [최종 검증 보고](../.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-13-report.md)

현재 Mac의 전체 결과 보고서는
`/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md`에
있다. 이 절대 경로는 다른 컴퓨터에서는 달라질 수 있다.

### Exp409 설계·실행·준비성

- [사전등록](EXP409_PREREGISTRATION_KO.md)
- [실행 runbook](EXP409_RUNBOOK_KO.md)
- [설계 명세](superpowers/specs/2026-07-28-exp409-prospective-readiness-design.md)

현재 Mac의 representative smoke 보고서는
`/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/EXP409_READINESS_REPORT_KO.md`에
있다. 이 보고서는 성능표가 아니라 준비성 검증 기록이다.

### Exp410 family 견고성 감사

- [사전등록](EXP410_PREREGISTRATION_KO.md)
- [실행 runbook](EXP410_RUNBOOK_KO.md)
- [설계 명세](superpowers/specs/2026-07-28-exp410-frozen-hard-family-robustness-audit-design.md)

현재 Mac의 상세 결과와 파생 metric은
`/Users/mac/Documents/Dataset/exp410_family_robustness/`에 로컬 전용으로
보존한다. 공개 GitHub 문서에는 상세 파생 수치를 게시하지 않는다.

### 새 실험 작성 규칙

- 먼저 현재 문서의 [후속 작업의 불변 원칙](#10-후속-작업의-불변-원칙)을
  확인한다.
- 새 가설은 기존 동결 결과를 수정하지 않고 새 실험 번호로 분리한다.
- prediction 입력, 고정 parameter grid, gate, claim boundary를 실행 전에
  사전등록한다.
- 구현 명령과 실패 폐쇄 조건은 별도의 runbook으로 남긴다.

## 12. 알려진 문서 결손과 충돌 처리

`docs/EXP403_408_PREREGISTRATION_KO.md`는 handoff에서 먼저 읽을 문서로
참조되지만 현재 working tree와 확인 가능한 전체 Git 이력에는 없다. 이
문서를 임의로 복원하거나 존재하는 근거처럼 인용하지 않는다. Exp403-408에
대해서는 현재 남아 있는 설계 명세, runbook, 구현 보고, 동결 결과로 확인
가능한 범위만 사용한다.

초기 [README](../README.md)의 947개 데이터셋은 초기 VAE 연구 단계의
역사적 모집단이다. 현재 비교 기준은 1,117개 동결 모집단이다. 두 숫자를
하나가 틀린 것으로 합치지 않고 각각 당시 범위와 현재 범위로 구분한다.

Exp409 설계의 “genuinely new data가 도착할 때”라는 전제도 당시 준비성
설계의 역사적 가정으로 보존한다. 그러나 신규 반도체 실데이터가 영구적으로
없다는 현재의 확정 조건 때문에 이는 실행 가능한 다음 연구 계획이 아니다.
Exp409 산출물은 governance smoke로는 재사용할 수 있지만 prospective 성능
증거로는 사용할 수 없다.

문서나 수치가 충돌할 때는 다음 순서로 확인한다.

1. 동결된 결과 artifact와 독립 감사 보고서
2. 해당 실험의 사전등록 문서와 실행 runbook
3. handoff와 재개 문서
4. 과거 시점의 종합 보고서
5. 초기 README

다만 사전등록은 “무엇을 하기로 했는가”, 결과 보고서는 “실제로 무엇이
나왔는가”에 대한 근거다. 결과로 과거 계획을 바꾸거나, 계획으로 실제 결과를
덮지 않는다. 확인할 원문이 없으면 기억으로 보완하지 않고 “미확인” 또는
“문서 결손”으로 표시한다.
