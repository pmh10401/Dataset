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
연구는 현재 데이터 안에서 family 중복, TRAIN/TEST 중복, 설정 선택과 최종
평가의 혼입을 최대한 줄이는 방향으로 설계한다.

## 3. 데이터셋과 평가의 주요 위험

현재 데이터는 사용할 수 없는 데이터가 아니다. 다만 다음 위험 때문에
전체 평균 하나를 실제 일반화 성능으로 바로 해석해서는 안 된다.

| 위험 | 확인된 현상 | 해석 시 주의점 |
|---|---|---|
| 작은 TEST와 단일 이상 | 788개 데이터셋은 이상이 1개이고 TEST 크기가 50 이하 | 한 샘플의 성공·실패가 F1을 크게 바꿀 수 있음 |
| 작은 TRAIN | 478개 데이터셋은 TRAIN이 20 미만 | calibration과 conformal 해상도가 제한됨 |
| 정상 TRAIN/TEST 중복 | 460개 데이터셋은 정상 TEST의 90% 이상이 TRAIN 정상과 정확히 겹침 | 일반화 성능이 실제보다 좋아 보일 수 있음 |
| 전체 정상 중복 | 정상 TEST 221,284개 중 179,652개, 약 81.2%가 TRAIN 정상과 정확히 겹침 | 무작위 dataset 평균만으로 독립성을 주장할 수 없음 |
| family 반복 | `ShapesAll` 60개, Pig 세 family 각각 52개, `FiftyWords` 50개처럼 동일 원자료 계열이 반복됨 | family 유사성 누수와 큰 family의 과대대표를 막아야 함 |
| label 의미 | `labels_blob`은 instance label을 전체 series 길이에 반복한 값 | 실제 이상 시점 위치로 해석하면 안 됨 |
| 가변 길이 | 153개 데이터셋에서 길이 관련 메타데이터와 실제 blob의 불일치가 보고됨 | 고정 길이 가정을 적용하기 전에 원자료 확인이 필요함 |
| 난이도 구분 | model-hard 218개, evaluation-hard 789개, 상대적으로 신뢰 가능한 집합 110개 | 모델 실패와 평가 불안정을 같은 문제로 취급하지 않음 |

작은 표본, 작은 TRAIN, 90% 이상 정상 중복, 가변 길이, 난이도 분류 수치는
[이상치 검증 난이도 보고서](anomaly_validation_difficulty_report_20260707.md)에
정리되어 있다. 전체 정상 중복률, family 반복, `labels_blob`의 의미는
[과거 데이터 구조 감사 기록](archive/task_history_20260709_before_compaction.md)에
근거한다.

이 위험의 결론은 “데이터를 버린다”가 아니다. 어떤 모델 문제인지, 어떤
평가 문제인지 구분하고, family와 중복 정도에 따라 결과를 나누어 보고하며,
주장의 범위를 현재 벤치마크 안으로 제한해야 한다는 뜻이다.

## 4. 연구가 현재 구조에 도달한 과정

1. **초기 VAE 중심 탐색:** 초기
   [README](../README.md)는 947개 데이터셋에서 VAE, 증강, threshold,
   hybrid routing을 비교한 당시 연구 단계의 기록이다. 현재 1,117개 동결
   모집단의 공식 종합 결과가 아니라 역사적 출발점으로 읽는다.
2. **데이터와 평가 구조 감사:** 이후 작은 TEST, 단일 이상, 정상
   TRAIN/TEST 중복, one-vs-all family 반복이 확인되면서 단일 평균 성능의
   해석 범위가 축소되었다. 평가 불안정과 실제 모델 실패도 분리되었다.
3. **넓은 교체와 합집합의 한계:** global replacement와 candidate union은
   놓친 이상을 일부 복구했지만 false positive가 더 빠르게 늘어나는 경우가
   반복되었다. 정답 후보 수만 늘리는 방식은 사용자 신뢰라는 목표와 맞지
   않았다.
4. **family별 신호와 누수 경계:** family별 표현과 score source에서 유용한
   신호가 발견되었지만, 과거 family별 TEST 성능을 보고 routing하는 것은
   prediction에 정답 정보를 간접 사용하므로 금지되었다.
5. **보수적인 lane 분리:** 자동 경보인 Hard를 동결하고 Standard와
   Priority를 사람 검토 영역으로 분리했다. 새 근거는 먼저 Shadow 또는
   review-only로 검토하며 자동 경보에 바로 합치지 않는다.
6. **Exp403-409:** Exp403-408은 feature 안정성, confidence, demotion,
   self-supervised·synthetic 근거와 독립 consensus를 검증했다. Exp409는
   이 근거를 새 데이터에 적용할 준비가 되었는지 label isolation, lane
   불변성, budget, provenance 관점에서 확인했다.

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
