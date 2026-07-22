# Exp265 신규 run Shadow 검증 사전등록

## 1. 목적과 상태

- 실험 ID: `experiment_265_shadow_validation_ledger`
- 정책 버전: `exp265-shadow-v1`
- 작성일: 2026-07-21
- 실제 수집 활성화 시각: **미설정**
- 검증 성격: 신규 데이터에 대한 prospective shadow validation
- 목표: Exp253 자동 경보를 바꾸지 않고 Exp264 다중 점수원 근거가 검토 품질과 우선순위에 도움이 되는지 확인한다.

코드 완성과 실제 수집 활성화는 다르다. 활성화 시각 이후 처음 관찰된 신규 run만 정확히 100-run 분모에 포함한다. 과거 1,117개 TEST 데이터셋, 재생 데이터, smoke 데이터는 포함하지 않는다.

## 2. 동결된 정책

| 항목 | 동결값 |
|---|---|
| `SCHEMA_VERSION` | `1` |
| `COLLECTION_MODE` | `prospective_shadow` |
| `ALPHA` | `0.005` |
| Shadow 최소 TRAIN 수 | `100` |
| Shadow 동의 점수원 수 | `3/4 이상` |
| 우선 검토 동의 점수원 수 | `2/4 이상` |
| 목표 신규 run | `100` |
| 점수원 | 통계, 구조, 변화점, 마스킹 복원 |

Exp253 Hard index는 그대로 유지한다. 기존 Hard 후보에만 A/B/C 신뢰도를 붙인다. 기존 Standard 후보 중 TRAIN 수 100 이상, 품질 경고 없음, 3개 이상 점수원 동의인 후보만 Shadow로 보낸다. 2개 동의는 우선 검토로 보낸다. Shadow, 우선 검토, 일반 검토는 자동 경보가 아니다.

## 3. 입력 금지 정보

예측·후보 선택·lane 분류에는 다음을 사용하지 않는다.

- TEST 라벨, 실제 이상 위치
- TP, FP, FN, Precision, Recall, F1
- 과거 family별 TEST 성능
- TEST 길이 기반 top-N budget
- tail replacement

`run_id`, 설비·레시피·센서·step 식별자는 조회와 기록에만 사용하며 정책 분기에 사용하지 않는다.

## 4. 평가 계약

- 후보 Precision: 사용자가 `실제 이상` 또는 `오탐`으로 판정한 후보만 사용한다. `보류`는 분모에서 제외한다.
- lane별 지표: Hard, Shadow, 우선 검토, 일반 검토를 분리한다.
- Recall/F1: 한 run의 모든 후보가 확정 판정을 받고, 사용자가 놓친 이상 개수를 정확히 입력한 run에 한해서만 사람 보조 지표로 계산한다.
- 자동 성능: Hard lane만 autonomous 범위다.
- 100 run 도달: 보고서를 생성할 수 있지만 정책을 자동 승격하거나 변경하지 않는다.

## 5. 사전 고정된 주장 경계

검증 가능:

- Exp253 Hard 후보가 Exp265에서 변하지 않았는지
- 신규 run에서 lane별 검토량과 사용자 판정 Precision
- 완전 주석 run에서만 계산한 사람 보조 Recall/F1

이 실험만으로 주장할 수 없음:

- 전체 파이프라인의 end-to-end strict TRAIN-only provenance 완료
- 실제 설비 배포 적합성 또는 생산 성능 보장
- Shadow 후보의 자동 경보 승격
- 특정 점수원이나 임계값이 최적이라는 결론

## 6. 중단 조건

- 신규 label-free upstream payload를 안정적으로 만들 수 없음
- 예측 payload에 TEST 라벨·평가 지표가 포함됨
- Exp253 Hard index가 변경됨
- 동일 `run_id`에 서로 다른 prediction hash가 발생함
- 실제 run과 재생/smoke run을 구분할 수 없음
- 보안 정책 또는 append-only 기록이 검증되지 않음

결과가 기대보다 나빠도 상수나 정책을 변경하지 않는다. 변경은 새 실험 번호와 새 사전등록으로만 수행한다.
