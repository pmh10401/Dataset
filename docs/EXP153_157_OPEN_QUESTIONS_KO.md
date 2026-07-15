# Exp153–157 실행 전 확인이 필요한 질문

아래 질문은 코드 오류가 아니라 **정책 해석과 연구 범위 결정**에 필요한 사항입니다. 답이 없어도 모든 variant를 sensitivity로 계산할 수 있지만, primary 정책을 선언해서는 안 됩니다.

## Q1. Cross-dataset pooling을 연구 후보로 허용하는가?

Exp154는 서로 다른 가상 설비 dataset의 TRAIN 정상 score를 robust-z로 정규화한 뒤 leave-one-dataset-out 방식으로 pool합니다.

- 허용: Exp154 global/group/hybrid를 모두 실행
- 불허: Exp154는 진단용으로만 실행하거나 제외하고 Exp153/155/156/157만 진행

## Q2. 정상 false-alert target의 primary 값이 있는가?

코드는 다음을 모두 sensitivity로 실행합니다.

```text
0.5 / 100 normal runs
1.0 / 100 normal runs
2.0 / 100 normal runs
```

사용자 신뢰나 운영 정책으로 사전에 정한 primary target이 있다면 알려주세요. 현재 TEST 결과를 보고 선택해서는 안 됩니다.

## Q3. One-sided 95% Wilson upper bound가 적절한가?

Exp157은 작은 pseudo-normal 표본에서 단순 observed rate보다 보수적인 Wilson upper bound를 사용합니다. 다른 confidence 수준을 원한다면 TEST 평가 전에 정해야 합니다.

## Q4. TRAIN-only group 정의가 적절한가?

기본 그룹:

- TRAIN count bin: 5–19, 20–49, 50–99, 100+
- TRAIN target length quartile

Family 이름은 사용하지 않습니다. 다른 TRAIN-only grouping 근거가 있다면 실행 전에 알려주세요.

## Q5. Pool 최소 reference 200을 유지할 것인가?

200은 alpha=0.005를 경험적으로 해상할 수 있는 최소 규모를 고려한 고정값입니다. 변경하려면 TEST 결과 전에 정해야 합니다.

## Q6. `n_train < 5`는 계속 완전 abstention할 것인가?

현재 코드는 source score를 안정적으로 cross-fit할 수 없다고 보고 Hard/Review 후보를 모두 생성하지 않습니다. 극소표본 dataset까지 pooled fallback을 허용하려면 별도 신규 실험으로 분리하는 것이 안전합니다.

## Q7. Exp156 confirmation sensitivity grid를 승인하는가?

```text
legacy_countcap
conformal_strict: B=0.01, C=0.02
conformal_balanced: B=0.02, C=0.05
```

이 값은 모두 사전등록 sensitivity이며 TEST로 하나를 고르지 않습니다.

## Q8. 계산 자원과 예상 실행 시간을 허용하는가?

Shared cache는 네 source OOF/TEST score와 Block-B/C OOF/TEST score를 1,117개 dataset에서 계산합니다. Exp152 1회보다 계산량이 큽니다. 충분한 디스크와 장시간 실행이 가능한지 확인이 필요합니다.

## Q9. D2 configuration provenance를 병행할 것인가?

Exp153–157은 현재 고정 source configuration을 사용합니다. 해당 configuration이 과거 TEST 성능으로 선택되었는지는 별도 감사가 필요합니다.

## Q10. 신규 잠금 benchmark를 언제 확보할 수 있는가?

현재 TEST는 반복 관찰된 retrospective benchmark입니다. 개선 정책의 최종 선택은 신규 잠금 데이터 또는 이후 추가한 공개 데이터셋에서 1회 평가해야 합니다.
