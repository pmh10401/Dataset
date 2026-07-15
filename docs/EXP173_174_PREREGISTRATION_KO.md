# Exp173/174 균형형 TRAIN-only selector 사전등록

## 목적

Exp171은 TRAIN 정상 OOB 검토 위험의 동시 신뢰상한을 1% 이하로 제한했지만,
위험 예산 안에서 정상 후보 수를 가장 많이 발생시키는 설정을 선택했다. Exp172의
사후 결과는 검토 후보 40개, TP 30개, FP 10개, precision 75%였으며 316개 중
284개 데이터셋에서는 후보가 없었다.

Exp173은 정상 후보 수를 탐지력의 대리 지표로 사용하지 않는다. 세 점수원 중
두 개가 동의하는 구조와 TRAIN fold 간 정상 오경보 안정성을 사용해 정책 하나를
동결한다. Exp174는 그 정책을 변경하지 않고 한 번 사후 평가한다.

## 동결된 규칙

1. 입력은 Exp169 TRAIN-normal OOB audit뿐이다.
2. 60개 설정 전체에 Bonferroni 보정 단측 95% Wilson 상한을 계산한다.
3. 동시 상한이 1% 이하이고 resolution blocker가 0인 설정만 허용한다.
4. 세 점수원 중 두 개가 동의하는 `2of3` 설정만 선택 대상으로 삼는다.
5. 허용된 `2of3` 설정 중 가장 큰 alpha를 먼저 고른다.
6. 데이터셋 이름 SHA256으로 결정되는 5개 fold의 정상 검토율 표준편차가 가장
   작은 pool을 고른다.
7. 동률이면 최대 fold 검토율, 평균 reference 크기, 고정 pool 우선순위 순으로
   결정한다.
8. 선택된 policy와 label-free prediction을 각각 SHA256으로 동결한 뒤에만 TEST
   라벨을 읽는다.

세 점수원은 주파수 패턴, 형상 패턴, 구간 통계다. `2of3`은 이 중 임의의 두
점수원이 같은 후보를 지지해야 검토 후보가 된다는 뜻이다.

## 금지 사항

- TEST label, 이상 위치, TEST 길이, family 이름, 과거 family TEST 성능으로
  alpha, pool, 점수원 또는 후보를 선택하지 않는다.
- Exp174 결과를 본 뒤 1%, alpha, fold 수, tie-break 또는 `2of3`을 수정하지 않는다.
- Priority review 또는 Exp174 review 후보를 자동 Hard alert로 승격하지 않는다.
- tail replacement를 사용하지 않는다.
- autonomous와 human-assisted 지표를 합치지 않는다.

## 해석 범위

Exp170의 60개 TEST 결과는 과거 연구에서 이미 관찰됐다. 따라서 Exp173 코드가
구조적으로 TRAIN audit만 사용하더라도 Exp174는 독립 holdout 검증이 아니다.
모든 결과는 retrospective counterfactual이며, 실제 설비 성능이나 prospective
false-alarm 보장을 뜻하지 않는다.

운영 채택에는 시간적으로 이후인 잠금 데이터 또는 shadow mode가 필요하다.
그 검증에서는 selector와 모든 상수를 라벨 및 사용자 피드백을 열기 전에 동결한다.

## 중단 조건

- 60개 설정 또는 설정별 316개 데이터셋 coverage가 완전하지 않다.
- Exp169 입력 hash가 manifest와 다르다.
- 허용되는 `2of3` 설정이 없다.
- prediction에 라벨 또는 TP/FP/F1 같은 평가 필드가 포함된다.
- policy 또는 prediction hash가 평가 전에 달라진다.
- 누락 데이터셋을 제외해야만 결과를 만들 수 있다.

성능이 Exp172보다 나빠도 실험은 성공적으로 기록하고 규칙은 변경하지 않는다.
