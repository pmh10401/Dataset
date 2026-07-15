# Exp171/172 TRAIN 정상 위험 정책 사전등록

## 목적

Exp169/170은 작은 TRAIN 데이터셋 316개에서 conformal 해상도 문제를
해결했지만 60개 설정 중 하나를 선택하지 않았다. Exp171은 TEST 결과를
사용하지 않고 TRAIN 정상 OOB audit만으로 일반 검토 정책 하나를 동결한다.
Exp172는 동결된 prediction hash를 확인한 뒤 기존 TEST에서 사후 평가한다.

## 고정 운영 계약

- 운영 단위: 가상 wafer-run trace 1개
- 정상 검토 요청 목표: 정상 run 100개당 최대 1개 (`0.01`)
- 신뢰수준: 단측 95% Wilson upper bound
- 60개 동시 비교: Bonferroni 보정
- alpha: `0.005, 0.01, 0.02, 0.05` 고정
- pool 3개와 점수원 조합 5개는 Exp169와 동일
- 선택 입력: TRAIN 정상 OOB count만 허용
- TEST label, TEST 이상 위치, TEST metric, family 이름, dataset 이름,
  TEST 길이는 정책 선택에 사용하지 않음

위험 상한을 만족한 설정 가운데 TRAIN 정상 review count가 가장 큰 설정을
동결한다. 이는 허용된 위험 예산을 사용하는 보수성 완화 규칙이며 이상 탐지
성능이 가장 좋다는 의미가 아니다. 동률 규칙도 실행 전에 코드로 고정한다.

## 해석 제한

Exp172 후보는 자동 경보가 아닌 연구용 일반 검토 후보다. 결과는 기존 TEST를
사용한 retrospective counterfactual이며 prospective 또는 실제 설비 검증이
아니다. 전체 feature configuration의 end-to-end TRAIN-only provenance도 아직
검증 완료로 표현하지 않는다.

