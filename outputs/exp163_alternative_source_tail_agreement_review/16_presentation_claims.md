# Exp163 발표 문구 판정

## 검증 완료

- Exp163 prediction은 TEST label과 실제 이상 위치를 읽지 않는다.
- 세 대안 점수는 원시값 합산 없이 TRAIN 경험적 상위 분위수와 동일 index 합의로 사용했다.
- Exp160 자동 경보와 우선 검토 index는 Exp163에서 변경되지 않았다.
- Exp163은 후보 0 데이터셋 일부에 일반 검토 후보를 추가했다.

## 조건부 사용

- 3-of-3 합의는 낮은 alpha에서 비교적 높은 신규 review precision을 보였다.
  - 동일 benchmark의 retrospective diagnostic이라는 조건을 함께 표시해야 한다.
- Exp163은 TRAIN-normal score threshold를 사용한다.
  - candidate-zero gate가 frozen TEST score 결과에 의존하므로 policy 전체를 strict TRAIN-only라고 부르지 않는다.

## 사용 금지

- Exp163이 자동 경보 성능을 개선했다.
- 특정 alpha 또는 합의 variant가 운영 최적값으로 확정됐다.
- 전체 pipeline이 end-to-end strict TRAIN-only다.
- 실제 반도체 설비에서 prospective 검증됐다.
- `mean_combined_f1`이 autonomous 모델 F1이다.
