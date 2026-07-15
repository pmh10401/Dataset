# Exp167/168 불변성 및 실행 계약 검증

## 결과

| 항목 | 결과 | 근거 |
|---|---|---|
| fold 대응 p-value와 `+1` 보정 | PASS | core 단위 테스트 |
| fold 순서 변경 불변 | PASS | core 단위 테스트 |
| TEST 라벨 변경 시 prediction 불변 | PASS | candidate 생성은 label 인자를 받지 않음 |
| prediction 함수의 `y_test` 인자 부재 | PASS | structure 단위 테스트 |
| 고정 4 alpha x 5 조합 | PASS | 316개 x 20행 = 6,320행 |
| 중복 및 범위 밖 candidate 차단 | PASS | core 및 전체 출력 독립 검사 |
| prediction 평가 필드 부재 | PASS | 6,320행 전체 검사 |
| prediction hash tamper 감지 | PASS | manifest 검증 단위 테스트 |
| 동일 입력 직렬화 결정성 | PASS | deterministic JSON hash 단위 테스트 |
| 상대/절대 output 경로 지원 | PASS | 경로 회귀 테스트 |
| family 이름 정책 미사용 | PASS | contract 및 structure 검사 |
| TEST 길이 budget 미사용 | PASS | contract 및 structure 검사 |
| tail replacement 미사용 | PASS | structure 검사 |

## 전체 출력 검증

- audit 데이터셋: 316
- audit 오류: 0
- source audit 행: 948
- prediction 행: 6,320
- 데이터셋당 prediction 행: 모두 20
- candidate count 불일치: 0
- candidate index 범위 오류: 0
- research-review 이외 lane: 0
- prediction 평가 필드: 0
- prediction SHA256:
  `2eb0a80273f822e76a6e78f1a2a31ccb9df63f454e7421b359cf8cf0d521504b`
- manifest SHA 일치: PASS
- dataset metric에서 summary 독립 재집계 불일치: 0

## 해석 제한

이 검증은 현재 고정된 세 점수원의 score-calibration parity와 prediction/evaluation
분리를 확인한다. 점수원 configuration의 최초 선정 계보 전체가 end-to-end
TRAIN-only였음을 증명하지 않으며 prospective 검증도 아니다.
