# Exp169/170 검증 보고서

## 실행 범위

- 코드 커밋: `ed94f41d`
- DB SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- 대상: 316개
- frozen prediction: 18,960행
- prediction SHA256: `836a773001c97888dd218372e3579c36dd1811c275d7cd418f46fd39aeeef4a4`

## 검증 결과

| 항목 | 결과 | 근거 |
|---|---:|---|
| OOB split 결정성 | PASS | 동일 seed와 n_train에서 같은 32개 split |
| 모든 TRAIN row OOB coverage | PASS | 최소 OOB component count가 1 이상 |
| target donor 제외 | PASS | 모든 pool에서 target 이름 제외 |
| donor 중복 방지 | PASS | distance tie shell의 unique donor 검사 |
| 실제 source reference 200 이상 | PASS | source 배열의 실제 길이로 검사 |
| cache coverage | PASS | 316/316, error 0 |
| cache hash와 shape | PASS | manifest SHA256 및 source 배열 shape 재검증 |
| pool manifest 재현 | PASS | donor, reference SHA256, policy config hash 일치 |
| prediction label-free | PASS | TEST label 및 평가 metric 필드 없음 |
| prediction 고정 격자 | PASS | 316 x 60 = 18,960, 중복 없음 |
| prediction hash tamper guard | PASS | 평가 전에 SHA256 검증 |
| DB hash gate | PASS | cache와 evaluation 전에 고정 DB hash 확인 |
| 독립 재집계 | PASS | mismatch 0 |
| autonomous Hard alert 평가 | 해당 없음 | 모든 후보는 research review lane |

29개 단위·구조 테스트와 10개 smoke를 통과한 뒤 full run을 실행했다. prediction 함수는 TEST label을 입력받지 않으며, evaluation이 frozen prediction과 의존 manifest의 hash를 검증한 뒤에만 TEST label을 읽는다.

## 제한

- 이번 결과는 기존 TEST를 사용한 retrospective counterfactual evaluation이다.
- feature/source configuration의 전체 개발 계보가 end-to-end TRAIN-only였는지는 별도 provenance 검증 범위다.
- 실제 설비의 신규 데이터에서 prospective 또는 shadow validation을 수행하지 않았다.
- 60개 결과 중 운영값을 선택하지 않았다.

