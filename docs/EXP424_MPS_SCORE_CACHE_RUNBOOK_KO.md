# Exp424 실행 런북

실행 전 MPS availability, 깨끗한 Git HEAD, DB와 Exp291/Exp292 SHA-256을 확인한다.
Exp423의 중단된 checkpoint는 보존하지만 변경된 Exp424 provenance로 재사용하지 않는다.
output과 checkpoint는 매 시도마다 새 경로를 사용한다.

MPS-heavy 프로세스는 하나만 실행한다. 진행 확인은 다음 두 파일을 사용한다.

- checkpoint `progress.json`: dataset 완료 회계
- output `runtime_progress.json`: `current_dataset`, `stage`, query 진행

완료 후 manifest에서 다음을 확인한다.

- `state=review_only`
- `official_prediction=false`
- `contains_test_labels=false`
- `test_labels_opened=false`
- `optimization_adopted=true`
- parity·fallback·입력 hash·prediction hash 기록

혼동행렬과 성능은 prediction manifest/hash를 확인한 후 별도 post-hoc evaluator로만
계산한다.
