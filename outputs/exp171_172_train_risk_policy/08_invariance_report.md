# Exp171/172 불변성 검증

| 항목 | 결과 | 근거 |
|---|---|---|
| 60개 고정 grid | PASS | policy core가 3 pool x 4 alpha x 5 source configuration을 강제함 |
| TRAIN-only 정책 선택 | PASS | Exp171 입력은 Exp169의 `04_exp169_dataset_audit.csv`, `05_exp169_global_summary.csv`뿐임 |
| 다중 비교 보정 | PASS | 단측 95% Wilson bound에 60-way Bonferroni 적용 |
| 입력 행 순서 불변 | PASS | shuffled global/dataset audit unit test 통과 |
| TEST metric 선택 금지 | PASS | policy core와 audit 경로는 Exp170 metrics 파일을 참조하지 않음 |
| Prediction label-free | PASS | 316개 prediction row에 label/TP/FP/F1 필드가 없음 |
| Prediction hash gate | PASS | tamper unit test 및 evaluation 전 SHA256 검증 통과 |
| Policy hash gate | PASS | evaluation과 독립 재집계가 frozen policy SHA256을 검증함 |
| Dataset coverage | PASS | prediction 및 metrics 316/316, 중복 0 |
| Candidate index 범위 | PASS | 독립 재집계 범위 오류 0 |
| 독립 재집계 | PASS | global summary mismatch 0 |
| Family/dataset 정책 분기 | PASS | family와 dataset 이름은 정책 선택 조건에 사용되지 않음 |
| TEST-length budget | PASS | top-N 및 TEST 길이 기반 budget 없음 |
| Autonomous Hard alert | NOT EVALUATED | Exp172는 research review candidate 전용 |
| Prospective validation | NOT PERFORMED | 기존 TEST를 사용한 retrospective counterfactual |

검증 명령은 `docs/EXP171_172_RUNBOOK_KO.md`에 기록했다. 이 결과는 과거
feature/source configuration의 end-to-end TRAIN-only provenance를 증명하지 않는다.

