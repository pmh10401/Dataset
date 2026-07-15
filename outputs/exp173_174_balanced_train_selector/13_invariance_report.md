# Exp173/174 불변성 검증

| 항목 | 결과 | 근거 |
|---|---|---|
| 60개 고정 grid | PASS | selector core가 Exp169의 3 pool x 4 alpha x 5 source configuration을 강제함 |
| 316개 TRAIN audit coverage | PASS | 각 policy key에서 데이터셋 316개 및 중복 0을 검증함 |
| TRAIN-only 정책 선택 | PASS | Exp173 입력은 Exp169의 dataset/global TRAIN OOB audit뿐임 |
| 다중 비교 보정 | PASS | 단측 95% Wilson bound에 60-way Bonferroni 적용 |
| 정상 위험 상한 | PASS | 선택 정책 동시 상한 `0.006722` <= `0.01` |
| Source agreement | PASS | 선택 가능한 source configuration을 `2of3`으로 고정함 |
| 최대 eligible alpha | PASS | 선택 alpha `0.01`; eligible `2of3` 중 최대값과 일치 |
| 5-fold 안정성 순위 | PASS | 선택 pool의 fold-rate 표준편차가 alpha 0.01 `2of3` 중 최소 |
| 입력 행 순서 불변 | PASS | global/dataset audit 역순 unit test 통과 |
| TEST metric 선택 금지 | PASS | selector와 audit 경로는 Exp170 metric CSV를 참조하지 않음 |
| Prediction label-free | PASS | 316개 prediction row에 label/TP/FP/F1 필드가 없음 |
| Prediction hash gate | PASS | tamper unit test 및 evaluation 전 SHA256 검증 통과 |
| Policy hash gate | PASS | prediction, evaluation, 독립 재집계가 frozen policy SHA256을 검증함 |
| Dataset coverage | PASS | prediction 및 metrics 316/316, 중복 0 |
| Candidate index 범위 | PASS | 독립 재집계 범위 오류 0 |
| 독립 재집계 | PASS | `mismatch_count=0` |
| Family/dataset 정책 분기 | PASS | family와 dataset 이름을 policy branch에 사용하지 않음 |
| TEST-length budget | PASS | top-N 및 TEST 길이 기반 budget 없음 |
| Autonomous Hard alert | NOT EVALUATED | Exp174는 research review candidate 전용 |
| 독립 holdout | NOT PERFORMED | Exp170 TEST 결과가 과거 연구에서 이미 관찰됨 |
| Prospective validation | NOT PERFORMED | 기존 TEST를 사용한 retrospective counterfactual |

검증 명령은 `docs/EXP173_174_RUNBOOK_KO.md`에 기록했다. 구조적으로 Exp173
selector는 TRAIN audit만 읽지만, 이 규칙이 과거 TEST 결과에 대한 인간의 지식과
완전히 독립적으로 개발됐다고 주장할 수는 없다. feature/source configuration의
end-to-end TRAIN-only provenance 역시 별도 미해결 과제다.
