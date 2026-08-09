# Exp443 정상 TRAIN 결측 처리 정책 비교 사전등록

## 동결 원본과 연구 범위

Exp443은 사용자 승인을 받은 정상 TRAIN 결측 처리 정책 비교의 구현 전
사전등록이다. 이 문서는 승인된 설계의 과학 계약을 요약하고, 구현·실행 전에
바꿀 수 없는 입력·정책·평가·해석 경계를 기록한다. 설계 바이트가 바뀌면 이
실험을 실행하지 않고 새 실험 번호와 새 사전등록을 만든다.

```text
experiment_id = Exp443
run_identity = exp443-missing-normal-train-policy-comparison-antigravity-20260809
design_source = docs/superpowers/specs/2026-08-09-exp443-missing-normal-train-policy-comparison-antigravity-design.md
design_source_sha256 = b52142a0a48f9c557a56180ee96dc1a7a8e46760a15a547aa92f8b5999808f9b
sha256_verification = passed_before_preregistration_copy
run_root = /Users/mac/Documents/Dataset/exp443_missing_normal_train_policy_comparison_antigravity_20260809
dataset_count = 1117
affected_dataset_count = 19
affected_family_count = 4
affected_normal_train_rows = 164
nonfinite_scalar_count = 983
all_missing_row_count = 0
recompute_units = 57
```

질문은 정상 TRAIN 결측 처리만 바꾸었을 때 기존 자동 Hard의 공식 MiniROCKET
승격 부분이 더 많은 놓친 이상을 복구하면서 전체 오경보와 강한 데이터셋
성능을 보존하는지다. Exp249의 MiniROCKET scorer, TEST 전처리, 탐지 규칙,
Exp236 lane, frozen exact-normal 보호 규칙은 바꾸지 않는다. Exp443은 같은
1,117개 벤치마크를 반복하는 adaptive follow-up이며, 모든 결과는
`review-only` 회고 증거다. 독립 검증, 외부 일반화, 반도체 현장 성능, 운영
자동 경보, 과학적 champion을 주장하지 않는다.

## Global Constraints

다음 항목은 승인 설계와 구현 계획에서 그대로 동결한다.

1. 새 파일 이름에는 반드시 `antigravity`를 포함하고 `__init__.py`는 만들지 않는다.
2. 실험 ID는 `Exp443`, 실행 ID는 위의 고정 문자열이다.
3. 최종 검토 설계와 SHA256은 위에서 고정한 한 쌍이며, 내용이 달라지면 실행을 중단한다.
4. 고정 run root와 별도 Python 3.14.6 환경 root는 저장소 밖에 둔다.
5. prediction은 원시 정상 TRAIN 값, label-free TEST 값과 identity, 동결된 Exp249/291/414/404 산출물, 검토된 Exp443 코드·설정만 읽는다.
6. prediction은 evaluator, TEST `label`, `labels_blob`, 실제 이상 위치, Exp416/Exp442 평가, 과거 family TEST 성능을 import하거나 읽지 않는다.
7. TEST는 기존 Exp249 선형 보간과 가장 가까운 유한값의 edge extension을 그대로 사용하고 정상 TRAIN 처리만 바꾼다.
8. target length는 원본 정상 TRAIN 길이로 먼저 계산하여 세 정책에서 동일하게 고정한다.
9. 정책 순서는 `linear_edge_nearest_control`, `exclude_nonfinite_row`, `leave_one_out_peer_median`이며 작업 순서는 정책-major, case-sensitive dataset 정렬이다.
10. 모집단은 1,117개 dataset, 결측 영향 19개·4개 family, 결측 정상 TRAIN 164행, 비유한 scalar 983개, 전부 결측 행 0개로 고정한다. 불일치는 scoring 전에 실패한다.
11. Policy C는 같은 원래 길이의 다른 정상 TRAIN 행만 참조하고, 자기 행을 제외한 유한 peer의 ordinary NumPy median을 사용한다. 지원 peer가 하나라도 없으면 fallback 없이 행 전체를 제외한다.
12. 현재 예상되는 Policy C 미지원 행은 `MelbournePedestrian_normal_8` instance 1과 `MelbournePedestrian_normal_9` instance 242의 두 행이며, 네 개 좌표는 audit 증거로만 보존한다.
13. MiniROCKET은 기존 공식 `experiments.exp249_250_official_mini_multirocket.scoring.score_configuration`을 그대로 사용한다. transformer를 fork·rewrite·vectorize·번역하지 않는다.
14. kernel 요청 수는 1,024와 4,096, 실제 차원은 1,008과 4,032, 최대 dilation은 32, seed는 20260721과 20261721, five-fold OOF, KNN-3 local-gap, 최대 reference 1,000, query chunk 128, per-instance z-normalization, alpha 0.005 seed intersection이다.
15. LocalGapModel의 fold-fit 중심은 median이고 RobustScaler 범위는 `(10, 90)`이다. `(25, 75)` IQR로 바꾸지 않는다.
16. frozen routing은 `Exp236 Hard OR (Exp236 Standard AND (Mini1024 OR Mini4096)) MINUS frozen exact-normal`이다. 동결 수는 `2,116 + 137 - 35 = 2,218`이다.
17. Policy A는 B/C 전에 finite score, 두 kernel 후보 index 완전 일치, scalar 최대 절대오차 `1e-6` 이하, Exp291/Exp404 최종 Hard 좌표 완전 일치를 모두 재현해야 한다. 하나라도 실패하면 `control_reproduction_failed`로 닫고 B/C·TEST evaluator를 만들지 않는다.
18. 공식 CPU topology는 논리 CPU 12개에서 outer worker 11개, 각 numerical inner thread 1개다. 1-worker와 11-worker payload·candidate·routing·canonical prediction bytes가 같아야 한다.
19. 공식 aeon MiniROCKET은 MLX/MPS로 다시 구현하지 않는다. acceleration 상태는 `inapplicable_official_cpu_transformer`이며 MLX/MPS-heavy process는 실행하지 않는다.
20. checkpoint는 57개 recompute unit의 ordered prefix와 payload SHA에 묶인 create-only 기록이다. 오류 record를 성공으로 덮거나 완료 record를 덮어쓰지 않는다.
21. 세 정책의 label-free prediction을 각각 동결·inspect한 뒤에만 별도 evaluator가 TEST label을 연다. prediction 파일에는 label·confusion·F1을 넣지 않는다.
22. baseline은 TP 1,915 / FP 303 / FN 2,003 / TN 191,679, micro F1 `0.6241851368970013`을 재현해야 한다.
23. dataset precision·recall·F1·false-positive rate의 분모가 0이면 `0.0`을 사용하고, macro는 1,117개 dataset 단순 평균이다.
24. `rescued`는 baseline TP가 0이고 정책 TP가 1 이상인 dataset, `regressed`는 정책 dataset F1이 baseline보다 감소한 dataset이다.
25. bootstrap은 NumPy 2.3.5 `PCG64`, seed `44320260809`, `int64`, `endpoint=False`, quantile method `linear`, 20,000 paired dataset samples로 고정한다.
26. 세 정책은 각각 독립적으로 분류·보고하고 TEST 결과를 보고 같은 Exp443 안에서 하나를 승자로 재선택하지 않는다.
27. Exp443은 adaptive, retrospective, `review-only`, independently unvalidated, non-operational이다. Exp404/Exp416은 검증된 reference로 유지한다.
28. 결과를 기록한 뒤 Exp444를 시작하지 않고 방향 논의를 위해 멈춘다.
29. 코드 변경 전 `docs/codemap/codemap.lock`을 확인하고, mapped module을 수정할 때 `codemap.json`의 callers/effects/tests를 확인한다. 경계·의존성·major flow 변경 시 map triplet을 함께 갱신한다.

## 동결 입력과 SHA256

모든 파일은 descriptor-bound read-only로 열고 동일 bytes의 SHA256을 재검증한다.
Exp249 cache manifest가 선언한 1,117개 NPZ도 scalar score를 읽기 전에 확인한다.
symlink·FIFO·inventory 변경·중복 dataset·범위 밖 좌표는 TEST label 접근 전에
fail closed한다.

| 입력 | 경로 | SHA256 |
|---|---|---|
| SQLite DB | `/Users/mac/Documents/Dataset/univariate_ts.db` | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |
| Exp249 prediction | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/frozen_predictions.jsonl` | `fc72d2dfbaf6a1653ec804dbe349397214992387083b22bdab0e8c26a66876c2` |
| Exp249 prediction manifest | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/prediction_manifest.json` | `25ade7c3f8362559a29c15e43fead6070a7fac6cab648f03f62bcb20382b8c8f` |
| Exp249 artifact manifest | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/MANIFEST.json` | `42fee6f9a26b2c6fd19747f83887a431bf5edad144e2af3464033256a0108da9` |
| Exp249 cache manifest | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/cache_manifest.csv` | `44c0ecc80ff7cbae383d49d2b8b0cefed442c49b8a12a7bd0cf8569ce121c971` |
| Exp249 audit | `/Users/mac/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/audit.json` | `144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee` |
| Exp291 routing | `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl` | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` |
| Exp291 manifest | `/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json` | `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e` |
| Exp414 repair audit | `/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat/exp414_missing_repair_audit.csv` | `548f493a1608521bc81e4b742cc061fab7298f0d03c69f23845e37528b26ab91` |
| Exp414 manifest | `/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat/MANIFEST.json` | `99556922dc8f5178feaf03bc61eca4bfae628ae8f6b6092c2f6159da8c5dd3cc` |
| Exp404 Hard comparator | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` |
| Exp404 Hard manifest | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json` | `41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8` |

## 세 정책 정의

세 정책은 정상 TRAIN에만 적용한다. 유한 정상 TRAIN 행은 값과 순서를 유지하고,
TEST preprocessing은 세 정책에서 동일하다.

### A. `linear_edge_nearest_control`

기존 Exp249 전처리를 그대로 재현한다. 내부 `NaN`, `+Inf`, `-Inf`는 원래
시간좌표의 선형 보간으로 채우고, 시작·끝 결측은 가장 가까운 유한값으로
연장한다. 이 정책은 B/C 실행을 허용하는 control reproduction gate다.

### B. `exclude_nonfinite_row`

비유한값이 하나라도 있는 정상 TRAIN 행 전체를 model fit과 OOF calibration에서
제외한다. dataset, instance index, 원래 길이, 결측 위치와 원본 blob SHA256을
기록한다. 현재 inventory에서는 영향 dataset마다 최소 15행이 남는다.

### C. `leave_one_out_peer_median`

원래 길이가 같은 다른 정상 TRAIN 행만 참조한다. 각 결측 위치에서 자기 행을
제외한 유한 peer의 leave-one-out 중앙값으로 복원한다. 한 위치라도 유한 peer가
0개면 fallback 보간 없이 행 전체를 제외한다. mask는 복원·audit에만 쓰고
MiniROCKET feature로 전달하지 않는다. 현재 inventory에서는 162행이 복원되고
두 행은 무지원으로 제외된다.

전부 결측인 행이 실행 중 발견되면 세 정책 모두 `unexpected_all_missing_train`
으로 종료하며 0 대체를 하지 않는다.

## 고정 MiniROCKET·routing 계약

공식 aeon MiniROCKET의 요청 kernel, 실제 차원, seed, dilation, OOF cross-fit,
fold-fit TRAIN reference 최대 1,000, query chunk 128, median/10th-to-90th
robust scaling, KNN-3 local-gap, `seed_intersection`, alpha `0.005`,
per-instance z-normalization은 세 정책에서 동일하다. TEST 전처리·alignment·
normalization 순서도 동일하다.

동결 집합은 다음과 같이 재생한다.

```text
promotion(d)
  = exp236_standard(d)
    AND (mini1024_seed_intersection_alpha005(d)
         OR mini4096_seed_intersection_alpha005(d))
pre_exact_hard(d) = exp236_hard(d) OR promotion(d)
automatic_hard(d) = pre_exact_hard(d) MINUS frozen_exact_normal(d)
```

정확히 `2,116 + 137 - 35 = 2,218` 좌표를 control로 재현하고, 결측 없는
1,098개 dataset 좌표가 세 정책에서 baseline과 완전히 같은지 확인한다.

## Control fail-closed gate

Policy A를 결측 영향 19개에서 먼저 계산한다. 다음 네 조건을 모두 만족해야
Policy B/C와 TEST evaluation으로 진행한다.

1. 모든 OOF와 TEST score가 finite다.
2. 두 kernel 설정과 alpha `0.005`에서 Exp249의 후보 index가 완전히 같다.
3. 대응 scalar score의 최대 절대오차가 `1e-6` 이하다.
4. A를 기존 1,098개 동결 결과와 결합한 final Hard가 Exp291/Exp404의 2,218개 좌표와 완전히 같다.

실패 상태는 `control_reproduction_failed`이며 B/C prediction·TEST 평가는 만들지
않는다. 환경·코드 변경은 같은 official attempt를 덮지 않고 새 attempt에서
readiness부터 다시 시작한다.

## 사후 평가와 성공 게이트

세 prediction SHA가 모두 동결되고 `inspect`가 통과한 뒤에만 별도 evaluator가
같은 TEST label snapshot으로 baseline과 A/B/C를 paired 평가한다. 각 정책에 대해
TP, FP, FN, TN, precision, recall, micro F1, dataset-balanced macro precision/
recall/F1, false-positive rate, 추가·제거 좌표의 TP/FP, 19개 영향 dataset과
전체 1,117개 결과, rescued/regressed, strong-dataset 악화, family concentration,
19개 밖 좌표 불변성을 기록한다.

정책 하나가 `retrospective_improvement_signal`이 되려면 다음 여덟 조건을
모두 만족해야 한다.

1. 전체 1,117개 micro F1 delta `> 0`.
2. 전체 1,117개 dataset-balanced macro F1 delta `> 0`.
3. micro와 macro paired-bootstrap 95% 구간 하한이 모두 `> 0`.
4. 전체 FP가 baseline `303`보다 증가하지 않는다.
5. 새로 복구한 dataset이 최소 1개다.
6. baseline F1 `>= 0.8`인 strong dataset의 F1 감소가 0개다.
7. 결측 없는 1,098개의 final Hard 좌표가 baseline과 완전히 같다.
8. prediction 이전 label 접근 0회, 오류 0개, 비유한 출력 0개다.

여러 정책이 통과해도 TEST 결과로 하나를 Exp443의 승자로 고르지 않는다. 통과한
정책은 모두 review-only 신호로 보고하며 후속 사용은 새 실험 번호와 새
사전등록에서 하나의 규칙을 다시 동결해야 한다.

## 거버넌스 해석 (flat block)

아래 아홉 key는 preregistration·runbook·catalog·state·champion history에서
같은 flat key/type/value로 유지한다. 중첩 object, alias, truthy 대체값은 쓰지
않는다.

```text
official_run_purpose = controlled_missing_normal_train_policy_comparison
adaptive_followup = true
independent_validation = false
external_generalization_supported = false
confirmatory_statistical_claim = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
test_informed_reselection_allowed = false
```

현재 상태는 `design_approved_implementation_planned`이며 공식 readiness,
prediction, publication, evaluation은 아직 생성되지 않았다. 이 문서는
구현을 승인하는 과학 계약이지 실행 성공을 주장하는 보고서가 아니다.
