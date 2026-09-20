# Exp445 Exp444 publication source authority 복구 사전등록

## 신원과 동결 authority

Exp445는 Exp444의 publication source authority 결함만 복구하는 engineering-only,
review-only 실험이다. 알고리즘, 입력 모집단, feature, score, strict threshold,
mask, view, lane, ranking, worker topology, backend 선택과 평가 해석을 바꾸지 않는다.

```text
experiment_id = Exp445
artifact_schema = exp445
run_identity = exp445-exp444-publication-source-authority-repair-antigravity-20260810
run_root = /Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810
design_source = docs/superpowers/specs/2026-08-10-exp445-exp444-publication-source-authority-repair-antigravity-design.md
design_source_sha256 = 099c0d412d2375764a7e30440feca17d70a2a6bbfdd14e3e53999fe3c458d70d
plan_source = docs/superpowers/plans/2026-08-11-exp445-exp444-publication-source-authority-repair-antigravity.md
plan_source_sha256 = 4b369f044456ce9e3cca0abad141beb28b5b3431691677f19a2900997dc5ac9a
isolated_interpreter = /Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python
inherited_environment_lock_sha256 = 8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94
inherited_scientific_configuration_sha256 = 8ba0a9524600bef7479df0626834bfeef4b3410f64d3f2d1cbd3885c784eb11a
logical_cpu_count = 12
cpu_worker_rule = max(1, logical CPU - 1)
cpu_workers = 11
inner_numerical_threads = 1
backend = cpu
```

Exp445는 새 score source가 아니다. 따라서 backend를 새로 benchmark하거나 선택하지
않고, 동결 Exp444 CPU backend-decision authority와 30-candidate sequential/11-worker
parity authority를 SHA256으로 재검증한다. PyTorch/PyTorch MPS는 동결 환경 lock에
없어 unavailable/inapplicable이고, MLX exact-float64 probe 및 Numba `0.9968x < 1.1x`
fallback 근거를 그대로 상속한다.

## 불변 predecessor와 record-parity gate

Exp444는 immutable predecessor다. 다음 다섯 authority 조건은 readiness와 `new`가
모두 읽기 전용으로 확인하며, Exp444 또는 Exp442 파일을 수정·이름변경·재작성·삭제하지
않는다.

1. reviewed commit은 `f4ae53c46422a46cc880d5d204f68a6a3971e92d`이고 predecessor root는 `/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810`, attempt는 `attempt_0000_antigravity`다.
2. terminal context/state/progress/PID/events/text-log SHA256은 각각 `dcbc3755bf94d771c3f506746785e544d88940cc36bb75c0da72e99506ec9926`, `bcb524292f3ce2332a06b05fa6205bafce268f0d6d1ca6ff5622eea0e72be8a4`, `bba24f851ede8aa498a5d2f94d4696792c81754290a48b8138e9c9ecd704f6ae`, `e93fdbb36dd0e3339cf0900df22a14ee7627f74c25c6af31018adf8e0ca3397e`, `e58780ef8b127f7ad103278ac47494176b56c4653d2498405c8e125cd69042ff`, `8876d3db294294ecbb3c60ab170ed44776b61bbe9bcf3735bf32f27c156feef5`다.
3. predecessor state는 정확히 `complete`, `completed_units=30`, `total_units=30`, `error_count=0`, `error=null`이며 full ordered `record_sha256_prefix`가 authority다.
4. ordered 30-record prefix payload SHA256은 `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`다. Exp445는 쓰기 전 각 index의 canonical wrapper SHA256을 같은 predecessor SHA256과 비교하고, 30번째 쓰기 직전에 predecessor와 aggregate prefix를 다시 검증한다.
5. predecessor에는 `publication_antigravity`나 evaluation artifact가 없고 source database link count는 정확히 1이다.

30개 모두 index별·aggregate별로 일치하고 `30/30`, error `0`, labels `0`, CPU `11`,
inner thread `1`이 기록될 때만 publication staging을 시작한다. 불일치는 root allocation
뒤 immutable error record와 failed progress로 보존하며 terminal complete를 되돌리지
않고 publication/evaluation을 만들지 않는다.

## label firewall, failure 보존, operator surface

readiness, `new`, checkpoint, record, publication, inspect에는 TEST `label`,
`labels_blob`, 실제 anomaly 위치, 과거 TEST metric, family 성능, TEST missingness,
candidate index 위치와 tail rule이 없다. `evaluate`만 Exp445 publication을 독립적으로
두 번 inspect한 뒤 lazy evaluator import와 TEST label 접근을 할 수 있다.

`new`는 single-use다. resume, same-root retry, attempt/path/profile/source/seed/mask/
science/worker/backend/evaluation override는 없다. root allocation 전 provenance/source/
predecessor failure는 mutation-free이고 root를 만들지 않는다. allocation 뒤 failure는
새 Exp445 root의 create-only record/failure/publication/evaluation evidence로 보존한다.
단, 같은 run의 checkpoint state, progress, PID만은 기존 동결 방식대로 atomic
replacement를 허용한다. 이 예외는 create-only record, failure, publication,
evaluation artifact에는 적용되지 않는다.

```text
same-run mutable artifacts = checkpoint state, progress, PID (atomic replacement only)
create-only artifacts = record, failure, publication, evaluation
```

다음 wrapper만 operator surface다.

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py new --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py inspect
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py evaluate
```

`REVIEWED_COMMIT`은 clean HEAD와 같음을 증명하는 operator-provided 40-hex argument이며
scientific configuration은 아니다. readiness/new 외 명령은 이를 받지 않는다.

## 거버넌스 해석 (literal block)

```text
scientific_success_criterion = none
evaluation_status = descriptive_complete_no_promotion
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
test_informed_reselection_allowed = false
```

평가는 post-freeze descriptive record일 뿐 score target, candidate retuning, source
replacement, threshold/mask/view 조정, champion selection 또는 operational promotion의
근거가 될 수 없다. canonical bytes가 아직 없는 미래 preregistration/authority artifact
SHA256은 미리 선언하지 않으며, 실제 bytes가 확정된 뒤 별도 verifier가 계산한 64-hex
SHA256만 받아들인다.
