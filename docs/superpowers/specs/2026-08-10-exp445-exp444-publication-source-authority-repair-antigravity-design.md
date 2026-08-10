# Exp445 Exp444 publication source authority repair 설계

## 1. 결정과 목적

Exp445는 Exp444의 **publication source authority 결함만** 고치는 engineering-only
복구다. 과학적 알고리즘, 입력 모집단, 특징, score, threshold, mask, view, lane,
ranking, worker topology, backend 선택 또는 평가 해석을 바꾸지 않는다.

- experiment identity: `Exp445`
- artifact schema: `exp445`
- run identity: `exp445-exp444-publication-source-authority-repair-antigravity-20260810`
- fixed run root:
  `/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810`
- inherited isolated environment lock SHA256:
  `8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94`
- inherited scientific configuration SHA256:
  `8ba0a9524600bef7479df0626834bfeef4b3410f64d3f2d1cbd3885c784eb11a`

Exp445의 성공은 새 과학 결과가 아니라, 동일한 30개 Exp444 record를 새 namespace에서
재계산하고 전임 record bytes와 정확히 일치시킨 뒤 source-authoritative publication
경계를 안전하게 통과하는 것이다.

## 2. 불변 전임 authority

Exp444는 publication 전에 실패한 immutable predecessor다. 그것은 `attempt_0000_antigravity`
에서 30/30 record를 complete로 보존했지만 publication과 evaluation은 모두 없다. Exp445는
이 terminal authority를 읽기 전용으로 확인하고, Exp444 또는 Exp442의 어떤 파일도 수정,
이름변경, 재작성, 삭제하지 않는다.

| Authority | 고정 값 |
| --- | --- |
| Exp444 reviewed commit | `f4ae53c46422a46cc880d5d204f68a6a3971e92d` |
| Exp444 run root | `/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810` |
| predecessor attempt | `attempt_0000_antigravity` |
| terminal context SHA256 | `dcbc3755bf94d771c3f506746785e544d88940cc36bb75c0da72e99506ec9926` |
| terminal state SHA256 | `bcb524292f3ce2332a06b05fa6205bafce268f0d6d1ca6ff5622eea0e72be8a4` |
| terminal progress SHA256 | `bba24f851ede8aa498a5d2f94d4696792c81754290a48b8138e9c9ecd704f6ae` |
| terminal PID SHA256 | `e93fdbb36dd0e3339cf0900df22a14ee7627f74c25c6af31018adf8e0ca3397e` |
| terminal events SHA256 | `e58780ef8b127f7ad103278ac47494176b56c4653d2498405c8e125cd69042ff` |
| terminal text-log SHA256 | `8876d3db294294ecbb3c60ab170ed44776b61bbe9bcf3735bf32f27c156feef5` |
| ordered 30-record prefix payload SHA256 | `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034` |

The predecessor state must be exactly `complete`, `completed_units=30`,
`total_units=30`, `error_count=0`, and `error=null`. Its full ordered
`record_sha256_prefix` is authority, not merely the aggregate payload hash.
The predecessor has no `publication_antigravity` or evaluation artifact, and the
source database link count is exactly one before Exp445 readiness.

The frozen Exp442 source artifacts and Exp444 terminal files are preserved
byte-for-byte. Exp445 may read descriptor-pinned authorities but never treats an
Exp442 sibling pathname or a database-wide query as a substitute for them.

## 3. 닫힌 lifecycle profile 전략

구현은 기존 Exp444 wrapper를 유지하고, 하나의 shared implementation 안에 정확히 두
closed profile만 둔다.

| Profile | Fixed identity/root | 허용된 profile 차이 |
| --- | --- | --- |
| Exp444 | 기존 Exp444 identity 및 기존 root | 기존 동결 문서와 terminal predecessor authority |
| Exp445 | `Exp445`, `exp445`, Exp445 run identity 및 새 root | identity, schema, root, governing documents, predecessor authority |

Profile은 identity/root/docs/predecessor authority만 고를 수 있다. 다음은 두 profile
모두에서 동결되며 CLI, path, environment 또는 runtime override로 바꿀 수 없다.

- source hashes 및 source population
- Exp440 feature/calibration/scoring equation과 strict threshold rule
- seeds, omission masks, maximum views, stability fraction 및 lane rule
- candidate count 30, rank order 및 candidate coordinate
- CPU worker 수 11, inner numeric thread 수 1 및 backend gate
- post-freeze evaluation formula, query order 및 descriptive interpretation

Exp444 wrapper는 Exp445 root나 identity를 받을 수 없고, Exp445 wrapper는 Exp444 root를
받을 수 없다. 모든 public readiness/runtime/profile entrypoint는 profile object만 받고
arbitrary root, attempt, source, threshold, seed, worker, backend 또는 evaluation override를
거절한다.

## 4. 결함 수정: publication summary authority

Exp444의 결함은 publication source authority가 frozen Exp442 sibling summary를 다시
파생하는 데 있었다. Exp445는 이 경로, fallback, synthesis 및 database full query를
모두 제거한다.

새 publication helper의 고정 interface는 다음과 같다.

```python
def _publish_source_authority_antigravity(
    dataset_summary_bytes: bytes,
) -> _PublishSourceAuthorityAntigravity:
```

`dataset_summary_bytes`는 Exp445 provenance verifier가 descriptor-pinned Exp440
publication directory에서 단 한 번 capture한 canonical
`dataset_summary_antigravity.jsonl` bytes다. Helper는 전달된 bytes만 canonical JSONL로
parse하고, 1,117개 ordered summary row, Exp440 provenance SHA256 및 label-free schema를
검증한다. Helper 내부에서 pathname을 열거나 SQLite를 열지 않으며, Exp442 summary를
찾거나 만들어내지 않는다.

Provenance capture는 Exp440 summary, manifest, artifact map을 같은 retained descriptor
authority로 확인한다. Capture가 실패하거나 byte identity가 달라지면 publication 전에
실패한다. Exp445 runtime은 그 validated bytes와 terminal predecessor authority로만
publication payload를 만들고, public publication API에 다른 source-bytes 인자를 허용하지
않는다.

## 5. 새 root 실행과 publication 전 parity

Exp445 `new`는 새 fixed root에서 정상 label-free runtime을 처음부터 실행한다. resume은
없다. 30개의 frozen coordinate를 동결 Exp444 규칙으로 재계산하고, 새 checkpoint가
complete가 된 다음 publication 전에 다음을 모두 요구한다.

1. 새 context가 Exp445 profile identity/root/document authority를 갖고, inherited
   scientific configuration/environment authority는 전임과 정확히 같다.
2. 새 ordered 30 record wrapper SHA256 배열이 predecessor
   `record_sha256_prefix`와 index별로 정확히 일치한다.
3. 새 array의 canonical payload SHA256이
   `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`와 정확히
   일치한다.
4. 30/30, error 0, labels 0, CPU 11 workers 및 inner thread 1이 terminal authority에
   기록된다.
5. descriptor-pinned Exp440 summary bytes와 predecessor terminal authorities가 모두
   revalidated된 뒤에만 create-only publication staging을 시작한다.

어느 하나라도 다르면 new root는 failed evidence로 보존하고, publication/evaluation은
만들지 않는다. source authority 오류를 맞추기 위해 candidate, score, threshold, mask,
view 또는 backend를 조정하는 행위는 금지된다.

## 6. 고정된 네 명령

다음 wrapper만 Exp445 operator surface다. `REVIEWED_COMMIT`은 readiness/new가
현재 clean HEAD와 동일함을 증명하기 위해 operator가 제공하는 40-hex reviewed commit
argument이며, 어떤 scientific configuration value도 아니다.

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py new --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py inspect
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py evaluate
```

- readiness와 new만 `--reviewed-commit`을 받는다.
- resume, attempt/path, profile, source, seed, mask, science, worker, backend 및
  evaluation override는 public CLI에 없다.
- inspect는 Exp445 publication만 read-only로 검사하며 Exp444 publication을 찾거나
  해석하지 않는다.
- evaluate는 Exp445 publication을 서로 독립적으로 두 번 inspect한 뒤에만 lazy evaluator
  import를 허용한다. 그 전에는 label-capable module과 TEST label을 import/open/query하지
  않는다.

## 7. 가속과 실행 topology

각 새 score-source experiment에는 full run 전에 AGENTS.md의 MLX gate를 적용한다.
MLX는 정확한 동결 algorithm과 data definition을 표현하고, 동일 input/seed CPU parity,
finite output, 모든 preregistered alpha의 동일 candidate index, preregistered tolerance
이내 score difference, transfer/synchronization 포함 exact-stage 1.1x 이상 speed를
모두 만족할 때만 선택한다. Manifest에는 MLX/mlx-metal version, device, timing, parity 및
fallback reason을 기록한다.

어느 조건이라도 충족하지 않으면 CPU를 사용한다. 현재 설계의 default는 CPU 11 outer
workers와 worker당 inner numerical thread 1이다. 한 번에 하나의 MLX/MPS-heavy process만
허용하며 outer multiprocessing와 concurrent GPU job을 함께 사용하지 않는다.

## 8. 실패 보존과 거절한 대안

모든 readiness/runtime/source/parity/publication/evaluation failure는 새 Exp445 root 안의
create-only evidence로 보존한다. Exp444 및 Exp442 authority에는 failure marker조차
쓰지 않는다. 실패한 Exp445 root는 재시도하거나 덮어쓰지 않으며, operator는 새
실험/새 preregistration 없이는 동일 root에서 continuation을 만들 수 없다.

다음 두 대안은 명시적으로 거절한다.

1. **Exp444 same-root `attempt_0001_antigravity`**: immutable predecessor namespace에
   새 attempt를 추가하면 30/30 terminal source와 failed prepublication authority를
   혼합하고, Exp445의 새 identity/root 및 complete-parity gate를 우회한다. 따라서
   Exp444 root에는 `attempt_0001_antigravity`를 만들지 않는다.
2. **publication bypass**: Exp444의 30 record files나 terminal state를 직접
   publication API에 넣는 방식은 Exp445 provenance capture, fresh recomputation,
   ordered record parity 및 new-root lifecycle을 건너뛴다. 따라서 허용하지 않는다.

Task 11에서 시도된 fresh worker smoke는 unrelated authorized Exp236 seven-worker run과
겹치지 않도록 sequential rank 7에서 중단되었고 parallel phase 전에 끝났다. artifact나
새 parity authority를 만들지 않았으므로 Exp445의 source, runtime 또는 publication
evidence로 사용하지 않는다.

## 9. 검증 및 테스트 gate

후속 구현은 다음 test/gate를 갖는다.

- Exp444와 Exp445 closed-profile identity/root/document/predecessor separation,
  wrapper compatibility, 그리고 모든 forbidden override rejection
- descriptor-pinned Exp440 summary/manifest/artifact-map capture의 byte SHA 및
  directory replacement/symlink/sibling substitution rejection
- `_publish_source_authority_antigravity(dataset_summary_bytes)`가 전달된 bytes만
  사용하고 Exp442 sibling, fallback, synthesis, database full query를 호출하지 않는 spy
  및 failure tests
- 30 ordered record SHA index parity, aggregate prefix payload SHA parity, 30/30 terminal
  state, CPU/11/inner-1 topology, zero-label boundary
- source or record parity failure가 publication 전에 failed root를 보존하는 test
- Exp445-only inspect의 source-free/read-only replay, double inspection before lazy
  evaluator import, and no evaluation publication before those gates
- create-only publication/evaluation, failure preservation, no-clobber retry rejection,
  and Exp444/Exp442 byte identity regression
- CPU-versus-MLX fresh gate parity/speed evidence와 CPU fallback regression

Likely implementation files are the existing
`experiments/exp444_train_only_stability_precision_gate_antigravity/config_antigravity.py`,
`contracts_antigravity.py`, `reporting_antigravity.py`,
`readiness_antigravity.py`, `runtime_antigravity.py`, and
`cli_antigravity.py`; a new closed-profile module and Exp445 wrapper such as
`profiles_antigravity.py` and
`scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py`; and focused
`test_*_antigravity.py` files under the same experiment test directory. Exact file ownership
and a test-first task sequence belong to the later implementation plan.

## 10. 연구 거버넌스와 hash 선언 규칙

Exp445는 review-only, descriptive-only다.

```text
scientific_success_criterion = none
evaluation_status = descriptive_complete_no_promotion
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
test_informed_reselection_allowed = false
```

Evaluation은 post-freeze descriptive record일 뿐 score target, candidate retuning, source
replacement, threshold/mask/view adjustment, champion selection 또는 operational promotion의
근거가 될 수 없다. Exp445가 실패해도 Exp444/Exp442 결과를 수정하거나 재해석하지 않는다.

Exp445 design, preregistration 및 future authority artifact의 SHA256은 각각의 canonical
file bytes가 실제로 존재한 뒤에만 계산한다. 이 design에는 미래 파일의 hash value를
미리 선언하지 않으며, zero/dummy/temporary hash를 기록하지 않는다. 이후 preregistration
또는 authority artifact는 자기 canonical bytes가 확정된 뒤 별도 verifier가 계산한
64-hex SHA256만 받아들인다.

## 11. 설계 self-review

이 문서는 미결 항목이나 임시 표기를 남기지 않았다. identity, root, predecessor
authority, repair interface, lifecycle, parity gate, failure path, acceleration policy 및
governance를 각각 하나의 해석으로 고정했다. Exp444 same-root retry와 publication
bypass는 명시적으로 금지했고, 새로운 과학 규칙을 추가하지 않았다.
