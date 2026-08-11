# Exp446 Exp445 publication materialization repair 설계

## 1. 목적

Exp446는 Exp445의 과학 계산을 변경하지 않고, 출판 직전 frozen JSON handoff의
중첩 tuple을 canonical JSON list로 materialize하지 않아 발생한 publication 오류만
복구하는 engineering-only, label-free 실험이다.

Exp446는 새 score source, 새 threshold, 새 mask, 새 view, 새 backend 또는 성능 개선
실험이 아니다. Exp445의 30개 좌표를 같은 CPU 규칙으로 처음부터 다시 계산하고,
Exp444 및 Exp445와 record wrapper SHA가 index별로 정확히 같을 때만 출판한다.

## 2. Exp445 불변 실패 증거

Exp445 공식 root는 다음 위치에 보존한다.

```text
/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810
```

`attempt_0000_antigravity`는 complete/inactive 상태이며 다음 사실을 갖는다.

- 계산 완료: `30/30`
- runtime error: `0`
- label access: `0`
- backend: CPU, outer worker `11`, inner numerical thread `1`
- elapsed: `913.3950564998668` seconds
- abstention: `4`
- ordered record-prefix aggregate SHA256:
  `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`
- publication: absent
- evaluation: absent
- immutable publication failure:
  `ValueError: Exp444 original-control replay is invalid`

Frozen terminal artifact SHA256 values:

```text
checkpoint context = e7ea9465327ae4b71dfe06c8a83f7c3eb77e4d1445b0598da3de6afa4a5a53c3
checkpoint state   = b8a4d6f1bb1cb30680f4244c7e025c43367a41ed90518b6e49dc4d09e890ede4
progress           = 8d34590450e2fcd2e21dcc6f03454ac7d4cd9fc1d15e1ecb25fc858c20615eb5
PID                = e0f2d8ecc65008decdb8ba8292c7204c377bebeccb8ab564bb225bae307b3fbf
publication failure= 9c58eddaa152bde221a5b7673be3d413204ff3dfe9fd2b4262e6d9065b2c6db0
```

Exp446 readiness와 `new`는 descriptor-relative, no-follow, exact-inventory 방식으로
이 증거와 30개 record wrapper SHA를 검증한다. Exp445 root에는 write, rename,
unlink, retry, resume, publication 또는 evaluation을 수행하지 않는다.

## 3. 확정된 원인

값 전용 publication handoff는 mutable JSON container를 안전하게 분리하기 위해
중첩 list를 tuple로 동결한다.

```text
control row list
→ _freeze_json_value_antigravity
→ nested tuple
```

하지만 `_payloads_antigravity`는 바깥 tuple만 `list(...)`로 바꾸고 중첩
`depth_scores`와 `replayed_depths`는 tuple로 남긴다. 이어지는
`_parse_control_replays_antigravity`는 canonical JSON payload 계약에 따라 두 필드가
실제 list일 때만 허용하므로 모든 정상 control row를 거절한다.

순수 재현 결과:

```text
frozen depth_scores      = tuple
frozen replayed_depths   = tuple
parser accepts frozen    = false
materialized fields      = list / list
parser accepts materialized = true
```

readiness는 `OriginalReplayAntigravity` 객체를 직접 검사하므로 이 serialization
경계를 지나지 않아 통과했다. 기존 real-composition 테스트도 handoff 생성과 독립
serialization만 검사하고 실제 `_payloads_antigravity`까지 연결하지 않아 결함을
놓쳤다.

## 4. 유일한 과학 외 수정

shared reporting의 parser 입력에서 이미 존재하는 recursive JSON materializer를
사용한다.

```python
control_rows = _json_payload_value_antigravity(authority.control_rows)
_parse_control_replays_antigravity(control_rows, source)
```

같은 materialized `control_rows`를 canonical control-replay bytes 생성에도 사용한다.
parser가 tuple을 직접 허용하도록 느슨하게 변경하지 않는다. 수정 전후 source
coordinates, control semantic fields, rank, record bytes, score, threshold, mask,
view, lane 및 hash DAG는 동일해야 한다.

## 5. Exp446 고정 identity

```text
experiment_id = Exp446
artifact_schema = exp446
run_identity = exp446-exp445-publication-materialization-repair-antigravity-20260811
run_root = /Users/mac/Documents/Dataset/exp446_exp445_publication_materialization_repair_antigravity_20260811
attempt = attempt_0000_antigravity
predecessor = immutable Exp445 failed attempt_0000_antigravity
backend = cpu
logical_cpu_count = 12
outer_workers = max(1, logical_cpu_count - 1) = 11
inner_numerical_threads = 1
label_access_count = 0
```

Profile 선택은 Exp444, Exp445, Exp446 세 closed singleton만 허용한다. arbitrary
identity, schema, root, source, attempt, worker, backend, seed, threshold, mask,
view 또는 path override는 없다.

## 6. 닫힌 명령과 데이터 흐름

Exp446 operator surface는 정확히 세 명령이다.

```text
readiness --reviewed-commit REVIEWED_COMMIT
new --reviewed-commit REVIEWED_COMMIT
inspect
```

`resume`, `retry`, `evaluate`는 존재하지 않는다. TEST label-capable evaluator와
SQLite label query 경로는 Exp446 package와 launcher에서 import하지 않는다.

Data flow:

```text
clean reviewed HEAD + frozen documents + absent Exp446 root
→ Exp445 immutable failure/predecessor verification
→ descriptor-pinned Exp440 summary and values-only source verification
→ allocate single-use Exp446 root
→ recompute ordered 30 candidates on CPU/11
→ compare every wrapper SHA with Exp445 and Exp444
→ revalidate 30-prefix aggregate and Exp445 predecessor
→ recursively materialize frozen control rows
→ validate canonical control replay
→ create-only staging and atomic publication
→ source-free, label-free inspect
```

## 7. 가속 정책

Exp446는 새 score-source 실험이 아니며 동결 Exp445 계산을 재현한다. PyTorch,
MPS, MLX 또는 Numba를 재검토·재benchmark·재선택하지 않는다. 기존 environment lock,
CPU backend, 11 outer workers, one inner thread와 sequential/11-worker parity authority를
그대로 재검증한다.

## 8. 실패 보존

readiness 또는 `new`의 root allocation 전 실패는 mutation-free이며 Exp446 root를
만들지 않는다. Allocation 후 record, materialization, parity 또는 publication 실패는
새 Exp446 root에 create-only failure evidence로 보존한다. 실패한 Exp446 root는 같은
실험 번호와 root에서 재시도하지 않는다.

Exp445와 Exp444 root 및 artifact는 항상 읽기 전용이다.

## 9. TDD와 검증

첫 RED는 실제 frozen values-only handoff를 parser helper까지만 보내는 테스트가 아니다.
다음 전체 경계를 호출해야 한다.

```text
real values-only capture
→ compose source authority
→ real terminal Exp445-compatible fixture
→ _payloads_antigravity
→ control-replay bytes inspection
```

수정 전에는 정확히 `Exp444 original-control replay is invalid`로 실패해야 한다. 수정 후에는:

- 30개 control replay가 rank와 coordinate 검증을 통과한다.
- serialized `depth_scores`와 `replayed_depths`는 list다.
- 수정 전후 materialized canonical control bytes가 의미상 동일하다.
- Exp444와 Exp445 공개 wrapper 및 기존 frozen artifact bytes가 변하지 않는다.
- Exp446 30 wrapper SHA와 aggregate가 Exp445/Exp444와 정확히 같다.
- sequential/11-worker canonical bytes가 같다.
- readiness/new/inspect label access가 `0`이다.
- Exp446 launcher에 evaluation import와 command가 없다.

Focused test, inherited Exp444/445 regression, Exp446 full regression, clean-tree
map validation과 fresh Sol High review가 모두 통과한 뒤에만 공식 명령을 허용한다.

## 10. 공식 실행 성공 조건과 중단 조건

성공 조건:

- readiness는 clean reviewed HEAD, fixed document SHA, DB link count `1`, absent
  Exp446 root, Exp445 exact frozen authority, CPU/11, labels `0`을 확인한다.
- `new`는 `30/30`, error `0`, labels `0`으로 끝난다.
- 30 record wrapper SHA와 aggregate는 Exp445/Exp444와 정확히 같다.
- create-only Exp446 publication이 완성된다.
- 두 번의 source-free inspect 결과와 publication/prediction SHA가 동일하다.
- evaluation과 TEST-label access는 수행하지 않는다.

즉시 중단 조건:

- Exp445 inventory, inode, terminal SHA, record prefix 또는 failure marker 변경
- readiness 전 Exp446 root 존재
- clean HEAD, document SHA, environment lock, CPU/11 또는 parity authority 불일치
- record/control canonical 의미 또는 30-wrapper SHA 변경
- TEST label-capable import/open/query 발생
- allocation 후 Exp446 실패

## 11. 거절한 대안

Exp445 30 records를 복사해 publication만 만드는 방법은 계산 비용을 줄이지만 새 root의
runtime/context/record provenance를 실제 실행 없이 구성하는 publication bypass다. 사용하지
않는다.

Parser가 list와 tuple을 모두 허용하도록 변경하는 방법은 canonical JSON 계약을 약화한다.
사용하지 않는다.

Exp445 root에서 retry/resume 또는 publication을 추가하는 방법은 immutable failed-root
계약을 위반한다. 사용하지 않는다.
