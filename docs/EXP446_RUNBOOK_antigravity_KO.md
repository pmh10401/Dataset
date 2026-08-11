# Exp446 Exp445 publication materialization 복구 실행 안내서

## 고정 경로와 preflight

```text
RUN_ROOT=/Users/mac/Documents/Dataset/exp446_exp445_publication_materialization_repair_antigravity_20260811
ATTEMPT=$RUN_ROOT/attempts_antigravity/attempt_0000_antigravity
PREDECESSOR=/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810/attempts_antigravity/attempt_0000_antigravity
```

Exp446는 `official_completed_publication_succeeded_cli_serialization_failed_inspection_pending`이다.
공식 CPU/11 실행과 create-only publication은 완료됐지만, 원래 CLI wrapper는 반환 뒤 nested
`MappingProxyType`을 `dataclasses.asdict`가 deepcopy하면서 실패했다. 이 문서는 retry, resume,
새 readiness/new 또는 재-inspect 권한이 아니다. immutable attempt를 바꾸지 않으며 fresh Sol High
read-only review 전에는 source-free official inspect도 다시 실행하지 않는다. evaluation, promotion 및
TEST-label 접근은 미수행이다.

## 공식 완료 사실과 보존 경계

- execution commit `c5278ba600a2d7ee259f5f5f987e1f9cf20c04cb`, PID `74950`, elapsed
  `471.76320550008677` seconds; completed/success/abstention/error is `30/30/4/0`.
- prediction SHA256 is `602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100`;
  publication artifact-map SHA256 is `d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d`;
  manifest SHA256 is `bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11`.
- labels remain `0`, promotion remains false, DB nlink is `1`, and the official Exp446 root is
  immutable. Inspection is pending fresh Sol review; it is not a retry path.

## 구현 검증 증거 (공식 실행 아님)

- 불변 Exp445 30개 record wrapper를 strict 재구성해 실제 순차/11-worker scheduler와
  serialization 경로에만 통과시킨 bounded parity에서 각 wrapper byte/SHA가 같고 aggregate는
  `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`였다. 이 검증은
  과학 score를 재계산하지 않았고 label access는 `0`이다.
- CPU는 Apple M2 Max, logical CPU `12`, worker `11`, inner numerical thread `1`이다.
- 회귀 boundary는 Exp444 `206` tests, Exp445 `39` tests, Exp446 `47` tests 모두 exit `0`이다.
  Exp445의 immutable predecessor와 source DB single-link (`nlink=1`) 및 absent Exp446 root를
  read-only로 확인했다.
- 초기 bounded harness는 frozen wrapper의 `{sequence,result}` schema preflight에서
  fail-closed했고 보존했다. 수정된 harness는 그 frozen schema를 검증하는 scheduler/serialization
  replay만 수행했으며 production algorithm이나 scoring rule을 변경하지 않았다.

## 정확히 세 명령

```bash
python scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
python scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py new --reviewed-commit REVIEWED_COMMIT
python scratch/run_exp446_exp445_publication_materialization_repair_antigravity.py inspect
```

`REVIEWED_COMMIT`은 operator-supplied clean reviewed HEAD의 40-hex 값이다. `resume`,
`retry`, `evaluate`와 identity/root/source/attempt/worker/backend/seed/threshold/mask/view
override는 없다.

## 닫힌 실행 계약

- `readiness`는 Exp445 inventory/terminal SHA/failure marker, 30 wrapper SHA, aggregate,
  absent Exp446 root, CPU/11 inner-1, labels 0, clean reviewed HEAD를 descriptor-relative
  no-follow exact-inventory 방식으로 검증한다. root는 만들지 않는다.
- `new`는 동결 Exp445 계산을 CPU/11로 재실행하고 모든 wrapper SHA와 aggregate
  `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`를 검증한다.
  그 뒤 recursive materialization을 parser input과 canonical control-replay bytes에 공통으로
  사용해 create-only staging과 atomic publication을 수행한다.
- `inspect`는 source-free, label-free로 Exp446 publication만 읽는다.

Exp445/Exp444 root는 항상 read-only다. readiness/new의 allocation 전 failure는
mutation-free이고, allocation 후 failure는 새 Exp446 root에만 create-only evidence로 남긴다.
TEST label 접근은 모든 Exp446 command에서 0이어야 한다.
