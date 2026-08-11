# Exp446 Exp445 publication materialization 복구 실행 안내서

## 고정 경로와 preflight

```text
RUN_ROOT=/Users/mac/Documents/Dataset/exp446_exp445_publication_materialization_repair_antigravity_20260811
ATTEMPT=$RUN_ROOT/attempts_antigravity/attempt_0000_antigravity
PREDECESSOR=/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810/attempts_antigravity/attempt_0000_antigravity
```

Exp446는 아직 `preregistered_implementation_pending`이다. 이 문서는 공식 lifecycle
실행 권한이 아니다. product implementation, focused/inherited/full regression, clean-head
review와 별도 final review가 모두 완료되기 전에는 아래 명령을 실행하지 않는다.

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
