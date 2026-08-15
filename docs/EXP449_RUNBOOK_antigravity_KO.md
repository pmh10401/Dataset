# Exp449 Smoke A 실행 안내서 — grant 승인 대기

## 현재 상태와 신뢰 한계

현재 상태는 `implementation_ready_awaiting_grant_authorization`이다. Smoke A lifecycle
구현은 review 대상 `I`에만 있고, grant/digest와 실행 승인은 아직 없다. 고정 digest는
서명이나 OS 계정 격리를 대체하지 않는다. 사용자가 **정확한 authorization commit SHA**를
별도로 승인할 때만 그 한 번의 Git delta와 grant bytes를 신뢰한다. self-hash, readiness
출력, 작업 디렉터리의 파일, 또는 기존 root는 authority가 아니다.

현재 implementation-review 시점에는 여섯 production root, Smoke A claim, 그리고 다음
두 authority JSON이 모두 없다.

- `docs/authorities/smoke_a_execution_grant_antigravity.json`
- `docs/authorities/smoke_a_execution_grant_digest_antigravity.json`

## 고정 경로와 실행 profile

```text
Smoke A scientific: /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814
Smoke A operations: /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814_ops_antigravity
Smoke B scientific: /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_b_antigravity_20260814
Smoke B operations: /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_b_antigravity_20260814_ops_antigravity
Official scientific: /Users/mac/Documents/Dataset/paper_proposal_exp449_official_antigravity_20260814
Official operations: /Users/mac/Documents/Dataset/paper_proposal_exp449_official_antigravity_20260814_ops_antigravity
Smoke A claim: /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814_claim_antigravity.lock
```

CPU가 canonical backend다. detected logical CPU `12`에서 하나를 보존하므로 dataset
worker는 `11` (`max(1, logical_cpu_count - 1)`)이다. PyTorch/MPS와 MLX는 이 lifecycle
wrapper에서 canonical path가 아니며, MLX가 CPU를 대체하려면 동일 input/seed parity,
finite output, preregistered alpha의 candidate index 동일성, tolerance, 그리고 transfer/
synchronization을 포함한 `>=1.1x` speed benchmark를 별도 grant-ready evidence로 통과해야
한다.

## 정확한 비실행 승인 순서

1. authority JSON 없이 clean implementation commit `I`를 독립 Sol review에서 `SHIP`으로
   확인한다. 이 단계의 `run-smoke-a`와 `resume-smoke-a`는 실행하지 않는다.
2. 별도 grant-preparation 단계가 clean `I`에서 read-only readiness facts를 대조하고 정확히
   두 canonical authority JSON만 create-only로 만든다. scoring, label access, root/claim/
   checkpoint/log 생성은 허용되지 않는다.
3. 그 두 파일만 추가한 direct child authorization commit `A`를 만든다. `parent(A) == I`와
   exact two-file delta, canonical byte digest, closure/lock 불변성을 독립 Sol review에서
   다시 `SHIP`으로 확인한다.
4. 사용자에게 exact `I`, exact `A`, grant SHA-256, roots, claim, CPU `12→11`, backend
   `cpu`, 그리고 launch command를 제시한다. 사용자가 정확히 `Smoke A 실행 승인`이라고
   승인하기 전에는 실행하지 않는다.
5. 그 이후의 별도 execution turn에서만 다음 fixed-path command를 쓸 수 있다.

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py \
  run-smoke-a --approved-authority-commit EXACT_A
```

`I` 또는 `A` review가 `FIX-FIRST`이면 새 implementation `I`와 새 direct-child `A`를
만든다. grant를 수정 commit으로 누적하거나 자동 실행하지 않는다.

## read-only 관찰과 재개

아래는 execution을 시작하지 않는 fixed-path 관찰 명령이다. `readiness`는 production DB를
read-only로 열 수 있으므로 ordinary implementation verification에서는 호출하지 않고,
별도 grant-preparation review에서만 사용한다.

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py status

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py verify-proposal --target smoke-a
```

승인 후 fresh run은 verified `A→I` snapshot을 다시 확인하고, absent fixed roots/claim을
lstat한 뒤 one-shot claim과 operations consumption receipt를 만들며 최대 두 Task4/5 wave만
수행한다. progress JSONL append/fsync가 실패하면 다음 dataset scheduling 전에 멈춘다.
`status`는 scorer나 DB를 열지 않고 claim, receipt, bounded progress tail, checkpoint,
manifest, terminal metadata만 읽는다.

승인 후 재개는 다음 명령만 사용하며 새 root를 만들지 않는다.

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py \
  resume-smoke-a --approved-authority-commit EXACT_A
```

재개는 같은 `A`, `I`, grant/digest, consumed receipt, context/catalog/database/closure/
environment/backend/worker, inactive claim inode, checkpoint/attempt ledger만 허용한다.
Verified success는 재계산하지 않으며 drift나 mismatch를 repair/migrate하지 않고 거부한다.
freeze 완료 뒤 complete terminal 전에 중단됐다면 scorer-free verification과 terminal
publication만 복구한다.

두 wave 뒤 exact catalog completion이 아니면 create-only
`proposal_score_incomplete` terminal을 기록하고 Task6/7, Candidate C, freeze를 실행하지
않는다. incomplete terminal은 scientific proposal artifact나 complete terminal을 의미하지
않는다.

## 계속 차단된 범위

Smoke B, official 1,117-dataset run, TEST label module/import/read, AP/AUROC/evaluation,
그리고 모든 non-A target은 이 approval sequence 밖이며 계속 blocked다. Smoke A가 complete
and independently verified 된 뒤에도 Smoke B와 official에는 각각 별도 authorization and
user approval이 필요하다.
