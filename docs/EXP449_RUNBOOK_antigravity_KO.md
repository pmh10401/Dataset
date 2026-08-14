# Exp449 실행 안내서 — 현재 실행 비인가

## 현재 상태와 절대 경계

현재 상태는 `implementation_verified_awaiting_smoke_authorization`이다. 모든 여섯
root는 없어야 하며, smoke와 official은 아직 시작되지 않았다. 현재 구현의 execution
authority는 정확히 `execution_authority_not_configured`다. 따라서 아래의
`run-smoke-a`, `run-smoke-b`, `run-official`은 **지금 실행하지 않는다**. 현재 호출하면
root를 만들거나 DB/scoring/label 경로를 열기 전에 fail-closed로 그 상태를 반환한다.

자기 자신을 해시한 artifact, 작업 디렉터리의 임의 파일, readiness 출력, 또는 기존
root는 trusted execution authority가 아니다. 실행하려면 먼저 trusted authority digest
또는 signature를 pin하는 **새로운 별도 code/config 변경**을 작성하고 독립 review를
통과시킨 뒤, 그 변경에 대해 사용자의 명시적 승인을 받아야 한다. 이 요구는 A, B,
official 각각의 실행 승인보다 먼저 충족해야 한다.

## 고정 interpreter와 read-only 명령

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py status

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py verify-proposal --target smoke-a

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py verify-proposal --target smoke-b

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py verify-proposal --target official
```

`REVIEWED_COMMIT`은 clean reviewed HEAD의 정확한 lowercase 40-hex여야 한다. `status`와
`verify-proposal`은 read-only이며 absent root만 보고한다. `readiness`는 later authority
review에서만, clean reviewed commit에 대해 실행한다. 현재 no-execution Task8은 database,
scoring, label 접근을 하지 않으므로 readiness를 실행 증거로 사용하지 않았다.

Readiness가 훗날 authority review에서 허용될 때에는 scoring을 호출하지 않지만, read-only
catalog identity 확인을 위해 production DB를 열 수 있다. 그 record는 full normalized
installed-distribution inventory (algorithm/schema/canonical SHA256), explicit 30-file scientific
closure, `lineage_lock_paper.json` SHA256, 그리고 **codemap lock** SHA256를 bind한다.
`codemap_sha256`은 JSON이 아니라 `docs/codemap/codemap_exp449_antigravity.lock` bytes의
SHA256이다. `seeds_and_constants`는 `exp449-seeds-and-constants-antigravity.v1`로 all
seed/alpha/threshold/routing/top-k/policy/preprocessing/count/tolerance/two-wave/schema/verifier
constants를 source-derived manifest로 bind한다. 이 실행 안내의 문구나 self-hash는 authority가
아니다.

## 미래 명령 — 현재 모두 fail-closed

```bash
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py run-smoke-a --reviewed-commit REVIEWED_COMMIT

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py run-smoke-b --reviewed-commit REVIEWED_COMMIT

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp449_reproducible_rebaseline_antigravity.py run-official --reviewed-commit REVIEWED_COMMIT
```

세 명령은 현재 `execution_authority_not_configured`로 실패해야 한다. self-hash artifact는
이를 해제하지 않는다. 새 trusted-authority code/config이 pinㆍreview된 뒤에도 자동 실행은
없다. 사용자는 Smoke A에 별도 승인하고 A complete/independent verification이 끝난 뒤
Smoke B에 별도 승인한다. A/B는 독립 roots/complete checkpoints이고 exact coordinate equality
및 all-policy score delta `<=1e-12`가 필요하다. 그 후에만 1,117 dataset official에 대한
별도 승인을 요청한다.

공식 실행은 최대 두 wave의 Task4/5 recovery만 허용한다. full catalog의 정확한 1,117
bijection이 없으면 Task6/7/Candidate C freeze로 진행할 수 없다. 어느 command도 TEST label
access나 post-freeze evaluation을 허용하지 않는다.
