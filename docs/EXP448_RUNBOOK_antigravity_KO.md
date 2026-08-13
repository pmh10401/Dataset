# Exp448 Exp447 TEXT label 경계 수리 실행 안내서

```text
approved_design_sha256 = ed2b6e5e8b8ee63ab41a15e54f679147f7e14aeca601f4d62f694ca104ed3ce4
approved_plan_sha256 = b79b3b7fc43aae7b15e3c7c52e648dd4404d50930f5bd71ce106c244d21fe934
```

## 현재 상태

`implementation_verified_awaiting_official_readiness`이며 fresh Sol High `ship` verdict를
기다린다. 그 verdict 전에는 TEST label, DB, Exp446 publication, Exp447 lifecycle 또는
dashboard를 열도록 허가하지 않는다.

## 선행 gate

1. Exp448 package/tests/launcher가 구현되고 controlled-fork differential proof가 통과한다.
2. candidate HEAD가 clean이고 fresh Sol final review가 `ship`이다.
3. Exp447 audit identity와 absent/non-symlink root, Exp446/Exp404 bytes, DB SHA/inode/nlink,
   Exp448 root absence가 모두 exact match다.

## 허용될 수 있는 명령

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py evaluate --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py inspect
```

정확한 순서는 readiness 1회 → evaluate 1회 → bare inspect 1회다. readiness와 evaluate만
exact lowercase 40-hex reviewed commit을 받고 inspect는 인자를 받지 않는다. `new`, `resume`, `retry`,
override, abbreviation은 없다. 하나라도 nonzero면 Exp448을 영구 중단하며 같은 root/experiment를
다시 실행하지 않는다. Exp447 command는 어떤 경우에도 실행하지 않는다.

## 실행 후

independent verification과 truth/map closeout을 먼저 한다. 성공했을 때도 결과는
descriptive-only/non-promoted다. private dashboard는 clean site repository와 sanitized aggregate
payload가 모두 있을 때만 별도로 진행하며, dashboard failure는 scientific artifact나 retry 권한을
바꾸지 않는다.

## Paper r4 operations surface (separate from official Exp448)

Implementation verification of `scratch/manage_paper_r4_proposal_antigravity.py` does
**not** authorize creation of the official r4 root, the smoke root, or the ops
root. Do not run `smoke` or `launch` until a later explicit user approval.

Fixed interpreter and paths:

```text
interpreter = /Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python
database    = /Users/mac/Documents/Dataset/univariate_ts.db
r3          = /Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813
r4          = /Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813
ops         = /Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity
smoke       = /Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813
failed_smoke = /Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813
failed_ops   = /Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813_ops_antigravity
```

The failed 20260813 smoke/ops pair is immutable read-only evidence. Do not migrate,
reuse, edit, or delete it. `check` requires those roots present and the v2 smoke/ops
roots absent. Official r4 remains unauthorized.

Read-only check (allowed after the operations code is committed; creates no roots).
The command must be invoked with that exact lexical venv path so `sys.executable`
and `sys.prefix` stay the venv identity. Authority records `executable`,
`executable_realpath`, `python_prefix`, `python_base_prefix`,
`python_identity_sha256`, and the frozen scientific distribution inventory
(exact version or `absent` for numpy, scipy, scikit-learn, aeon, pandas,
numba, joblib, scikit-image, pycatch22, torch, mlx, mlx-metal) plus its
canonical SHA. `check` also rejects a symlinked smoke-ops derived root.
Worker/pipeline commands keep the same venv path and do not resolve it to
base Python.

```bash
/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python \
  scratch/manage_paper_r4_proposal_antigravity.py check --reviewed-commit REVIEWED_COMMIT
```

Future prelaunch smoke, still unauthorized by this implementation. The v2
smoke authority is the frozen sorted 23-name set and includes
`MelbournePedestrian_normal_10` so the official TEST-only union member is
exercised on the real Task4+5 path. The first
completed invocation is non-authorizing (`smoke_confirmation_required`) and
does not publish `smoke_result`. A second same-authority verification-only
invocation is required before create-only authorization. An incomplete first
wave plus a later completing resume still requires one additional
verification-only invocation. The CLI exits 0 only for top-level
`smoke_verified`; `smoke_confirmation_required` and `smoke_incomplete` are
nonzero.

```bash
/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python \
  scratch/manage_paper_r4_proposal_antigravity.py smoke --reviewed-commit REVIEWED_COMMIT
```

Operator observation after a future authorized process exists:

```bash
/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python \
  scratch/manage_paper_r4_proposal_antigravity.py status --reviewed-commit REVIEWED_COMMIT
tail -f /Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity/progress_antigravity.log
```

Stop and resume. There is no automatic retry. After a `proposal_score_incomplete`
wave, analyze the full failed/pending batch first. Resume requires another
explicit approval and the identical stored authority. It also accepts only
verified pre-genesis shapes: an empty r4 root, or a root containing only an
empty canonical journal directory; the runner then initializes. Every other
partial or polluted shape is rejected. Checkpoints are never migrated into a
new closure/config.

```bash
/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python \
  scratch/manage_paper_r4_proposal_antigravity.py stop --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python \
  scratch/manage_paper_r4_proposal_antigravity.py resume --reviewed-commit REVIEWED_COMMIT
```

Final post-review launch remains blocked until a new explicit user approval.
The future command, not to be run now, is:

```bash
/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python \
  scratch/manage_paper_r4_proposal_antigravity.py launch --reviewed-commit REVIEWED_COMMIT
```
