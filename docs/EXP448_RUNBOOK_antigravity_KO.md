# Exp448 Exp447 TEXT label 경계 수리 실행 안내서

```text
approved_design_sha256 = 42166c0a43365e7e3a92ab0ed73a8a208ee24630ef070cc519b77f8491cdee76
approved_plan_sha256 = 7f9f646f55eb13791d94f094013b0149b247c127e63f94625e7a261a606db0c5
```

## 현재 상태

`implementation_not_started_no_official_authorization`이다. 이 문서는 미래의 closed
lifecycle을 고정할 뿐, 지금 TEST label, DB, Exp446 publication, Exp447 lifecycle 또는
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
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py inspect --reviewed-commit REVIEWED_COMMIT
```

정확한 순서는 readiness 1회 → evaluate 1회 → inspect 1회다. `new`, `resume`, `retry`,
override, abbreviation은 없다. 하나라도 nonzero면 Exp448을 영구 중단하며 같은 root/experiment를
다시 실행하지 않는다. Exp447 command는 어떤 경우에도 실행하지 않는다.

## 실행 후

independent verification과 truth/map closeout을 먼저 한다. 성공했을 때도 결과는
descriptive-only/non-promoted다. private dashboard는 clean site repository와 sanitized aggregate
payload가 모두 있을 때만 별도로 진행하며, dashboard failure는 scientific artifact나 retry 권한을
바꾸지 않는다.
