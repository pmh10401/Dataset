# Exp448 Exp447 TEXT label 경계 수리 실행 안내서

```text
approved_design_sha256 = ed2b6e5e8b8ee63ab41a15e54f679147f7e14aeca601f4d62f694ca104ed3ce4
approved_plan_sha256 = b79b3b7fc43aae7b15e3c7c52e648dd4404d50930f5bd71ce106c244d21fe934
```

## 현재 상태

`official_evaluation_inspected_complete_dashboard_published_owner_only`다. fresh Sol High `ship`
gate 뒤 reviewed commit `2f5229ccd471bba907562b3131d293b46b0b3afc`에서 readiness, evaluate,
bare inspect가 각각 정확히 한 번 exit 0으로 성공했다. Exp448 lifecycle은 끝났고 같은
experiment/root에서 어떤 명령도 다시 실행하지 않는다. 결과는
`descriptive_complete_no_promotion`; champion, pending candidate, promotion 또는 winner는 없다.

## 완료된 선행 gate

1. Exp448 package/tests/launcher가 구현되고 controlled-fork differential proof가 통과한다.
2. candidate HEAD가 clean이고 fresh Sol final review가 `ship`이다.
3. Exp447 audit identity와 absent/non-symlink root, Exp446/Exp404 bytes, DB SHA/inode/nlink,
   Exp448 root absence가 모두 exact match였다.

## 실행된 closed command surface

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py evaluate --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py inspect
```

정확한 순서 readiness 1회 → evaluate 1회 → bare inspect 1회가 모두 exit 0으로 완료됐다.
readiness와 evaluate만 exact lowercase 40-hex reviewed commit을 받고 inspect는 인자를 받지
않는다. `new`, `resume`, `retry`, override, abbreviation은 없으며 이제 어떤 command도 재실행할 수
없다. Exp447 command는 어떤 경우에도 실행하지 않는다.

## 실행 후

Independent immutable-root verification과 truth/map closeout이 완료됐다. 성공했어도 결과는
descriptive-only/non-promoted다. sanitized aggregate `exp448.public-evaluation.v1` snapshot은
`2026-08-12T05:15:11+09:00`에 기존 owner-only 연구 관측소 Sites version `6`으로 게시됐다.
접근은 한 명의 owner와 group 0개로 유지되며 snapshot은 live source connection이 아니다.
Dashboard 상태는 scientific artifact, champion/pending candidate, promotion 또는 retry 권한을 바꾸지 않는다.
