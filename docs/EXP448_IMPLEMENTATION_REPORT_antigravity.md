# Exp448 구현 보고

```text
approved_design_sha256 = ed2b6e5e8b8ee63ab41a15e54f679147f7e14aeca601f4d62f694ca104ed3ce4
approved_plan_sha256 = b79b3b7fc43aae7b15e3c7c52e648dd4404d50930f5bd71ce106c244d21fe934
```

```text
implementation_status = official_evaluation_inspected_complete_dashboard_published_owner_only
official_readiness = true
official_evaluate = true
official_inspect = true
test_labels_opened = true
official_label_access_count = 1117
evaluation_sha256 = a58a74f69dcc13a64efb2a679ee7e93eca40ac8c5db87b29b12ed11ea068b21c
artifact_sha256 = ea9856c7de039697ed4f352864def0fde1c0d336714e68c9b0cd586d2c796f05
decision_sha256 = 54ac58d89367ec5396dafdb180777f787384ffbd21ead1a96764bfe627bbddd5
champion_sha256 = null
promotion_sha256 = null
```

Exp448은 승인된 engineering-only controlled fork로서 Exp447의 immutable TEXT-label
boundary failure를 보존하고 converter 경계만 수리했다. reviewed commit
`2f5229ccd471bba907562b3131d293b46b0b3afc`에서 official closed lifecycle은 정확히
readiness 1회 (exit 0), evaluate 1회 (exit 0), bare inspect 1회 (exit 0)로 종료됐다.
결과는 `descriptive_complete_no_promotion`이며 champion, pending candidate, winner,
promotion 또는 operational alert는 만들지 않았다. 별도로 governed된 aggregate-only dashboard는
기존 owner-only 연구 관측소에 검증된 snapshot으로 게시됐으며 과학 결과나 재평가 권한을 바꾸지 않는다.

## Owner-only dashboard publication

- Snapshot schema is `exp448.public-evaluation.v1`; it was published at
  `2026-08-12T05:15:11+09:00` to the existing owner-only research observatory.
- The published surface contains only the exact 12-field completed status, four aggregate
  confusion/precision/recall/F1/FPR rows, aggregate transition counts `15/15/0/0`, and three
  paired-bootstrap delta/interval rows.
- Raw TEST labels, indices, dataset names or lists, per-dataset rows, hashes, local paths, logs,
  database/checkpoint details, credentials, winner, promotion, and retry controls are absent.
- The site source commit is `500ccbe12b07475e51e8cdcbfa8619d9fbdefddb`, Sites version is `6`,
  and access remains custom owner-only with one allowed account and no groups. The displayed data is
  a published snapshot, not a live source connection.

## Official truth closeout

- Immutable root is `/Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812`:
  regular directory, inode `19850406`, with exactly `11` regular single-link leaves. Its artifact map
  SHA256 is `ea9856c7de039697ed4f352864def0fde1c0d336714e68c9b0cd586d2c796f05`; evaluation and
  decision SHA256 are respectively `a58a74f69dcc13a64efb2a679ee7e93eca40ac8c5db87b29b12ed11ea068b21c`
  and `54ac58d89367ec5396dafdb180777f787384ffbd21ead1a96764bfe627bbddd5`.
- The successful evaluate invocation opened one closure-private TEST-label session, completed the
  fixed `1117` ordered queries, converted and consumed `195900` labels, and closed the session.
  Readiness had label-access count `0`; no failure audit exists for the successful Exp448 evaluation.
- Stored immutable evidence was independently rechecked without invoking any official command:
  strict duplicate-key parsing, all 11 leaf names and hashes, schema/counts, method order, confusion,
  diagnostic leaves, and bootstrap comparisons agree with the recorded evaluate/inspect truth.
- Aggregate results: baseline confusion `[1915, 303, 2003, 191679]`; source-30
  `[1930, 318, 1988, 191664]`; stable-26 `[1930, 314, 1988, 191668]`; unstable-4
  `[1915, 307, 2003, 191675]`. Source-30 has `30` transitions: `15` rescued and `15` added false
  alarms (`11` stable plus `4` unstable). The primary stable-26 diagnostic leaves are `15` rescued,
  `11` false alarms, `0` regressed, and `0` strong regressions.
- Primary stable-26 micro/macro F1 is `0.6264199935086011` / `0.6214265552464396`.
  Stable-minus-baseline micro delta is `0.0022348566115998336` (95% CI
  `[0.0004830992298546444, 0.004215589161616776]`) and macro delta is
  `0.012444046553267696` (95% CI `[0.006624888093106462, 0.019158460161145974]`).
  Stable-minus-source-30 micro delta is `0.0004063704142125424` (95% CI
  `[9.289898787021578e-05, 0.0008597004339725767]`); macro delta and CI are `0`.
- The DB remains SHA256 `8458e678...ef3c`, inode `1048138`, nlink `1`; Exp447's immutable failure
  audit remains SHA256 `9f8f7d3d...2b97`, inode `19812443`, nlink `1`. Exp447 is not rerun.

## Task5 closed command evidence

- Functional CLI commit: `cb46a6bcc233d0cfefe226e1c3054dfaca5081a8`
  (`feat(exp448): add closed evaluation CLI`). The public surface is only
  `readiness --reviewed-commit LOWER40HEX`, `evaluate --reviewed-commit LOWER40HEX`,
  and bare `inspect`; parser abbreviation and lifecycle/override commands are rejected.
- The focused CLI suite passed `6/6`. It exercises exact commit parsing, forbidden override
  matrix, clean-HEAD-before-evaluator ordering, source-free fresh-process inspect, explicit
  JSON serialization, and the outside-CWD launcher’s seven `=1` numerical-thread pins.
- Full Exp448 discovery completed twice: `62` tests in `81.123s` and `62` tests in
  `81.402s`; both are `OK`. The first captured log has a post-observation `exit=0` sentinel
  because the harness detached its wrapper before its trailing write. The second recovery run
  retained its unified session and exact `exit=0` sentinel. This is engineering verification
  only, never an official command.
- Sequential inherited discovery was `66` Exp447 tests in `76.974s`, `50` Exp446 tests in
  `9.456s`, `39` Exp445 tests in `1.241s`, and `206` Exp444 tests in `316.121s`, all `OK`.
  The exact Exp447 `7e85d5c...` package-and-launcher byte guard and `git diff --check` both
  exited zero.

## Pre-official implementation authority facts

- Python/NumPy are `3.14.6` / `2.4.6`; design, plan, preregistration, and runbook SHA256 values
  remain respectively `ed2b6e5e...3ce4`, `b79b3b7f...fe934`, `af2b8b10...c612f`, and
  `474d4133...c51ca`.
- The DB SHA256 is `8458e678...ef3c`, inode `1048138`, nlink `1`. The immutable Exp447 audit
  SHA256 is `9f8f7d3d...2b97`, inode `19812443`, nlink `1`.
- Descriptor-pinned Exp446 identity remains `1117` datasets with manifest
  `bbaf9506...c11` and prediction `602ed1b...6100`; Exp404 remains `2218` Hard coordinates.
  Before the lifecycle, Exp447 and Exp448 official roots were absent/non-symlink; the completed
  Exp448 root is now the immutable regular directory recorded above.
- Before the closed official lifecycle, no real TEST label value was queried, no official command was
  invoked, no dashboard/network module was imported by the no-label command paths, and no root/result/
  audit was created. Exp447 audit truth and champion `pending_candidate` remained unchanged. The later
  dashboard publication used only the already-closed sanitized aggregate snapshot.
