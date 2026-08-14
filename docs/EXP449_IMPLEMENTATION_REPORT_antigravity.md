# Exp449 구현 증거 보고 — 비실행

## 선언

```text
implementation_status = implementation_verified_awaiting_smoke_authorization
smoke_started = false
official_run_started = false
labels_opened = false
official_label_access_count = 0
all_six_fixed_roots = absent
```

이 보고서는 implementation evidence만 기록한다. Exp449 smoke, official 실행, TEST label
access, AP, AUROC, scientific result는 수행하거나 주장하지 않는다.

## 구현 commit 계보

승인 design commit은 `26830622`이다. Exp449 Task1–7 구현 계보는
`327323d4`, `4ad53f25`, `12360279`, `9d48747c`, `8522151d`, `201c6334`,
`1760bcb5`, `8c81e9c7`, `b485ec36`, `016013ae`, `145b35fc`, `ba836ce2`,
`48c12953`, `bc92c8be`, `165deaaa`, `c76bb29d`, `3c9fd2cb`, `d01ff6ed`,
`012d23fe`, `c99e277b`, `527b4978`, `7a41dfd2`, `9293a632`, `c9071b79`,
`815cb98a`, `0d8d4cd2`이다. 마지막 Task7 commit `0d8d4cd2`는 execution authority를
`execution_authority_not_configured`로 fail-close한다.

## Task8 검증 closeout

- Exp449 discovery: `81/81 OK`, 9.025초.
- paper-public-baselines discovery: `79/79 OK`, 6.460초.
- curated paper discovery: 발견 `161`, 실행 `160/160 OK`. 유일한 의도적 제외는
  `test_hard_route_paper.HardRouteTests.test_capable_ten_frozen_db_raw_smoke_freezes_then_compares_exp137_and_exp236_artifacts`다.
  이 test만 frozen production DB raw smoke를 읽으므로 Task8 비실행 경계에서 실행하지
  않았다. 나머지 paper tests의 temporary fixture SQLite와 mock/scoring unit tests는
  기존 test sandbox 안에서만 수행되었다.
- JSON state/map/lock parse, HTML embedded payload equality, 실제 node/edge/flow referential
  integrity (`52` nodes, `78` edges, `4` flows), `sha256-v1` module fingerprints, Exp448
  design-commit blob inventory equality, `git diff --check`, 및 `py_compile`은 closeout
  verification에 포함된다.

검증 당시 source SHA256은 config `1959f2f06542924605561d9b30c3bf5c9127eb8dc4bceb41f65510e8e797b337`,
contracts `6a6f80717ecc79881e4e9f973e516c133aace63e9cd2e8105b8da3ef30c75ac4`,
runner `6a8b6f5636cfad82f970611947e5057a77fbcf7b628ce42d2893812279f2a525`,
artifacts `279bcb4c1d010f289aab728837975ee89824cbbc61a5167adb575d39bbefefde`,
operations `cbab8799c09758ac9ebccf415dac83fb263bc05a0af4bdf46957f9aaf2657d6a`,
validation `7818fe9de5fdf2ff620dcbbe9605c4ab43dbb7a2d9a5e4920916959581f6990d`,
verifier `dfa7e739cc3f757b3c933c889840bc3a8111a5ead8e573673c3485057ccfad01`이다.
Current map SHA256은 JSON `d548b01079babd0e3f23f20888e69227d8e15628c46243eb384cc3edf758b2dd`,
HTML `b0f2e2e24aba3ae94fc6c4be36dc14a598b61bf586a0bfa194481410fc805deb`이며, lock의
sha256-v1 current-tree fingerprints는 closeout에서 별도로 exact 검증한다.
RESEARCH_STATE SHA256은 `59b6021d02bf5f7d0833b755f69a5bc939abe4232d0ae952688639f80bb35888`이다.
이 파일의 작성이나 self-hash는 execution authority가 아니다.

## 고정 runtime facts

pin된 capable interpreter는
`/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`이고
Python `3.14.6`, NumPy `2.3.5`, pycatch22 `0.5.0`, aeon `1.5.0`, PyTorch `2.13.0`,
SciPy `1.17.1`, scikit-learn `1.8.0`이다. MLX와 mlx-metal은 `absent`다. 호스트는
macOS `26.5.2` arm64, logical CPU `12`; PyTorch MPS는 built/available 모두 `true`다.
Task8에서 이 facts는 non-execution verification으로만 읽고, accelerated path는 canonical로
선택하지 않는다. canonical backend는 CPU이며 worker는 `11` (`max(1, 12 - 1)`)이다.
PyTorch, MLX, MPS의 availability,
transferㆍcompilationㆍsynchronization overhead와 parity/speed gate는 future authority
review의 readiness record에 남겨야 한다. MLX candidate는 same-input/seed, finite output,
candidate index identity, score delta `<=1e-5`, end-to-end `>=1.1x`를 만족하기 전에는
CPU를 대체하지 않는다.

## 비실행 경계와 알려진 제한

six fixed scientific/operations roots는 absent이며 생성하지 않았다. `labels_opened=false`,
`smoke_started=false`, `official_run_started=false`, `official_label_access_count=0`이다.
Task7에서 pre-existing broad paper discovery가 frozen raw DB smoke 값을 한 번 읽은 사실은
공개한다. 그것은 Exp449 run, root 생성, label access, score 실행 또는 mutation이 아니며,
Task8은 그 discovery를 재실행하지 않았다.

현재 execution authority가 구성되지 않았으므로 다음 action은 정확히
`design_and_review_trusted_smoke_a_execution_authority_then_request_user_approval`이다.
그 전에 smoke/official command, self-hash artifact 기반 enablement, DB/scoring/labels는
허용되지 않는다.
