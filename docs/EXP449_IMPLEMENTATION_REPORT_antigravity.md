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

Task8 documentation/state/test commit은 `2c9646b9d1bcf373a7f7a0b48f9a2f97801b9c69`
(`docs(exp449): preregister proposal readiness`)이며, 후속 map refresh는 이 clean
reviewed commit을 source로 bind한다.

## Task8 final-review fix closeout

- 이 final-review fix는 full scientific closure를 명시적인 정렬 30 production-path SHA256
  mapping으로 교체했다. 범위는 Exp449 config/contracts/operations/runner/artifacts/validation/
  verifier, paper config/data/contracts/catalog/calibration, Task4/5 source scoring/conformal/
  preprocessing/MiniRocket/legacy route, hard route, Task6/Task7, routing/proposal composition,
  typed serialization/freeze, acceleration이다.
- `lineage_lock_paper.json` SHA256과 actual Exp449 codemap **lock** SHA256을 함께 bind한다.
  context `codemap_sha256`은 lock SHA256이며 codemap JSON SHA256은 별도 진단 값이다.
- installed distributions의 normalized name/exact-version full inventory와 canonical-byte SHA256
  (schema/algorithm 포함)을 readiness facts와 every Exp449Context environment에 bind한다.
  기존 exact pycatch22/aeon/Hydra/torch preflight는 계속 fail-closed다.
- `exp449-seeds-and-constants-antigravity.v1`은 source-derived seed, alpha/threshold,
  route/promotion/top-k/rate, policy/order, preprocessing, counts, tolerances, two-wave,
  typed shard/checkpoint/manifest/parity/acceleration schema와 root-snapshot verifier를 bind한다.
  source-hash-only constant 예외는 없다.
- Exp449 discovery: `87/87 OK`를 두 번 연속 실행했다 (10.200초, 10.162초). package-inventory/context
  regression은 fresh interpreter 50회 isolated repeat에서 `50/50 OK`였다. root inspection은 exact
  historical-root/component authority만 거부하며 benign `r4`/`exp448` substring temporary roots는 허용한다.
  Codemap 27 Exp449 node의 Python module callers는 independent AST import graph로, logical/external callers와
  coverage tokens는 explicit verifier table로 확인했다. paper-public-baselines discovery: `79/79 OK`, 6.187초.
- curated paper discovery: 발견 `161`, 실행 `160/160 OK`. 유일한 의도적 제외는
  `test_hard_route_paper.HardRouteTests.test_capable_ten_frozen_db_raw_smoke_freezes_then_compares_exp137_and_exp236_artifacts`다.
  이 test만 frozen production DB raw smoke를 읽으므로 Task8 비실행 경계에서 실행하지
  않았다. 나머지 paper tests의 temporary fixture SQLite와 mock/scoring unit tests는
  기존 test sandbox 안에서만 수행되었다.
- JSON state/map/lock parse, HTML embedded payload equality, 실제 node/edge/flow referential
  integrity (`52` nodes, `78` edges, `4` flows), `sha256-v1` module fingerprints, Exp448
  design-commit blob inventory equality, `git diff --check`, 및 `py_compile`은 closeout
  verification에 포함된다.

Final source/map SHA256, test count, HTML embedded-map equality와 lock fingerprint
recalculation은 fresh commands로 검증했다. complete source SHA256 mapping, paper lineage-lock
SHA256, codemap JSON diagnostic SHA256, 그리고 actual codemap-lock SHA256은 readiness context가
canonical JSON으로 bind한다. state flags/status와
`execution_authority_not_configured`는 변경하지 않았다. 이 파일의 작성이나 self-hash는
execution authority가 아니다.

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
