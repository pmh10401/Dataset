# Exp448 Exp447 TEXT label 경계 수리 구현 계획

> **For agentic workers:** execute one task at a time; preserve unrelated concurrent changes.

**목표:** Exp447의 불변 실패 사실을 보존하면서, SQLite TEST label의 exact TEXT 저장
형식을 유일한 새 Exp448 label boundary에서 엄격히 0/1 정수로 변환해 독립 평가한다.

**구조:** Exp448은 Exp447의 열 개 모듈과 launcher를 controlled fork로 복제한다. identity,
failure audit 검증, absent predecessor root, DB schema metadata, exact TEXT converter만
allowlist delta다. prediction과 과학 계산은 바뀌지 않는다.

## 전역 제약

- design SHA는 `42166c0a43365e7e3a92ab0ed73a8a208ee24630ef070cc519b77f8491cdee76`이고
  경로는 `docs/superpowers/specs/2026-08-12-exp448-exp447-text-label-boundary-repair-antigravity-design.md`다.
- identity/schema/kind/run identity/root/package/launcher/interpreter는 승인 설계의 exact
  값만 쓴다. 새 파일명은 `antigravity`를 포함하고 `__init__.py`는 없다.
- Exp447 reviewed failure truth commit `7e85d5c7948f8592db80f958d746f31138c67bfb`와
  official attempt commit `33e7dc9c18c179a64bc0020b0bb939c5e4caa394`을 보존한다.
  Exp447 readiness/evaluate/inspect는 절대 실행하지 않는다.
- Exp447 sibling audit의 path/SHA/inode/nlink/schema/label_snapshot/1-0-0/closed/absent/
  `ValueError: TEST label row is invalid`과 Exp447 absent non-symlink root는 preflight다.
- Exp446/Exp404/DB fixed hashes, dataset 1117, rows 195900, 30/26/4 lane, baseline and
  source30 confusion, bootstrap PCG64 seed/draw/SHA는 설계의 exact 값에서 바꾸지 않는다.
- label은 Exp447 audit verify, two equal Exp446 captures, Exp404 validate 후에만 1117 ordered
  query/195900 row로 연다. exact `int` index와 exact `str` `'0'|'1'`만 받고,
  `0 if raw_label == '0' else 1`으로 변환한다. 그 외 type/value는 거절하고 raw string은
  evaluator/artifact에 전달하지 않는다.
- score source, prediction, candidate, threshold, source/lane/budget/method/backend/training,
  bootstrap, diagnostics, leaves/order, create-only/replay/governance를 바꾸지 않는다.
- CLI는 `readiness --reviewed-commit`, `evaluate --reviewed-commit`, `inspect`만 제공한다.
  any official nonzero는 permanent stop이며 same-experiment retry가 없다.
- success는 항상 `descriptive_complete_no_promotion`, success criterion은 null이다. winner,
  champion, pending candidate, alert, promotion, reselection은 만들지 않는다.
- evaluator에는 network/dashboard import가 없고, readiness/inspect에는 label-capable SQLite가
  없다. dashboard는 Exp448 truth closeout 성공 뒤 clean private site에서만 aggregate-only로
  쓴다; failure는 scientific artifact/retry를 바꾸지 않는다.

### Task 1: Governance, preregistration, ledgers, and map

Create the approved plan, preregistration, operator runbook, not-started implementation report,
planned catalog/state/champion history records, and the Exp448 versioned code-map triplet. Preserve
the Exp447 immutable failure record and `pending_candidate`. Do not add Exp448 to FAILURE_LEDGER.
The map has at most 20 nodes and explicitly records that no Exp448 source edge exists yet.

### Task 2: Controlled-fork foundations and differential proof

Create the allowed Exp448 config/contracts/artifacts/bootstrap/CLI foundations plus paired tests.
Implement canonical profile equality ignoring identity keys, normalized AST and exact constant checks.
Only the approved A--E delta allowlist may differ from Exp447. Prove synthetic 1,117-dataset TEXT
runtime equals Exp447 integer pure-evaluator output when identity/hash fields are ignored.

### Task 3: Predecessor, source, readiness, and converter

Implement Exp447 audit/absent-root verification, two equal Exp446 captures, Exp404 validation and
immutable DB identity checks. Admit label capability only after those gates. Verify schema metadata
`instance_index INTEGER`, `label TEXT`; apply the exact converter and rejection matrix. Do not inspect
real label values in unit tests.

### Task 4: Evaluator, replay, and failure-audit parity

Implement post-freeze evaluator/replay with the exact four method order, eleven leaves, diagnostics,
bootstrap and create-only behavior. Preserve label close, source recheck, error audit and immutable
replay semantics; raw TEXT does not leave its conversion boundary.

### Task 5: Closed CLI/launcher, complete verification, and fresh review

Implement only the closed three-command CLI and launcher. Run controlled-fork differential tests,
Exp448 tests, inherited relevant regression, code-map validation, and a fresh Sol final review. No
official DB or TEST-label action occurs in this task.

### Task 6: One-shot official truth closeout and conditional dashboard

After clean candidate and fresh Sol `ship`, run exactly once in order:
`readiness` → `evaluate` → `inspect` → independent verification → truth/map → final review.
Any nonzero permanently stops Exp448. Only complete immutable truth plus label-free inspect and a
clean owner-only site allow a separate sanitized `exp448.public-evaluation.v1` dashboard action.
