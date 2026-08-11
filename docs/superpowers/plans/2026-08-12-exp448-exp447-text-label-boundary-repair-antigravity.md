# Exp448 Exp447 TEXT label 경계 수리 구현 계획

> **For agentic workers:** 한 task씩만 수행한다. 각 functional commit 뒤 map-only child
> commit과 fresh reviewer verdict를 남기며, 다른 작업자의 변경을 되돌리지 않는다.

## 목표와 구조

Exp448은 Exp447 evaluate가 SQLite TEST label의 exact TEXT를 정수로 잘못 요구해 실패한
경계만 수리하는 engineering-only controlled fork다. prediction, candidate, score, threshold,
source, lane, budget, method, backend, training과 수학은 바꾸지 않는다. Exp447은 immutable
failure truth로 보존하며 절대 다시 실행하지 않는다.

Exp448 package는 `experiments/exp448_exp447_text_label_boundary_repair_antigravity/`, launcher는
`scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py`다. `__init__.py`는 만들지
않으며 새 파일명은 모두 `antigravity`를 포함한다.

## 모든 task의 고정 경계

- Approved design은 `docs/superpowers/specs/2026-08-12-exp448-exp447-text-label-boundary-repair-antigravity-design.md`, SHA
  `42166c0a43365e7e3a92ab0ed73a8a208ee24630ef070cc519b77f8491cdee76`다.
- Exp447 truth commit `7e85d5c7948f8592db80f958d746f31138c67bfb`, official attempt commit
  `33e7dc9c18c179a64bc0020b0bb939c5e4caa394`, sibling audit, and absent root are immutable
  predecessor gates. Exp447 readiness/evaluate/inspect는 영구 금지다.
- Exp446/Exp404/DB fixed hashes, 1,117 datasets, 195,900 rows, 30/26/4 lanes, baseline/source30
  confusion, PCG64 seed/draw SHA, four method order, eleven leaves, diagnostics, bootstrap,
  create-only/replay/governance는 design의 exact values를 그대로 쓴다.
- label은 audit verify → two equal Exp446 captures → Exp404 validate 뒤에만 연다. index는
  exact `int`, raw label은 exact `str` `'0'|'1'`만 받고 `0 if raw_label == '0' else 1`로
  즉시 변환한다. int/bool/float/bytes/whitespace/null/other string은 거절하며 raw string은
  evaluator/artifact로 전달하지 않는다.
- 결과는 항상 `descriptive_complete_no_promotion`, criterion null이다. winner, champion,
  pending candidate, alert, promotion, reselection은 만들지 않는다.
- evaluator는 network/dashboard import가 없고 readiness/inspect에는 label-capable SQLite가
  없다. MLX/MPS/workers/training을 새로 쓰지 않는다.
- 각 functional source/test commit 뒤에는 Exp448 map triplet만 담은 child commit을 만든다.
  Reviewer는 직전 clean parent부터 functional+map child까지의 package를 read-only로 검토한다.

## Task 1: Governance freeze (완료)

**Files:** plan/preregistration/runbook/not-started report, catalog/state/champion ledger, and
versioned Exp448 map triplet. FAILURE_LEDGER에는 Exp448 entry를 추가하지 않는다.

**Result:** Exp448은 planned, labels 0, official false다. Exp447 immutable failure와
`pending_candidate`는 unchanged다. 아직 Exp448 product source edge는 없다.

## Task 2: Controlled-fork foundations only

**Owns exact files:**

- Create `config_antigravity.py`, `contracts_antigravity.py`, `artifacts_antigravity.py`,
  `bootstrap_antigravity.py`.
- Create `tests/test_config_antigravity.py`, `tests/test_contracts_antigravity.py`,
  `tests/test_artifacts_antigravity.py`, `tests/test_bootstrap_antigravity.py`,
  `tests/test_controlled_fork_differential_antigravity.py`.
- Modify the Exp448 map triplet only in the child map commit.

**Does not own:** `publication_identity`, `sources`, `readiness`, `evaluation`, `replay`, `cli`,
launcher or their runtime tests. Task2 must not claim a TEXT runtime/evaluator proof.

**Interfaces:** config exports fixed Exp448 identity/constants; contracts export exact-type immutable
objects; artifacts export canonical JSON/create-only helpers; bootstrap exports project F1/fixed PCG64
draw validation. Differential skeleton compares canonical profiles while ignoring identity keys,
named constants, and normalized AST for these four present paired modules only.

**RED tests and expected failure:**

```python
def test_profile_changes_only_allowed_identity_keys_antigravity():
    assert compare_present_profile_antigravity(exp447, exp448).equal

def test_confusion_rejects_bool_and_uses_project_f1_antigravity():
    with self.assertRaises(ValueError):
        ConfusionAntigravity(tp=True, fp=0, fn=0, tn=0)

def test_paired_ast_rejects_unallowlisted_constant_delta_antigravity():
    self.assertFalse(normalized_pair_equal_antigravity(exp447_source, changed_source))
```

RED exits with import errors because these modules/helpers are absent.

**GREEN:** port only Exp447 semantics for those four modules; substitute the approved Exp448
identity/root/audit names. Preserve bootstrap algorithm/seed/draw contract. Reject malformed canonical
JSON and pre-label label-like records. Do not import evaluator, SQLite, network, dashboard or CLI.

**Commands:**

```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_config_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_contracts_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_artifacts_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_bootstrap_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_controlled_fork_differential_antigravity
python3 -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
git diff --check
```

Commit `feat(exp448): add controlled fork foundations`; regenerate three map files and commit only
them as `docs(codemap): map Exp448 fork foundations`. Fresh Sol review scope: ownership allowlist,
exact constants/AST, and absence of runtime-evaluator claim.

## Task 3: Predecessor, sources, readiness, exact TEXT converter

**Owns exact files:** create `publication_identity_antigravity.py`, `sources_antigravity.py`,
`readiness_antigravity.py`; create `tests/test_publication_identity_antigravity.py`,
`tests/test_sources_antigravity.py`, `tests/test_readiness_antigravity.py`; extend only
`test_controlled_fork_differential_antigravity.py`; map child only changes the triplet.

**Interfaces:** publication identity captures descriptor-pinned Exp446/Exp404 bytes; sources verifies
Exp447 audit/absent root, supplies two independent Exp446 captures and closure-private label authority;
readiness verifies clean reviewed HEAD, document hashes, DB identity and absent Exp448 root without
importing a label session.

**RED:** audit SHA/inode/nlink/schema/phase mismatch or non-absent predecessor root fails before DB
access. Different capture bytes/order fails before label capability. Schema metadata mismatch fails.
Converter accepts only `('0',0)` and `('1',1)`; `0`, `True`, `0.0`, `b'0'`, `' 0'`, `'0 '`, `None`,
`'2'` fail. Extend differential AST proof for these three modules and allow only audit/root/metadata/
converter nodes.

**GREEN:** verify fixed Exp447 source byte guard first; authenticate audit/root with no-follow checks.
Schema inspection asserts only `instance_index INTEGER`, `label TEXT`, never values/counts. Convert
valid raw strings immediately. Do not add evaluation/replay/CLI imports.

**Commands:**

```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_publication_identity_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_sources_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_readiness_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_controlled_fork_differential_antigravity
python3 -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
shasum -a 256 experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/*_antigravity.py scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py
git diff --check
```

Commit `feat(exp448): add predecessor text boundary`, map child
`docs(codemap): map Exp448 source boundary`, then fresh Sol review: label reachability, metadata-only
schema check, rejection matrix and Exp447 byte guard.

## Task 4: Evaluator, replay, complete runtime differential

**Owns exact files:** create `evaluation_antigravity.py`, `replay_antigravity.py`,
`tests/test_evaluation_antigravity.py`, `tests/test_replay_antigravity.py`; extend
`test_controlled_fork_differential_antigravity.py`; map only in child commit.

**Interfaces:** evaluator receives converted integer snapshot through closure-private authority,
computes exact four methods/transitions/eleven leaves/three bootstrap comparisons; replay reads only
complete immutable artifacts. It preserves failure audit, label close, source recheck and create-only
semantics.

**RED:** synthetic 1,117-dataset TEXT snapshot fails because evaluator/replay are absent. Fail raw
TEXT reaching evaluator, wrong 1117/195900 counts, noncontiguous index, lane 30 != 26+4, wrong method
or leaf order, source drift, existing root, close failure, or replay SQLite import. Failure parity
requires audit phase/count/session/destination values.

**GREEN:** port Exp447 evaluator/replay exactly except approved Exp448 identity/audit and incoming
already-converted int labels. Complete the sole 1,117 synthetic runtime differential: Exp448 TEXT path
and Exp447 integer pure evaluator must produce identical science after identity/hash fields are removed.
It never uses real TEST labels.

**Commands:**

```bash
python3 -m unittest experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_evaluation_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_replay_antigravity experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_controlled_fork_differential_antigravity
python3 -m unittest discover -s experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests -p 'test_*.py'
python3 -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
git diff --check
```

Commit `feat(exp448): add evaluation replay parity`, map child
`docs(codemap): map Exp448 evaluation boundary`, fresh Sol review: full synthetic proof, failure parity,
label close/recheck, and raw-string non-escape.

## Task 5: Closed CLI, launcher, full verification, final Sol gate

**Owns exact files:** create `cli_antigravity.py`,
`scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py`,
`tests/test_cli_antigravity.py`; map only in child commit.

**RED:** parser accepts exactly `readiness --reviewed-commit`, `evaluate --reviewed-commit`,
and bare `inspect`; rejects `new`, `resume`, `retry`, inspect arguments, abbreviations and every override.
Readiness/inspect must prove no evaluator/label SQLite import. Dirty HEAD fails before lazy evaluator.

**GREEN:** closed dispatcher and launcher only set numerical thread environment before importing CLI.
No arbitrary path/hash/source/lane/threshold/budget input. Evaluator remains lazy for evaluate; tests
run no official command.

**Commands:**

```bash
python3 -m unittest discover -s experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests -p 'test_*.py'
python3 -m unittest discover -s experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity/tests -p 'test_*.py'
python3 -m unittest discover -s experiments/exp446_exp445_publication_materialization_repair_antigravity/tests -p 'test_*.py'
python3 -m unittest discover -s experiments/exp445_exp444_publication_source_authority_repair_antigravity/tests -p 'test_*.py'
git diff --check
```

Commit `feat(exp448): close label boundary lifecycle`, map child
`docs(codemap): map Exp448 closed lifecycle`. Package Task5 changes, then require fresh Sol final
`ship`; official preflight is forbidden before that verdict.

## Task 6: One-shot official truth closeout and conditional dashboard

Require clean candidate, fresh Sol ship, byte guard, audit/root/DB/Exp446/Exp404 preflight and absent
Exp448 root. Then exactly once:

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py evaluate --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py inspect
```

Any nonzero permanently stops Exp448; no retry/root reuse. All-success only permits independent truth
verification, ledger/report/map closeout and final read-only review. A clean owner-only site can then
consume only sanitized aggregate `exp448.public-evaluation.v1`; it never receives raw labels/indices/
dataset names/lists/per-dataset rows/hashes/paths/logs/DB/checkpoints/secrets. Dashboard failure cannot
alter science or authorize retry.
